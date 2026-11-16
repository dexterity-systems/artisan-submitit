"""Exercise the workflow's scheduler lifecycle without contacting Docker."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

CONTAINER_ID = "a" * 64
REPLACEMENT_ID = "b" * 64
FAKE_DOCKER = r"""#!/usr/bin/python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
assert arguments[:2] == ["--context", "default"], arguments
arguments = arguments[2:]
state_path = Path(os.environ["FAKE_DOCKER_STATE"])
state = json.loads(state_path.read_text())
container = os.environ["QUALIFICATION_CONTAINER"]
current = state["id"]
if arguments == ["container", "ls", "--all", "--no-trunc", "--filter", f"name=^/{container}$", "--format", "{{.ID}}"]:
    action = "list"
elif arguments == ["container", "inspect", container, "--format", "{{.Id}}"]:
    action = "inspect"
else:
    assert arguments[:5] == ["compose", "-p", container, "-f", "docker/qualification-compose.yml"], arguments
    command = arguments[5:]
    if command == ["create", "--no-build", "--pull", "never"]:
        assert current is None
        action = "create"
        current = state["id"] = "a" * 64
    elif command == ["up", "--no-recreate", "--no-build", "--pull", "never", "--wait", "--wait-timeout", "180"]:
        action = "up"
        bound = os.environ["QUALIFICATION_CONTAINER_ID"]
        assert bound == "a" * 64
        assert f"QUALIFICATION_CONTAINER_ID={bound}\n" in Path(os.environ["GITHUB_ENV"]).read_text()
    elif command == ["down", "--volumes"]:
        action = "down"
        state["id"] = None
    else:
        raise AssertionError(arguments)
with Path(os.environ["FAKE_DOCKER_EVENTS"]).open("a") as stream:
    stream.write(f"{action}:{current or 'absent'}\n")
state_path.write_text(json.dumps(state))
if action in {"list", "inspect"}:
    if current is not None:
        print(current)
    elif action == "inspect":
        sys.exit(1)
if action == "up" and os.environ.get("FAKE_HEALTH_FAIL") == "1":
    sys.exit(1)
"""


@dataclass
class SchedulerFixture:
    """Private files and environment used by the three workflow steps."""

    environment: dict[str, str]
    scripts: dict[str, str]
    state_path: Path
    event_path: Path
    github_env: Path

    def run(self, name: str) -> subprocess.CompletedProcess[str]:
        """Run one Bash body and apply its GitHub environment file."""
        result = subprocess.run(
            ["/bin/bash", "-c", self.scripts[name]],
            cwd=self.state_path.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        for line in self.github_env.read_text().splitlines():
            key, value = line.split("=", 1)
            self.environment[key] = value
        return result


@pytest.fixture
def scheduler(tmp_path: Path) -> SchedulerFixture:
    """Use only the fake executable on PATH, with no Docker fallback."""
    workflow_path = Path(__file__).parents[1] / ".github/workflows/slurm-docker.yaml"
    workflow = yaml.safe_load(workflow_path.read_text())
    scripts = {
        step["name"]: step["run"]
        for step in workflow["jobs"]["slurm"]["steps"]
        if "run" in step
    }
    executable = tmp_path / "bin/docker"
    executable.parent.mkdir()
    executable.write_text(FAKE_DOCKER)
    executable.chmod(0o755)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"id": None}))
    event_path = tmp_path / "events"
    github_env = tmp_path / "github-env"
    github_env.touch()
    environment = {
        "PATH": str(executable.parent),
        "QUALIFICATION_CONTAINER": "artisan-slurm-test",
        "GITHUB_ENV": str(github_env),
        "FAKE_DOCKER_STATE": str(state_path),
        "FAKE_DOCKER_EVENTS": str(event_path),
    }
    return SchedulerFixture(environment, scripts, state_path, event_path, github_env)


def test_scheduler_binds_before_start_and_removes_owned_container(
    scheduler: SchedulerFixture,
) -> None:
    assert scheduler.run("Create isolated scheduler").returncode == 0
    assert scheduler.environment["QUALIFICATION_CONTAINER_ID"] == CONTAINER_ID
    assert scheduler.run("Start isolated scheduler").returncode == 0
    assert scheduler.run("Stop isolated scheduler").returncode == 0
    assert json.loads(scheduler.state_path.read_text())["id"] is None
    assert scheduler.event_path.read_text().splitlines() == [
        "list:absent",
        f"create:{CONTAINER_ID}",
        f"inspect:{CONTAINER_ID}",
        f"inspect:{CONTAINER_ID}",
        f"up:{CONTAINER_ID}",
        f"list:{CONTAINER_ID}",
        f"down:{CONTAINER_ID}",
    ]


def test_scheduler_failed_health_preserves_binding_for_cleanup(
    scheduler: SchedulerFixture,
) -> None:
    assert scheduler.run("Create isolated scheduler").returncode == 0
    scheduler.environment["FAKE_HEALTH_FAIL"] = "1"
    assert scheduler.run("Start isolated scheduler").returncode != 0
    assert scheduler.environment["QUALIFICATION_CONTAINER_ID"] == CONTAINER_ID
    assert json.loads(scheduler.state_path.read_text())["id"] == CONTAINER_ID
    assert scheduler.run("Stop isolated scheduler").returncode == 0
    assert json.loads(scheduler.state_path.read_text())["id"] is None
    assert scheduler.event_path.read_text().splitlines()[-1] == f"down:{CONTAINER_ID}"


def test_scheduler_cleanup_succeeds_without_binding_or_existing_container(
    scheduler: SchedulerFixture,
) -> None:
    assert scheduler.run("Stop isolated scheduler").returncode == 0
    assert not scheduler.event_path.exists()
    assert scheduler.run("Create isolated scheduler").returncode == 0
    scheduler.state_path.write_text(json.dumps({"id": None}))
    assert scheduler.run("Stop isolated scheduler").returncode == 0
    assert scheduler.event_path.read_text().splitlines()[-1] == "down:absent"


def test_scheduler_refuses_start_and_cleanup_for_same_name_replacement(
    scheduler: SchedulerFixture,
) -> None:
    assert scheduler.run("Create isolated scheduler").returncode == 0
    scheduler.state_path.write_text(json.dumps({"id": REPLACEMENT_ID}))
    assert scheduler.run("Start isolated scheduler").returncode != 0
    cleanup = scheduler.run("Stop isolated scheduler")
    assert cleanup.returncode != 0
    assert "Refusing to stop a replacement qualification container" in cleanup.stderr
    assert json.loads(scheduler.state_path.read_text())["id"] == REPLACEMENT_ID
    events = scheduler.event_path.read_text().splitlines()
    assert not any(event.startswith(("up:", "down:")) for event in events)
