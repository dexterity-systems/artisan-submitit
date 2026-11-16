"""Pure host-fixture validation tests; every Docker interaction is mocked."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.qualification import fixture_identity as subject

CONTAINER = "artisan-submitit-qualification-123-1"
CONTAINER_ID = "c" * 64
IMAGE_ID = "sha256:" + "d" * 64
COMMAND = [
    "/qualification/profile/.pixi/envs/controller/bin/python",
    "probe.py",
    "literal; value",
]
PATHS = ["/usr/bin/python3", "/etc/slurm/slurm.conf", "/etc/slurm/slurmdbd.conf"]


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    receipt = {
        "schema_version": 1,
        "image_id": IMAGE_ID,
        "user": "qualification",
        "uid": 1001,
        "gid": 1002,
        "node_python_sha256": "1" * 64,
        "slurm_conf_sha256": "2" * 64,
        "slurmdbd_conf_sha256": "3" * 64,
    }
    (tmp_path / "config/fixture.json").write_text(json.dumps(receipt))
    return tmp_path


def _receipt_digest(root: Path) -> str:
    return hashlib.sha256((root / "config/fixture.json").read_bytes()).hexdigest()


def _mock_docker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    image: str = IMAGE_ID,
    name: str = CONTAINER,
    container_id: str = CONTAINER_ID,
    sums: str | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    calls: list[list[str]] = []
    executions: list[list[str]] = []
    outputs = [
        json.dumps([{"Id": container_id, "Image": image, "Name": "/" + name}]),
        sums
        if sums is not None
        else "\n".join(
            f"{number * 64}  {path}" for number, path in zip("123", PATHS, strict=True)
        )
        + "\n",
    ]

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, outputs[len(calls) - 1])

    monkeypatch.setattr(subject.subprocess, "run", run)
    monkeypatch.setattr(
        subject.os, "execv", lambda _executable, argv: executions.append(argv)
    )
    return calls, executions


def _launch(
    root: Path, *, digest: str | None = None, container: str = CONTAINER
) -> None:
    subject.launch(
        COMMAND,
        root=root,
        container=container,
        receipt_sha256=digest or _receipt_digest(root),
        expected_container_id=CONTAINER_ID,
    )


def test_matching_fixture_pins_id_and_runs_only_as_unprivileged_user(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, executions = _mock_docker(monkeypatch)
    _launch(fixture_root)
    assert calls == [
        [*subject.DOCKER, "container", "inspect", CONTAINER],
        [
            *subject.DOCKER,
            "exec",
            "--user",
            "0",
            "--",
            CONTAINER_ID,
            "/usr/bin/sha256sum",
            "--",
            *PATHS,
        ],
    ]
    assert executions == [
        [
            *subject.DOCKER,
            "exec",
            "--user",
            "1001:1002",
            "--env",
            "ARTISAN_SLURM_FIXTURE_RECEIPT=/qualification-config/fixture.json",
            "--env",
            "ARTISAN_SLURM_FIXTURE_SHA256=" + _receipt_digest(fixture_root),
            "--",
            CONTAINER_ID,
            *COMMAND,
        ]
    ]


def test_changed_receipt_blocks_even_docker_inspection(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, executions = _mock_docker(monkeypatch)
    digest = _receipt_digest(fixture_root)
    path = fixture_root / "config/fixture.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        _launch(fixture_root, digest=digest)
    assert not calls
    assert not executions


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uid", 0),
        ("gid", True),
        ("uid", "1001"),
        ("schema_version", True),
        ("schema_version", 2),
        ("user", "root"),
        ("image_id", "image:mutable"),
        ("node_python_sha256", "bad"),
        ("extra", "unexpected"),
    ],
)
def test_invalid_receipt_never_runs_docker(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    calls, executions = _mock_docker(monkeypatch)
    path = fixture_root / "config/fixture.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=r"fixture|Fixture"):
        _launch(fixture_root)
    assert not calls
    assert not executions


@pytest.mark.parametrize(
    "changes", [{"image": "sha256:" + "e" * 64}, {"name": "other-container"}]
)
def test_container_or_image_drift_blocks_science(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, str]
) -> None:
    calls, executions = _mock_docker(monkeypatch, **changes)
    with pytest.raises(ValueError, match="container or immutable image changed"):
        _launch(fixture_root)
    assert len(calls) == 1
    assert not executions


@pytest.mark.parametrize("index", [0, 1, 2])
def test_node_python_or_config_drift_blocks_science(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    rows = [f"{number * 64}  {path}" for number, path in zip("123", PATHS, strict=True)]
    rows[index] = "f" * 64 + "  " + PATHS[index]
    calls, executions = _mock_docker(monkeypatch, sums="\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="Python or scheduler configuration changed"):
        _launch(fixture_root)
    assert len(calls) == 2
    assert not executions


@pytest.mark.parametrize("name", ["", "*", "name; rm", "--privileged", "/container"])
def test_nonliteral_container_name_blocks_docker(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls, executions = _mock_docker(monkeypatch)
    with pytest.raises(ValueError, match="literal container name"):
        _launch(fixture_root, container=name)
    assert not calls
    assert not executions


def test_same_name_and_image_cannot_replace_prepared_container(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, executions = _mock_docker(monkeypatch, container_id="e" * 64)
    with pytest.raises(ValueError, match="container or immutable image changed"):
        _launch(fixture_root)
    assert len(calls) == 1
    assert executions == []


@pytest.mark.parametrize("identifier", ["", "not-an-id", "e" * 63, "e" * 65])
def test_invalid_expected_container_id_fails_before_docker(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    calls, executions = _mock_docker(monkeypatch)
    with pytest.raises(ValueError, match="container ID is required"):
        subject.launch(
            COMMAND,
            root=fixture_root,
            container=CONTAINER,
            receipt_sha256=_receipt_digest(fixture_root),
            expected_container_id=identifier,
        )
    assert calls == executions == []
