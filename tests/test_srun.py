"""An intra-allocation pack owns one numeric step and one synchronous client."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from artisan_submitit.control import EntryIdentity
from artisan_submitit.srun import _SrunPack


def _pack(tmp_path: Path) -> _SrunPack:
    pack = object.__new__(_SrunPack)
    pack.directory = tmp_path / ("a" * 32)
    pack.directory.mkdir()
    pack.allocation = "41"
    pack.arguments = ["--cpu-bind=cores"]
    pack.step_id = None
    pack.process = None
    pack._submission_started = False
    pack.assignments = ()
    pack.software = SimpleNamespace(
        gpu=None,
        scheduler=SimpleNamespace(
            executable=lambda _name: Path("/usr/bin/srun"),
            executables=(SimpleNamespace(path=Path("/usr/bin/srun")),),
            bootstrap_shell=SimpleNamespace(path=Path("/bin/bash")),
        ),
        local=SimpleNamespace(
            worker=SimpleNamespace(prefix=tmp_path / "worker"),
            system_env=Path("/usr/bin/env"),
        ),
        bootstrap_script=tmp_path / "worker/bootstrap.py",
        resources=SimpleNamespace(time_limit="00:10:00", cpus=2, memory_gb=4, gpus=0),
    )
    return pack


def test_command_preserves_exact_allocation_and_single_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    popen = Mock(return_value=Mock(poll=lambda: None))
    monkeypatch.setattr("artisan_submitit.srun.subprocess.Popen", popen)
    request = SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64)
    pack.submit(request)
    command = popen.call_args.args[0]
    assert command[:5] == [
        "/usr/bin/srun",
        "--jobid=41",
        "--nodes=1",
        "--ntasks=1",
        "--exclusive",
    ]
    assert "--cpus-per-task=2" in command
    assert "--mem=4G" in command
    assert "--gres=none" in command
    assert "--gpus=0" in command
    assert command[command.index("--pack-sha256") + 1] == request.digest
    assert str(pack.software.bootstrap_script) in command
    assert popen.call_args.kwargs["start_new_session"]
    assert pack.scheduler_id() is None
    assert not pack.transport_closed()


@pytest.mark.parametrize("step", ["41", "42.0"])
def test_unknown_or_foreign_step_never_becomes_cancellation_target(
    tmp_path: Path, step: str
) -> None:
    pack = _pack(tmp_path)
    identity = EntryIdentity(
        pack_id="a" * 32, entry_id="b" * 32, job_id="41", step_id=step
    )
    with pytest.raises(ValueError, match="exact srun"):
        pack.observe_identity(identity)
    assert pack.scheduler_id() is None


def test_cancellation_target_is_only_the_validated_step(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    identity = EntryIdentity(
        pack_id="a" * 32, entry_id="b" * 32, job_id="41", step_id="41.3"
    )
    pack.observe_identity(identity)
    assert pack.scheduler_id() == "41.3"
    with pytest.raises(ValueError, match="multiple step"):
        pack.observe_identity(identity.model_copy(update={"step_id": "41.4"}))
    assert pack.scheduler_id() == "41.3"


def test_missing_step_cleanup_reaps_only_owned_client_without_canceling_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys
    import time

    pack = _pack(tmp_path)
    pack._unknown = True
    monkeypatch.setattr(pack, "_publish_requests", lambda: None)
    cancel = Mock()
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", cancel)
    client = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    sibling = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    pack.process = client
    try:
        pack._cleanup_only(time.monotonic() + 2)
        assert client.poll() is not None
        assert sibling.poll() is None
        assert pack.scheduler_id() is None
        assert pack._unknown
        cancel.assert_not_called()
    finally:
        for process in (client, sibling):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def test_gpu_probe_owns_exact_step_on_concrete_allocation_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artisan_submitit.gpu import GPUProbeSpec

    pack = _pack(tmp_path)
    pack.assignments = GPUProbeSpec(node="gpu-node", device_count=1)
    popen = Mock(return_value=Mock(poll=lambda: None))
    monkeypatch.setattr("artisan_submitit.srun.subprocess.Popen", popen)
    pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    command = popen.call_args.args[0]
    assert "--gpus=1" in command
    assert not any(part.startswith("--gres=") for part in command)
    assert "--nodelist=gpu-node" in command
    assert "--jobid=41" in command


@pytest.mark.parametrize("gpus", [0, 1])
def test_declared_gpu_count_overrides_inherited_allocation_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpus: int
) -> None:
    monkeypatch.setenv("SLURM_GPUS", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    before = dict(os.environ)
    pack = _pack(tmp_path)
    pack.software.resources.gpus = gpus
    pack.software.gpu = SimpleNamespace(node="gpu-node") if gpus else None
    popen = Mock(return_value=Mock(poll=lambda: None))
    monkeypatch.setattr("artisan_submitit.srun.subprocess.Popen", popen)
    pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    command = popen.call_args.args[0]
    assert [part for part in command if part.startswith("--gpus=")] == [
        f"--gpus={gpus}"
    ]
    assert "--jobid=41" in command
    assert ("--gres=none" in command) is (gpus == 0)
    assert ("--nodelist=gpu-node" in command) is (gpus == 1)
    environment = popen.call_args.kwargs.get("env") or os.environ
    assert environment["SLURM_JOB_ID"] == "41"
    assert environment["SLURM_JOB_GPUS"] == "0"
    assert dict(os.environ) == before
