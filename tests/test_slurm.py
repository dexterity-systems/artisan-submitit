"""The pinned SDK binds its actual outer pickle before emitting sbatch."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import submitit
from submitit.core.utils import CommandFunction

from artisan_submitit.control import ReadyRecord
from artisan_submitit.slurm import _SubmititPack


def _pack(tmp_path: Path) -> _SubmititPack:
    pack = object.__new__(_SubmititPack)
    pack.directory = tmp_path / ("a" * 32)
    pack.directory.mkdir()
    pack.parameters = {"slurm_partition": "cpu"}
    pack._stop = threading.Event()
    pack._submission_started = False
    pack._job_id = None
    pack.handles = ()
    pack.assignments = ()
    pack.software = SimpleNamespace(
        gpu=None,
        scheduler=Mock(
            executables=(SimpleNamespace(path=Path("/usr/bin/srun")),),
            bootstrap_shell=SimpleNamespace(path=Path("/bin/bash")),
        ),
        local=SimpleNamespace(
            worker=SimpleNamespace(prefix=tmp_path / "worker"),
            system_env=Path("/usr/bin/env"),
        ),
        bootstrap_script=tmp_path / "worker/bootstrap.py",
        resources=SimpleNamespace(time_limit="00:10:30", cpus=3, memory_gb=7, gpus=0),
    )
    return pack


def _scheduler(
    monkeypatch: pytest.MonkeyPatch, callback: Callable[[], None] | None = None
) -> list[str]:
    scripts: list[str] = []
    monkeypatch.setattr(
        "submitit.slurm.slurm.shutil.which", lambda _name: "/usr/bin/srun"
    )

    def submit(command: CommandFunction) -> str:
        assert command.command[0] == "sbatch"
        scripts.append(Path(command.command[1]).read_text())
        if callback is not None:
            callback()
        return "Submitted batch job 41"

    monkeypatch.setattr(CommandFunction, "__call__", submit)
    return scripts


def test_real_pinned_sdk_binds_original_bytes_before_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    scripts = _scheduler(monkeypatch)
    request = SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64)
    pack.submit(request)
    assert len(scripts) == 1
    script = scripts[0]
    encoded = (pack.directory / "outer.bin").read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    assert "--outer-sha256 " + digest in script
    assert "ARTISAN_CAPTURED_SUBMITIT_SHA256" not in script
    assert "--pack-sha256 " + request.digest in script
    assert str(pack.software.bootstrap_script) in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=7GB" in script
    ready = ReadyRecord.model_validate_json(
        (pack.directory / "ready.json").read_bytes()
    )
    assert ready.outer_sha256 == digest
    assert ready.job_id == "41"
    sdk_pickle = pack.directory / "sdk/41_submitted.pkl"
    assert sdk_pickle.read_bytes() == encoded
    sdk_pickle.write_bytes(b"SDK checkpoint")
    assert (pack.directory / "outer.bin").read_bytes() == encoded


def test_sdk_layout_change_fails_before_scheduler_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    scripts = _scheduler(monkeypatch)
    sdk = pack.directory / "sdk"
    sdk.mkdir()
    (sdk / "unexpected.pkl").write_bytes(b"another payload")
    with pytest.raises(ValueError, match="pre-submission pickle layout"):
        pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    assert not scripts
    assert not pack._submission_started
    assert not (pack.directory / "ready.json").exists()


def test_cancel_during_sdk_promotion_is_published_before_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    scripts = _scheduler(monkeypatch, pack._stop.set)
    pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    assert len(scripts) == 1
    assert (pack.directory / "cancel").exists()
    assert (pack.directory / "ready.json").exists()


def test_cancel_before_submit_never_calls_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    scripts = _scheduler(monkeypatch)
    pack._stop.set()
    with pytest.raises(CancelledError):
        pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    assert not scripts
    assert not pack._submission_started


def test_effective_sdk_topology_rejects_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path)
    scripts = _scheduler(monkeypatch)
    original = submitit.AutoExecutor.update_parameters

    def change(executor: submitit.AutoExecutor, **kwargs: object) -> None:
        original(executor, **kwargs)
        executor.parameters["ntasks_per_node"] = 2

    monkeypatch.setattr(submitit.AutoExecutor, "update_parameters", change)
    with pytest.raises(ValueError, match="single-task"):
        pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    assert not scripts


def test_real_sdk_gpu_probe_requests_exact_node_and_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artisan_submitit.gpu import GPUProbeSpec

    pack = _pack(tmp_path)
    pack.assignments = GPUProbeSpec(node="gpu-node", device_count=1)
    scripts = _scheduler(monkeypatch)
    pack.submit(SimpleNamespace(pack_id=pack.directory.name, digest="b" * 64))
    assert "#SBATCH --gpus-per-node=1" in scripts[0]
    assert "#SBATCH --nodelist=gpu-node" in scripts[0]
    assert "#SBATCH --cpus-per-task=1" in scripts[0]
