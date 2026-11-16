"""Reject changed scheduler evidence before any SDK payload is loaded."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from artisan.orchestration.runner_api import LocalWorkerProfile, PreparedLocalBindings

from artisan_submitit import bootstrap
from artisan_submitit.control import PackRequest, ReadyRecord, write_once
from artisan_submitit.software import (
    SchedulerSoftware,
    SlurmSoftware,
    verify_submitit_plugins,
)


def _request() -> PackRequest:
    return cast(PackRequest, SimpleNamespace(pack_id="a" * 32, digest="b" * 64))


@pytest.mark.parametrize(
    "change",
    [
        {"SLURM_JOB_ID": ""},
        {"SLURM_JOB_ID": "41_2"},
        {"SLURM_ARRAY_JOB_ID": "40"},
        {"SLURM_PROCID": "1"},
        {"SLURM_NTASKS": "2"},
    ],
)
def test_job_identity_rejects_unowned_topology(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, str]
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    for key, value in change.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="requires"):
        bootstrap._job_identity()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "42"),
        ("pack_id", "c" * 32),
        ("pack_digest", "c" * 64),
        ("outer_sha256", "c" * 64),
    ],
)
def test_outer_capture_rejects_changed_binding(
    tmp_path: Path, field: str, value: str
) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    payload = b"exact SDK payload"
    (sdk / "41_submitted.pkl").write_bytes(payload)
    (tmp_path / "outer.bin").write_bytes(payload)
    ready = ReadyRecord(
        pack_id="a" * 32,
        pack_digest="b" * 64,
        job_id="41",
        outer_sha256=hashlib.sha256(payload).hexdigest(),
    )
    changed = ready.model_dump()
    changed[field] = value
    write_once(tmp_path / "ready.json", json.dumps(changed).encode())
    with pytest.raises(ValueError, match="differs"):
        bootstrap._capture_outer(
            tmp_path, sdk, "41", _request(), hashlib.sha256(payload).hexdigest()
        )


def test_outer_capture_returns_only_the_verified_bytes(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    path = sdk / "41_submitted.pkl"
    path.write_bytes(b"captured")
    (tmp_path / "outer.bin").write_bytes(b"captured")
    ready = ReadyRecord(
        pack_id="a" * 32,
        pack_digest="b" * 64,
        job_id="41",
        outer_sha256=hashlib.sha256(b"captured").hexdigest(),
    )
    write_once(tmp_path / "ready.json", ready.model_dump_json().encode())
    captured = bootstrap._capture_outer(
        tmp_path, sdk, "41", _request(), hashlib.sha256(b"captured").hexdigest()
    )
    path.write_bytes(b"changed SDK checkpoint")
    assert captured == b"captured"
    assert (
        bootstrap._capture_outer(
            tmp_path, sdk, "41", _request(), hashlib.sha256(b"captured").hexdigest()
        )
        == b"captured"
    )


def test_bootstrap_checks_runtime_before_provider_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pack.json").write_text(json.dumps({"profile": "opaque"}))
    calls = []

    def reject_profile() -> None:
        calls.append("profile")
        raise ValueError("changed prefix")

    profile = SimpleNamespace(verify=reject_profile)
    monkeypatch.setattr(LocalWorkerProfile, "from_json", lambda _encoded: profile)
    monkeypatch.setattr(
        "artisan_submitit.control.decode_pack",
        lambda *_args: pytest.fail(
            "Provider request parsed before runtime verification"
        ),
    )
    with pytest.raises(ValueError, match="changed prefix"):
        bootstrap._verified_request(
            tmp_path, hashlib.sha256((tmp_path / "pack.json").read_bytes()).hexdigest()
        )
    assert calls == ["profile"]


def test_sdk_plugins_are_rejected_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Plugin:
        def load(self) -> None:
            pytest.fail("An unqualified plugin was imported")

    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda **_kwargs: (Plugin(),)
    )
    with pytest.raises(ValueError, match="entry-point"):
        verify_submitit_plugins()


def test_scheduler_client_drift_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_kwargs: ())
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    for name in ("sacct", "sbatch", "scancel", "scontrol", "srun"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nprintf 'slurm 22.05.8\\n'\n")
        path.chmod(0o755)
    identity = SchedulerSoftware.capture("sbatch", tmp_path)
    identity.verify()
    (tmp_path / "srun").write_text(
        "#!/bin/sh\nprintf 'slurm 22.05.8\\n'\n# changed bytes\n"
    )
    with pytest.raises(ValueError, match="changed"):
        identity.verify()


def test_ready_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        ReadyRecord.model_validate(
            {
                "pack_id": "a" * 32,
                "pack_digest": "b" * 64,
                "job_id": "41",
                "outer_sha256": "c" * 64,
                "callable": "untrusted.module",
            }
        )


def test_whole_pack_swap_rejects_before_profile_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b'{"profile":"original"}'
    (tmp_path / "pack.json").write_bytes(b'{"profile":"changed"}')
    monkeypatch.setattr(
        LocalWorkerProfile,
        "from_json",
        lambda _encoded: pytest.fail("Changed profile was executed"),
    )
    with pytest.raises(ValueError, match="exact launch binding"):
        bootstrap._verified_request(tmp_path, hashlib.sha256(original).hexdigest())


@pytest.mark.parametrize(
    "arguments",
    [
        ["--async"],
        ["--jobid=99"],
        ["--export=ALL"],
        ["--multi-prog"],
        ["--chdir=/tmp"],
        ["--mpi=external"],
        ["--container-image=unqualified"],
        ["--cpu-b=cores"],
        ["python", "unowned.py"],
        ["--"],
        [":"],
        ["--label=1"],
        ["--cpu-bind"],
        ["--cpu-bind", "--async"],
        ["--cpu-bind=cores", "unowned.py"],
    ],
)
def test_srun_arguments_cannot_replace_owned_launch(arguments: list[str]) -> None:
    from artisan_submitit.utils import validate_srun_arguments

    with pytest.raises(ValueError, match="srun_args"):
        validate_srun_arguments(arguments)


def test_srun_affinity_and_logging_arguments_remain_literal() -> None:
    from artisan_submitit.utils import validate_srun_arguments

    arguments = ["--cpu-bind=cores", "--hint", "nomultithread", "--label"]
    assert validate_srun_arguments(arguments) == tuple(arguments)


@pytest.fixture
def prepared_software(tmp_path: Path) -> SlurmSoftware:
    from unittest.mock import Mock

    from artisan.orchestration.runner_api import PreparedNativeEnvironment

    worker = Mock(spec=PreparedNativeEnvironment)
    worker.prefix = tmp_path / "worker"
    script = (
        worker.prefix / "lib/python3.12/site-packages/artisan_submitit/bootstrap.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# installed provider bootstrap\n")
    local = SimpleNamespace(
        worker_is_observed=False,
        native_gpu=None,
        worker=worker,
        compatibility=SimpleNamespace(
            python_version="3.12.12",
            packages=(SimpleNamespace(name="artisan-submitit"),),
        ),
    )
    return SlurmSoftware(cast(PreparedLocalBindings, local), Mock(), "{}")


def test_bootstrap_reuses_installed_path_without_reading_payload(
    prepared_software: SlurmSoftware, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        prepared_software.local.worker.prefix
        / "lib/python3.12/site-packages/artisan_submitit/bootstrap.py"
    )
    script.write_text("# manual installed-file edits are not audited\n")
    monkeypatch.setattr(
        Path, "open", lambda *_args, **_kwargs: pytest.fail("Installed payload read")
    )
    assert prepared_software.bootstrap_script == script


def test_bootstrap_requires_provider_package_metadata(
    prepared_software: SlurmSoftware,
) -> None:
    cast(SimpleNamespace, prepared_software.local.compatibility).packages = ()
    with pytest.raises(ValueError, match="provider package"):
        _ = prepared_software.bootstrap_script


@pytest.mark.parametrize("replacement", ["missing", "outside"])
def test_bootstrap_requires_an_available_in_prefix_file(
    prepared_software: SlurmSoftware, tmp_path: Path, replacement: str
) -> None:
    script = prepared_software.bootstrap_script
    script.unlink()
    if replacement == "outside":
        outside = tmp_path / "outside.py"
        outside.write_text("# outside prepared installation\n")
        script.symlink_to(outside)
    with pytest.raises(ValueError, match="in-prefix provider bootstrap"):
        _ = prepared_software.bootstrap_script


def test_replacing_both_outer_payload_and_adjacent_receipt_still_rejects(
    tmp_path: Path,
) -> None:
    expected = hashlib.sha256(b"controller SDK capture").hexdigest()
    replacement = b"different outer payload"
    (tmp_path / "outer.bin").write_bytes(replacement)
    ready = ReadyRecord(
        pack_id="a" * 32,
        pack_digest="b" * 64,
        job_id="41",
        outer_sha256=hashlib.sha256(replacement).hexdigest(),
    )
    (tmp_path / "ready.json").write_text(ready.model_dump_json())
    with pytest.raises(ValueError, match="exact controller capture"):
        bootstrap._capture_outer(tmp_path, tmp_path / "sdk", "41", _request(), expected)


@pytest.mark.parametrize("name", ["SLURM_CONF", "SLURM_CONF_SERVER", "SLURM_CLUSTERS"])
def test_unqualified_scheduler_endpoint_override_rejects(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "other-cluster")
    with pytest.raises(ValueError, match="endpoint overrides"):
        verify_submitit_plugins()
