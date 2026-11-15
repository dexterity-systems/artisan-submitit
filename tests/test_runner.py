"""The provider reuses qualified local resolution and explicit application roots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import (
    LocalRunner,
    PreparedLocalBindings,
    RunnerResources,
    merge_runner_resources,
)

from artisan_submitit.runner import _validate_parameters
from artisan_submitit.slurm import SlurmRunner


def _runner() -> SlurmRunner:
    runner = object.__new__(SlurmRunner)
    object.__setattr__(runner, "parameters", {})
    object.__setattr__(runner, "scheduler_bin", Path("/usr/bin"))
    return runner


@pytest.mark.parametrize("curator", [False, True])
def test_resolution_preserves_measured_application_selection(
    monkeypatch: pytest.MonkeyPatch, curator: bool
) -> None:
    local = Mock(spec=PreparedLocalBindings)
    local.worker_is_observed = False
    local.native_gpu = None
    local.application_modules = ("example_application",)
    resolve = Mock(return_value=local)
    method = "resolve_curator" if curator else "resolve"
    monkeypatch.setattr(LocalRunner, method, resolve)
    scheduler = Mock()
    monkeypatch.setattr(
        "artisan_submitit.runner.SchedulerSoftware.capture", lambda *_args: scheduler
    )
    operation = Mock()
    resources = RunnerResources(cpus=2, memory_gb=3, gpus=0)
    software = getattr(_runner(), method)(
        operation, resources, application_modules=local.application_modules
    )
    assert resolve.call_args.kwargs == {
        "application_modules": local.application_modules
    }
    assert software.local is local
    assert software.scheduler is scheduler
    assert software.application_modules == local.application_modules
    assert software.resources == resources


@pytest.mark.parametrize(
    "key",
    [
        "max_array_size",
        "slurm_array_parallelism",
        "slurm_python",
        "slurm_setup",
        "nodes",
        "cpus_per_task",
    ],
)
def test_owned_scheduler_options_reject_explicitly(key: str) -> None:
    with pytest.raises(ValueError, match=r"owned|Unsupported"):
        _validate_parameters({key: 2})


def test_capture_rejects_application_mismatch_before_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from artisan_submitit.calls import capture_pack

    serialize = Mock()
    monkeypatch.setattr("artisan_submitit.calls.serialize_worker_call", serialize)
    with pytest.raises(ValueError, match="Application selection"):
        capture_pack(
            (),
            SimpleNamespace(application_modules=("measured",)),
            SimpleNamespace(application_modules=("other",)),
            tmp_path,
        )
    serialize.assert_not_called()


@pytest.mark.parametrize("curator", [False, True])
@pytest.mark.parametrize("override", [None, RunnerResources(memory_gb=96)])
def test_preparation_keeps_authored_defaults_without_login_node_gpu_selection(
    monkeypatch: pytest.MonkeyPatch,
    curator: bool,
    override: RunnerResources | None,
) -> None:
    method = "prepare_curator" if curator else "prepare"
    prepare = Mock()
    monkeypatch.setattr(LocalRunner, method, prepare)
    subject = Mock()
    before = None if override is None else override.model_dump()
    getattr(_runner(), method)(
        subject,
        runner_resources=override,
        application_modules=("example_application",),
    )
    defaults = RunnerResources(
        cpus=8,
        memory_gb=64,
        gpus=2,
        time_limit="02:00:00",
        extra={"slurm_partition": "gpu"},
    )
    effective = merge_runner_resources(
        defaults, prepare.call_args.kwargs["runner_resources"]
    )
    assert effective == RunnerResources(
        cpus=8,
        memory_gb=64 if override is None else 96,
        gpus=0,
        time_limit="02:00:00",
        extra={},
    )
    assert prepare.call_args.args == (subject,)
    assert prepare.call_args.kwargs["application_modules"] == ("example_application",)
    assert (None if override is None else override.model_dump()) == before
    assert defaults.gpus == 2
    assert defaults.extra == {"slurm_partition": "gpu"}
