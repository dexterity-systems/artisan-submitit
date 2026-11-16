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
from artisan_submitit.srun import SlurmIntraRunner


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


@pytest.mark.parametrize("curator", [False, True])
@pytest.mark.parametrize("step_extra", [False, True])
def test_cpu_intra_dispatch_preserves_declared_node_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, curator: bool, step_extra: bool
) -> None:
    runner = object.__new__(SlurmIntraRunner)
    selection = {"slurm_nodelist": "selected-node"}
    object.__setattr__(runner, "parameters", {} if step_extra else selection)
    object.__setattr__(runner, "scheduler_bin", Path("/usr/bin"))
    object.__setattr__(runner, "log_folder", tmp_path)
    local = Mock(spec=PreparedLocalBindings)
    local.worker_is_observed = False
    local.native_gpu = None
    method = "resolve_curator" if curator else "resolve"
    monkeypatch.setattr(LocalRunner, method, Mock(return_value=local))
    capture = Mock()
    monkeypatch.setattr("artisan_submitit.runner.SchedulerSoftware.capture", capture)
    resources = RunnerResources(gpus=0, extra=selection if step_extra else {})
    software = getattr(runner, method)(Mock(), resources)
    owner = Mock()
    monkeypatch.setattr("artisan_submitit.srun._SrunPack", owner)
    monkeypatch.setenv("SLURM_JOB_ID", "41")

    runner.create_owner(Mock() if curator else (), software, Mock())

    assert owner.call_args.args[-1] == ["--nodelist=selected-node"]
    assert runner.parameters == ({} if step_extra else selection)
    assert software.resources == resources


@pytest.mark.parametrize("curator", [False, True])
def test_gpu_intra_preparation_keeps_node_selection_without_probing(
    monkeypatch: pytest.MonkeyPatch, curator: bool
) -> None:
    runner = object.__new__(SlurmIntraRunner)
    object.__setattr__(runner, "parameters", {"slurm_nodelist": "gpu-node"})
    object.__setattr__(runner, "scheduler_bin", Path("/usr/bin"))
    local = Mock(spec=PreparedLocalBindings)
    local.worker_is_observed = False
    local.native_gpu = None
    resolve_method = "resolve_curator" if curator else "resolve"
    prepare_method = "prepare_curator" if curator else "prepare"
    monkeypatch.setattr(LocalRunner, resolve_method, Mock(return_value=local))
    monkeypatch.setattr("artisan_submitit.runner.SchedulerSoftware.capture", Mock())
    probe = Mock()
    monkeypatch.setattr(SlurmIntraRunner, "_measure_gpu", probe)

    def prepare(self: SlurmIntraRunner, subject: object, **options: object) -> None:
        getattr(self, resolve_method)(subject, options["runner_resources"])

    monkeypatch.setattr(LocalRunner, prepare_method, prepare)
    getattr(runner, prepare_method)(Mock(), runner_resources=RunnerResources(gpus=2))

    probe.assert_not_called()
    assert runner.parameters == {"slurm_nodelist": "gpu-node"}
