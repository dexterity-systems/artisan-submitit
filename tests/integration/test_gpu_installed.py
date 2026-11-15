"""Actual GPU allocation, command activation and cancellation through public runners."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
from artisan.orchestration import (
    CancellationStatus,
    PipelineManager,
    StepDisposition,
    StepStatus,
)
from artisan.orchestration.engine.current_outputs import read_current_step_outputs
from artisan.schemas import MetricArtifact
from artisan.storage.core.retained_reader import AcceptedArtifact

from artisan_submitit import SlurmIntraRunner, SlurmRunner
from artisan_submitit.control import PackRequest, UnitEvidence
from artisan_submitit.gpu import GPUProbeSpec, ScheduledGPUProfile
from artisan_submitit.runner import SchedulerRunner
from artisan_submitit.software import SlurmSoftware

from .helpers import (
    APPLICATIONS,
    RESOURCES,
    assert_closed,
    pipeline,
    runner,
    wait_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(480)]
GPU = {"cpus": 1, "memory_gb": 2, "gpus": 1, "time_limit": "00:05:00"}
BATCH = {"artifacts_per_unit": 1, "units_per_worker": 2, "max_workers": 1}


def _mode() -> Literal["sbatch", "srun"]:
    value = os.environ["ARTISAN_SLURM_MODE"]
    assert value in {"sbatch", "srun"}
    return cast(Literal["sbatch", "srun"], value)


def _runner(root: Path) -> SlurmRunner | SlurmIntraRunner:
    return replace(
        runner(root, _mode()),
        parameters={"slurm_nodelist": os.environ["SLURM_TEST_NODE"]},
    )


def _outputs(owner: PipelineManager, step_id: str) -> tuple[AcceptedArtifact, ...]:
    (role,) = read_current_step_outputs(owner.committer, step_id)
    return role.occurrences


def _packs(root: Path) -> dict[Path, PackRequest]:
    return {
        path.parent: PackRequest.model_validate_json(path.read_bytes())
        for path in root.glob("*/pack.json")
    }


def _fresh_probe(root: Path, before: set[Path]) -> None:
    fresh = {
        path: request for path, request in _packs(root).items() if path not in before
    }
    assert len(fresh) == 1
    ((path, request),) = fresh.items()
    assert request.probe is not None
    assert (path / "closed").is_file()


@pytest.fixture(autouse=True)
def forbid_controller_gpu_and_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    def controller_capture(count: int) -> None:
        assert count == 0, "Controller attempted a GPU measurement"

    def forbidden(*_args: object, **_kwargs: object) -> None:
        msg = "Execution attempted login GPU measurement or installation"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "artisan.orchestration.runners.local.capture_native_gpu", controller_capture
    )
    monkeypatch.setattr("artisan_submitit.gpu.capture_native_gpu", forbidden)
    monkeypatch.setattr(
        "artisan.orchestration.runners.local.prepare_native_environment", forbidden
    )


@pytest.mark.parametrize("kind", ["native", "command"])
def test_gpu_kernel_activation_and_current_probe_cache(
    tmp_path: Path, kind: str
) -> None:
    from slurm_qualification_app import (
        CommandGPUIncrement,
        GPUMerge,
        NativeGPUIncrement,
    )

    selected = _runner(tmp_path)
    definition = NativeGPUIncrement if kind == "native" else CommandGPUIncrement
    params = (
        {} if kind == "native" else {"manifest": os.environ["ARTISAN_SLURM_MANIFEST"]}
    )
    inputs = [
        MetricArtifact.draft({"value": value}, f"input-{value}.json").finalize()
        for value in (41, 43)
    ]
    with pipeline(tmp_path, selected) as owner:
        arguments = {
            "inputs": {"source": inputs},
            "params": params,
            "runner_resources": GPU,
            "batch_strategy": BATCH,
        }
        first = owner.run(definition, **arguments)
        assert first.status == StepStatus.SUCCEEDED, first.error
        assert first.step_run_id is not None
        outputs = _outputs(owner, first.step_run_id)
        assert len(outputs) == 2
        handles = owner.executions[-1].handles
        assert len(handles) == 2
        assert handles[0].worker_owner is handles[1].worker_owner
        assert all(isinstance(item.artifact, MetricArtifact) for item in outputs)
        values = [cast(MetricArtifact, item.artifact).values for item in outputs]
        assert [value["input_value"] for value in values] == [41, 43]
        assert len({value["preprocess_pid"] for value in values}) == 1
        for value in values:
            assert value["status"] == "passed"
            assert value["output"] == 42
            assert value["device_count"] == 1
            assert value["cuda_visible_devices"].startswith("GPU-")
            allocation_field = (
                "slurm_step_gpus" if _mode() == "srun" else "slurm_job_gpus"
            )
            assert value[allocation_field] is not None
            assert value["pythonpath"] is None
            assert value["preprocess_pid"] == value["postprocess_pid"]
            if kind == "native":
                assert value["worker_pid"] == value["preprocess_pid"]
            else:
                assert value["tool_pid"] != value["preprocess_pid"]
                assert value["packaging"] == "26.3"
                assert value["activation"] == "prepared-native-gpu"
                assert value["literal"] is None
                assert value["selected"] is None
                assert value["loader_path"].endswith("/command-runtime/gpu-driver")
                assert value["executable"].endswith("/.pixi/envs/gpu-tool/bin/python")
                assert "execute" in Path(value["cwd"]).parts
                assert len(value["driver_payloads"]) == 2
        before = set(_packs(selected.log_folder))
        cached = owner.run(definition, **arguments)
        assert cached.status == StepStatus.SUCCEEDED, cached.error
        assert cached.disposition == StepDisposition.CACHE_HIT
        assert not owner.executions[-1].handles
        _fresh_probe(selected.log_folder, before)
        assert cached.step_run_id is not None
        reused = _outputs(owner, cached.step_run_id)
        assert [(item.reference, item.artifact.artifact_id) for item in reused] == [
            (item.reference, item.artifact.artifact_id) for item in outputs
        ]
        if kind == "native":
            curator = owner.run(
                GPUMerge,
                inputs={"source": first.output("metrics")},
                runner_resources=GPU,
            )
            assert curator.status == StepStatus.SUCCEEDED, curator.error
            assert curator.metadata["gpu_kernel"]["output"] == 42
            before = set(_packs(selected.log_folder))
            repeated = owner.run(
                GPUMerge,
                inputs={"source": first.output("metrics")},
                runner_resources=GPU,
            )
            assert repeated.disposition == StepDisposition.CACHE_HIT, repeated.error
            _fresh_probe(selected.log_folder, before)
    assert_closed(owner)
    requests = _packs(selected.log_folder)
    assert requests
    assert all((path / "closed").is_file() for path in requests)
    for request in requests.values():
        assert all(
            value["cuda_visible_devices"] not in request.model_dump_json()
            for value in values
        )
    if kind == "command":
        scientific = next(
            request
            for request in requests.values()
            if request.gpu is not None and request.probe is None
        )
        expected = {item.sha256 for item in scientific.gpu.software.libraries}
        assert all(
            set(value["driver_payloads"].values()) == expected for value in values
        )
    (tmp_path / "gpu-result.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "kernels": values,
                "fresh_probe_on_cache": True,
                "all_closed": True,
            },
            indent=2,
        )
        + "\n"
    )


def test_gpu_changed_profile_rejects_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slurm_qualification_app import NativeGPUIncrement

    original = SchedulerRunner._measure_gpu
    observed: list[ScheduledGPUProfile] = []

    def changed(
        self: SchedulerRunner,
        probe: GPUProbeSpec,
        software: SlurmSoftware,
        *,
        curator: bool,
    ) -> ScheduledGPUProfile:
        actual = original(self, probe, software, curator=curator)
        observed.append(actual)
        values = actual.model_dump()
        values["software"]["driver"]["driver_version"] += "-qualification-stale"
        return ScheduledGPUProfile.model_validate(values)

    monkeypatch.setattr(SchedulerRunner, "_measure_gpu", changed)
    selected = _runner(tmp_path)
    with PipelineManager.create(
        "gpu-stale-profile",
        str(tmp_path / "delta"),
        str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        default_step_runner=selected,
        application_modules=APPLICATIONS,
        preserve_working=True,
    ) as owner:
        result = owner.run(
            NativeGPUIncrement,
            inputs={
                "source": [MetricArtifact.draft({"value": 41}, "input.json").finalize()]
            },
            runner_resources=GPU,
        )
        assert result.status == StepStatus.FAILED
        assert "GPU software changed" in result.error
        assert result.step_run_id is not None
        assert read_current_step_outputs(owner.committer, result.step_run_id) == ()
    assert_closed(owner)
    assert len(observed) == 1
    assert not tuple((tmp_path / "working").rglob("source.json"))
    scientific = [
        (path, request)
        for path, request in _packs(selected.log_folder).items()
        if request.probe is None
    ]
    assert len(scientific) == 1
    path, request = scientific[0]
    (unit_path,) = (path / "entries").glob("*/unit-0.json")
    evidence = UnitEvidence.model_validate_json(unit_path.read_bytes())
    assert evidence.ownership == "closed"
    assert "GPU software changed" in evidence.reply
    assert (path / "closed").is_file()
    (tmp_path / "stale-result.json").write_text(
        json.dumps({"pack_id": request.pack_id, "closed": True, "no_science": True})
        + "\n"
    )


def test_gpu_worker_cancel_preserves_independent_cpu_worker(tmp_path: Path) -> None:
    from slurm_qualification_app import CommandGPUIncrement, ManagedWait

    first_root, second_root = tmp_path / "gpu", tmp_path / "cpu"
    marker, heartbeat = tmp_path / "gpu-pids.json", tmp_path / "cpu-marker"
    first, second = (
        pipeline(first_root, _runner(first_root)),
        pipeline(second_root, _runner(second_root)),
    )
    with first, second:
        cancelled = first.submit(
            CommandGPUIncrement,
            inputs={
                "source": [MetricArtifact.draft({"value": 1}, "input.json").finalize()]
            },
            params={
                "manifest": os.environ["ARTISAN_SLURM_MANIFEST"],
                "hold": 120,
                "marker": str(marker),
            },
            runner_resources=GPU,
        )
        deadline = time.monotonic() + 90
        # The tool creates the marker before its JSON write finishes.
        while True:
            assert time.monotonic() < deadline, "First GPU tool did not start"
            try:
                pids = json.loads(marker.read_bytes())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            break
        assert set(pids) == {"parent", "child"}
        for pid in pids.values():
            assert Path(f"/proc/{pid}").is_dir()
            assert os.readlink(f"/proc/{pid}/ns/pid") == os.readlink(
                "/proc/self/ns/pid"
            )
        survivor = second.submit(
            ManagedWait,
            params={"marker_root": str(heartbeat), "duration": 120},
            runner_resources=RESOURCES,
        )
        try:
            wait_ready(heartbeat)
            first.cancel()
            result = cancelled.result(timeout=35)
            assert result.status == StepStatus.CANCELLED, result.error
            assert result.cancellation_status == CancellationStatus.CONFIRMED
            for pid in pids.values():
                assert not Path(f"/proc/{pid}").exists()
            before = (heartbeat / "heartbeat").read_bytes()
            time.sleep(0.15)
            assert (heartbeat / "heartbeat").read_bytes() != before
        finally:
            heartbeat.mkdir(parents=True, exist_ok=True)
            (heartbeat / "stop").touch()
        survived = survivor.result(timeout=60)
        assert survived.status == StepStatus.SUCCEEDED, survived.error
    assert_closed(first)
    assert_closed(second)


def test_gpu_unit_cancel_and_typed_literal_injections(tmp_path: Path) -> None:
    from ..qualification.gpu_cancellation import qualify

    literal = "literal:$(touch should-not-exist)"
    selected = os.environ["ARTISAN_GPU_SELECTED_TEST"]
    with pipeline(tmp_path, _runner(tmp_path)) as owner:
        proof = qualify(
            owner,
            tmp_path / "gpu-pids.json",
            manifest=Path(os.environ["ARTISAN_SLURM_MANIFEST"]),
            literal=literal,
            selected=selected,
        )
    assert_closed(owner)
    assert proof["ownership"] == ["closed"]
    assert not tuple(tmp_path.rglob("should-not-exist"))
    (tmp_path / "gpu-unit-cancel.json").write_text(json.dumps(proof, indent=2) + "\n")
