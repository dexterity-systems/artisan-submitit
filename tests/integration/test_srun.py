"""Real-allocation integration tests for ``SlurmIntraRunner``."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import (
    CancellationStatus,
    PipelineManager,
    StepResult,
    StepStatus,
)

from artisan_submitit import SlurmIntraRunner
from artisan_submitit.cancellation import cancel_jobs
from artisan_submitit.srun import SlurmIntraLifecycleRouter

from .operations import ManagedWait, ObservableWait

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(120)]


def _pipeline(tmp_path: Path, name: str) -> PipelineManager:
    return PipelineManager.create(
        name=name,
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
    )


def test_srun_batched_pipeline_inside_allocation(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "srun_multi_batch")
    output = pipeline.output
    pipeline.run(DataGenerator, name="generate", params={"count": 4, "seed": 42})

    step = pipeline.run(
        DataTransformer,
        name="transform",
        inputs={"dataset": output("generate", "datasets")},
        params={"scale_factor": 0.5, "variants": 1, "seed": 100},
        step_runner=SlurmIntraRunner(srun_launch_concurrency=2),
        batch_strategy={"artifacts_per_unit": 1, "units_per_worker": 2},
        runner_resources={"cpus": 1, "memory_gb": 1, "time_limit": "00:02:00"},
    )
    summary = pipeline.finalize()

    assert summary["overall_success"] is True
    assert step.succeeded_count == 4
    assert step.failed_count == 0


@pytest.mark.parametrize("operation", [ObservableWait, ManagedWait])
def test_srun_cancel_terminates_inflight_process(
    tmp_path: Path, operation: type[ObservableWait], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path, "srun_cancel")
    markers = tmp_path / "markers"
    runner = SlurmIntraRunner(cancel_grace_period=1)
    routers = _observe_routers(runner, monkeypatch)
    future = pipeline.submit(
        operation,
        params={"duration": 60, "marker_root": str(markers)},
        step_runner=runner,
        runner_resources={"cpus": 1, "memory_gb": 1, "time_limit": "00:02:00"},
    )
    try:
        _wait_ready(markers, heartbeat=operation is ManagedWait)
        started = time.monotonic()
        pipeline.cancel()
        result = future.result(timeout=35)
        assert time.monotonic() - started < 32
        _assert_confirmed(tmp_path / "delta", result)
        if operation is ManagedWait:
            before = (markers / "heartbeat").read_text()
            time.sleep(0.2)
            assert (markers / "heartbeat").read_text() == before
    finally:
        _cleanup(pipeline, routers)


def test_srun_cancellation_does_not_stop_sibling_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled_pipeline = _pipeline(tmp_path / "cancelled", "srun_cancelled")
    sibling_pipeline = _pipeline(tmp_path / "sibling", "srun_sibling")
    cancelled_markers, sibling_markers = tmp_path / "first", tmp_path / "second"
    resources = {"cpus": 1, "memory_gb": 1, "time_limit": "00:02:00"}
    cancelled_runner, sibling_runner = SlurmIntraRunner(), SlurmIntraRunner()
    cancelled_routers = _observe_routers(cancelled_runner, monkeypatch)
    sibling_routers = _observe_routers(sibling_runner, monkeypatch)
    cancelled = cancelled_pipeline.submit(
        ObservableWait,
        name="same_name",
        params={"duration": 60, "marker_root": str(cancelled_markers)},
        step_runner=cancelled_runner,
        runner_resources=resources,
    )
    sibling = sibling_pipeline.submit(
        ManagedWait,
        name="same_name",
        params={"duration": 60, "marker_root": str(sibling_markers)},
        step_runner=sibling_runner,
        runner_resources=resources,
    )
    try:
        _wait_ready(cancelled_markers)
        _wait_ready(sibling_markers, heartbeat=True)
        cancelled_pipeline.cancel()
        _assert_confirmed(
            tmp_path / "cancelled" / "delta", cancelled.result(timeout=35)
        )
        before = (sibling_markers / "heartbeat").read_text()
        time.sleep(0.2)
        assert (sibling_markers / "heartbeat").read_text() != before
        (sibling_markers / "stop").touch()
        assert sibling.result(timeout=15).status is StepStatus.SUCCEEDED
    finally:
        sibling_markers.mkdir(parents=True, exist_ok=True)
        (sibling_markers / "stop").touch()
        _cleanup(cancelled_pipeline, cancelled_routers)
        _cleanup(sibling_pipeline, sibling_routers)


def _observe_routers(
    runner: SlurmIntraRunner, monkeypatch: pytest.MonkeyPatch
) -> list[SlurmIntraLifecycleRouter]:
    """Keep exact admitted launchers for cleanup after an assertion failure."""
    routers: list[SlurmIntraLifecycleRouter] = []
    create = runner.create_lifecycle_router

    def record(*args: Any, **kwargs: Any) -> SlurmIntraLifecycleRouter:
        router = create(*args, **kwargs)
        assert isinstance(router, SlurmIntraLifecycleRouter)
        routers.append(router)
        return router

    monkeypatch.setattr(runner, "create_lifecycle_router", record)
    return routers


def _cleanup(
    pipeline: PipelineManager, routers: list[SlurmIntraLifecycleRouter]
) -> None:
    """Cancel only this test's exact owned steps, including assertion failures."""
    pipeline.cancel()
    step_ids = [
        step_id
        for router in routers
        for item in router._submitted
        if (step_id := router._step_identity(item)) is not None
    ]
    cancel_jobs(step_ids, timeout=1)
    pipeline.finalize()


def _wait_ready(root: Path, *, heartbeat: bool = False) -> None:
    """Wait for the actual remote operation/command to publish startup."""
    deadline = time.monotonic() + 30
    while not (root / "ready").exists() or (
        heartbeat and not (root / "heartbeat").exists()
    ):
        assert time.monotonic() < deadline, (
            "srun worker did not report execution startup"
        )
        time.sleep(0.05)


def _assert_confirmed(delta_root: Path, result: StepResult) -> None:
    """Compare returned cancellation with the durable current step snapshot."""
    assert result.status is StepStatus.CANCELLED
    assert result.cancellation_status is CancellationStatus.CONFIRMED
    assert result.succeeded_count == 0
    assert not result.output_roles
    rows = (
        pl.read_delta(delta_root / "orchestration" / "steps")
        .filter(pl.col("step_run_id") == result.step_run_id)
        .sort("state_sequence")
    )
    assert rows["status"][-1] == "cancelled"
    assert rows["cancellation_status"][-1] == "confirmed"
    assert rows["logical_commit_id"].null_count() == rows.height
