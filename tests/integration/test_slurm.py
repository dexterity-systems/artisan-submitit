"""Real-scheduler integration tests for ``SlurmRunner``."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import submitit
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import (
    CancellationStatus,
    PipelineManager,
    StepResult,
    StepStatus,
)
from artisan.schemas.enums import FailurePolicy
from artisan.visualization import inspect_worker_log

from artisan_submitit import SlurmRunner
from artisan_submitit.slurm import SlurmLifecycleRouter

from .operations import FailingWait, ManagedWait, ObservableWait

pytestmark = [pytest.mark.integration, pytest.mark.slurm]

PARTITION = os.environ.get("SLURM_TEST_PARTITION", "cpu")


def _pipeline(tmp_path: Path, name: str) -> PipelineManager:
    return PipelineManager.create(
        name=name,
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
    )


def _unrelated_heartbeat(heartbeat: Path, stop: Path) -> str:
    """Expose remote liveness until the test explicitly releases this job."""
    deadline = time.monotonic() + 120
    while not stop.exists() and time.monotonic() < deadline:
        with heartbeat.open("a") as stream:
            stream.write("alive\n")
        time.sleep(0.05)
    return "unrelated job completed"


def _wait_for(predicate: Callable[[], bool], description: str) -> None:
    """Bound readiness waits so cancellation tests cannot pass before admission."""
    deadline = time.monotonic() + 45
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {description}")
        time.sleep(0.05)


def _observe_routers(
    runner: SlurmRunner, monkeypatch: pytest.MonkeyPatch
) -> list[SlurmLifecycleRouter]:
    """Retain actual routers for exact cleanup without replacing transport."""
    routers: list[SlurmLifecycleRouter] = []
    create = runner.create_lifecycle_router

    def record(*args: Any, **kwargs: Any) -> SlurmLifecycleRouter:
        router = create(*args, **kwargs)
        routers.append(router)
        return router

    monkeypatch.setattr(runner, "create_lifecycle_router", record)
    return routers


def _cancel_exact_jobs(job_ids: list[str]) -> None:
    """Clean up only handles admitted by this test, including failure paths."""
    if job_ids:
        subprocess.run(
            ["scancel", *job_ids], capture_output=True, check=False, timeout=5
        )


def _assert_cancellation(
    delta_root: Path,
    result: StepResult,
    status: StepStatus = StepStatus.CANCELLED,
    cancellation: CancellationStatus = CancellationStatus.CONFIRMED,
) -> None:
    """Check returned and persisted cancellation without accepting outputs."""
    assert result.status is status, result
    assert result.cancellation_status is cancellation, result
    assert result.succeeded_count == 0
    rows = (
        pl.read_delta(delta_root / "orchestration" / "steps")
        .filter(pl.col("step_run_id") == result.step_run_id)
        .sort("state_sequence")
    )
    terminal = rows.row(-1, named=True)
    assert terminal["status"] == status.value
    assert terminal["cancellation_status"] == cancellation.value
    assert rows["logical_commit_id"].null_count() == rows.height


def _accounting(job_id: str) -> dict[str, str] | None:
    """Read fresh independent scheduler state for the exact top-level job."""
    completed = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            job_id,
            "--format=JobID%64,State%40,Submit,Start,End",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
        env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
    )
    for line in completed.stdout.splitlines():
        identifier, state, submitted, start, end = line.split("|")[:5]
        if identifier == job_id:
            return {"state": state, "submit": submitted, "start": start, "end": end}
    return None


def _is_pending(job_id: str) -> bool:
    """Observe held admission even when accounting has not recorded it yet."""
    completed = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return completed.stdout.strip() == "PENDING"


@pytest.mark.parametrize("max_array_size", [1, 2])
def test_multi_batch_arrays_cross_scheduler_limit(
    tmp_path: Path, max_array_size: int
) -> None:
    """Five units packed by two cross the limit, including a real task array."""
    runner = SlurmRunner(slurm_partition=PARTITION, max_array_size=max_array_size)
    pipeline = _pipeline(tmp_path, "slurm_multi_batch")
    output = pipeline.output
    pipeline.run(DataGenerator, name="generate", params={"count": 5, "seed": 42})

    step = pipeline.run(
        DataTransformer,
        name="transform",
        inputs={"dataset": output("generate", "datasets")},
        params={"scale_factor": 0.5, "variants": 1, "seed": 100},
        step_runner=runner,
        batch_strategy={"artifacts_per_unit": 1, "units_per_worker": 2},
        runner_resources={"extra": {"slurm_partition": PARTITION}},
    )
    summary = pipeline.finalize()

    assert summary["overall_success"] is True
    assert step.succeeded_count == 5
    assert step.failed_count == 0


def test_custom_runner_can_be_pipeline_default_resume_and_cache(
    tmp_path: Path,
) -> None:
    runner = SlurmRunner(slurm_partition=PARTITION)
    first_pipeline = PipelineManager.create(
        name="slurm_default",
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        default_step_runner=runner,
    )
    first = first_pipeline.run(DataGenerator, params={"count": 2, "seed": 42})
    first_pipeline.finalize()
    run_id = first_pipeline.config.pipeline_run_id
    executions_path = tmp_path / "delta" / "orchestration" / "executions"
    first_execution_count = pl.read_delta(executions_path).height

    resumed_pipeline = PipelineManager.resume(
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        pipeline_run_id=run_id,
        default_step_runner=runner,
    )
    resumed = resumed_pipeline.run(
        DataGenerator,
        name="generate_after_resume",
        params={"count": 1, "seed": 43},
    )
    resumed_pipeline.finalize()

    second_pipeline = PipelineManager.create(
        name="slurm_default",
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        default_step_runner=runner,
    )
    second = second_pipeline.run(DataGenerator, params={"count": 2, "seed": 42})
    summary = second_pipeline.finalize()

    assert summary["overall_success"] is True
    assert first.succeeded_count == 1
    assert resumed.succeeded_count == 1
    assert second.succeeded_count == 1
    assert pl.read_delta(executions_path).height == first_execution_count + 1


def test_operation_failure_and_worker_log_are_preserved(tmp_path: Path) -> None:
    pipeline = PipelineManager.create(
        name="slurm_operation_failure",
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        failure_policy=FailurePolicy.CONTINUE,
    )
    step = pipeline.run(
        FailingWait,
        params={"duration": 0},
        step_runner=SlurmRunner(slurm_partition=PARTITION),
    )
    summary = pipeline.finalize()

    records = pl.read_delta(tmp_path / "delta" / "orchestration" / "executions")
    failed = records.filter(pl.col("origin_step_number") == 0)
    assert summary["overall_success"] is False
    assert step.failed_count == 1
    assert "intentional scheduler operation failure" in failed["error"][0]
    worker_log = inspect_worker_log(
        str(tmp_path / "delta"), failed["execution_run_id"][0]
    )
    assert worker_log is not None
    assert "artisan-submitit worker log marker" in worker_log


def test_submission_failure_preserves_unresolved_admission(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "slurm_submission_failure")
    step = pipeline.run(
        DataGenerator,
        params={"count": 1, "seed": 42},
        step_runner=SlurmRunner(slurm_partition="not-a-real-partition"),
    )
    summary = pipeline.finalize()

    assert summary["overall_success"] is False
    _assert_cancellation(
        tmp_path / "delta", step, StepStatus.FAILED, CancellationStatus.UNKNOWN
    )


@pytest.mark.parametrize("operation", [ObservableWait, ManagedWait])
@pytest.mark.timeout(120)
def test_running_cancel_preserves_unrelated_same_named_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: type[ObservableWait] | type[ManagedWait],
) -> None:
    """Confirm real worker/command cleanup while an unrelated job stays alive."""
    heartbeat = tmp_path / "unrelated-heartbeat"
    stop = tmp_path / "unrelated-stop"
    unrelated = submitit.AutoExecutor(folder=tmp_path / "unrelated" / "%j")
    unrelated.update_parameters(
        slurm_partition=PARTITION,
        timeout_min=5,
        cpus_per_task=1,
        mem_gb=1,
        slurm_job_name="artisan-cancel-isolation",
    )
    unrelated_job = unrelated.submit(_unrelated_heartbeat, heartbeat, stop)
    pipeline = _pipeline(tmp_path / "pipeline", "cancel_isolation")
    runner = SlurmRunner(
        slurm_partition=PARTITION, slurm_job_name="artisan-cancel-isolation"
    )
    routers = _observe_routers(runner, monkeypatch)
    marker = tmp_path / "owned-marker"
    try:
        _wait_for(
            lambda: heartbeat.exists() and heartbeat.stat().st_size > 0,
            "unrelated job execution",
        )
        future = pipeline.submit(
            operation,
            name="wait",
            params={"duration": 60, "marker_root": str(marker)},
            step_runner=runner,
            runner_resources={"cpus": 1, "memory_gb": 1, "time_limit": "00:05:00"},
        )
        _wait_for(
            lambda: (marker / "ready").exists()
            and (operation is ObservableWait or (marker / "heartbeat").exists()),
            "owned worker execution",
        )
        started = time.monotonic()
        pipeline.cancel()
        result = future.result(timeout=40)
        assert time.monotonic() - started < 35
        pipeline.finalize()
        _assert_cancellation(tmp_path / "pipeline" / "delta", result)
        assert routers
        assert all(router.is_done() for router in routers)
        assert all(
            router._thread is None or not router._thread.is_alive()
            for router in routers
        )
        before = heartbeat.stat().st_size
        time.sleep(0.2)
        assert heartbeat.stat().st_size > before
        if operation is ManagedWait:
            settled = (marker / "heartbeat").read_bytes()
            time.sleep(0.2)
            assert (marker / "heartbeat").read_bytes() == settled
        stop.touch()
        _wait_for(unrelated_job.done, "unrelated job natural completion")
        assert unrelated_job.result() == "unrelated job completed"
    finally:
        stop.touch()
        try:
            pipeline.cancel()
            pipeline.finalize()
        finally:
            _cancel_exact_jobs(
                [
                    str(unrelated_job.job_id),
                    *[
                        job_id
                        for router in routers
                        for job_id in router.submitted_job_ids
                    ],
                ]
            )


@pytest.mark.timeout(120)
def test_queued_cancel_has_non_start_accounting_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    """Qualify non-start proof, preserving uncertainty when it is unavailable."""
    pipeline = _pipeline(tmp_path, "queued_cancellation")
    runner = SlurmRunner(
        slurm_partition=PARTITION, slurm_additional_parameters={"hold": True}
    )
    routers = _observe_routers(runner, monkeypatch)
    marker = tmp_path / "never-started"
    submitted_after = datetime.now(UTC).replace(microsecond=0)
    try:
        future = pipeline.submit(
            ObservableWait,
            params={"duration": 60, "marker_root": str(marker)},
            step_runner=runner,
        )
        _wait_for(
            lambda: bool(routers and routers[0].submitted_job_ids),
            "held job submission",
        )
        (job_id,) = routers[0].submitted_job_ids
        _wait_for(
            lambda: _is_pending(job_id),
            "held job pending in the scheduler",
        )
        assert not (marker / "ready").exists()
        pipeline.cancel()
        result = future.result(timeout=40)
        pipeline.finalize()
        _wait_for(
            lambda: (_accounting(job_id) or {})
            .get("state", "")
            .startswith("CANCELLED"),
            "terminal cancellation accounting",
        )
        state = _accounting(job_id)
        assert state is not None
        try:
            submitted = datetime.fromisoformat(state["submit"]).replace(tzinfo=UTC)
            ended = datetime.fromisoformat(state["end"]).replace(tzinfo=UTC)
            qualified = (
                state["start"] == "Unknown" and submitted_after <= submitted <= ended
            )
        except ValueError:
            qualified = False
        record_testsuite_property("queued_accounting", str(state))
        record_testsuite_property("queued_non_start_proof", qualified)
        _assert_cancellation(
            tmp_path / "delta",
            result,
            StepStatus.CANCELLED if qualified else StepStatus.FAILED,
            CancellationStatus.CONFIRMED if qualified else CancellationStatus.UNKNOWN,
        )
        assert not (marker / "ready").exists()
        assert routers[0]._thread is None or not routers[0]._thread.is_alive()
    finally:
        try:
            pipeline.cancel()
            pipeline.finalize()
        finally:
            _cancel_exact_jobs(
                [job_id for router in routers for job_id in router.submitted_job_ids]
            )
