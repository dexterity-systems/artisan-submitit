"""Real-scheduler integration tests for ``SlurmRunner``."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import polars as pl
import pytest
import submitit
from artisan.operations.examples import DataGenerator, DataTransformer, Wait
from artisan.orchestration import PipelineManager
from artisan.schemas.enums import FailurePolicy

from artisan_submitit import SlurmRunner

from .operations import FailingWait

pytestmark = [pytest.mark.integration, pytest.mark.slurm]

PARTITION = os.environ.get("SLURM_TEST_PARTITION", "cpu")


def _pipeline(tmp_path: Path, name: str) -> PipelineManager:
    return PipelineManager.create(
        name=name,
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
    )


def _sleep_and_return(seconds: float) -> str:
    time.sleep(seconds)
    return "unrelated job completed"


def test_multi_batch_arrays_cross_scheduler_limit(tmp_path: Path) -> None:
    """Five units packed by two become three jobs across three arrays."""
    runner = SlurmRunner(slurm_partition=PARTITION, max_array_size=1)
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
    assert "artisan-submitit worker log marker" in failed["worker_log"][0]


def test_submission_failure_becomes_aligned_step_failure(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "slurm_submission_failure")
    step = pipeline.run(
        DataGenerator,
        params={"count": 1, "seed": 42},
        step_runner=SlurmRunner(slurm_partition="not-a-real-partition"),
    )
    summary = pipeline.finalize()

    assert summary["overall_success"] is False
    assert step.failed_count == 1


def test_cancel_does_not_cancel_unrelated_same_named_job(tmp_path: Path) -> None:
    """Provider cancellation uses its handles, never a broad job-name query."""
    unrelated = submitit.AutoExecutor(folder=tmp_path / "unrelated" / "%j")
    unrelated.update_parameters(
        slurm_partition=PARTITION,
        timeout_min=2,
        cpus_per_task=1,
        mem_gb=1,
        slurm_job_name="s0_wait",
    )
    unrelated_job = unrelated.submit(_sleep_and_return, 2.0)

    pipeline = _pipeline(tmp_path / "pipeline", "cancel_isolation")

    def cancel_pipeline() -> None:
        time.sleep(1.0)
        pipeline.cancel()

    threading.Thread(target=cancel_pipeline, daemon=True).start()
    started = time.monotonic()
    pipeline.submit(
        Wait,
        name="wait",
        params={"duration": 30},
        step_runner=SlurmRunner(slurm_partition=PARTITION),
    )
    pipeline.finalize()

    assert time.monotonic() - started < 20
    assert unrelated_job.result() == "unrelated job completed"
