"""Real-allocation integration tests for ``SlurmIntraRunner``."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from artisan.operations.examples import DataGenerator, DataTransformer, Wait
from artisan.orchestration import PipelineManager

from artisan_submitit import SlurmIntraRunner

pytestmark = (pytest.mark.integration, pytest.mark.slurm)


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


def test_srun_cancel_terminates_inflight_process(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "srun_cancel")

    def cancel_pipeline() -> None:
        time.sleep(1.0)
        pipeline.cancel()

    threading.Thread(target=cancel_pipeline, daemon=True).start()
    started = time.monotonic()
    pipeline.submit(
        Wait,
        params={"duration": 30},
        step_runner=SlurmIntraRunner(cancel_grace_period=1),
        runner_resources={"cpus": 1, "memory_gb": 1, "time_limit": "00:02:00"},
    )
    pipeline.finalize()

    assert time.monotonic() - started < 20
