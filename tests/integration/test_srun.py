"""Public pipeline and exact worker cancellation inside an existing allocation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import StepStatus

from .helpers import (
    RESOURCES,
    assert_closed,
    cancelled_worker_keeps_sibling,
    pipeline,
    runner,
)

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(600)]


def test_srun_batched_pipeline_inside_allocation(tmp_path: Path) -> None:
    with pipeline(tmp_path, runner(tmp_path, "srun")) as owner:
        generated = owner.run(
            DataGenerator, params={"count": 4, "seed": 42}, runner_resources=RESOURCES
        )
        assert generated.status == StepStatus.SUCCEEDED, generated.error
        result = owner.run(
            DataTransformer,
            inputs={"dataset": generated.output("datasets")},
            params={"variants": 1, "seed": 100},
            runner_resources=RESOURCES,
            batch_strategy={
                "artifacts_per_unit": 1,
                "units_per_worker": 2,
                "max_workers": 2,
            },
        )
        assert result.status == StepStatus.SUCCEEDED, result.error
        assert result.succeeded_count == 4
        handles = owner.executions[-1].handles
        assert len(handles) == 4
        assert sorted(
            Counter(id(handle.worker_owner) for handle in handles).values()
        ) == [
            2,
            2,
        ]
    assert_closed(owner)


def test_srun_cancellation_does_not_stop_sibling_worker(tmp_path: Path) -> None:
    cancelled_worker_keeps_sibling(tmp_path, "srun")
