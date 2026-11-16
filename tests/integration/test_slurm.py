"""Public pipeline behavior through actual installed Submitit scheduler workers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from artisan.inspection import inspect_evidence
from artisan.inspection.models import LogsResponse
from artisan.operations.curator import Merge
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import PipelineManager, StepDisposition, StepStatus
from artisan.orchestration.engine.current_outputs import read_current_step_outputs
from artisan.visualization import inspect_failures, inspect_step

from .helpers import (
    APPLICATIONS,
    RESOURCES,
    assert_closed,
    cancelled_worker_keeps_sibling,
    pipeline,
    runner,
)

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(600)]


def test_ordered_packs_curator_and_cache(tmp_path: Path) -> None:
    selected = runner(tmp_path, "sbatch")
    with pipeline(tmp_path, selected) as owner:
        generated = owner.run(
            DataGenerator, params={"count": 5, "seed": 42}, runner_resources=RESOURCES
        )
        assert generated.status == StepStatus.SUCCEEDED, generated.error
        arguments = {
            "inputs": {"dataset": generated.output("datasets")},
            "params": {"scale_factor": 2.0, "variants": 1, "seed": 100},
            "runner_resources": RESOURCES,
            "batch_strategy": {
                "artifacts_per_unit": 1,
                "units_per_worker": 2,
                "max_workers": 2,
            },
        }
        transformed = owner.run(DataTransformer, **arguments)
        assert transformed.status == StepStatus.SUCCEEDED, transformed.error
        assert transformed.succeeded_count == 5
        handles = owner.executions[-1].handles
        assert len(handles) == 5
        assert sorted(
            Counter(id(handle.worker_owner) for handle in handles).values()
        ) == [
            1,
            2,
            2,
        ]
        cached = owner.run(DataTransformer, **arguments)
        assert cached.disposition == StepDisposition.CACHE_HIT, cached.error
        assert not owner.executions[-1].handles
        merged = owner.run(
            Merge,
            inputs={
                "first": transformed.output("dataset"),
                "second": cached.output("dataset"),
            },
            runner_resources=RESOURCES,
        )
        assert merged.status == StepStatus.SUCCEEDED, merged.error
        frames = [
            inspect_step(
                str(tmp_path / "delta"),
                step.step_number,
                pipeline_run_id=owner.config.pipeline_run_id,
            )
            for step in (transformed, cached)
        ]
        assert frames[0].drop("step").equals(frames[1].drop("step"))
        assert transformed.step_run_id is not None
        assert cached.step_run_id is not None
        originals, reused = (
            read_current_step_outputs(owner.committer, step_id)
            for step_id in (transformed.step_run_id, cached.step_run_id)
        )
        assert tuple(role.role for role in originals) == tuple(
            role.role for role in reused
        )
        for first_role, cached_role in zip(originals, reused, strict=True):
            assert tuple(
                (item.reference, item.artifact.artifact_id, item.artifact.artifact_type)
                for item in first_role.occurrences
            ) == tuple(
                (item.reference, item.artifact.artifact_id, item.artifact.artifact_type)
                for item in cached_role.occurrences
            )
    assert_closed(owner)


def test_configured_runner_resume_appends(tmp_path: Path) -> None:
    selected = runner(tmp_path, "sbatch")
    with pipeline(tmp_path, selected) as first:
        source = first.run(
            DataGenerator, params={"count": 1}, runner_resources=RESOURCES
        )
        assert source.status == StepStatus.SUCCEEDED, source.error
    with PipelineManager.resume(
        str(tmp_path / "delta"),
        str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        pipeline_run_id=first.config.pipeline_run_id,
        default_step_runner=selected,
        application_modules=APPLICATIONS,
    ) as resumed:
        cached = resumed.run(
            DataGenerator, params={"count": 1}, runner_resources=RESOURCES
        )
        assert cached.step_number == source.step_number + 1
        assert cached.disposition == StepDisposition.CACHE_HIT, cached.error
    assert_closed(first)
    assert_closed(resumed)


def test_closed_scientific_failure_keeps_error_and_log(tmp_path: Path) -> None:
    from slurm_qualification_app import ObservableWait

    with pipeline(tmp_path, runner(tmp_path, "sbatch")) as owner:
        result = owner.run(
            ObservableWait,
            params={"marker_root": str(tmp_path / "marker"), "fail": True},
            runner_resources=RESOURCES,
        )
        assert result.status == StepStatus.FAILED, result
    assert_closed(owner)
    failures = inspect_failures(
        str(tmp_path / "delta"), pipeline_run_id=owner.config.pipeline_run_id
    )
    assert failures.height == 1
    assert "intentional scheduler operation failure" in failures["error"][0]
    assert failures["log"][0] is not None
    log = inspect_evidence(
        "get_step_logs",
        {
            "pipeline_run_id": owner.config.pipeline_run_id,
            "step_name": result.step_name,
            "tail_lines": 200,
        },
        delta_root=str(tmp_path / "delta"),
    )
    assert isinstance(log, LogsResponse)
    assert "intentional scheduler operation failure" in "\n".join(log.lines)


def test_running_cancel_preserves_independent_worker(tmp_path: Path) -> None:
    cancelled_worker_keeps_sibling(tmp_path, "sbatch")
