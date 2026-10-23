"""Provider cancellation evidence survives actual pipeline publication."""

from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import cloudpickle
import polars as pl
import pytest
from artisan.operations.examples import DataGenerator
from artisan.orchestration import PipelineManager
from artisan.orchestration.run_status import run_status
from artisan.orchestration.runner_api import (
    BatchStrategy,
    CancellationAcknowledgement,
    CancellationStatus,
    ExecutionUnit,
    LifecycleRouter,
    RunnerBase,
    RunnerResources,
    RuntimeEnvironment,
    execute_unit_batch,
    failure_results_for_units,
)
from artisan.schemas.enums import TablePath
from artisan.schemas.orchestration.step_lifecycle import StepStatus
from artisan.visualization import inspect_step

from artisan_submitit import SlurmIntraRunner, SlurmRunner
from artisan_submitit import srun as srun_module
from artisan_submitit.cancellation import SchedulerObservation
from artisan_submitit.slurm import SlurmLifecycleRouter
from artisan_submitit.utils import write_atomic
from artisan_submitit.worker import WorkerOutcome


def _publish_outcome(
    pipeline: PipelineManager,
    mode: str,
    units: list[ExecutionUnit],
    runtime: RuntimeEnvironment,
    folder: Path,
) -> None:
    """Provide transport evidence while keeping core execution/persistence real."""
    if mode in {"rejected", "missing", "nested_unknown"}:
        results = execute_unit_batch(units, runtime)
        assert all(result.success for result in results), results
        assert all(result.execution_run_ids for result in results)
        assert list(Path(runtime.staging_root).rglob("executions.parquet"))
    else:
        results = failure_results_for_units(units, "test worker cancelled")
    acknowledgement = (
        None
        if mode == "rejected"
        else CancellationAcknowledgement(CancellationStatus.CONFIRMED)
    )
    if mode == "nested_unknown":
        results = [
            replace(
                result,
                cancellation_acknowledgement=CancellationAcknowledgement(
                    CancellationStatus.UNKNOWN, "Nested remote closure is unknown"
                ),
            )
            for result in results
        ]
    pipeline.cancel()
    if mode != "missing":
        write_atomic(
            folder / "outcome.pkl",
            pickle.dumps(WorkerOutcome(results, acknowledgement)),
        )


def _runner_with_transport_double(
    provider: str,
    mode: str,
    pipeline: PipelineManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RunnerBase, LifecycleRouter]:
    """Replace scheduler transport while exercising each actual router."""
    if provider == "slurm":
        executor = MagicMock()
        job = MagicMock(job_id="123_0")
        job.paths = SimpleNamespace(
            stdout=tmp_path / "stdout", stderr=tmp_path / "stderr"
        )
        job.paths.stdout.write_text("transport worker output")
        job.paths.stderr.touch()

        def submit(
            _call: Any,
            units: list[ExecutionUnit],
            runtime: RuntimeEnvironment,
            folder: Path,
        ) -> Any:
            _publish_outcome(pipeline, mode, units, runtime, folder)
            return job

        executor.submit.side_effect = submit
        runner: RunnerBase = SlurmRunner(max_array_size=1)
        router: LifecycleRouter = SlurmLifecycleRouter(
            executor, 1, 1, tmp_path / "control"
        )
        now = datetime.now(UTC)
        observation = SchedulerObservation("COMPLETED", now, now, now)
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states",
            lambda *_args, **_kwargs: {"123_0": observation},
        )
    else:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        runner = SlurmIntraRunner(log_folder=str(tmp_path / "logs"), poll_interval=0.01)
        router = runner.create_lifecycle_router(
            RunnerResources.model_validate({}),
            BatchStrategy.model_validate({}),
            0,
            "test",
        )

        def popen(command: list[str], **kwargs: Any) -> Any:
            folder = Path(command[-1])
            with (folder / "job.pkl").open("rb") as stream:
                call = cloudpickle.load(stream)
            _publish_outcome(pipeline, mode, call.args[0], call.args[1], folder)
            process = MagicMock(pid=12345, returncode=0)
            process.poll.return_value = 0
            return process

        # Isolate the transport module; real core execution still creates its
        # native command supervisor through Python's actual subprocess module.
        monkeypatch.setattr(srun_module, "subprocess", SimpleNamespace(Popen=popen))
    monkeypatch.setattr(
        runner, "create_lifecycle_router", lambda *_args, **_kwargs: router
    )
    return runner, router


@pytest.mark.parametrize("provider", ["slurm", "srun"])
@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_cancellation"),
    [
        ("confirmed", StepStatus.CANCELLED, CancellationStatus.CONFIRMED),
        ("missing", StepStatus.FAILED, CancellationStatus.UNKNOWN),
        ("nested_unknown", StepStatus.FAILED, CancellationStatus.UNKNOWN),
        ("rejected", StepStatus.SUCCEEDED, CancellationStatus.REJECTED),
    ],
)
def test_cancellation_outcome_matches_durable_pipeline_state(
    provider: str,
    mode: str,
    expected_status: StepStatus,
    expected_cancellation: CancellationStatus,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = PipelineManager.create(
        name=f"{provider}_{mode}",
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        skip_cache=True,
    )
    runner, router = _runner_with_transport_double(
        provider, mode, pipeline, tmp_path, monkeypatch
    )
    try:
        future = pipeline.submit(
            DataGenerator,
            name="cancellation evidence",
            params={"count": 1, "rows_per_file": 1},
            step_runner=runner,
            compact=False,
        )
        result = future.result(timeout=15)
        pipeline.finalize()
        assert result.status == future.status == expected_status
        assert result.cancellation_status == expected_cancellation
        assert router.is_done()
        assert router._thread is not None
        assert not router._thread.is_alive()
        history = pl.read_delta(tmp_path / "delta" / TablePath.STEPS).sort(
            "state_sequence"
        )
        final = history.row(-1, named=True)
        assert final["status"] == expected_status
        assert final["cancellation_status"] == expected_cancellation
        assert final["step_run_id"] == result.step_run_id
        assert "requested" in history["cancellation_status"].to_list()
        public_status = run_status(
            pipeline.config.delta_root, pipeline.config.pipeline_run_id
        )
        assert public_status.steps[0].status == expected_status
        accepted = inspect_step(
            pipeline.config.delta_root,
            0,
            pipeline_run_id=pipeline.config.pipeline_run_id,
            step_run_id=result.step_run_id,
        )
        if mode == "rejected":
            assert result.succeeded_count == 1
            assert accepted.height == 1
            assert final["logical_commit_id"] is not None
        else:
            assert not result.output_roles
            assert accepted.is_empty()
            assert history["logical_commit_id"].null_count() == history.height
        if mode == "nested_unknown":
            assert result.error == "Nested remote closure is unknown"
    finally:
        pipeline.finalize()
