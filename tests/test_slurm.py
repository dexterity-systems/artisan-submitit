"""Tests for Submitit array routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from artisan.orchestration.runner_api import (
    BatchStrategy,
    RunnerResources,
    UnitResult,
)

from artisan_submitit.slurm import SlurmLifecycleRouter, SlurmRunner


def _units(count: int) -> list[MagicMock]:
    return [MagicMock(name=f"unit_{index}") for index in range(count)]


def _result(index: int) -> UnitResult:
    return UnitResult(
        success=True,
        error=None,
        item_count=1,
        execution_run_ids=[str(index)],
    )


class TestSlurmRunner:
    def test_traits(self) -> None:
        assert SlurmRunner.name == "slurm"
        assert SlurmRunner.worker_traits.worker_id_env_var == "SLURM_ARRAY_TASK_ID"
        assert SlurmRunner.worker_traits.shared_filesystem is True
        assert SlurmRunner.orchestrator_traits.needs_staging_verification is True

    def test_maps_portable_resources_to_submitit_names(self) -> None:
        runner = SlurmRunner(slurm_partition="cpu")
        resources = RunnerResources(
            cpus=4,
            memory_gb=16,
            gpus=2,
            time_limit="02:00:30",
            extra={"slurm_partition": "gpu", "slurm_constraint": "a100"},
        )
        parameters = runner._executor_parameters(
            resources,
            BatchStrategy(max_workers=8),
            step_number=3,
            job_name="train",
        )

        assert parameters == {
            "slurm_partition": "gpu",
            "timeout_min": 121,
            "mem_gb": 16,
            "cpus_per_task": 4,
            "gpus_per_node": 2,
            "slurm_array_parallelism": 8,
            "slurm_job_name": "s3_train",
            "slurm_srun_args": ["--export=ALL"],
            "slurm_constraint": "a100",
        }

    def test_slurm_gres_takes_precedence_over_portable_gpu(self, caplog) -> None:
        runner = SlurmRunner(slurm_gres="gpu:a100:1")
        parameters = runner._executor_parameters(
            RunnerResources(gpus=2), BatchStrategy(), 0, "op"
        )
        assert parameters["slurm_gres"] == "gpu:a100:1"
        assert "gpus_per_node" not in parameters
        assert "using slurm_gres" in caplog.text

    def test_max_workers_conflict_fails_fast(self) -> None:
        runner = SlurmRunner(slurm_array_parallelism=5)
        with pytest.raises(ValueError, match="conflicts"):
            runner._executor_parameters(
                RunnerResources(), BatchStrategy(max_workers=3), 0, "op"
            )

    def test_non_positive_max_workers_fails_fast(self) -> None:
        runner = SlurmRunner()
        with pytest.raises(ValueError, match="max_workers"):
            runner._executor_parameters(
                RunnerResources(), BatchStrategy(max_workers=0), 0, "op"
            )

    @pytest.mark.parametrize("alias", ["partition", "time_limit"])
    def test_removed_parameter_alias_fails_fast(self, alias: str) -> None:
        with pytest.raises(ValueError, match="Submitit-native"):
            SlurmRunner(**{alias: "bad"})

    @patch("artisan_submitit.slurm.get_cluster_max_array_size", return_value=100)
    @patch("artisan_submitit.slurm.submitit.AutoExecutor")
    def test_create_router_configures_auto_executor(
        self,
        executor_type: MagicMock,
        max_array_size: MagicMock,
        tmp_path,
    ) -> None:
        del max_array_size
        runner = SlurmRunner()
        router = runner.create_lifecycle_router(
            RunnerResources(),
            BatchStrategy(units_per_worker=2),
            step_number=4,
            job_name="op",
            log_folder=str(tmp_path),
        )

        assert isinstance(router, SlurmLifecycleRouter)
        executor_type.assert_called_once_with(folder=tmp_path / "%j")
        executor_type.return_value.update_parameters.assert_called_once()


class TestSlurmLifecycleRouter:
    def test_chunks_arrays_and_preserves_unit_order(self, tmp_path) -> None:
        executor = MagicMock()
        jobs = []
        for index, batch_results in enumerate(
            [[_result(0), _result(1)], [_result(2), _result(3)], [_result(4)]]
        ):
            job = MagicMock()
            job.job_id = f"99_{index}"
            job.result.return_value = batch_results
            job.paths = SimpleNamespace(
                stdout=tmp_path / f"out_{index}",
                stderr=tmp_path / f"err_{index}",
            )
            job.paths.stdout.write_text("completed")
            jobs.append(job)
        executor.submit.side_effect = jobs
        router = SlurmLifecycleRouter(
            executor=executor,
            units_per_worker=2,
            max_array_size=2,
        )

        results = router.run(_units(5), MagicMock())

        assert [result.execution_run_ids for result in results] == [
            ["0"],
            ["1"],
            ["2"],
            ["3"],
            ["4"],
        ]
        assert executor.batch.call_count == 2
        assert executor.submit.call_count == 3
        assert router.submitted_job_ids == ("99_0", "99_1", "99_2")

    def test_malformed_batch_becomes_one_failure_per_unit(self, tmp_path) -> None:
        job = MagicMock()
        job.job_id = "12_0"
        job.result.return_value = [_result(0)]
        job.paths = SimpleNamespace(
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
        )
        job.paths.stdout.write_text("completed")
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=2, max_array_size=10)

        results = router.run(_units(2), MagicMock())

        assert len(results) == 2
        assert all(not result.success for result in results)
        assert all(result.error for result in results)

    def test_job_exception_becomes_ordered_failure_results(self, tmp_path) -> None:
        job = MagicMock()
        job.job_id = "12_0"
        job.result.side_effect = RuntimeError("node failed")
        job.paths = SimpleNamespace(
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
        )
        job.paths.stdout.write_text("failed")
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=2, max_array_size=10)

        results = router.run(_units(2), MagicMock())

        assert len(results) == 2
        assert all(not result.success for result in results)
        assert all("node failed" in result.error for result in results)

    def test_partial_array_submission_cancels_retained_exact_handle(self) -> None:
        first_job = MagicMock(job_id="21_0")
        executor = MagicMock()
        executor.submit.side_effect = [first_job, RuntimeError("submit failed")]
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=10)

        results = router.run(_units(3), MagicMock())

        assert len(results) == 3
        assert all(not result.success for result in results)
        assert all("submit failed" in result.error for result in results)
        first_job.cancel.assert_called_once_with(check=False)

    def test_dispatch_precondition_failure_completes_with_failures(self) -> None:
        router = SlurmLifecycleRouter(
            MagicMock(),
            units_per_worker=0,
            max_array_size=10,
        )

        results = router.run(_units(2), MagicMock())

        assert router.is_done()
        assert len(results) == 2
        assert all(not result.success for result in results)

    def test_logs_are_attached_to_every_result_in_job(self, tmp_path) -> None:
        stdout = tmp_path / "stdout"
        stderr = tmp_path / "stderr"
        stdout.write_text("worker output")
        stderr.write_text("worker warning")
        job = MagicMock()
        job.job_id = "12_0"
        job.result.return_value = [_result(0), _result(1)]
        job.paths = SimpleNamespace(stdout=stdout, stderr=stderr)
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=2, max_array_size=10)

        results = router.run(_units(2), MagicMock())

        assert all(result.worker_log is not None for result in results)
        assert all("worker warning" in result.worker_log for result in results)

    def test_cancel_uses_exact_submitit_handles(self, tmp_path) -> None:
        executor = MagicMock()
        jobs = []
        for index in range(2):
            job = MagicMock()
            job.job_id = f"123_{index}"
            job.result.return_value = [_result(index)]
            job.paths = SimpleNamespace(
                stdout=tmp_path / f"out_{index}",
                stderr=tmp_path / f"err_{index}",
            )
            job.paths.stdout.write_text("completed")
            jobs.append(job)
        executor.submit.side_effect = jobs
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=10)
        router._submit_array([_units(1), _units(1)], MagicMock())

        router.cancel()
        router.cancel()

        for job in jobs:
            job.cancel.assert_called_once_with(check=False)

    def test_cancel_after_collection_does_not_cancel_completed_jobs(
        self, tmp_path
    ) -> None:
        job = MagicMock(job_id="123_0")
        job.result.return_value = [_result(0)]
        job.paths = SimpleNamespace(
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
        )
        job.paths.stdout.write_text("completed")
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=10)

        router.run(_units(1), MagicMock())
        router.cancel()

        job.cancel.assert_not_called()

    def test_job_added_after_cancel_is_not_leaked(self) -> None:
        job = MagicMock(job_id="123_0")
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=10)
        router.cancel()

        router._submit_array([_units(1)], MagicMock())

        job.cancel.assert_called_once_with(check=False)
