"""Tests for Submitit array routing."""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import submitit
from artisan.orchestration.runner_api import (
    BatchStrategy,
    RunnerResources,
    UnitResult,
)

from artisan_submitit.slurm import (
    SlurmLifecycleRouter,
    SlurmRunner,
    _validate_effective_executor,
)


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

    @pytest.mark.parametrize("slurm_gres", [None, ""])
    def test_empty_slurm_gres_does_not_drop_portable_gpu(
        self, slurm_gres: str | None
    ) -> None:
        runner = SlurmRunner(slurm_gres=slurm_gres)

        parameters = runner._executor_parameters(
            RunnerResources(gpus=2), BatchStrategy(), 0, "op"
        )

        assert parameters["gpus_per_node"] == 2
        assert "slurm_gres" not in parameters

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

    @pytest.mark.parametrize(
        "alias",
        [
            "additional_parameters",
            "array_parallelism",
            "gres",
            "partition",
            "srun_args",
            "time_limit",
        ],
    )
    def test_removed_parameter_alias_fails_fast(self, alias: str) -> None:
        with pytest.raises(ValueError, match="Submitit-native"):
            SlurmRunner(**{alias: "bad"})

    @pytest.mark.parametrize(
        "parameter",
        [
            "cpus_per_gpu",
            "cpus_per_task",
            "gpus_per_node",
            "gpus_per_task",
            "mem",
            "mem_gb",
            "mem_per_cpu",
            "mem_per_gpu",
            "num_gpus",
            "time",
            "timeout_min",
        ],
    )
    def test_deprecated_portable_alias_fails_fast(self, parameter: str) -> None:
        with pytest.raises(ValueError, match="RunnerResources"):
            SlurmRunner(**{parameter: 2})

    @pytest.mark.parametrize(
        "parameter",
        [
            "nodes",
            "ntasks_per_node",
            "tasks_per_node",
            "slurm_nodes",
            "slurm_tasks_per_node",
            "slurm_ntasks_per_node",
        ],
    )
    def test_multi_rank_topology_fails_fast(self, parameter: str) -> None:
        with pytest.raises(ValueError, match="one task"):
            SlurmRunner(**{parameter: 2})

    @pytest.mark.parametrize(
        "parameter",
        [
            "slurm_cpus_per_task",
            "slurm_cpus_per_gpu",
            "slurm_gpus_per_node",
            "slurm_gpus_per_task",
            "slurm_num_gpus",
            "slurm_mem",
            "slurm_mem_gb",
            "slurm_mem_per_cpu",
            "slurm_mem_per_gpu",
            "slurm_time",
            "slurm_timeout_min",
        ],
    )
    def test_prefixed_portable_resource_override_fails_fast(
        self, parameter: str
    ) -> None:
        with pytest.raises(ValueError, match="RunnerResources"):
            SlurmRunner(**{parameter: 2})

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--ntasks=2"],
            ["--nodes", "2"],
            ["-n2"],
            ["--cpus-per-task", "8"],
            ["-c8"],
            ["--mem=8G"],
            ["--gres", "gpu:2"],
            ["-G2"],
            ["--gpus-per-socket=2"],
            ["--tres-per-task=gres/gpu:2"],
            ["--sockets-per-node=2"],
            ["--cores-per-socket=4"],
            ["--threads-per-core=2"],
            ["--core-spec=1"],
            ["-B2:4:1"],
            ["-S1"],
            ["--time=01:00:00"],
            ["-t60"],
        ],
    )
    def test_managed_slurm_srun_argument_fails_fast(self, arguments: list[str]) -> None:
        with pytest.raises(ValueError, match="Artisan-managed"):
            SlurmRunner(slurm_srun_args=arguments)

    @pytest.mark.parametrize(
        "parameters",
        [
            {"array": "0-3"},
            {"nodes": 2},
            {"ntasks_per_node": 2},
            {"cpus-per-task": 8},
            {"mem": "8G"},
            {"gres": "gpu:2"},
            {"gpus-per-socket": 2},
            {"tres-per-task": "gres/gpu:2"},
            {"sockets-per-node": 2},
            {"cores-per-socket": 4},
            {"threads-per-core": 2},
            {"core-spec": 1},
            {"time": 60},
        ],
    )
    def test_managed_additional_parameter_fails_fast(
        self, parameters: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="Artisan-managed"):
            SlurmRunner(slurm_additional_parameters=parameters)

    def test_unmanaged_native_parameters_remain_available(self) -> None:
        runner = SlurmRunner(
            slurm_srun_args=[
                "--export=ALL",
                "--exclusive",
                "--cpu-bind=cores",
                "--constraint",
                "gpus-per-socket",
                "--tres-bind=gres/gpu:closest",
            ],
            slurm_additional_parameters={
                "constraint": "a100",
                "mail-type": "FAIL",
            },
        )

        parameters = runner._executor_parameters(
            RunnerResources(), BatchStrategy(), 0, "op"
        )

        assert parameters["slurm_srun_args"] == [
            "--export=ALL",
            "--exclusive",
            "--cpu-bind=cores",
            "--constraint",
            "gpus-per-socket",
            "--tres-bind=gres/gpu:closest",
        ]
        assert parameters["slurm_additional_parameters"] == {
            "constraint": "a100",
            "mail-type": "FAIL",
        }

    @patch("artisan_submitit.slurm.get_cluster_max_array_size", return_value=100)
    @patch("artisan_submitit.slurm.submitit.AutoExecutor")
    @patch("artisan_submitit.slurm.uuid.uuid4")
    def test_create_router_configures_auto_executor(
        self,
        uuid4: MagicMock,
        executor_type: MagicMock,
        max_array_size: MagicMock,
        tmp_path,
    ) -> None:
        del max_array_size
        uuid4.return_value.hex = "router-abc"
        executor_type.return_value.parameters = {
            "nodes": 1,
            "ntasks_per_node": None,
            "time": 60,
            "cpus_per_task": 1,
            "array_parallelism": 1000,
            "mem": "4GB",
            "gres": None,
            "gpus_per_node": None,
            "srun_args": ["--export=ALL"],
            "additional_parameters": None,
        }
        runner = SlurmRunner()
        router = runner.create_lifecycle_router(
            RunnerResources(),
            BatchStrategy(units_per_worker=2),
            step_number=4,
            job_name="op",
            log_folder=str(tmp_path),
        )

        assert isinstance(router, SlurmLifecycleRouter)
        executor_type.assert_called_once_with(
            folder=tmp_path / "router-abc" / "%j", cluster="slurm"
        )
        executor_type.return_value.update_parameters.assert_called_once()

    def test_effective_submitit_state_is_validated(self) -> None:
        executor = MagicMock()
        executor.parameters = {
            "nodes": 1,
            "ntasks_per_node": 4,
            "time": 5,
            "cpus_per_task": 1,
            "array_parallelism": 2,
        }
        expected = {
            "timeout_min": 5,
            "cpus_per_task": 1,
            "slurm_array_parallelism": 2,
        }

        with pytest.raises(ValueError, match="single-task topology"):
            _validate_effective_executor(executor, expected)

    @patch("submitit.slurm.slurm.SlurmExecutor.affinity", return_value=0)
    def test_create_router_fails_instead_of_falling_back_locally(
        self,
        affinity: MagicMock,
        tmp_path,
    ) -> None:
        del affinity
        with pytest.raises(RuntimeError, match='Could not detect "srun"'):
            SlurmRunner().create_lifecycle_router(
                RunnerResources(), BatchStrategy(), 0, "op", str(tmp_path)
            )

    @pytest.mark.parametrize(
        ("alias", "value"),
        [
            ("time", 99),
            ("mem", "8G"),
            ("ntasks_per_node", 4),
            ("srun_args", ["-n4"]),
            ("additional_parameters", {"nodes": 2}),
            ("array_parallelism", 999),
        ],
    )
    @patch("submitit.slurm.slurm.SlurmExecutor.affinity", return_value=1)
    def test_real_submitit_canonical_aliases_are_rejected(
        self,
        affinity: MagicMock,
        alias: str,
        value: object,
        tmp_path,
    ) -> None:
        del affinity
        executor = submitit.AutoExecutor(folder=tmp_path, cluster="slurm")
        with pytest.warns(UserWarning, match="deprecated"):
            executor.update_parameters(**{alias: value})
        assert executor.parameters[alias] == value

        with pytest.raises(
            ValueError, match=r"RunnerResources|Submitit-native|one task"
        ):
            SlurmRunner(**{alias: value})


class TestSlurmLifecycleRouter:
    def test_dispatch_returns_while_scheduler_submission_is_blocked(
        self, tmp_path
    ) -> None:
        entered_batch = threading.Event()
        release_batch = threading.Event()
        dispatch_returned = threading.Event()
        stdout = tmp_path / "stdout"
        stdout.write_text("completed")
        job = MagicMock(job_id="99_0")
        job.result.return_value = [_result(0)]
        job.paths = SimpleNamespace(stdout=stdout, stderr=tmp_path / "stderr")
        executor = MagicMock()

        @contextlib.contextmanager
        def blocking_batch():
            entered_batch.set()
            release_batch.wait(timeout=2)
            yield

        executor.batch.side_effect = blocking_batch
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)

        def dispatch() -> None:
            router.dispatch(_units(1), MagicMock())
            dispatch_returned.set()

        caller = threading.Thread(target=dispatch)
        caller.start()
        try:
            assert entered_batch.wait(timeout=1)
            assert dispatch_returned.wait(timeout=1)
        finally:
            release_batch.set()
            caller.join(timeout=2)
        assert router._thread is not None
        router._thread.join(timeout=2)
        assert len(router.collect()) == 1

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

    def test_scheduler_arrays_run_as_sequential_global_concurrency_waves(
        self, tmp_path
    ) -> None:
        first_collection_started = threading.Event()
        release_first = threading.Event()
        executor = MagicMock()
        jobs = []
        for index in range(3):
            stdout = tmp_path / f"wave_out_{index}"
            stdout.write_text("completed")
            job = MagicMock(job_id=f"44_{index}")
            job.paths = SimpleNamespace(
                stdout=stdout,
                stderr=tmp_path / f"wave_err_{index}",
            )
            job.result.return_value = [_result(index)]
            jobs.append(job)

        def first_result():
            first_collection_started.set()
            release_first.wait(timeout=2)
            return [_result(0)]

        jobs[0].result.side_effect = first_result
        executor.submit.side_effect = jobs
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)

        router.dispatch(_units(3), MagicMock())
        try:
            assert first_collection_started.wait(timeout=1)
            assert executor.submit.call_count == 1
        finally:
            release_first.set()
        assert router._thread is not None
        router._thread.join(timeout=2)

        results = router.collect()
        assert [result.execution_run_ids for result in results] == [
            ["0"],
            ["1"],
            ["2"],
        ]
        assert executor.batch.call_count == 3

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

    def test_later_wave_submission_failure_preserves_ordered_cardinality(
        self, tmp_path
    ) -> None:
        first_job = MagicMock(job_id="21_0")
        first_job.result.return_value = [_result(0)]
        first_job.paths = SimpleNamespace(
            stdout=tmp_path / "first_stdout",
            stderr=tmp_path / "first_stderr",
        )
        first_job.paths.stdout.write_text("completed")
        executor = MagicMock()
        executor.submit.side_effect = [first_job, RuntimeError("submit failed")]
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)

        results = router.run(_units(3), MagicMock())

        assert len(results) == 3
        assert results[0].success is True
        assert all(not result.success for result in results[1:])
        assert all("submit failed" in result.error for result in results[1:])
        first_job.cancel.assert_not_called()

    def test_batch_context_failure_never_publishes_delayed_jobs(self, tmp_path) -> None:
        executor = submitit.LocalExecutor(folder=tmp_path / "submitit")
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=2)

        with patch.object(
            executor,
            "_internal_process_submissions",
            side_effect=RuntimeError("scheduler submit failed"),
        ):
            results = router.run(_units(3), MagicMock())

        assert router.submitted_job_ids == ()
        assert len(results) == 3
        assert all(not result.success for result in results)
        assert all("scheduler submit failed" in result.error for result in results)

    def test_cancel_does_not_touch_job_until_batch_promotion_finishes(
        self, tmp_path
    ) -> None:
        promotion_started = threading.Event()
        finish_promotion = threading.Event()
        stdout = tmp_path / "promotion_stdout"
        stdout.write_text("completed")
        job = MagicMock(job_id="54_0")
        job.result.return_value = [_result(0)]
        job.paths = SimpleNamespace(
            stdout=stdout,
            stderr=tmp_path / "promotion_stderr",
        )
        executor = MagicMock()

        @contextlib.contextmanager
        def delayed_promotion():
            yield
            promotion_started.set()
            finish_promotion.wait(timeout=2)

        executor.batch.side_effect = delayed_promotion
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)

        router.dispatch(_units(1), MagicMock())
        try:
            assert promotion_started.wait(timeout=1)
            router.cancel()
            job.cancel.assert_not_called()
        finally:
            finish_promotion.set()
        assert router._thread is not None
        router._thread.join(timeout=2)

        router.collect()
        job.cancel.assert_called_once_with(check=True)

    def test_cancellation_synthesizes_failures_for_unsubmitted_waves(
        self, tmp_path
    ) -> None:
        collection_started = threading.Event()
        release_collection = threading.Event()
        stdout = tmp_path / "cancel_stdout"
        stdout.write_text("cancelled")
        job = MagicMock(job_id="55_0")
        job.paths = SimpleNamespace(stdout=stdout, stderr=tmp_path / "cancel_stderr")

        def cancelled_result():
            collection_started.set()
            release_collection.wait(timeout=2)
            raise RuntimeError("scheduler job cancelled")

        job.result.side_effect = cancelled_result
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)

        router.dispatch(_units(3), MagicMock())
        assert collection_started.wait(timeout=1)
        router.cancel()
        release_collection.set()
        assert router._thread is not None
        router._thread.join(timeout=2)

        results = router.collect()
        assert executor.submit.call_count == 1
        assert len(results) == 3
        assert all(not result.success for result in results)
        assert "scheduler job cancelled" in results[0].error
        assert all("before submission" in result.error for result in results[1:])
        job.cancel.assert_called_once_with(check=True)

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
        router._dispatch_started = True
        router._submit_array([_units(1), _units(1)], MagicMock())

        router.cancel()
        router.cancel()

        for job in jobs:
            job.cancel.assert_called_once_with(check=True)

    def test_failed_cancellation_is_retried(self) -> None:
        job = MagicMock(job_id="123_0")
        job.done.return_value = False
        job.cancel.side_effect = [RuntimeError("scancel failed"), None]
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)
        router._dispatch_started = True
        router._submit_array([_units(1)], MagicMock())

        router.cancel()
        router.cancel()

        assert job.cancel.call_count == 2
        assert job.cancel.call_args_list[0].kwargs == {"check": True}

    def test_terminal_job_is_not_cancelled(self) -> None:
        job = MagicMock(job_id="123_0")
        job.done.return_value = True
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=1)
        router._dispatch_started = True
        router._submit_array([_units(1)], MagicMock())

        router.cancel()

        job.cancel.assert_not_called()

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

    def test_idle_cancel_does_not_poison_later_dispatch(self) -> None:
        job = MagicMock(job_id="123_0")
        executor = MagicMock()
        executor.submit.return_value = job
        router = SlurmLifecycleRouter(executor, units_per_worker=1, max_array_size=10)
        router.cancel()

        router._submit_array([_units(1)], MagicMock())

        job.cancel.assert_not_called()
        assert not router._cancelled.is_set()
