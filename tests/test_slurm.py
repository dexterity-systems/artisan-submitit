"""Tests for Submitit array routing."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import submitit
from artisan.orchestration.runner_api import (
    BatchStrategy,
    CancellationAcknowledgement,
    CancellationStatus,
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


def _observation(state="COMPLETED", *, never_started=False):
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    now = datetime.now(UTC)
    return SchedulerObservation(
        state, now, None if never_started else now, now, never_started
    )


@pytest.fixture
def transport(tmp_path, monkeypatch):
    """A transport double publishes the same atomic evidence as a real worker."""
    import pickle

    from artisan_submitit.worker import WorkerOutcome

    executor = MagicMock()
    jobs = []
    folders = []

    def submit(fn, units, runtime_env, folder):
        index = len(jobs)
        job = MagicMock(job_id=f"99_{index}")
        job.paths = SimpleNamespace(
            stdout=tmp_path / f"out_{index}", stderr=tmp_path / f"err_{index}"
        )
        job.paths.stdout.write_text("completed")
        job.paths.stderr.write_text("worker warning")
        results = [
            _result(int(unit._mock_name.removeprefix("unit_"))) for unit in units
        ]
        (folder / "outcome.pkl").write_bytes(pickle.dumps(WorkerOutcome(results, None)))
        jobs.append(job)
        folders.append(folder)
        return job

    executor.submit.side_effect = submit
    monkeypatch.setattr(
        "artisan_submitit.slurm.query_job_states",
        lambda ids, **kw: {value: _observation() for value in ids},
    )
    cancelled = MagicMock(return_value=None)
    monkeypatch.setattr("artisan_submitit.slurm.cancel_jobs", cancelled)
    return SimpleNamespace(
        executor=executor, jobs=jobs, folders=folders, cancel=cancelled, submit=submit
    )


def _router(transport, tmp_path, *, units_per_worker=1, max_array_size=1):
    return SlurmLifecycleRouter(
        transport.executor, units_per_worker, max_array_size, tmp_path / "control"
    )


class TestSlurmLifecycleRouter:
    def test_dispatch_returns_while_scheduler_submission_is_blocked(
        self, tmp_path, transport
    ) -> None:
        entered, release = threading.Event(), threading.Event()

        @contextlib.contextmanager
        def blocked():
            entered.set()
            assert release.wait(2)
            yield

        transport.executor.batch.side_effect = blocked
        router = _router(transport, tmp_path)
        router.dispatch(_units(1), MagicMock())
        try:
            assert entered.wait(1)
            assert not router.is_done()
        finally:
            release.set()
        router._thread.join(3)
        assert len(router.collect()) == 1

    def test_chunks_arrays_and_preserves_unit_order(self, tmp_path, transport) -> None:
        router = _router(transport, tmp_path, units_per_worker=2, max_array_size=2)
        results = router.run(_units(5), MagicMock())
        assert [r.execution_run_ids for r in results] == [[str(i)] for i in range(5)]
        assert transport.executor.batch.call_count == 2
        assert router.submitted_job_ids == ("99_0", "99_1", "99_2")
        for job in transport.jobs:
            job.done.assert_not_called()
            job.result.assert_not_called()
            job.cancel.assert_not_called()

    def test_scheduler_arrays_run_as_sequential_global_concurrency_waves(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        from artisan_submitit.worker import read_worker_outcome

        entered, release = threading.Event(), threading.Event()

        def read(folder):
            if folder == transport.folders[0]:
                entered.set()
                assert release.wait(2)
            return read_worker_outcome(folder)

        monkeypatch.setattr("artisan_submitit.slurm.read_worker_outcome", read)
        router = _router(transport, tmp_path)
        router.dispatch(_units(3), MagicMock())
        try:
            assert entered.wait(1)
            assert transport.executor.submit.call_count == 1
        finally:
            release.set()
        router._thread.join(3)
        assert [r.execution_run_ids for r in router.collect()] == [
            [str(i)] for i in range(3)
        ]
        assert transport.executor.batch.call_count == 3

    def test_malformed_batch_becomes_one_failure_per_unit(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        from artisan_submitit.worker import WorkerOutcome

        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome",
            lambda folder: WorkerOutcome([_result(0)], None),
        )
        router = _router(transport, tmp_path, units_per_worker=2)
        results = router.run(_units(2), MagicMock())
        assert len(results) == 2
        assert all(not r.success and r.error for r in results)
        assert router.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN

    def test_job_failure_without_worker_proof_is_unknown(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome", lambda folder: None
        )
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states",
            lambda ids, **kw: {i: _observation("NODE_FAIL") for i in ids},
        )
        router = _router(transport, tmp_path, units_per_worker=2)
        results = router.run(_units(2), MagicMock())
        assert len(results) == 2
        assert all(not r.success and "NODE_FAIL" in r.error for r in results)
        assert router.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN

    def test_later_wave_submission_failure_preserves_ordered_cardinality(
        self, tmp_path, transport
    ) -> None:
        def submit(*args):
            if transport.jobs:
                raise RuntimeError("submit failed")
            return transport.submit(*args)

        transport.executor.submit.side_effect = submit
        router = _router(transport, tmp_path)
        results = router.run(_units(3), MagicMock())
        assert len(results) == 3
        assert results[0].success
        assert all(not r.success and "submit failed" in r.error for r in results[1:])

    def test_batch_context_failure_never_publishes_delayed_jobs(self, tmp_path) -> None:
        executor = submitit.LocalExecutor(folder=tmp_path / "submitit")
        router = SlurmLifecycleRouter(executor, 1, 2, tmp_path / "control")
        with patch.object(
            executor,
            "_internal_process_submissions",
            side_effect=RuntimeError("scheduler submit failed"),
        ):
            results = router.run(_units(3), MagicMock())
        assert router.submitted_job_ids == ()
        assert len(results) == 3
        assert all(
            not r.success and "scheduler submit failed" in r.error for r in results
        )
        assert router.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN
        assert all(
            folder.joinpath("cancel").exists()
            for folder in (tmp_path / "control").iterdir()
        )

    def test_cancel_during_promotion_remains_requested_until_collection(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        entered, release = threading.Event(), threading.Event()

        @contextlib.contextmanager
        def blocked_promotion():
            yield
            entered.set()
            assert release.wait(2)

        transport.executor.batch.side_effect = blocked_promotion
        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome", lambda folder: None
        )
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states",
            lambda ids, **kw: {
                i: _observation("CANCELLED", never_started=True) for i in ids
            },
        )
        router = _router(transport, tmp_path)
        router.dispatch(_units(3), MagicMock())
        try:
            assert entered.wait(1)
            assert router.cancel().status == CancellationStatus.REQUESTED
            transport.cancel.assert_not_called()
        finally:
            release.set()
        router._thread.join(3)
        results = router.collect()
        assert len(results) == 3
        assert all(not r.success for r in results)
        assert transport.executor.submit.call_count == 1
        assert (transport.folders[0] / "cancel").exists()
        assert router.cancel().status == CancellationStatus.CONFIRMED
        assert not router._thread.is_alive()

    def test_promotion_failure_keeps_admission_uncertainty_and_exact_cleanup(
        self, tmp_path, transport
    ) -> None:
        entered, release = threading.Event(), threading.Event()

        @contextlib.contextmanager
        def failed_promotion():
            yield
            entered.set()
            assert release.wait(2)
            raise RuntimeError("partial promotion failed")

        transport.executor.batch.side_effect = failed_promotion
        router = _router(transport, tmp_path)
        router.dispatch(_units(3), MagicMock())
        try:
            assert entered.wait(1)
            assert router.cancel().status == CancellationStatus.REQUESTED
        finally:
            release.set()
        router._thread.join(3)
        assert len(router.collect()) == 3
        assert router.cancel().status == CancellationStatus.UNKNOWN
        assert router.submitted_job_ids == ("99_0",)
        assert (transport.folders[0] / "cancel").exists()
        transport.cancel.assert_called_once_with(["99_0"])

    def test_invalid_outcome_overrides_queued_non_start_proof(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        entered, release = threading.Event(), threading.Event()

        def invalid_outcome(folder):
            entered.set()
            assert release.wait(2)
            raise ValueError("malformed final evidence")

        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome", invalid_outcome
        )
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states",
            lambda ids, **kw: {
                i: _observation("CANCELLED", never_started=True) for i in ids
            },
        )
        router = _router(transport, tmp_path)
        router.dispatch(_units(1), MagicMock())
        try:
            assert entered.wait(1)
            router.cancel()
        finally:
            release.set()
        router._thread.join(3)
        assert router.cancel().status == CancellationStatus.UNKNOWN
        assert "malformed final evidence" in router.collect()[0].error

    def test_query_and_cancel_share_remaining_deadline(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        from datetime import UTC, datetime

        from artisan_submitit.slurm import _SubmittedBatch

        now = [100.0]
        monkeypatch.setattr(
            "artisan_submitit.cancellation.time.monotonic", lambda: now[0]
        )
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        router = _router(transport, tmp_path)
        router._cancellation.request()
        now[0] += 24.2

        def query(ids, **kwargs):
            now[0] += 0.5
            return {}

        monkeypatch.setattr("artisan_submitit.slurm.query_job_states", query)
        item = _SubmittedBatch(
            MagicMock(job_id="99_0"), _units(1), tmp_path, datetime.now(UTC)
        )
        router._observe_jobs([item])
        assert transport.cancel.call_args.kwargs["timeout"] == pytest.approx(0.3)

    def test_cancel_does_not_join_stalled_log_collection(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        entered, release = threading.Event(), threading.Event()

        def logs(*args):
            entered.set()
            assert release.wait(2)

        monkeypatch.setattr("artisan_submitit.slurm._read_submitit_job_log", logs)
        router = _router(transport, tmp_path)
        router.dispatch(_units(1), MagicMock())
        try:
            assert entered.wait(1)
            assert router.cancel().status == CancellationStatus.REQUESTED
            assert not router.is_done()
        finally:
            release.set()
        router._thread.join(3)
        assert router.cancel().status == CancellationStatus.REJECTED
        assert router.collect()[0].success

    def test_cancel_uses_exact_ids_and_retries_failed_requests(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome", lambda folder: None
        )
        transport.cancel.side_effect = ["request failed", None]

        def states(ids, **kwargs):
            if transport.cancel.call_count >= 2:
                return {i: _observation("CANCELLED", never_started=True) for i in ids}
            return {}

        monkeypatch.setattr("artisan_submitit.slurm.query_job_states", states)
        router = _router(transport, tmp_path, max_array_size=2)
        router.dispatch(_units(2), MagicMock())
        deadline = time.monotonic() + 2
        while len(transport.jobs) != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(transport.jobs) == 2
        assert router.cancel().status == CancellationStatus.REQUESTED
        router._thread.join(5)
        assert len(router.collect()) == 2
        assert router.cancel().status == CancellationStatus.CONFIRMED
        assert transport.cancel.call_args_list[0].args == (["99_0", "99_1"],)
        assert transport.cancel.call_count == 2
        assert all((folder / "cancel").exists() for folder in transport.folders)

    @pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", None])
    def test_accepted_request_or_missing_accounting_never_confirms(
        self, tmp_path, transport, monkeypatch, state
    ) -> None:
        monkeypatch.setattr("artisan_submitit.cancellation.CANCELLATION_TIMEOUT", 0.15)
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        monkeypatch.setattr(
            "artisan_submitit.slurm.read_worker_outcome", lambda folder: None
        )
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states",
            lambda ids, **kw: {}
            if state is None
            else {i: _observation(state) for i in ids},
        )
        router = _router(transport, tmp_path)
        router.dispatch(_units(1), MagicMock())
        router.cancel()
        router._thread.join(2)
        assert router.cancel().status == CancellationStatus.UNKNOWN
        assert all(not r.success for r in router.collect())

    def test_missing_accounting_preserves_received_results_and_nested_uncertainty(
        self,
        tmp_path: Path,
        transport: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from artisan_submitit.worker import WorkerOutcome

        monkeypatch.setattr("artisan_submitit.cancellation.CANCELLATION_TIMEOUT", 0.15)
        monkeypatch.setattr(
            "artisan_submitit.slurm.query_job_states", lambda ids, **kw: {}
        )
        monkeypatch.setattr(
            "artisan_submitit.slurm._read_submitit_job_log", lambda *args: None
        )
        nested = CancellationAcknowledgement(
            CancellationStatus.UNKNOWN, "remote owner is unresolved"
        )
        original = [
            replace(_result(0), worker_log="first native log"),
            replace(
                _result(1),
                success=False,
                error="nested operation diagnostic",
                worker_log="second native log",
                cancellation_acknowledgement=nested,
            ),
        ]
        outcome = WorkerOutcome(
            original, CancellationAcknowledgement(CancellationStatus.CONFIRMED)
        )
        received = threading.Event()

        def read(folder: Path) -> WorkerOutcome:
            received.set()
            return outcome

        monkeypatch.setattr("artisan_submitit.slurm.read_worker_outcome", read)
        router = _router(transport, tmp_path, units_per_worker=2)
        router.dispatch(_units(2), MagicMock())
        assert received.wait(2)
        router.cancel()
        router._thread.join(2)
        results = router.collect()

        assert [result.execution_run_ids for result in results] == [["0"], ["1"]]
        assert [result.worker_log for result in results] == [
            "first native log",
            "second native log",
        ]
        assert all(not result.success for result in results)
        assert all(
            result.cancellation_acknowledgement.status is CancellationStatus.UNKNOWN
            for result in results
        )
        assert results[1].error == "nested operation diagnostic"
        assert results[1].cancellation_acknowledgement == nested
        assert router.cancel().status is CancellationStatus.UNKNOWN

    def test_late_promotion_after_deadline_still_cancels_exact_job(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.setattr("artisan_submitit.cancellation.CANCELLATION_TIMEOUT", 0.1)
        entered, release = threading.Event(), threading.Event()

        @contextlib.contextmanager
        def blocked_promotion():
            yield
            entered.set()
            assert release.wait(2)

        transport.executor.batch.side_effect = blocked_promotion
        router = _router(transport, tmp_path)
        router.dispatch(_units(1), MagicMock())
        try:
            assert entered.wait(1)
            router.cancel()
            time.sleep(0.12)
            assert router.cancel().status == CancellationStatus.UNKNOWN
            assert not router.is_done()
        finally:
            release.set()
        router._thread.join(2)
        transport.cancel.assert_called_once_with(["99_0"])
        assert router.cancel().status == CancellationStatus.UNKNOWN
        assert len(router.collect()) == 1

    def test_dispatch_precondition_failure_completes_with_failures(
        self, tmp_path, transport
    ) -> None:
        router = _router(transport, tmp_path, units_per_worker=0)
        results = router.run(_units(2), MagicMock())
        assert router.is_done()
        assert len(results) == 2
        assert all(not r.success for r in results)

    def test_logs_are_attached_to_every_result_in_job(
        self, tmp_path, transport
    ) -> None:
        results = _router(transport, tmp_path, units_per_worker=2).run(
            _units(2), MagicMock()
        )
        assert all("worker warning" in r.worker_log for r in results)

    def test_cancel_after_collection_rejects_and_preserves_results(
        self, tmp_path, transport
    ) -> None:
        router = _router(transport, tmp_path)
        results = router.run(_units(1), MagicMock())
        assert router.cancel().status == CancellationStatus.REJECTED
        assert results[0].success
        transport.cancel.assert_not_called()

    def test_idle_cancel_does_not_poison_later_dispatch(
        self, tmp_path, transport
    ) -> None:
        router = _router(transport, tmp_path)
        assert router.cancel().status == CancellationStatus.REJECTED
        assert router.run(_units(1), MagicMock())[0].success
        assert not router._cancellation.event.is_set()
