"""Tests for intra-allocation srun routing."""

from __future__ import annotations

import json
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from artisan.orchestration.runner_api import (
    BatchStrategy,
    CancellationAcknowledgement,
    CancellationStatus,
    RunnerResources,
    UnitResult,
)

from artisan_submitit.cancellation import SchedulerObservation
from artisan_submitit.srun import (
    SlurmIntraLifecycleRouter,
    SlurmIntraRunner,
    _SrunProcess,
)
from artisan_submitit.worker import WorkerOutcome


def _router(tmp_path, **overrides) -> SlurmIntraLifecycleRouter:
    values = {
        "units_per_worker": 1,
        "max_workers": 2,
        "work_folder": tmp_path / "work",
        "cpus": 2,
        "memory_gb": 8,
        "gpus": 1,
        "time_limit": "00:30:00",
        "slurm_gres": None,
        "srun_args": [],
        "poll_interval": 0.01,
        "max_poll_time": 60.0,
        "cancel_grace_period": 0.0,
    }
    values.update(overrides)
    return SlurmIntraLifecycleRouter(**values)


class TestSlurmIntraRunner:
    def test_traits(self) -> None:
        assert SlurmIntraRunner.name == "slurm_intra"
        assert SlurmIntraRunner.worker_traits.worker_id_env_var == "SLURM_STEP_ID"
        assert SlurmIntraRunner.worker_traits.shared_filesystem is True

    def test_create_router_maps_resources_and_uses_unique_folder(
        self, tmp_path
    ) -> None:
        runner = SlurmIntraRunner(log_folder=str(tmp_path))
        resources = RunnerResources(
            cpus=4,
            memory_gb=16,
            gpus=2,
            time_limit="02:00:00",
            extra={"slurm_gres": "gpu:a100:1", "srun_args": ["--exclusive"]},
        )
        first = runner.create_lifecycle_router(
            resources, BatchStrategy(max_workers=3), 2, "train"
        )
        second = runner.create_lifecycle_router(
            resources, BatchStrategy(max_workers=3), 2, "train"
        )

        assert isinstance(first, SlurmIntraLifecycleRouter)
        assert first.work_folder != second.work_folder
        assert first._max_workers == 3
        assert first._cpus == 4
        assert first._slurm_gres == "gpu:a100:1"
        assert first._srun_args == ["--exclusive"]

    def test_concurrency_conflict_fails_fast(self) -> None:
        runner = SlurmIntraRunner(srun_launch_concurrency=4)
        with pytest.raises(ValueError, match="conflicts"):
            runner.create_lifecycle_router(
                RunnerResources(), BatchStrategy(max_workers=2), 0, "op"
            )

    def test_non_positive_max_workers_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(ValueError, match="max_workers"):
            runner.create_lifecycle_router(
                RunnerResources(), BatchStrategy(max_workers=0), 0, "op"
            )

    def test_non_positive_poll_timeout_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="max_poll_time"):
            SlurmIntraRunner(max_poll_time=0)

    @pytest.mark.parametrize("value", [True, 1.5, "2"])
    def test_non_integer_launch_concurrency_fails_fast(self, value: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            SlurmIntraRunner(srun_launch_concurrency=value)  # type: ignore[arg-type]

    def test_unsupported_resource_extra_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(ValueError, match="Unsupported"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"slurm_constraint": "a100"}),
                BatchStrategy(),
                0,
                "op",
            )

    def test_non_string_gres_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(TypeError, match="slurm_gres"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"slurm_gres": 1}),
                BatchStrategy(),
                0,
                "op",
            )

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--ntasks=2"],
            ["--nodes", "2"],
            ["-n2"],
            ["-N2"],
            ["--cpus-per-task", "8"],
            ["-c8"],
            ["--mem=8G"],
            ["--gres", "gpu:2"],
            ["-G2"],
            ["--gpus-per-socket=2"],
            ["--tres-per-task=gres/gpu:2"],
            ["--sockets-per-node=2"],
            ["-B2:4:1"],
            ["--time=01:00:00"],
            ["-t60"],
            ["--threads=2"],
            ["--async"],
            ["--asy"],
            ["--jobid=12"],
            ["--jobi", "12"],
        ],
    )
    def test_managed_srun_argument_fails_fast(self, arguments: list[str]) -> None:
        runner = SlurmIntraRunner()

        with pytest.raises(ValueError, match="Artisan-managed"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"srun_args": arguments}),
                BatchStrategy(),
                0,
                "op",
            )

    @patch.dict("os.environ", {}, clear=True)
    def test_validate_operation_warns_outside_allocation(self) -> None:
        operation = MagicMock(name="operation")
        operation.name = "train"
        with pytest.warns(UserWarning, match="SLURM_JOB_ID"):
            SlurmIntraRunner().validate_operation(operation)


class TestSlurmIntraLifecycleRouter:
    def test_command_maps_resources(self, tmp_path) -> None:
        router = _router(
            tmp_path,
            slurm_gres="gpu:a100:1",
            srun_args=["--exclusive"],
        )
        command = router._build_command(tmp_path / "step")
        assert command[:4] == ["srun", "--exact", "--mpi=none", "-n1"]
        assert command[4:6] == ["--gres", "gpu:a100:1"]
        assert "8G" in command
        assert "2" in command
        assert "00:30:00" in command
        assert "--exclusive" in command
        assert "artisan_submitit.srun_worker" in command

    @patch.dict("os.environ", {}, clear=True)
    def test_dispatch_fails_without_allocation(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
            _router(tmp_path).dispatch([], MagicMock())

    def test_cancel_is_prompt_and_does_not_touch_processes(self, tmp_path) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step")
        router._submitted.append(item)
        router._dispatch_started = True

        first = router.cancel()
        second = router.cancel()

        assert first.status is second.status is CancellationStatus.REQUESTED
        item.process.poll.assert_not_called()
        item.process.terminate.assert_not_called()
        assert not (item.folder / "cancel").exists()

    def test_idle_cancel_does_not_poison_later_dispatch(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        router = _router(tmp_path)
        assert router.cancel().status is CancellationStatus.REJECTED
        assert not router._cancelled.is_set()
        assert router.run([], MagicMock()) == []
        assert router.cancel().status is CancellationStatus.REJECTED

    def test_cancellation_before_launch_prevents_popen(self, tmp_path) -> None:
        router = _router(tmp_path)
        router._dispatch_started = True
        router.cancel()
        with patch("artisan_submitit.srun.subprocess.Popen") as popen:
            with pytest.raises(RuntimeError, match="cancelled before launch"):
                router._launch(0, [_unit()], MagicMock())
        popen.assert_not_called()

    def test_process_launched_during_cancel_receives_persistent_request(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path)
        router._dispatch_started = True
        process = _process()

        def launched(*args, **kwargs):
            router.cancel()
            return process

        with patch("artisan_submitit.srun.subprocess.Popen", side_effect=launched):
            item = router._launch(0, [_unit()], MagicMock())

        assert (item.folder / "cancel").exists()
        assert router.submitted_process_ids == (10,)
        process.terminate.assert_not_called()

    def test_every_owned_process_receives_request_before_collection(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path)
        router._dispatch_started = True
        items = [_submitted(tmp_path / str(index)) for index in range(3)]
        router._submitted.extend(items)
        router.cancel()

        router._advance_processes()

        assert all((item.folder / "cancel").exists() for item in items)
        assert all(item.terminated_at is None for item in items)

    def test_wait_for_slot_honors_process_cap(self, tmp_path) -> None:
        router = _router(tmp_path, max_workers=1)
        item = _submitted(tmp_path / "step")
        router._submitted.append(item)
        with patch.object(
            router,
            "_sleep",
            side_effect=lambda: setattr(item.process.poll, "return_value", 0),
        ) as sleep:
            router._wait_for_slot()
        sleep.assert_called_once()

    def test_process_deadline_requests_native_cleanup_before_signalling(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path, max_poll_time=1)
        item = _submitted(tmp_path / "step")
        item.launched_at -= 2
        router._submitted.append(item)

        router._advance_processes()

        assert item.timeout_error
        assert "exceeded" in item.timeout_error
        assert (item.folder / "cancel").exists()
        item.process.terminate.assert_not_called()

    @pytest.mark.parametrize(
        ("job", "step"),
        [
            ("999", "7"),
            ("123", "batch"),
            ("123", "extern"),
            ("123", "7.0"),
            ("123", "7;8"),
        ],
    )
    def test_step_identity_rejects_unowned_or_nonnumeric_receipt(
        self, tmp_path, job, step
    ) -> None:
        router = _router(tmp_path)
        router._allocation_id = "123"
        item = _submitted(tmp_path / "step")
        _identity(item, job, step)
        assert router._step_identity(item) is None

    def test_fallback_cancels_only_validated_step_and_retries_failure(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        router = _router(tmp_path)
        router._allocation_id = "123"
        router._dispatch_started = True
        item = _submitted(tmp_path / "step")
        _identity(item, "123", "7")
        router._submitted.append(item)
        router.cancel()
        with (
            patch(
                "artisan_submitit.srun.cancel_jobs", side_effect=["unavailable", None]
            ) as cancel,
            patch(
                "artisan_submitit.srun.query_job_states",
                return_value={"123.7": _terminal()},
            ),
        ):
            router._advance_processes()
            router._next_scheduler_poll = 0
            router._advance_processes()
        assert cancel.call_count == 2
        assert all(call.args == (["123.7"],) for call in cancel.call_args_list)
        assert item.scheduler_terminal
        item.process.terminate.assert_called_once()
        item.process.kill.assert_called_once()
        item.process.wait.assert_not_called()

    def test_missing_receipt_never_cancels_entire_allocation(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        router = _router(tmp_path)
        router._allocation_id = "123"
        router._dispatch_started = True
        item = _submitted(tmp_path / "step")
        router._submitted.append(item)
        router.cancel()
        with patch("artisan_submitit.srun.cancel_jobs") as cancel:
            router._advance_processes()
        cancel.assert_not_called()
        item.process.terminate.assert_called_once()

    def test_launcher_termination_error_retries_without_waiting(
        self, tmp_path: Path
    ) -> None:
        router = _router(tmp_path, cancel_grace_period=1)
        item = _submitted(tmp_path / "step")
        item.process.terminate.side_effect = [OSError("retry"), None]
        router._signal_launcher(item, time.monotonic())
        assert item.terminated_at is None
        router._signal_launcher(item, time.monotonic())
        assert item.terminated_at is not None
        assert item.process.terminate.call_count == 2
        item.process.wait.assert_not_called()

    @pytest.mark.parametrize("owner_unknown", [True, False])
    def test_process_timeout_retains_nested_or_owner_uncertainty(
        self, tmp_path: Path, owner_unknown: bool
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=0)
        item.timeout_error = "normal process timeout"
        acknowledgement = CancellationAcknowledgement(
            CancellationStatus.UNKNOWN, "nested cleanup unresolved"
        )
        result = replace(
            _result("first", success=False),
            worker_log="preserved native log",
            cancellation_acknowledgement=None if owner_unknown else acknowledgement,
        )
        _publish(
            item.folder,
            [result],
            CancellationStatus.UNKNOWN
            if owner_unknown
            else CancellationStatus.CONFIRMED,
        )
        router._observe_outcome(item)

        results, owner = router._process_results(item)

        assert results[0].execution_run_ids == ["first"]
        assert results[0].worker_log == "preserved native log"
        assert (
            results[0].cancellation_acknowledgement.status is CancellationStatus.UNKNOWN
        )
        assert owner.status is (
            CancellationStatus.UNKNOWN
            if owner_unknown
            else CancellationStatus.CONFIRMED
        )

    def test_failed_launcher_retains_valid_result_diagnostics(
        self, tmp_path: Path
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=1)
        _publish(
            item.folder,
            [replace(_result("first"), worker_log="native diagnostic")],
        )
        router._observe_outcome(item)

        results, owner = router._process_results(item)

        assert owner.status is CancellationStatus.UNKNOWN
        assert results[0].execution_run_ids == ["first"]
        assert results[0].worker_log == "native diagnostic"
        assert not results[0].success

    @pytest.mark.parametrize("returncode", [None, 0])
    def test_per_process_timeout_bounds_large_poll_interval(
        self, tmp_path: Path, returncode: int | None
    ) -> None:
        router = _router(tmp_path, poll_interval=60)
        item = _submitted(tmp_path / "step", returncode=returncode)
        item.requested_at = time.monotonic()
        router._submitted.append(item)

        with patch.object(router._cancelled, "wait") as wait:
            router._sleep()

        assert 0 < wait.call_args.kwargs["timeout"] <= 0.1

    def test_later_outer_cancel_does_not_restart_existing_cleanup_deadline(
        self, tmp_path: Path
    ) -> None:
        router = _router(tmp_path)
        router._dispatch_started = True
        item = _submitted(tmp_path / "step")
        item.requested_at = time.monotonic() - 24

        router.cancel()

        assert 0 < router._remaining(item) <= 1
        assert not router._cooperating(item)

    def test_malformed_bootstrap_pickle_yields_aligned_unknown(
        self, tmp_path: Path
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=1)
        (item.folder / "result.pkl").write_bytes(b"\x80\xff")

        results, owner = router._process_results(item)

        assert len(results) == 1
        assert not results[0].success
        assert owner.status is CancellationStatus.UNKNOWN

    def test_collection_error_requests_cleanup_and_preserves_unknown(
        self, tmp_path: Path
    ) -> None:
        router = _router(tmp_path)
        router._dispatch_started = True
        item = _submitted(tmp_path / "step")
        router._submitted.append(item)

        with patch.object(router, "_execute_batches", side_effect=OSError("lost")):
            results = router._run_batches([item.units], MagicMock())

        assert len(results) == 1
        assert not results[0].success
        assert (item.folder / "cancel").exists()
        assert router.cancellation_acknowledgement.status is CancellationStatus.UNKNOWN

    def test_outcome_read_retries_visibility_without_transport_result(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=0)
        router._submitted.append(item)
        expected = _result("first")
        with patch.object(
            router, "_sleep", side_effect=lambda: _publish(item.folder, [expected])
        ):
            assert router._collect_process(item) == [expected]
        item.process.wait.assert_not_called()

    def test_transient_outcome_read_error_does_not_replace_valid_results(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=0)
        router._submitted.append(item)
        expected = _result("first")
        with patch(
            "artisan_submitit.srun.read_worker_outcome",
            side_effect=[ValueError("transient EOF"), WorkerOutcome([expected], None)],
        ):
            router._observe_outcome(item)
            router._observe_outcome(item)
        results, owner = router._process_results(item)
        assert results == [expected]
        assert owner is None

    def test_wrong_cardinality_cannot_confirm_cancellation(self, tmp_path) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=0)
        _publish(item.folder, [], CancellationStatus.CONFIRMED)
        router._observe_outcome(item)
        results, owner = router._process_results(item)
        assert owner.status is CancellationStatus.UNKNOWN
        assert len(results) == 1
        assert "cardinality" in results[0].error

    def test_normal_owner_unknown_preserves_uncertainty_and_logs(
        self, tmp_path
    ) -> None:
        router = _router(tmp_path)
        item = _submitted(tmp_path / "step", returncode=0)
        router._submitted.append(item)
        _publish(item.folder, [_result("first")], CancellationStatus.UNKNOWN)
        (item.folder / "stdout.log").write_text("owned worker log")
        results = router._collect_process(item)
        assert not results[0].success
        assert results[0].execution_run_ids == ["first"]
        assert (
            results[0].cancellation_acknowledgement.status is CancellationStatus.UNKNOWN
        )
        assert "owned worker log" in results[0].worker_log

    def test_actual_run_waits_for_worker_confirmation_and_joins_collection(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        router = _router(tmp_path)
        cancelled = threading.Event()
        workers: list[threading.Thread] = []

        def launched(command, **kwargs):
            folder = Path(command[-1])
            process = _process()

            def finish():
                cancelled.set()
                _wait_for(lambda: (folder / "cancel").exists())
                _publish(
                    folder,
                    [_result("first", success=False)],
                    CancellationStatus.CONFIRMED,
                )
                process.returncode = 0
                process.poll.return_value = 0

            worker = threading.Thread(target=finish)
            workers.append(worker)
            worker.start()
            return process

        with patch("artisan_submitit.srun.subprocess.Popen", side_effect=launched):
            results = router.run([_unit()], MagicMock(), cancelled)
        for worker in workers:
            worker.join(timeout=2)
        assert len(results) == 1
        assert (
            router.cancellation_acknowledgement.status is CancellationStatus.CONFIRMED
        )
        assert router._thread is not None
        assert not router._thread.is_alive()
        assert router.cancel().status is CancellationStatus.CONFIRMED

    def test_actual_run_rejects_cancel_after_natural_completion(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        router = _router(tmp_path)

        def launched(command, **kwargs):
            _publish(Path(command[-1]), [_result("first")])
            return _process(0)

        with patch("artisan_submitit.srun.subprocess.Popen", side_effect=launched):
            assert router.run([_unit()], MagicMock())[0].success
        assert router.cancel().status is CancellationStatus.REJECTED

    def test_actual_run_returns_unknown_for_surviving_launcher(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setattr("artisan_submitit.cancellation.CANCELLATION_TIMEOUT", 0.2)
        monkeypatch.setattr("artisan_submitit.cancellation.COOPERATION_TIMEOUT", 0)
        router = _router(tmp_path)
        cancelled = threading.Event()
        process = _process()

        def launched(command, **kwargs):
            cancelled.set()
            return process

        with patch("artisan_submitit.srun.subprocess.Popen", side_effect=launched):
            started = time.monotonic()
            results = router.run([_unit()], MagicMock(), cancelled)
        assert time.monotonic() - started < 2
        assert router.cancellation_acknowledgement.status is CancellationStatus.UNKNOWN
        assert (
            results[0].cancellation_acknowledgement.status is CancellationStatus.UNKNOWN
        )
        process.kill.assert_called_once()
        router._thread.join(timeout=2)
        assert not router._thread.is_alive()


def _unit() -> MagicMock:
    unit = MagicMock(step_run_id=None)
    unit.get_batch_size.return_value = 1
    return unit


def _result(run_id: str, success: bool = True) -> UnitResult:
    return UnitResult(success, None if success else "cancelled", 1, [run_id])


def _process(returncode: int | None = None) -> MagicMock:
    process = MagicMock(pid=10, returncode=returncode)
    process.poll.return_value = returncode
    return process


def _submitted(folder: Path, returncode: int | None = None) -> _SrunProcess:
    folder.mkdir()
    (folder / "stdout.log").touch()
    (folder / "stderr.log").touch()
    return _SrunProcess(_process(returncode), [_unit()], folder, time.monotonic())


def _publish(
    folder: Path, results: list[UnitResult], status: CancellationStatus | None = None
) -> None:
    acknowledgement = (
        CancellationAcknowledgement(status, "worker evidence") if status else None
    )
    with (folder / "outcome.pkl").open("wb") as stream:
        pickle.dump(WorkerOutcome(results, acknowledgement), stream)


def _identity(item: _SrunProcess, job: str, step: str) -> None:
    (item.folder / "identity.json").write_text(
        json.dumps(
            {
                "job_id": job,
                "step_id": step,
                "array_job_id": None,
                "array_task_id": None,
            }
        )
    )


def _terminal() -> SchedulerObservation:
    now = datetime.now(UTC)
    return SchedulerObservation("CANCELLED", now, now, now)


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "worker condition timed out"
        time.sleep(0.005)
