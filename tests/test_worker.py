"""Shared worker ownership, evidence publication, and real native execution."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cloudpickle
import polars as pl
import pytest
from artisan.operations.examples import DataGenerator, StreamingEcho, Wait
from artisan.orchestration.runner_api import (
    CancellationAcknowledgement,
    CancellationStatus,
    ExecutionUnit,
    RuntimeEnvironment,
    UnitResult,
)
from artisan.schemas.operation_config.environment_spec import LocalEnvironmentSpec
from artisan.schemas.specs.input_models import ExecuteInput
from artisan.storage.io.commit import DeltaCommitter
from artisan.storage.io.staging import StagingManager
from artisan.utils.external_tools import run_command

from artisan_submitit.utils import write_atomic
from artisan_submitit.worker import (
    WorkerOutcome,
    read_worker_identity,
    read_worker_outcome,
    run_worker_batch,
)


def _unit() -> ExecutionUnit:
    return ExecutionUnit(operation=Wait(), inputs={}, step_number=3)


def _result() -> UnitResult:
    return UnitResult(True, None, 1, ["execution-1"], worker_log="worker output")


def _runtime(tmp_path: Path, **values: Any) -> RuntimeEnvironment:
    return RuntimeEnvironment(
        delta_root=str(tmp_path / "delta"),
        staging_root=str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        **values,
    )


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_request_before_dispatch_never_starts_native_work(
    factory: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "7")
    (tmp_path / "cancel").touch()

    outcome = run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    factory.assert_not_called()
    assert outcome.cancellation_acknowledgement.status == CancellationStatus.CONFIRMED
    assert len(outcome.results) == 1
    assert not outcome.results[0].success
    assert read_worker_outcome(tmp_path) == outcome
    assert read_worker_identity(tmp_path)["job_id"] == "123"
    assert read_worker_identity(tmp_path)["step_id"] == "7"
    assert not list(tmp_path.glob("*.tmp"))


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_native_batch_preserves_runtime_and_original_results(
    factory: MagicMock, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path, worker_id=42, worker_id_env_var="SLURM_STEP_ID")
    router = factory.return_value
    router.run.return_value = [_result()]
    router.cancellation_acknowledgement = None

    outcome = run_worker_batch([_unit()], runtime, tmp_path)

    assert outcome.results == [_result()]
    assert outcome.cancellation_acknowledgement is None
    assert router.run.call_args.args[1] is runtime
    strategy = factory.call_args.args[1]
    assert strategy.max_workers == 1
    assert strategy.units_per_worker == 1
    assert factory.call_args.args[2] == 3
    assert read_worker_outcome(tmp_path) == outcome
    assert not any(
        thread.name == "artisan-submitit-cancel-watcher"
        for thread in threading.enumerate()
    )


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_request_during_native_startup_reaches_event(
    factory: MagicMock, tmp_path: Path
) -> None:
    router = factory.return_value
    router.cancellation_acknowledgement = CancellationAcknowledgement(
        CancellationStatus.CONFIRMED
    )

    def native_run(
        *args: Any, cancel_event: threading.Event, **kwargs: Any
    ) -> list[UnitResult]:
        (tmp_path / "cancel").touch()
        assert cancel_event.wait(timeout=2)
        return [_result()]

    router.run.side_effect = native_run
    outcome = run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    assert outcome.cancellation_acknowledgement.status == CancellationStatus.CONFIRMED
    assert outcome.results[0].cancellation_acknowledgement is None


@pytest.mark.parametrize(
    "status", [CancellationStatus.UNKNOWN, CancellationStatus.REQUESTED]
)
@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_unresolved_native_owner_fails_results_without_losing_diagnostics(
    factory: MagicMock, status: CancellationStatus, tmp_path: Path
) -> None:
    nested = CancellationAcknowledgement(
        CancellationStatus.UNKNOWN, "remote cleanup unknown"
    )
    original = replace(
        _result(), error="unit diagnostic", cancellation_acknowledgement=nested
    )
    router = factory.return_value
    router.run.return_value = [original]
    router.cancellation_acknowledgement = CancellationAcknowledgement(
        status, "owner cleanup unknown"
    )

    outcome = run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    assert outcome.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN
    result = outcome.results[0]
    assert not result.success
    assert result.execution_run_ids == original.execution_run_ids
    assert result.worker_log == original.worker_log
    assert result.error == "unit diagnostic"
    assert result.cancellation_acknowledgement == nested


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_watcher_error_requests_cleanup_and_preserves_uncertainty(
    factory: MagicMock, tmp_path: Path
) -> None:
    router = factory.return_value
    router.cancellation_acknowledgement = CancellationAcknowledgement(
        CancellationStatus.CONFIRMED
    )

    def native_run(
        *args: Any, cancel_event: threading.Event, **kwargs: Any
    ) -> list[UnitResult]:
        assert cancel_event.wait(timeout=2)
        return [_result()]

    router.run.side_effect = native_run
    with patch(
        "artisan_submitit.worker._request_exists",
        side_effect=[False, PermissionError("request denied")],
    ):
        outcome = run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    assert outcome.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN
    assert "request denied" in outcome.cancellation_acknowledgement.message
    assert (
        outcome.results[0].cancellation_acknowledgement.status
        == CancellationStatus.UNKNOWN
    )
    assert not outcome.results[0].success


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_native_exception_stops_watcher_without_publishing_closure(
    factory: MagicMock, tmp_path: Path
) -> None:
    factory.return_value.run.side_effect = RuntimeError("native bootstrap failed")

    with pytest.raises(RuntimeError, match="native bootstrap failed"):
        run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    assert read_worker_outcome(tmp_path) is None
    assert read_worker_identity(tmp_path) is not None
    assert not any(
        thread.name == "artisan-submitit-cancel-watcher"
        for thread in threading.enumerate()
    )


@patch("artisan_submitit.worker.Runner.LOCAL.create_lifecycle_router")
def test_native_unknown_does_not_wait_for_unresolved_collection(
    factory: MagicMock, tmp_path: Path
) -> None:
    router = factory.return_value
    router.run.return_value = [_result()]
    router.is_done.return_value = False
    router.cancellation_acknowledgement = CancellationAcknowledgement(
        CancellationStatus.UNKNOWN, "Native cleanup deadline expired"
    )

    outcome = run_worker_batch([_unit()], _runtime(tmp_path), tmp_path)

    assert outcome.cancellation_acknowledgement.status == CancellationStatus.UNKNOWN
    assert not outcome.results[0].success
    assert read_worker_outcome(tmp_path) == outcome
    router.collect.assert_not_called()
    router.cancel.assert_not_called()


@pytest.mark.parametrize("reader", [read_worker_outcome, read_worker_identity])
def test_missing_control_file_is_pending(reader: Any, tmp_path: Path) -> None:
    assert reader(tmp_path) is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        WorkerOutcome([], CancellationAcknowledgement(CancellationStatus.REQUESTED)),
        WorkerOutcome(["bad result"], None),
    ],
)
def test_invalid_outcome_is_rejected(value: object, tmp_path: Path) -> None:
    (tmp_path / "outcome.pkl").write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError, match=r"malformed|invalid"):
        read_worker_outcome(tmp_path)


def test_corrupt_pickle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "outcome.pkl").write_bytes(b"broken pickle")
    with pytest.raises(ValueError, match="unreadable"):
        read_worker_outcome(tmp_path)


def test_reader_exposes_owner_uncertainty(tmp_path: Path) -> None:
    acknowledgement = CancellationAcknowledgement(
        CancellationStatus.UNKNOWN, "native owner missing"
    )
    (tmp_path / "outcome.pkl").write_bytes(
        pickle.dumps(WorkerOutcome([_result()], acknowledgement))
    )
    outcome = read_worker_outcome(tmp_path)
    assert not outcome.results[0].success
    assert outcome.results[0].cancellation_acknowledgement == acknowledgement
    assert outcome.results[0].execution_run_ids == ["execution-1"]


@pytest.mark.parametrize("value", [{}, [], {"job_id": 123}, "invalid json"])
def test_invalid_identity_is_rejected(value: object, tmp_path: Path) -> None:
    (tmp_path / "identity.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="malformed"):
        read_worker_identity(tmp_path)


def test_atomic_publish_failure_preserves_previous_file(tmp_path: Path) -> None:
    destination = tmp_path / "outcome.pkl"
    destination.write_bytes(b"previous")
    with patch.object(Path, "replace", side_effect=OSError("rename failed")):
        with pytest.raises(OSError, match="rename failed"):
            write_atomic(destination, b"new")
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


class ObservedWait(Wait):
    """Expose actual execution admission before invoking the current Wait."""

    name = "observed_wait"

    def execute_function(self, inputs: ExecuteInput) -> dict[str, Any]:
        Path(os.environ["ARTISAN_SUBMITIT_TEST_MARKER"]).write_text(str(os.getpid()))
        return super().execute_function(inputs)


class ObservedCommand(Wait):
    """Expose the exact managed-command PID before waiting for cancellation."""

    name = "observed_command"

    def execute_function(self, inputs: ExecuteInput) -> dict[str, Any]:
        run_command(
            LocalEnvironmentSpec(),
            [
                sys.executable,
                "-c",
                "import os,time; from pathlib import Path; "
                "Path(os.environ['ARTISAN_SUBMITIT_TEST_MARKER']).write_text(str(os.getpid())); time.sleep(60)",
            ],
            cwd=inputs.execute_dir,
        )
        return {"elapsed": 60}


def _prepare_real_worker(
    tmp_path: Path, unit: ExecutionUnit, runtime: RuntimeEnvironment
) -> dict[str, str]:
    fs = runtime.storage.filesystem()
    DeltaCommitter(
        runtime.delta_root, StagingManager(runtime.staging_root, fs), fs=fs
    ).initialize_tables()
    with (tmp_path / "job.pkl").open("wb") as stream:
        cloudpickle.dump(partial(run_worker_batch, [unit], runtime, tmp_path), stream)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(__file__).parents[1]),
            str(tmp_path),
            environment.get("PYTHONPATH", ""),
        )
    )
    return environment


@pytest.mark.parametrize("identity_variable", ["SLURM_ARRAY_TASK_ID", "SLURM_STEP_ID"])
def test_real_worker_preserves_identity_and_explicit_applications(
    tmp_path: Path, identity_variable: str
) -> None:
    trusted, ambient = "provider_trusted_app", "provider_ambient_app"
    for module in (trusted, ambient):
        (tmp_path / f"{module}.py").write_text(
            f"from pathlib import Path\nPath({str(tmp_path / module)!r}).touch()\n"
        )
    runtime = _runtime(
        tmp_path,
        worker_id=42,
        worker_id_env_var=identity_variable,
        application_modules=(trusted,),
    )
    unit = ExecutionUnit(
        operation=DataGenerator(params=DataGenerator.Params(count=1, rows_per_file=1)),
        inputs={},
        step_number=0,
    )
    environment = _prepare_real_worker(tmp_path, unit, runtime)
    environment.update({identity_variable: "7", "ARTISAN_LOAD_MODULES": ambient})

    completed = subprocess.run(
        [sys.executable, "-m", "artisan_submitit.srun_worker", str(tmp_path)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    outcome = read_worker_outcome(tmp_path)
    assert outcome.cancellation_acknowledgement is None
    assert outcome.results[0].success, outcome.results[0].error
    assert (tmp_path / trusted).exists()
    assert not (tmp_path / ambient).exists()
    records = list(Path(runtime.staging_root).rglob("executions.parquet"))
    assert len(records) == 1
    assert pl.read_parquet(records[0]).item(0, "source_worker") == 7
    assert runtime.worker_id == 42


def test_real_worker_records_managed_command(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    unit = ExecutionUnit(
        operation=StreamingEcho(params=StreamingEcho.Params(seconds=1)),
        inputs={},
        step_number=0,
    )
    environment = _prepare_real_worker(tmp_path, unit, runtime)
    completed = subprocess.run(
        [sys.executable, "-m", "artisan_submitit.srun_worker", str(tmp_path)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    outcome = read_worker_outcome(tmp_path)
    assert outcome.results[0].success, outcome.results[0].error
    record = next(Path(runtime.staging_root).rglob("executions.parquet"))
    row = pl.read_parquet(record).row(0, named=True)
    assert json.loads(row["command_recording"])["commands"]
    assert "streaming_echo line 1 / 1" in row["tool_output"]


@pytest.mark.parametrize("operation", [ObservedWait, ObservedCommand])
def test_real_worker_cancellation_closes_native_work(
    tmp_path: Path, operation: type[Wait]
) -> None:
    runtime = _runtime(tmp_path)
    unit = ExecutionUnit(
        operation=operation(params=operation.Params(duration=60)),
        inputs={},
        step_number=0,
        step_run_id="worker_cancellation",
    )
    environment = _prepare_real_worker(tmp_path, unit, runtime)
    marker = tmp_path / "started"
    environment["ARTISAN_SUBMITIT_TEST_MARKER"] = str(marker)
    process = subprocess.Popen(
        [sys.executable, "-m", "artisan_submitit.srun_worker", str(tmp_path)],
        env=environment,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists(), process.communicate(timeout=1)
        work_pid = int(marker.read_text())
        (tmp_path / "cancel").touch()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        outcome = read_worker_outcome(tmp_path)
        assert (
            outcome.cancellation_acknowledgement.status == CancellationStatus.CONFIRMED
        )
        with pytest.raises(ProcessLookupError):
            os.kill(work_pid, 0)
        assert len(outcome.results) == 1
        assert not outcome.results[0].success
    finally:
        (tmp_path / "cancel").touch()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
