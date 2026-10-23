"""Shared compute-node ownership and cancellation evidence for both transports."""

from __future__ import annotations

import json
import os
import pickle
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from artisan.orchestration import Runner
from artisan.orchestration.runner_api import (
    BatchStrategy,
    CancellationAcknowledgement,
    CancellationStatus,
    ExecutionUnit,
    RunnerResources,
    RuntimeEnvironment,
    UnitResult,
    failure_results_for_units,
)

from artisan_submitit.utils import write_atomic

_NATIVE_CANCELLATION_SECONDS = 8.0
_WATCH_INTERVAL = 0.05
_WATCHER_JOIN_SECONDS = 1.0
_IDENTITY_ENVIRONMENT = {
    "job_id": "SLURM_JOB_ID",
    "step_id": "SLURM_STEP_ID",
    "array_job_id": "SLURM_ARRAY_JOB_ID",
    "array_task_id": "SLURM_ARRAY_TASK_ID",
}


@dataclass(frozen=True)
class WorkerOutcome:
    """Ordered unit results and separate native batch-owner evidence."""

    results: list[UnitResult]
    cancellation_acknowledgement: CancellationAcknowledgement | None


def run_worker_batch(
    units: list[ExecutionUnit],
    runtime_env: RuntimeEnvironment,
    control_folder: Path,
) -> WorkerOutcome:
    """Execute one packed batch under core ownership and publish its outcome.

    The parent creates the unique control directory before submission. Only the
    native router owns worker and managed-command cleanup; scheduler signals do
    not supply this evidence.
    """
    identity = {key: os.environ.get(env) for key, env in _IDENTITY_ENVIRONMENT.items()}
    write_atomic(control_folder / "identity.json", json.dumps(identity).encode())
    request = control_folder / "cancel"
    if _request_exists(request):
        acknowledgement = CancellationAcknowledgement(
            CancellationStatus.CONFIRMED, "Batch cancelled before native dispatch"
        )
        outcome = WorkerOutcome(
            failure_results_for_units(units, acknowledgement.message or "Cancelled"),
            acknowledgement,
        )
    else:
        outcome = _run_native_batch(units, runtime_env, request)
    write_atomic(
        control_folder / "outcome.pkl",
        pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL),
    )
    return outcome


def _run_native_batch(
    units: list[ExecutionUnit], runtime_env: RuntimeEnvironment, request: Path
) -> WorkerOutcome:
    """Compose the public native runner without changing worker runtime traits."""
    router = Runner.LOCAL.create_lifecycle_router(
        RunnerResources.model_validate({}),
        BatchStrategy.model_validate(
            {"max_workers": 1, "units_per_worker": max(1, len(units))}
        ),
        units[0].step_number if units else 0,
        "artisan-submitit batch",
    )
    cancelled, stopped = threading.Event(), threading.Event()
    errors: list[OSError] = []
    watcher = threading.Thread(
        target=_watch_request,
        args=(request, cancelled, stopped, errors),
        name="artisan-submitit-cancel-watcher",
        daemon=True,
    )
    watcher.start()
    try:
        results = router.run(
            units,
            runtime_env,
            cancel_event=cancelled,
            cancellation_confirmation_timeout=_NATIVE_CANCELLATION_SECONDS,
        )
    finally:
        stopped.set()
        watcher.join(timeout=_WATCHER_JOIN_SECONDS)
    acknowledgement = router.cancellation_acknowledgement
    if errors or watcher.is_alive():
        acknowledgement = CancellationAcknowledgement(
            CancellationStatus.UNKNOWN,
            f"Cancellation request observation failed: {errors[0]}"
            if errors
            else "Cancellation request watcher did not stop",
        )
    elif acknowledgement is not None and (
        not _valid_acknowledgement(acknowledgement)
        or acknowledgement.status == CancellationStatus.REQUESTED
    ):
        acknowledgement = CancellationAcknowledgement(
            CancellationStatus.UNKNOWN, "Native owner returned unresolved evidence"
        )
    return _preserve_owner_uncertainty(WorkerOutcome(results, acknowledgement))


def _request_exists(request: Path) -> bool:
    """Distinguish an absent request from inaccessible cancellation storage."""
    try:
        request.stat()
    except FileNotFoundError:
        return False
    return True


def _watch_request(
    request: Path,
    cancelled: threading.Event,
    stopped: threading.Event,
    errors: list[OSError],
) -> None:
    """Stop native admission when cancellation storage becomes unreadable."""
    while not stopped.is_set():
        try:
            requested = _request_exists(request)
        except OSError as exc:
            errors.append(exc)
            cancelled.set()
            return
        if requested:
            cancelled.set()
            return
        stopped.wait(_WATCH_INTERVAL)


def _valid_acknowledgement(value: object) -> bool:
    """Require actual lifecycle enum values in trusted transport evidence."""
    if not isinstance(value, CancellationAcknowledgement):
        return False
    status: object = value.status
    message: object = value.message
    return isinstance(status, CancellationStatus) and (
        message is None or isinstance(message, str)
    )


def _valid_unit_result(value: object) -> bool:
    """Check unit type and nested evidence at the deserialization boundary."""
    return isinstance(value, UnitResult) and (
        value.cancellation_acknowledgement is None
        or _valid_acknowledgement(value.cancellation_acknowledgement)
    )


def _preserve_owner_uncertainty(outcome: WorkerOutcome) -> WorkerOutcome:
    """Expose owner uncertainty even when the parent saw no cancellation event."""
    owner = outcome.cancellation_acknowledgement
    if owner is None or owner.status != CancellationStatus.UNKNOWN:
        return outcome
    return WorkerOutcome(
        [
            replace(
                result,
                success=False,
                error=result.error or owner.message or "Native ownership is unknown",
                cancellation_acknowledgement=(
                    result.cancellation_acknowledgement
                    if result.cancellation_acknowledgement is not None
                    and result.cancellation_acknowledgement.status
                    == CancellationStatus.UNKNOWN
                    else owner
                ),
            )
            for result in outcome.results
        ],
        owner,
    )


def read_worker_outcome(control_folder: Path) -> WorkerOutcome | None:
    """Read complete current-format evidence; only a missing file means pending.

    Raises:
        ValueError: If the envelope or its cancellation evidence is malformed.
        OSError: If the outcome cannot be read.
    """
    try:
        data = (control_folder / "outcome.pkl").read_bytes()
    except FileNotFoundError:
        return None
    try:
        outcome: object = pickle.loads(data)
    except Exception as exc:
        msg = "Worker returned an unreadable outcome"
        raise ValueError(msg) from exc
    if not isinstance(outcome, WorkerOutcome) or not isinstance(outcome.results, list):
        msg = "Worker returned a malformed outcome"
        raise ValueError(msg)
    owner = outcome.cancellation_acknowledgement
    if owner is not None and (
        not _valid_acknowledgement(owner)
        or owner.status == CancellationStatus.REQUESTED
    ):
        msg = "Worker returned invalid final ownership evidence"
        raise ValueError(msg)
    if any(not _valid_unit_result(result) for result in outcome.results):
        msg = "Worker returned malformed unit results"
        raise ValueError(msg)
    return _preserve_owner_uncertainty(outcome)


def read_worker_identity(control_folder: Path) -> dict[str, str | None] | None:
    """Read the receipt shape without authorizing any scheduler identity.

    Raises:
        ValueError: If the receipt is not four nullable identity strings.
        OSError: If the receipt cannot be read.
    """
    try:
        data = (control_folder / "identity.json").read_bytes()
    except FileNotFoundError:
        return None
    try:
        identity: object = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        msg = "Worker returned an unreadable identity receipt"
        raise ValueError(msg) from exc
    if (
        not isinstance(identity, dict)
        or set(identity) != set(_IDENTITY_ENVIRONMENT)
        or any(
            value is not None and not isinstance(value, str)
            for value in identity.values()
        )
    ):
        msg = "Worker returned a malformed identity receipt"
        raise ValueError(msg)
    return cast("dict[str, str | None]", identity)
