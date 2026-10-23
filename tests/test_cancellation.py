"""Cancellation clocks, nested evidence and fresh exact scheduler observations."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from artisan.orchestration.runner_api import (
    CancellationAcknowledgement,
    CancellationStatus,
    UnitResult,
)

from artisan_submitit.cancellation import (
    CancellationState,
    cancel_jobs,
    query_job_states,
    reduce_cancellation,
)


def _ack(status: str) -> CancellationAcknowledgement:
    return CancellationAcknowledgement(CancellationStatus(status), status)


def _result(status: str) -> UnitResult:
    return UnitResult(
        success=False,
        error="cancelled",
        item_count=1,
        execution_run_ids=[],
        cancellation_acknowledgement=_ack(status),
    )


@pytest.mark.parametrize(
    ("owners", "units", "prevented", "expected"),
    [
        ([], [], False, "rejected"),
        ([], [], True, "confirmed"),
        (["rejected", "confirmed"], [], False, "confirmed"),
        (["confirmed"], ["rejected"], False, "rejected"),
        (["confirmed"], ["unknown"], True, "unknown"),
        (["unknown"], ["rejected"], True, "unknown"),
        (["requested"], [], True, "unknown"),
        ([], ["requested"], True, "unknown"),
        ([], ["confirmed"], False, "confirmed"),
    ],
)
def test_evidence_precedence(owners, units, prevented, expected) -> None:
    evidence = reduce_cancellation(
        [_ack(s) for s in owners], [_result(s) for s in units], prevented
    )
    assert evidence.status == CancellationStatus(expected)


def test_required_missing_proof_overrides_confirmation() -> None:
    evidence = reduce_cancellation([_ack("confirmed")], [], True, "missing receipt")
    assert evidence.status == CancellationStatus.UNKNOWN
    assert evidence.message == "missing receipt"


def test_repeated_request_cannot_reset_deadline_and_unknown_is_sticky(
    monkeypatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr("artisan_submitit.cancellation.time.monotonic", lambda: now[0])
    state = CancellationState()
    assert state.request().status == CancellationStatus.REQUESTED
    now[0] += 20
    assert state.request().status == CancellationStatus.REQUESTED
    assert state.remaining == 5
    assert not state.cooperating
    now[0] += 5
    assert state.current().status == CancellationStatus.UNKNOWN
    assert (
        state.finish([_ack("confirmed")], [], True).status == CancellationStatus.UNKNOWN
    )


def test_finish_preserves_completion_when_cancel_arrives_late() -> None:
    state = CancellationState()
    state.finish([None], [], False)
    assert state.request().status == CancellationStatus.REJECTED
    assert not state.event.is_set()


def _query(monkeypatch: pytest.MonkeyPatch, rows: str) -> MagicMock:
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, rows, ""))
    monkeypatch.setattr("artisan_submitit.cancellation.subprocess.run", run)
    return run


def test_accounting_uses_exact_array_elements_and_bounded_query(monkeypatch) -> None:
    run = _query(
        monkeypatch,
        "123_4|CANCELLED by 1001|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|\n123_4.batch|COMPLETED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|\n",
    )
    states = query_job_states(
        ["123_4"], datetime(2026, 10, 23, 10, tzinfo=UTC), timeout=0.4
    )
    assert list(states) == ["123_4"]
    assert states["123_4"].terminal
    assert states["123_4"].never_started
    command = run.call_args.args[0]
    assert "--array" not in command
    assert command[command.index("--jobs") + 1] == "123_4"
    assert run.call_args.kwargs["timeout"] == 0.4
    assert run.call_args.kwargs["env"]["TZ"] == "UTC"


@pytest.mark.parametrize(
    "rows",
    [
        "123_[1-4]|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|",
        "123_4|CANCELLED+|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|",
        "123_4|CANCELLED|2026-10-22T10:00:00|Unknown|2026-10-23T10:01:00|",
        "123_4|CANCELLED|Unknown|Unknown|2026-10-23T10:01:00|",
        "123_4|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|\n123_4|RUNNING|2026-10-23T10:00:00|Unknown|Unknown|",
        "123_4|CANCELLED|Unknown|Unknown|",
        "123_4|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T09:59:59|",
        "123_4|COMPLETED|2026-10-23T10:00:00|2026-10-23T09:59:59|2026-10-23T10:01:00|",
        "123_4|COMPLETED|2026-10-23T10:00:00|2026-10-23T10:02:00|2026-10-23T10:01:00|",
    ],
)
def test_stale_truncated_aggregate_or_duplicate_rows_are_not_proof(
    monkeypatch, rows
) -> None:
    _query(monkeypatch, rows)
    assert query_job_states(["123_4"], datetime(2026, 10, 23, 10, tzinfo=UTC)) == {}


@pytest.mark.parametrize(
    ("state", "start", "end"),
    [
        ("RUNNING", "Unknown", "Unknown"),
        ("COMPLETING", "Unknown", "2026-10-23T10:01:00"),
        ("CANCELLED", "Unknown", "Unknown"),
        ("CANCELLED", "2026-10-23T10:00:01", "2026-10-23T10:01:00"),
        ("CANCELLED", "", "2026-10-23T10:01:00"),
    ],
)
def test_state_alone_never_proves_queued_non_start(
    monkeypatch, state, start, end
) -> None:
    _query(monkeypatch, f"123_4|{state}|2026-10-23T10:00:00|{start}|{end}|\n")
    observation = query_job_states(["123_4"])["123_4"]
    assert not observation.never_started


def test_failed_query_is_unresolved_and_cancel_failure_remains_retryable(
    monkeypatch,
) -> None:
    run = MagicMock(
        side_effect=[
            subprocess.TimeoutExpired("sacct", 1),
            OSError("unavailable"),
            subprocess.CompletedProcess([], 0),
        ]
    )
    monkeypatch.setattr("artisan_submitit.cancellation.subprocess.run", run)
    assert query_job_states(["123_4"]) == {}
    assert "unavailable" in cancel_jobs(["123_4"])
    assert cancel_jobs(["123_4"]) is None
    assert run.call_args.args[0] == ["scancel", "123_4"]


@pytest.mark.parametrize(
    "identifier",
    ["--name=job", "123_[1-4]", "123.batch", "123.extern", "123+", "job", "١٢٣"],
)
def test_scheduler_helpers_reject_nonexact_identifiers(monkeypatch, identifier) -> None:
    run = _query(monkeypatch, "")
    assert query_job_states([identifier]) == {}
    assert cancel_jobs([identifier]) is not None
    run.assert_not_called()
