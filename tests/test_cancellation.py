"""Fresh exact scheduler observations and narrowly addressed cancellation."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from artisan_submitit.cancellation import cancel_jobs, query_job_states


def _query(monkeypatch: pytest.MonkeyPatch, rows: str) -> MagicMock:
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, rows, ""))
    monkeypatch.setattr("artisan_submitit.cancellation.subprocess.run", run)
    return run


def test_accounting_uses_exact_array_elements_and_bounded_query(monkeypatch) -> None:
    run = _query(
        monkeypatch,
        "123_4|CANCELLED by 1001|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|\n123_4.batch|COMPLETED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|\n",
    )
    states = query_job_states(
        ["123_4"], datetime(2026, 10, 23, 10, tzinfo=UTC), timeout=0.4
    )
    assert list(states) == ["123_4"]
    assert states["123_4"].terminal
    assert not states["123_4"].never_started
    command = run.call_args.args[0]
    assert "--array" not in command
    assert command[command.index("--jobs") + 1] == "123_4"
    assert run.call_args.kwargs["timeout"] == 0.4
    assert run.call_args.kwargs["env"]["TZ"] == "UTC"


@pytest.mark.parametrize(
    "rows",
    [
        "123_[1-4]|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|",
        "123_4|CANCELLED+|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|",
        "123_4|CANCELLED|2026-10-22T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|",
        "123_4|CANCELLED|Unknown|Unknown|2026-10-23T10:01:00|0|None assigned|0|",
        "123_4|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T10:01:00|0|None assigned|0|\n123_4|RUNNING|2026-10-23T10:00:00|Unknown|Unknown|0|None assigned|0|",
        "123_4|CANCELLED|Unknown|Unknown|",
        "123_4|CANCELLED|2026-10-23T10:00:00|Unknown|2026-10-23T09:59:59|0|None assigned|0|",
        "123_4|COMPLETED|2026-10-23T10:00:00|2026-10-23T09:59:59|2026-10-23T10:01:00|0|None assigned|0|",
        "123_4|COMPLETED|2026-10-23T10:00:00|2026-10-23T10:02:00|2026-10-23T10:01:00|0|None assigned|0|",
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
    _query(
        monkeypatch,
        f"123_4|{state}|2026-10-23T10:00:00|{start}|{end}|0|None assigned|0|\n",
    )
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
    assert run.call_args.args[0] == ["/usr/bin/scancel", "123_4"]


@pytest.mark.parametrize(
    "identifier",
    ["--name=job", "123_[1-4]", "123.batch", "123.extern", "123+", "job", "١٢٣"],
)
def test_scheduler_helpers_reject_nonexact_identifiers(monkeypatch, identifier) -> None:
    run = _query(monkeypatch, "")
    assert query_job_states([identifier]) == {}
    assert cancel_jobs([identifier]) is not None
    run.assert_not_called()


_PENDING_CANCEL = "123|CANCELLED by 1001|2026-10-23T10:00:00|2026-10-23T10:01:00|2026-10-23T10:01:00|0|None assigned|0|\n"
_CONTROLLER_CANCEL = (
    "JobId=123 JobName=qualification JobState=CANCELLED Restarts=0 BatchFlag=1 "
    "RunTime=00:00:00 SubmitTime=2026-10-23T10:00:00 "
    "StartTime=2026-10-23T10:01:00 EndTime=2026-10-23T10:01:00 "
    "ReqNodeList=(null) NodeList=(null) Comment=ordinary\n"
)


def _pending_cancel_query(
    monkeypatch, accounting=_PENDING_CANCEL, control=_CONTROLLER_CANCEL
):
    run = MagicMock(
        side_effect=[
            subprocess.CompletedProcess([], 0, accounting, ""),
            subprocess.CompletedProcess([], 0, control, ""),
        ]
    )
    monkeypatch.setattr("artisan_submitit.cancellation.subprocess.run", run)
    return run


def test_equal_cancel_timestamps_require_exact_unallocated_controller_proof(
    monkeypatch,
) -> None:
    run = _pending_cancel_query(monkeypatch)
    observation = query_job_states(["123"], timeout=0.5)["123"]
    assert observation.never_started
    assert observation.unallocated_cancel_confirmed
    assert not observation.start_unknown
    assert "--duplicates" in run.call_args_list[0].args[0]
    assert run.call_args.args[0] == [
        "/usr/bin/scontrol",
        "show",
        "job",
        "123",
        "--oneliner",
    ]
    assert 0 < run.call_args.kwargs["timeout"] <= 0.5


@pytest.mark.parametrize(
    "changed",
    [
        _PENDING_CANCEL.replace("|0|None assigned|0|", "|1|node-a|0|"),
        _PENDING_CANCEL.replace("|0|None assigned|0|", "|0|node-a|0|"),
        _PENDING_CANCEL.replace("|0|None assigned|0|", "||None assigned|0|"),
        _PENDING_CANCEL.replace("|0|None assigned|0|", "|0|None assigned|1|"),
        _PENDING_CANCEL.replace("|0|None assigned|0|", "|0|None assigned||"),
        _PENDING_CANCEL.replace("123|", "123_4|"),
        _PENDING_CANCEL.replace("123|", "123.0|"),
    ],
)
def test_equal_timestamps_without_zero_allocation_do_not_query_controller(
    monkeypatch, changed
) -> None:
    run = _pending_cancel_query(monkeypatch, accounting=changed)
    observation = query_job_states([changed.split("|", 1)[0]])
    assert all(not item.never_started for item in observation.values())
    assert run.call_count == 1


@pytest.mark.parametrize(
    "changed",
    [
        _CONTROLLER_CANCEL.replace("Restarts=0", "Restarts=1"),
        _CONTROLLER_CANCEL.replace("Restarts=0 ", ""),
        _CONTROLLER_CANCEL.replace("Restarts=0", "Restarts=Unknown"),
        _CONTROLLER_CANCEL.replace("BatchFlag=1", "BatchFlag=2"),
        _CONTROLLER_CANCEL.replace(" NodeList=(null)", " NodeList=node-a"),
        _CONTROLLER_CANCEL.replace("JobId=123", "JobId=124"),
        _CONTROLLER_CANCEL.replace(
            "SubmitTime=2026-10-23T10:00:00", "SubmitTime=2026-10-23T09:00:00"
        ),
        _CONTROLLER_CANCEL.replace(
            "EndTime=2026-10-23T10:01:00", "EndTime=2026-10-23T10:01:01"
        ),
        _CONTROLLER_CANCEL.replace("JobState=CANCELLED", "JobState=COMPLETING"),
        _CONTROLLER_CANCEL.replace("Comment=ordinary", "Comment=ordinary Restarts=0"),
        _CONTROLLER_CANCEL.replace("Comment=ordinary", "ArrayJobId=123 ArrayTaskId=0"),
        _CONTROLLER_CANCEL.replace("Comment=ordinary", "HetJobId=123 HetJobOffset=0"),
        _CONTROLLER_CANCEL + _CONTROLLER_CANCEL,
        "",
    ],
)
def test_corroboration_rejects_prior_requeue_missing_and_changed_evidence(
    monkeypatch, changed
) -> None:
    _pending_cancel_query(monkeypatch, control=changed)
    observation = query_job_states(["123"])["123"]
    assert observation.terminal
    assert not observation.never_started


@pytest.mark.parametrize(
    "failure", [OSError("unavailable"), subprocess.TimeoutExpired("scontrol", 0.1)]
)
def test_corroboration_failure_preserves_only_terminal_observation(
    monkeypatch, failure
) -> None:
    run = _pending_cancel_query(monkeypatch)
    run.side_effect = [subprocess.CompletedProcess([], 0, _PENDING_CANCEL, ""), failure]
    observation = query_job_states(["123"])["123"]
    assert observation.terminal
    assert not observation.never_started


def test_late_valid_corroboration_cannot_extend_query_deadline(monkeypatch) -> None:
    _pending_cancel_query(monkeypatch)
    clock = iter((0.0, 0.2, 0.6))
    monkeypatch.setattr(
        "artisan_submitit.cancellation.time.monotonic", lambda: next(clock)
    )
    assert not query_job_states(["123"], timeout=0.5)["123"].never_started


@pytest.mark.parametrize("malformed_first", [True, False])
def test_malformed_exact_id_cannot_hide_duplicate_accounting(
    monkeypatch, malformed_first
) -> None:
    rows = ["123|malformed\n", _PENDING_CANCEL]
    if not malformed_first:
        rows.reverse()
    run = _pending_cancel_query(monkeypatch, accounting="".join(rows))
    assert query_job_states(["123"]) == {}
    assert run.call_count == 1


def test_unknown_start_requires_the_same_positive_unallocated_proof(
    monkeypatch,
) -> None:
    accounting = _PENDING_CANCEL.replace("|2026-10-23T10:01:00|", "|Unknown|", 1)
    controller = _CONTROLLER_CANCEL.replace(
        "StartTime=2026-10-23T10:01:00", "StartTime=Unknown"
    )
    _pending_cancel_query(monkeypatch, accounting, controller)
    observation = query_job_states(["123"])["123"]
    assert observation.start_unknown
    assert observation.unallocated_cancel_confirmed
    assert observation.never_started


@pytest.mark.parametrize("change", ["node", "elapsed", "restart", "malformed_start"])
def test_unknown_start_does_not_bypass_contradiction_or_requeue_checks(
    monkeypatch, change
) -> None:
    accounting = _PENDING_CANCEL.replace("|2026-10-23T10:01:00|", "|Unknown|", 1)
    controller = _CONTROLLER_CANCEL.replace(
        "StartTime=2026-10-23T10:01:00", "StartTime=Unknown"
    )
    if change == "node":
        accounting = accounting.replace("|0|None assigned|0|", "|1|node-a|0|")
    elif change == "elapsed":
        accounting = accounting.replace("|0|None assigned|0|", "|0|None assigned|1|")
    elif change == "restart":
        controller = controller.replace("Restarts=0", "Restarts=1")
    else:
        controller = controller.replace("StartTime=Unknown", "StartTime=garbage")
    _pending_cancel_query(monkeypatch, accounting, controller)
    assert not query_job_states(["123"])["123"].never_started
