"""Scheduler teardown observations never replace retained worker closure proof."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from .qualification import run_faults


def _record(**changes: str) -> str:
    values = {
        "JobId": "42",
        "JobState": "CANCELLED",
        "UserId": f"qualification({os.getuid()})",
        "BatchFlag": "1",
        "Restarts": "1",
        "EndTime": "2026-01-02T03:04:05",
    }
    return " ".join(f"{key}={value}" for key, value in (values | changes).items())


def _response(monkeypatch: pytest.MonkeyPatch, text: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert 0 < kwargs["timeout"] <= 1
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, text)

    monkeypatch.setattr(run_faults.subprocess, "run", run)
    return calls


def test_requeued_terminal_is_exact_scheduler_cleanup_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _response(monkeypatch, _record())
    result = run_faults._requeued_terminal("42", time.monotonic() + 10)
    assert result == dict(item.split("=", 1) for item in _record().split())
    assert calls == [["/usr/bin/scontrol", "show", "job", "42", "--oneliner"]]
    assert "ownership" not in result
    assert "never_started" not in result


@pytest.mark.parametrize(
    "text",
    [
        _record(JobId="43"),
        _record(JobState="RUNNING"),
        _record(JobState="COMPLETING"),
        _record(UserId=f"foreign({os.getuid() + 1})"),
        _record(BatchFlag="0"),
        _record(Restarts="0"),
        _record(Restarts="Unknown"),
        _record(EndTime="Unknown"),
        _record(EndTime="2026-02-30T00:00:00"),
        _record(ArrayJobId="42"),
        _record(HetJobId="42"),
        _record() + " JobId=42",
        _record() + " JobState=CANCELLED",
        _record() + "\n" + _record(),
        _record().replace(" Restarts=1", ""),
        _record() + " Comment=" + "x" * (64 * 1024),
    ],
)
def test_requeued_terminal_rejects_ambiguous_foreign_or_unfinished_record(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    _response(monkeypatch, text)
    assert run_faults._requeued_terminal("42", time.monotonic() + 10) is None


@pytest.mark.parametrize("job_id", ["42.0", "42_1", "42,43", "-42"])
def test_requeued_terminal_rejects_nonbatch_identity(
    monkeypatch: pytest.MonkeyPatch, job_id: str
) -> None:
    calls = _response(monkeypatch, _record())
    assert run_faults._requeued_terminal(job_id, time.monotonic() + 10) is None
    assert not calls


def test_requeued_terminal_rejects_late_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _response(monkeypatch, _record())
    clock = iter((1.0, 3.0))
    monkeypatch.setattr(run_faults.time, "monotonic", lambda: next(clock))
    assert run_faults._requeued_terminal("42", 2.0) is None


def test_requeued_terminal_never_queries_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _response(monkeypatch, _record())
    assert run_faults._requeued_terminal("42", time.monotonic() - 1) is None
    assert not calls


def test_requeued_terminal_missing_controller_remains_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("scontrol", 1)

    monkeypatch.setattr(run_faults.subprocess, "run", unavailable)
    assert run_faults._requeued_terminal("42", time.monotonic() + 10) is None


def test_cleanup_records_controller_teardown_without_rewriting_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _response(monkeypatch, _record())
    original: dict[str, object] = {}
    monkeypatch.setattr(run_faults, "query_job_states", lambda *a, **k: original)
    monkeypatch.setattr(run_faults, "_discover", lambda root: ({"42"}, [], []))
    cancelled: list[str] = []
    monkeypatch.setattr(
        run_faults, "cancel_jobs", lambda jobs, **kw: cancelled.extend(jobs)
    )
    result = run_faults.cleanup_known_jobs(tmp_path, time.monotonic() + 10)
    assert result["scheduler_terminal"] is True
    assert result["scheduler_states"] == {"42": "CANCELLED"}
    assert result["scheduler_controller_records"]["42"]["Restarts"] == "1"
    assert result["scheduler_observation_scope"].endswith(
        "worker ownership is unchanged"
    )
    assert original == {}
    assert not cancelled
    assert len(calls) == 1


def test_cleanup_does_not_replace_current_running_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artisan_submitit.cancellation import SchedulerObservation

    calls = _response(monkeypatch, _record())
    monkeypatch.setattr(
        run_faults,
        "query_job_states",
        lambda *a, **k: {"42": SchedulerObservation("RUNNING", None, None, None)},
    )
    assert run_faults._cleanup_states({"42"}, time.monotonic() + 10) == ({}, {})
    assert not calls


def test_cleanup_late_accounting_does_not_pass_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    calls = _response(monkeypatch, _record())
    monkeypatch.setattr(
        run_faults,
        "query_job_states",
        lambda *a, **k: {
            "42": SchedulerObservation("CANCELLED", None, None, datetime.now(UTC))
        },
    )
    clock = iter((1.0, 3.0))
    monkeypatch.setattr(run_faults.time, "monotonic", lambda: next(clock))
    assert run_faults._cleanup_states({"42"}, 2.0) == ({}, {})
    assert not calls
