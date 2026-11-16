"""Current requeue termination can bound uncertainty, never establish closure."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import OwnershipState

from artisan_submitit import cancellation
from artisan_submitit.cancellation import SchedulerObservation, requeued_job_ended
from artisan_submitit.control import PackRequest
from artisan_submitit.software import SlurmSoftware

from .test_prepared_transport import ProbeOwner, _entry, _owner

SUBMITTED_AFTER = datetime(2026, 1, 2, 3, tzinfo=UTC)


def _record(**changes: str) -> str:
    values = {
        "JobId": "41",
        "UserId": f"qualification({os.getuid()})",
        "JobState": "COMPLETED",
        "BatchFlag": "1",
        "Restarts": "1",
        "SubmitTime": "2026-01-02T03:00:02",
        "StartTime": "2026-01-02T03:02:03",
        "EndTime": "2026-01-02T03:02:04",
    }
    return " ".join(f"{key}={value}" for key, value in (values | changes).items())


def test_current_requeue_terminal_uses_only_the_remaining_query_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cancellation.time, "monotonic", lambda: 10)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, _record()))
    monkeypatch.setattr(cancellation.subprocess, "run", run)
    assert requeued_job_ended("41", SUBMITTED_AFTER, deadline=10.25)
    assert run.call_args.args[0] == [
        "/usr/bin/scontrol",
        "show",
        "job",
        "41",
        "--oneliner",
    ]
    assert run.call_args.kwargs["timeout"] == 0.25
    assert run.call_args.kwargs["env"]["TZ"] == "UTC"


@pytest.mark.parametrize(
    "record",
    [
        _record(JobId="42"),
        _record(UserId=f"foreign({os.getuid() + 1})"),
        _record(JobState="RUNNING"),
        _record(JobState="COMPLETING"),
        _record(JobState="COMPLETED+"),
        _record(Restarts="0"),
        _record(Restarts="Unknown"),
        _record(Restarts="\u0661"),
        _record(BatchFlag="0"),
        _record(BatchFlag="2"),
        _record(ArrayJobId="41"),
        _record(HetJobId="41"),
        _record(SubmitTime="2026-01-01T03:00:02"),
        _record(SubmitTime="Unknown"),
        _record(StartTime="Unknown"),
        _record(StartTime="2026-01-02T02:59:59"),
        _record(EndTime="Unknown"),
        _record(EndTime="2026-01-02T03:01:00"),
        _record(EndTime="2026-02-30T03:01:00"),
        _record() + " JobId=41",
        _record() + "\n" + _record(),
        _record().replace(" Restarts=1", ""),
        _record() + " Comment=" + "x" * (64 * 1024),
    ],
)
def test_requeue_terminal_rejects_foreign_stale_or_ambiguous_record(
    record: str,
) -> None:
    assert not cancellation._requeued_terminal_record(record, "41", SUBMITTED_AFTER)


@pytest.mark.parametrize("job_id", ["41.0", "41_1", "41+1", "٤١", "--name=probe"])
def test_requeue_terminal_rejects_nonordinary_identity_without_query(
    monkeypatch: pytest.MonkeyPatch, job_id: str
) -> None:
    run = Mock(side_effect=AssertionError("Invalid job must not query scheduler"))
    monkeypatch.setattr(cancellation.subprocess, "run", run)
    assert not requeued_job_ended(job_id, SUBMITTED_AFTER, deadline=float("inf"))
    run.assert_not_called()


@pytest.mark.parametrize(
    "failure", [OSError("unavailable"), subprocess.TimeoutExpired("scontrol", 1)]
)
def test_requeue_terminal_missing_transport_stays_unavailable(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr(cancellation.time, "monotonic", lambda: 10)
    monkeypatch.setattr(cancellation.subprocess, "run", Mock(side_effect=failure))
    assert not requeued_job_ended("41", SUBMITTED_AFTER, deadline=11)


def test_requeue_terminal_late_valid_response_is_not_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((10, 12))
    monkeypatch.setattr(cancellation.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        cancellation.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, _record())),
    )
    assert not requeued_job_ended("41", SUBMITTED_AFTER, deadline=11)


def test_requeue_terminal_does_not_query_expired_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cancellation.time, "monotonic", lambda: 10)
    run = Mock()
    monkeypatch.setattr(cancellation.subprocess, "run", run)
    assert not requeued_job_ended("41", SUBMITTED_AFTER, deadline=10)
    run.assert_not_called()


def _observing_owner(tmp_path: Path) -> ProbeOwner:
    owner = _owner(tmp_path)
    owner._submitted_after = SUBMITTED_AFTER
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    (tmp_path / "entries").mkdir()
    return owner


def test_current_running_accounting_cannot_be_superseded_by_requeue_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _observing_owner(tmp_path)
    owner.request_close(8)
    clock = [0.0]
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _: clock.__setitem__(0, clock[0] + 1),
    )
    monkeypatch.setattr(
        "artisan_submitit.transport.query_job_states",
        lambda *a, **kw: {
            "41": SchedulerObservation(
                "RUNNING", SUBMITTED_AFTER, SUBMITTED_AFTER, None
            )
        },
    )
    fallback = Mock(side_effect=AssertionError("RUNNING cannot use terminal fallback"))
    monkeypatch.setattr("artisan_submitit.transport.requeued_job_ended", fallback)
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", lambda *a, **kw: None)
    owner._observe(cast(PackRequest, SimpleNamespace()))
    fallback.assert_not_called()
    assert owner.ownership_state() == OwnershipState.UNKNOWN


def test_entry_appearing_during_final_query_retains_reply_but_never_closes_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _observing_owner(tmp_path)
    clock = [10.0]
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _: clock.__setitem__(0, clock[0] + 1),
    )
    monkeypatch.setattr(
        "artisan_submitit.transport.query_job_states", lambda *a, **kw: {}
    )
    owner.directory = tmp_path / "entries"
    first, request = _entry(owner)
    (first / "unit-0.json").unlink()
    (first / "outcome.json").unlink()
    owner.directory = tmp_path

    def terminal(*args: object, **kwargs: object) -> bool:
        if clock[0] == 15:
            owner.directory = tmp_path / "entries"
            _entry(owner)
            owner.directory = tmp_path
        return True

    monkeypatch.setattr("artisan_submitit.transport.requeued_job_ended", terminal)
    owner._observe(request)
    assert clock[0] == 15
    assert owner.handles[0].done()
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    assert not owner._stop.is_set()
    assert owner._deadline is None
    assert not (first / "outcome.json").exists()
    assert len(tuple((tmp_path / "entries").iterdir())) == 2
