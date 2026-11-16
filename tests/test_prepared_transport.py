"""Result availability cannot erase scheduler ownership or earlier redeliveries."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AssignmentAttempt,
    AssignmentDispatch,
    AssignmentReply,
    CommandOwnershipError,
    OwnershipState,
    SealedDelivery,
)

from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    UnitEvidence,
    write_once,
)
from artisan_submitit.software import SlurmSoftware
from artisan_submitit.transport import (
    PackOwner,
    SchedulerAssignmentHandle,
    SchedulerStartupError,
)


def _attempt() -> AssignmentAttempt:
    return AssignmentAttempt.model_validate(
        {
            "assignment": {"step_run_id": "a" * 32, "unit_index": 0},
            "attempt": 0,
            "generation": 0,
            "recipe_digest": "b" * 32,
            "software_digest": "c" * 64,
            "input_context_digest": "d" * 32,
        }
    )


class ProbeOwner(PackOwner):
    def submit(self, request: PackRequest) -> None:
        self._job_id = "41"


def _owner(tmp_path: Path) -> ProbeOwner:
    owner = object.__new__(ProbeOwner)
    owner.directory = tmp_path
    owner._job_id = "41"
    owner._unknown = False
    owner._lock = threading.Lock()
    owner._state = OwnershipState.PENDING
    owner._stop = threading.Event()
    owner._deadline = None
    owner._cleanup_thread = None
    owner._thread = Mock(is_alive=lambda: True)
    owner.dispatch = cast(AssignmentDispatch, SimpleNamespace(runtime_env={}))
    owner.handles = (SchedulerAssignmentHandle(owner, 0, _attempt()),)
    return owner


def _entry(owner: ProbeOwner, *, unknown: bool = False) -> tuple[Path, PackRequest]:
    entry_id = uuid.uuid4().hex
    directory = owner.directory / entry_id
    directory.mkdir()
    identity = EntryIdentity(
        pack_id="a" * 32, entry_id=entry_id, job_id="41", step_id="41"
    )
    receipt = SealedDelivery.model_validate(
        {
            "delivery": {
                "attempt": _attempt().model_dump(),
                "delivery_id": uuid.uuid4().hex,
            },
            "seal_uri": "memory://candidate/retained.json",
            "seal_digest": "e" * 32,
        }
    )
    reply = AssignmentReply(attempt=_attempt(), kind="success", receipt=receipt)
    write_once(directory / "identity.json", identity.model_dump_json().encode())
    write_once(
        directory / "unit-0.json",
        UnitEvidence(
            entry=identity,
            index=0,
            payload_sha256="f" * 64,
            reply=reply.model_dump_json(),
            ownership="unknown" if unknown else "closed",
        )
        .model_dump_json()
        .encode(),
    )
    write_once(
        directory / "outcome.json",
        EntryEvidence(entry=identity, ownership="unknown" if unknown else "closed")
        .model_dump_json()
        .encode(),
    )
    request = cast(
        PackRequest,
        SimpleNamespace(
            pack_id="a" * 32,
            payload_digest=lambda _index: "f" * 64,
            calls=(SimpleNamespace(header=SimpleNamespace(payload_sha256="f" * 64)),),
        ),
    )
    return directory, request


def test_result_does_not_close_pending_scheduler_owner(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    directory, request = _entry(owner)
    assert owner._collect_entry(directory, request)
    assert owner.handles[0].done()
    assert owner.handles[0].result().delivery.attempt == _attempt()
    assert owner.handles[0].ownership_state() == OwnershipState.PENDING


def test_later_redelivery_success_preserves_prior_unknown(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    uncertain, request = _entry(owner, unknown=True)
    assert not owner._collect_entry(uncertain, request)
    successful, request = _entry(owner)
    assert owner._collect_entry(successful, request)
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    with pytest.raises(CommandOwnershipError):
        owner.handles[0].result()


def test_unknown_deadline_stays_owned_after_result(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    directory, request = _entry(owner)
    owner._collect_entry(directory, request)
    owner.request_close(time.monotonic() - 1)
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    owner._state = OwnershipState.CLOSED
    assert owner.ownership_state() == OwnershipState.UNKNOWN


def test_thread_start_failure_retains_exact_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    admitted = cast(AdmittedAssignment, SimpleNamespace(attempt=_attempt()))
    with pytest.raises(SchedulerStartupError) as caught:
        ProbeOwner(
            (admitted,),
            cast(SlurmSoftware, Mock()),
            cast(AssignmentDispatch, SimpleNamespace(runtime_env={})),
            tmp_path,
        )
    owner = caught.value.owner
    assert owner.ownership_state() == OwnershipState.CLOSED
    assert owner.handles[0].worker_owner is owner
    with pytest.raises(RuntimeError, match="thread unavailable"):
        owner.handles[0].result()


def test_foreign_entry_cannot_supply_result(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    directory, request = _entry(owner)
    path = directory / "identity.json"
    identity = EntryIdentity.model_validate_json(path.read_bytes())
    path.write_text(identity.model_copy(update={"job_id": "99"}).model_dump_json())
    with pytest.raises(ValueError, match="scheduler owner"):
        owner._collect_entry(directory, request)
    assert not owner.handles[0].done()


def test_sdk_failure_after_acceptance_retains_unknown_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AcceptedThenFailed(ProbeOwner):
        def submit(self, request: PackRequest) -> None:
            self._submission_started = True
            raise RuntimeError("SDK lost the accepted handle")

    monkeypatch.setattr(
        "artisan_submitit.transport.capture_pack",
        lambda *_args: cast(PackRequest, SimpleNamespace()),
    )
    admitted = cast(AdmittedAssignment, SimpleNamespace(attempt=_attempt()))
    owner = AcceptedThenFailed(
        (admitted,),
        cast(SlurmSoftware, Mock()),
        cast(AssignmentDispatch, SimpleNamespace(runtime_env={})),
        tmp_path,
    )
    with pytest.raises(CommandOwnershipError):
        owner.handles[0].result(timeout=2)
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    assert owner.handles[0].worker_owner is owner
    assert not (owner.directory / "closed").exists()


def test_pre_submission_failure_closes_owner_without_scheduler_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_capture(*_args: object) -> PackRequest:
        raise ValueError("incompatible worker")

    monkeypatch.setattr("artisan_submitit.transport.capture_pack", reject_capture)
    admitted = cast(AdmittedAssignment, SimpleNamespace(attempt=_attempt()))
    owner = ProbeOwner(
        (admitted,),
        cast(SlurmSoftware, Mock()),
        cast(AssignmentDispatch, SimpleNamespace(runtime_env={})),
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="incompatible worker"):
        owner.handles[0].result(timeout=2)
    assert owner.ownership_state() == OwnershipState.CLOSED
    assert (owner.directory / "closed").exists()


def test_ownership_reply_overrides_false_closed_transport_claim(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    directory, request = _entry(owner)
    path = directory / "unit-0.json"
    evidence = UnitEvidence.model_validate_json(path.read_bytes())
    reply = AssignmentReply.failed(
        _attempt(), CommandOwnershipError("native cleanup unresolved")
    )
    path.write_text(
        evidence.model_copy(update={"reply": reply.model_dump_json()}).model_dump_json()
    )
    owner._collect_entry(directory, request)
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    with pytest.raises(CommandOwnershipError):
        owner.handles[0].result()


def test_pre_entry_error_is_infrastructure_and_does_not_expose_runtime_values() -> None:
    from artisan_submitit.transport import _transport_failure

    class AuthoredLookingError(RuntimeError):
        pass

    original = AuthoredLookingError("startup failed with synthetic-secret")
    original.__cause__ = RuntimeError("cause contains synthetic-secret")
    dispatch = cast(
        AssignmentDispatch, SimpleNamespace(runtime_env={"TOKEN": "synthetic-secret"})
    )
    sanitized = _transport_failure(original, dispatch)
    assert type(sanitized) is RuntimeError
    assert str(sanitized) == "startup failed with [REDACTED]"
    assert sanitized.__cause__ is None


def test_corrupt_entry_after_acceptance_still_cancels_exact_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    class CorruptAccepted(ProbeOwner):
        def submit(self, request: PackRequest) -> None:
            self._submission_started = True
            self._job_id = "41"
            entry = self.directory / "entries" / ("b" * 32)
            entry.mkdir(parents=True)
            (entry / "identity.json").write_text("corrupt")

    monkeypatch.setattr(
        "artisan_submitit.transport.capture_pack",
        lambda *_args: cast(PackRequest, SimpleNamespace()),
    )
    cancel = Mock(return_value=None)
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", cancel)
    now = datetime.now(UTC)
    monkeypatch.setattr(
        "artisan_submitit.transport.query_job_states",
        lambda *_args, **_kwargs: {
            "41": SchedulerObservation("CANCELLED", now, now, now)
        },
    )
    software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    owner = CorruptAccepted(
        (cast(AdmittedAssignment, SimpleNamespace(attempt=_attempt())),),
        software,
        cast(AssignmentDispatch, SimpleNamespace(runtime_env={})),
        tmp_path,
    )
    with pytest.raises(CommandOwnershipError):
        owner.handles[0].result(timeout=2)
    assert cancel.call_args.args == (["41"],)
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    assert not (owner.directory / "closed").exists()


def test_cancel_drains_then_forces_exact_owner_with_one_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    owner = _owner(tmp_path)
    owner._submitted_after = datetime.now(UTC)
    owner._stop.set()
    owner._deadline = 8.0
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    clock = [0.0]
    cancel = Mock(return_value=None)
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    monkeypatch.setattr(
        "artisan_submitit.transport.query_job_states", Mock(return_value={})
    )
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", cancel)
    owner._observe(cast(PackRequest, SimpleNamespace()))
    cancel.assert_called_once_with(
        ["41"], timeout=1.0, executable=Path("/usr/bin/scancel")
    )
    assert clock[0] == owner._deadline == 8.0
    assert owner.ownership_state() == OwnershipState.UNKNOWN


@pytest.mark.parametrize("race", ["none", "late_entry", "late_reply", "transport_open"])
def test_never_started_closure_rechecks_entries_deadline_and_local_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    owner = _owner(tmp_path)
    now = datetime.now(UTC)
    owner._submitted_after = now
    owner._stop.set()
    owner._deadline = 8.0
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    clock = [1.0]
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _delay: clock.__setitem__(0, 8.0),
    )
    monkeypatch.setattr(
        "artisan_submitit.transport.cancel_jobs", Mock(return_value=None)
    )
    monkeypatch.setattr(owner, "transport_closed", lambda: race != "transport_open")

    def observe(*_args, **_kwargs):
        if race == "late_entry":
            (tmp_path / "entries" / "late").mkdir(parents=True, exist_ok=True)
        elif race == "late_reply":
            clock[0] = 8.0
        return {
            "41": SchedulerObservation(
                "CANCELLED", now, now, now, unallocated_cancel_confirmed=True
            )
        }

    monkeypatch.setattr("artisan_submitit.transport.query_job_states", observe)
    owner._observe(cast(PackRequest, SimpleNamespace()))
    assert owner._unknown is (race != "none")


@pytest.mark.parametrize(
    "phase", ["queued", "entry_during_query", "running", "unavailable"]
)
def test_whole_queued_cancel_precedes_unit_deadline_without_shortening_active_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    owner = _owner(tmp_path)
    now = datetime.now(UTC)
    owner._submitted_after = now
    owner.request_close(10.0)
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    clock = [0.0]
    cancelled: list[float] = []
    transport_closed: list[float] = []
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    owner.handles[0].cancel()

    def cancel(ids: list[str], **kwargs: object) -> None:
        assert ids == ["41"]
        assert isinstance(kwargs["timeout"], float)
        assert kwargs["timeout"] <= 1.0
        cancelled.append(clock[0])

    def observe(*_args: object, **_kwargs: object) -> dict[str, SchedulerObservation]:
        owner.handles[0].cancel()
        if phase == "queued" and cancelled:
            return {
                "41": SchedulerObservation(
                    "CANCELLED", now, now, now, unallocated_cancel_confirmed=True
                )
            }
        if phase == "entry_during_query":
            (tmp_path / "entries" / "raced-entry").mkdir(parents=True, exist_ok=True)
        if phase == "unavailable":
            return {}
        return {
            "41": SchedulerObservation(
                "RUNNING" if phase == "running" else "PENDING", now, None, None
            )
        }

    monkeypatch.setattr("artisan_submitit.transport.query_job_states", observe)
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", cancel)
    monkeypatch.setattr(owner, "_collect_entry", lambda *_args: False)
    monkeypatch.setattr(
        owner, "close_transport", lambda _deadline: transport_closed.append(clock[0])
    )
    owner._observe(cast(PackRequest, SimpleNamespace()))
    assert cancelled == [0.0 if phase == "queued" else 8.0]
    assert transport_closed[0] == (0.0 if phase == "queued" else 8.0)
    assert owner._deadline == 10.0
    assert owner._unknown is (phase != "queued")


def test_one_cancelled_unit_does_not_cancel_queued_shared_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import CancelledError
    from datetime import UTC, datetime

    from artisan_submitit.cancellation import SchedulerObservation

    owner = _owner(tmp_path)
    now = datetime.now(UTC)
    owner._submitted_after = now
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    sibling = SchedulerAssignmentHandle(owner, 1, _attempt())
    owner.handles = (*owner.handles, sibling)
    clock = [0.0]
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.transport.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    owner.handles[0].cancel()
    cancel = Mock(return_value=None)
    monkeypatch.setattr("artisan_submitit.transport.cancel_jobs", cancel)

    def observe(*_args: object, **_kwargs: object) -> dict[str, SchedulerObservation]:
        if clock[0] >= 1:
            (tmp_path / "entries" / "started").mkdir(parents=True, exist_ok=True)
        return {
            "41": SchedulerObservation(
                "PENDING" if clock[0] == 0 else "COMPLETED",
                now,
                now,
                now if clock[0] else None,
            )
        }

    def collect(*_args: object) -> bool:
        owner.handles[0].future.set_exception(CancelledError())
        sibling.future.set_result(None)
        return True

    monkeypatch.setattr("artisan_submitit.transport.query_job_states", observe)
    monkeypatch.setattr(owner, "_collect_entry", collect)
    owner._observe(cast(PackRequest, SimpleNamespace()))
    cancel.assert_not_called()
    assert not owner._stop.is_set()
    assert not sibling.cancelled.is_set()
    assert sibling.done()
    assert not owner._unknown
    assert (tmp_path / "cancel-0").exists()
    assert not (tmp_path / "cancel-1").exists()


@pytest.mark.parametrize("first_closed", [False, True])
def test_natural_requeue_terminal_is_unknown_without_close_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_closed: bool
) -> None:
    import os
    import subprocess
    from datetime import UTC, datetime

    owner = _owner(tmp_path)
    owner._submitted_after = datetime(2026, 1, 2, 3, tzinfo=UTC)
    owner.software = cast(
        SlurmSoftware,
        SimpleNamespace(
            scheduler=SimpleNamespace(executable=lambda name: Path("/usr/bin") / name)
        ),
    )
    entries = tmp_path / "entries"
    entries.mkdir()
    owner.directory = entries
    first, request = _entry(owner)
    if not first_closed:
        (first / "unit-0.json").unlink()
        (first / "outcome.json").unlink()
    second, request = _entry(owner)
    owner.directory = tmp_path
    clock = [10.0]
    monkeypatch.setattr("artisan_submitit.transport.time.monotonic", lambda: clock[0])

    def sleep(_seconds: float) -> None:
        clock[0] += 1
        if clock[0] > 20:
            raise RuntimeError("Natural terminal requeue remained PENDING indefinitely")

    monkeypatch.setattr("artisan_submitit.transport.time.sleep", sleep)
    rows = (
        "41|REQUEUED|2026-01-02T03:00:00|2026-01-02T03:00:01|2026-01-02T03:00:02|1|node|1|\n"
        "41|COMPLETED|2026-01-02T03:00:02|2026-01-02T03:02:03|2026-01-02T03:02:04|1|node|1|\n"
    )
    record = (
        f"JobId=41 UserId=qualification({os.getuid()}) BatchFlag=1 Restarts=1 "
        "JobState=COMPLETED SubmitTime=2026-01-02T03:00:02 "
        "StartTime=2026-01-02T03:02:03 EndTime=2026-01-02T03:02:04\n"
    )
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert 0 < kwargs["timeout"] <= 1
        return subprocess.CompletedProcess(
            argv, 0, rows if Path(argv[0]).name == "sacct" else record
        )

    monkeypatch.setattr("artisan_submitit.cancellation.subprocess.run", run)
    owner._observe(request)
    assert 15 <= clock[0] <= 16
    assert not owner._stop.is_set()
    assert owner._deadline is None
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    assert owner.handles[0].result().delivery.attempt == _attempt()
    assert (first / "outcome.json").exists() is first_closed
    assert (second / "outcome.json").is_file()
    assert not (tmp_path / "closed").exists()
    assert any(Path(argv[0]).name == "scontrol" for argv in calls)
