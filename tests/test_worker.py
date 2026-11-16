"""Provider entry evidence preserves core cleanup and unit isolation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import (
    AssignmentReplyContext,
    CommandOwnershipError,
    LocalPackStartupError,
    OwnershipState,
)

from artisan_submitit.control import EntryEvidence, EntryIdentity, UnitEvidence
from artisan_submitit.worker import _collect, run_worker_pack
from tests.test_prepared_transport import _attempt


class _Request(SimpleNamespace):
    @property
    def result_count(self) -> int:
        return 1 if self.probe is not None else len(self.calls)

    def payload_digest(self, index: int) -> str:
        return (
            self.probe.digest
            if self.probe is not None
            else self.calls[index].header.payload_sha256
        )


def _request(tmp_path: Path) -> _Request:
    context = AssignmentReplyContext(
        attempt=_attempt(), payload_sha256="f" * 64, sensitive_values=()
    )
    return _Request(
        probe=None,
        gpu=None,
        pack_id="a" * 32,
        profile="profile",
        working_root=tmp_path / "working",
        runtime_env={},
        preserve_working=False,
        calls=(
            SimpleNamespace(
                reply_context=context.model_dump_json(),
                header=SimpleNamespace(payload_sha256="f" * 64),
                load=lambda _directory: None,
            ),
        ),
    )


def _identity() -> EntryIdentity:
    return EntryIdentity(pack_id="a" * 32, entry_id="b" * 32, job_id="41", step_id="41")


def test_unknown_reply_retains_scratch_even_when_owner_claims_closed(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    handle = Mock(done=lambda: True, ownership_state=lambda: OwnershipState.CLOSED)
    handle.result.side_effect = CommandOwnershipError("unknown descendants")
    owner = Mock(ownership_state=lambda: OwnershipState.CLOSED)
    assert not _collect(tmp_path, tmp_path, _identity(), request, (handle,), (owner,))
    owner.request_close.assert_not_called()
    assert (
        UnitEvidence.model_validate_json(
            (tmp_path / "unit-0.json").read_bytes()
        ).ownership
        == "unknown"
    )


def test_unit_cancel_does_not_kill_shared_owner_or_sibling(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.calls *= 2
    (tmp_path / "cancel-0").touch()
    first = Mock(done=lambda: True, ownership_state=lambda: OwnershipState.CLOSED)
    second = Mock(done=lambda: True, ownership_state=lambda: OwnershipState.CLOSED)
    first.result.return_value = second.result.return_value = b"{}"
    owner = Mock(ownership_state=lambda: OwnershipState.CLOSED)
    assert _collect(tmp_path, tmp_path, _identity(), request, (first, second), (owner,))
    first.cancel.assert_called_once()
    second.cancel.assert_not_called()
    owner.request_close.assert_not_called()


def test_request_before_entry_never_launches_core_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    launch = Mock()
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    (tmp_path / "cancel").touch()
    run_worker_pack(str(tmp_path), "c" * 64)
    launch.assert_not_called()
    entry = tmp_path / "entries" / ("b" * 32)
    assert (
        EntryEvidence.model_validate_json(
            (entry / "outcome.json").read_bytes()
        ).ownership
        == "closed"
    )
    assert UnitEvidence.model_validate_json(
        (entry / "unit-0.json").read_bytes()
    ).cancelled
    assert not (request.working_root / request.pack_id / ("b" * 32)).exists()


def test_zero_handle_startup_uncertainty_cannot_publish_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.LocalWorkerProfile.from_json", lambda _encoded: None
    )
    error = LocalPackStartupError((), cast(tuple, (SimpleNamespace(state="unknown"),)))
    monkeypatch.setattr(
        "artisan_submitit.worker.launch_inert_calls", Mock(side_effect=error)
    )
    run_worker_pack(str(tmp_path), "c" * 64)
    entry = tmp_path / "entries" / ("b" * 32)
    assert (
        EntryEvidence.model_validate_json(
            (entry / "outcome.json").read_bytes()
        ).ownership
        == "unknown"
    )
    assert (
        UnitEvidence.model_validate_json((entry / "unit-0.json").read_bytes()).ownership
        == "unknown"
    )
    assert (request.working_root / request.pack_id / ("b" * 32)).is_dir()


@pytest.mark.parametrize("preserve", [False, True])
def test_outer_cleanup_honors_core_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool
) -> None:
    request = _request(tmp_path)
    request.preserve_working = preserve
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.LocalWorkerProfile.from_json", lambda _encoded: None
    )
    owner = Mock(ownership_state=lambda: OwnershipState.CLOSED)
    handle = Mock(
        done=lambda: True,
        ownership_state=lambda: OwnershipState.CLOSED,
        worker_owner=owner,
    )
    handle.result.return_value = b"{}"
    launch = Mock(return_value=(handle,))
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    run_worker_pack(str(tmp_path), "c" * 64)
    assert launch.call_args.kwargs["preserve_working"] is preserve
    scratch = request.working_root / request.pack_id / ("b" * 32)
    assert scratch.exists() is preserve


def test_pack_cancel_drains_unit_reply_before_stopping_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    (tmp_path / "cancel").touch()
    events: list[str] = []
    clock = [0.0]
    owner_closed = [False]

    def stop_owner(deadline: float) -> None:
        assert "reply" in events
        assert deadline == 8.0
        owner_closed[0] = True
        events.append("owner")

    def reply(*_args: object, **_kwargs: object) -> bytes:
        events.append("reply")
        return b"{}"

    owner = Mock(
        ownership_state=lambda: OwnershipState.CLOSED
        if owner_closed[0]
        else OwnershipState.PENDING,
        request_close=stop_owner,
    )
    handle = Mock(
        done=lambda: clock[0] >= 1,
        ownership_state=owner.ownership_state,
        result=reply,
        cancel=lambda: events.append("unit"),
    )
    monkeypatch.setattr("artisan_submitit.worker.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.worker.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    assert _collect(tmp_path, tmp_path, _identity(), request, (handle,), (owner,))
    assert events.index("unit") < events.index("reply") < events.index("owner")


def test_pack_cancel_forces_blocked_native_at_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    (tmp_path / "cancel").touch()
    clock = [0.0]
    requests: list[tuple[float, float]] = []
    owner = Mock(
        ownership_state=lambda: OwnershipState.PENDING,
        request_close=lambda deadline: requests.append((clock[0], deadline)),
    )
    handle = Mock(done=lambda: False, ownership_state=owner.ownership_state)
    monkeypatch.setattr("artisan_submitit.worker.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.worker.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    assert not _collect(tmp_path, tmp_path, _identity(), request, (handle,), (owner,))
    assert requests[0] == (6.0, 8.0)
    assert all(deadline == 8.0 for _, deadline in requests)
    assert clock[0] == 8.0
    assert (
        UnitEvidence.model_validate_json(
            (tmp_path / "unit-0.json").read_bytes()
        ).ownership
        == "unknown"
    )


@pytest.mark.parametrize("failure", ["cancel", "state", "close"])
def test_cleanup_failure_still_attempts_sibling_owner_and_preserves_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    request = _request(tmp_path)
    (tmp_path / "cancel").touch()
    clock = [0.0]
    first = Mock(ownership_state=lambda: OwnershipState.PENDING)
    second = Mock(ownership_state=lambda: OwnershipState.PENDING)
    handle = Mock(done=lambda: True, ownership_state=lambda: OwnershipState.CLOSED)
    handle.result.return_value = b"{}"
    if failure == "cancel":
        handle.cancel.side_effect = RuntimeError("cancel unavailable")
    elif failure == "state":
        first.ownership_state = Mock(side_effect=RuntimeError("state unavailable"))
    else:
        first.request_close.side_effect = RuntimeError("close unavailable")
    monkeypatch.setattr("artisan_submitit.worker.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.worker.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    assert not _collect(
        tmp_path, tmp_path, _identity(), request, (handle,), (first, second)
    )
    assert clock[0] == 8.0
    assert first.request_close.call_count == second.request_close.call_count
    assert second.request_close.call_count > 0
    assert all(call.args == (8.0,) for call in second.request_close.call_args_list)
    if failure != "close":
        handle.result.assert_not_called()
        assert (
            UnitEvidence.model_validate_json(
                (tmp_path / "unit-0.json").read_bytes()
            ).ownership
            == "unknown"
        )


def test_result_write_failure_closes_all_owners_without_claiming_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    clock = [0.0]
    first = Mock(ownership_state=lambda: OwnershipState.PENDING)
    second = Mock(ownership_state=lambda: OwnershipState.PENDING)
    handle = Mock(done=lambda: True, ownership_state=lambda: OwnershipState.CLOSED)
    handle.result.return_value = b"{}"
    monkeypatch.setattr(
        "artisan_submitit.worker.write_once",
        Mock(side_effect=OSError("evidence unavailable")),
    )
    monkeypatch.setattr("artisan_submitit.worker.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "artisan_submitit.worker.time.sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1),
    )
    assert not _collect(
        tmp_path, tmp_path, _identity(), request, (handle,), (first, second)
    )
    assert first.request_close.call_count == second.request_close.call_count
    assert second.request_close.call_count > 0
    assert all(call.args == (8.0,) for call in second.request_close.call_args_list)
    assert clock[0] == 8.0
    assert not (tmp_path / "unit-0.json").exists()


def test_pre_entry_result_write_failure_cannot_publish_closed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from artisan_submitit.control import write_once

    request = _request(tmp_path)
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    (tmp_path / "cancel").touch()

    def fail_unit(path: Path, data: bytes) -> None:
        if path.name == "unit-0.json":
            raise OSError("result publication failed")
        write_once(path, data)

    monkeypatch.setattr("artisan_submitit.worker.write_once", fail_unit)
    run_worker_pack(str(tmp_path), "c" * 64)
    outcome = tmp_path / "entries" / ("b" * 32) / "outcome.json"
    assert (
        EntryEvidence.model_validate_json(outcome.read_bytes()).ownership == "unknown"
    )
    assert (request.working_root / request.pack_id / ("b" * 32)).is_dir()
