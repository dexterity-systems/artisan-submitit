"""Requeue waiting uses one bounded deadline without changing scheduler policy."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from .integration import test_ownership_faults as faults


def _clock(monkeypatch: pytest.MonkeyPatch) -> tuple[list[float], Callable[[], float]]:
    clock = [0.0]
    monkeypatch.setattr(faults.time, "monotonic", lambda: clock[0])

    def sleep(seconds: float) -> None:
        assert 0 <= seconds <= 0.05
        clock[0] += seconds

    monkeypatch.setattr(faults.time, "sleep", sleep)
    return clock, lambda: clock[0]


def test_requeue_can_enter_after_measured_credential_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, now = _clock(monkeypatch)
    faults._wait(lambda: now() >= 121, "not entered", deadline=230)
    assert 121 <= now() < 122


def test_other_waits_keep_their_existing_ninety_second_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, now = _clock(monkeypatch)
    with pytest.raises(AssertionError, match="not entered"):
        faults._wait(lambda: now() >= 121, "not entered")
    assert now() == 90


def test_requeue_uses_original_case_deadline_without_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, now = _clock(monkeypatch)
    clock[0] = 229.9
    with pytest.raises(AssertionError, match="not entered"):
        faults._wait(lambda: False, "not entered", deadline=230)
    assert now() == 230


def test_requeue_rejects_a_predicate_that_completes_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, _ = _clock(monkeypatch)

    def late() -> bool:
        clock[0] = 231
        return True

    with pytest.raises(AssertionError, match="late entry"):
        faults._wait(late, "late entry", deadline=230)
