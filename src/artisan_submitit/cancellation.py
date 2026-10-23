"""Bounded cancellation evidence and exact SLURM accounting observations."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from artisan.orchestration.runner_api import (
    CancellationAcknowledgement,
    CancellationStatus,
    UnitResult,
)

PROVIDER_CONFIRMATION_TIMEOUT = 30.0
CANCELLATION_TIMEOUT = 25.0
COOPERATION_TIMEOUT = 10.0
_EXACT_ID = re.compile(r"[0-9]+(?:_[0-9]+)?(?:\.[0-9]+)?")
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


class CancellationState:
    """Keep one cancellation clock and conservatively reduce owner evidence."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._started: float | None = None
        self._acknowledgement: CancellationAcknowledgement | None = None

    def request(self) -> CancellationAcknowledgement:
        """Record the first request without extending its deadline."""
        with self._lock:
            if self._acknowledgement is not None:
                return self._acknowledgement
            if self._started is None:
                self._started = time.monotonic()
            self.event.set()
        return self.current()

    def current(self) -> CancellationAcknowledgement:
        """Return pending evidence, or sticky uncertainty at the deadline."""
        with self._lock:
            if self._acknowledgement is None and self.expired:
                self._acknowledgement = CancellationAcknowledgement(
                    CancellationStatus.UNKNOWN,
                    "Cancellation deadline expired before ownership closure",
                )
            return self._acknowledgement or CancellationAcknowledgement(
                CancellationStatus.REQUESTED,
                "Waiting for worker ownership and transport closure",
            )

    @property
    def acknowledgement(self) -> CancellationAcknowledgement | None:
        """Return settled evidence without creating a cancellation request."""
        with self._lock:
            return self._acknowledgement

    @property
    def remaining(self) -> float:
        """Return the shared remaining provider cleanup budget in seconds."""
        if self._started is None:
            return CANCELLATION_TIMEOUT
        return max(0.0, CANCELLATION_TIMEOUT - (time.monotonic() - self._started))

    @property
    def expired(self) -> bool:
        """Whether an accepted request exhausted its original budget."""
        return self._started is not None and self.remaining <= 0

    @property
    def cooperating(self) -> bool:
        """Whether workers still have their native-owner cooperation allowance."""
        return (
            self._started is not None
            and time.monotonic() - self._started < COOPERATION_TIMEOUT
        )

    def finish(
        self,
        owner_outcomes: list[CancellationAcknowledgement | None],
        results: list[UnitResult],
        prevented: bool,
        error: str | None = None,
    ) -> CancellationAcknowledgement:
        """Settle once collection has accounted for every admitted owner."""
        evidence = reduce_cancellation(owner_outcomes, results, prevented, error)
        with self._lock:
            if (
                self._acknowledgement is not None
                and self._acknowledgement.status == CancellationStatus.UNKNOWN
            ):
                return self._acknowledgement
            if self.expired:
                evidence = CancellationAcknowledgement(
                    CancellationStatus.UNKNOWN,
                    "Cancellation deadline expired before collection finished",
                )
            self._acknowledgement = evidence
            return evidence


def reduce_cancellation(
    owners: list[CancellationAcknowledgement | None],
    results: list[UnitResult],
    prevented: bool,
    error: str | None = None,
) -> CancellationAcknowledgement:
    """Preserve nested uncertainty and distinguish owners from unit outcomes."""
    units = [r.cancellation_acknowledgement for r in results]
    for evidence in [*owners, *units]:
        if evidence is not None and evidence.status in {
            CancellationStatus.UNKNOWN,
            CancellationStatus.REQUESTED,
        }:
            return CancellationAcknowledgement(
                CancellationStatus.UNKNOWN,
                evidence.message or "Worker ownership closure is unresolved",
            )
    if error:
        return CancellationAcknowledgement(CancellationStatus.UNKNOWN, error)
    for evidence in units:
        if evidence is not None and evidence.status == CancellationStatus.REJECTED:
            return evidence
    if prevented or any(
        e is not None and e.status == CancellationStatus.CONFIRMED
        for e in [*owners, *units]
    ):
        return CancellationAcknowledgement(
            CancellationStatus.CONFIRMED, "Worker ownership and transport closed"
        )
    return CancellationAcknowledgement(
        CancellationStatus.REJECTED, "Work completed before cancellation took effect"
    )


@dataclass(frozen=True)
class SchedulerObservation:
    """An exact, unambiguous SLURM accounting row with UTC timestamps."""

    state: str
    submitted: datetime | None
    started: datetime | None
    ended: datetime | None
    start_unknown: bool = False

    @property
    def terminal(self) -> bool:
        """Whether accounting reports a final state and final end time."""
        return self.state in _TERMINAL_STATES and self.ended is not None

    @property
    def never_started(self) -> bool:
        """Recognize the qualified cancelled-before-start accounting form."""
        return (
            self.terminal
            and self.state == "CANCELLED"
            and self.submitted is not None
            and self.start_unknown
        )


def query_job_states(
    job_ids: list[str],
    submitted_after: datetime | None = None,
    timeout: float = 1.0,
) -> dict[str, SchedulerObservation]:
    """Read fresh exact accounting rows; missing or invalid evidence is omitted."""
    if not job_ids or not all(_EXACT_ID.fullmatch(value) for value in job_ids):
        return {}
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(job_ids),
        "--format",
        "JobID%64,State%40,Submit,Start,End",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return _parse_accounting(result.stdout, set(job_ids), submitted_after)


def _parse_accounting(
    output: str,
    job_ids: set[str],
    submitted_after: datetime | None,
) -> dict[str, SchedulerObservation]:
    """Ignore truncated, stale and duplicate records instead of guessing."""
    observations: dict[str, SchedulerObservation] = {}
    seen: set[str] = set()
    for line in output.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) != 5:
            continue
        job_id, raw_state, submitted, started, ended = (v.strip() for v in fields)
        if job_id not in job_ids:
            continue
        if job_id in seen:
            observations.pop(job_id, None)
            continue
        seen.add(job_id)
        state = raw_state.split(" ", 1)[0]
        submit_time = _timestamp(submitted)
        if "+" in raw_state or not state:
            continue
        if submitted_after is not None and (
            submit_time is None or submit_time < submitted_after
        ):
            continue
        start_time, end_time = _timestamp(started), _timestamp(ended)
        known_times = [t for t in (submit_time, start_time, end_time) if t is not None]
        if known_times != sorted(known_times):
            continue
        observations[job_id] = SchedulerObservation(
            state,
            submit_time,
            start_time,
            end_time,
            start_unknown=started == "Unknown",
        )
    return observations


def _timestamp(value: str) -> datetime | None:
    """Parse the explicitly requested SLURM UTC presentation format."""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def cancel_jobs(job_ids: list[str], timeout: float = 1.0) -> str | None:
    """Request cancellation only for exact numeric job or numeric step IDs."""
    if not job_ids:
        return None
    if not all(_EXACT_ID.fullmatch(value) for value in job_ids):
        return "Refusing cancellation with an invalid exact SLURM identifier"
    try:
        subprocess.run(
            ["scancel", *job_ids],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Exact SLURM cancellation failed: {exc}"
    return None
