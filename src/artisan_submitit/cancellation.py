"""Exact scheduler accounting and narrowly addressed cancellation requests."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

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


@dataclass(frozen=True)
class SchedulerObservation:
    """An exact, unambiguous SLURM accounting row with UTC timestamps."""

    state: str
    submitted: datetime | None
    started: datetime | None
    ended: datetime | None
    start_unknown: bool = False
    allocated_nodes: int | None = None
    node_list: str | None = None
    elapsed_seconds: int | None = None
    unallocated_cancel_confirmed: bool = False

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
            and self.unallocated_cancel_confirmed
        )


def query_job_states(
    job_ids: list[str],
    submitted_after: datetime | None = None,
    timeout: float = 1.0,
    *,
    executable: Path = Path("/usr/bin/sacct"),
    scontrol: Path = Path("/usr/bin/scontrol"),
) -> dict[str, SchedulerObservation]:
    """Read fresh exact accounting rows; missing or invalid evidence is omitted."""
    if (
        timeout <= 0
        or not job_ids
        or not all(_EXACT_ID.fullmatch(value) for value in job_ids)
    ):
        return {}
    deadline = time.monotonic() + timeout
    command = [
        str(executable),
        "--noheader",
        "--parsable2",
        "--duplicates",
        "--jobs",
        ",".join(job_ids),
        "--format",
        "JobID%64,State%40,Submit,Start,End,AllocNodes,NodeList%200,ElapsedRaw",
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
    observations = _parse_accounting(result.stdout, set(job_ids), submitted_after)
    for job_id, observation in observations.items():
        if _unallocated_candidate(job_id, observation) and _confirm_unallocated(
            job_id, observation, executable=scontrol, deadline=deadline
        ):
            observations[job_id] = replace(
                observation, unallocated_cancel_confirmed=True
            )
    return observations


def _unallocated_candidate(job_id: str, observation: SchedulerObservation) -> bool:
    """Equal timestamps need independent zero-allocation and restart evidence."""
    return (
        re.fullmatch(r"[0-9]+", job_id) is not None
        and observation.terminal
        and observation.state == "CANCELLED"
        and observation.submitted is not None
        and (
            observation.start_unknown
            or (
                observation.started is not None
                and observation.started == observation.ended
            )
        )
        and observation.allocated_nodes == 0
        and observation.node_list == "None assigned"
        and observation.elapsed_seconds == 0
    )


def _confirm_unallocated(
    job_id: str,
    observation: SchedulerObservation,
    *,
    executable: Path,
    deadline: float,
) -> bool:
    """Corroborate the pinned pending-cancel form within the original query budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    try:
        result = subprocess.run(
            [str(executable), "show", "job", job_id, "--oneliner"],
            check=True,
            capture_output=True,
            text=True,
            timeout=remaining,
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        _unallocated_record(result.stdout, job_id, observation)
        and time.monotonic() < deadline
    )


def _unallocated_record(
    output: str, job_id: str, observation: SchedulerObservation
) -> bool:
    """Reject ambiguity, arrays and prior requeues in the exact controller record."""
    if len(output.splitlines()) != 1 or not output.startswith(f"JobId={job_id} "):
        return False
    if re.search(r"(?:^|\s)(?:Array\w*|HetJob\w*)=", output):
        return False
    expected = {
        "JobId": job_id,
        "JobState": "CANCELLED",
        "Restarts": "0",
        "BatchFlag": "1",
        "NodeList": "(null)",
        "RunTime": "00:00:00",
    }
    for key, value in expected.items():
        if re.findall(rf"(?:^|\s){key}=([^\s]+)", output) != [value]:
            return False
    for key, expected_time in (
        ("SubmitTime", observation.submitted),
        ("StartTime", observation.started),
        ("EndTime", observation.ended),
    ):
        matches = re.findall(rf"(?:^|\s){key}=([^\s]+)", output)
        if len(matches) != 1:
            return False
        if expected_time is None:
            if matches[0] != "Unknown":
                return False
        elif _timestamp(matches[0]) != expected_time:
            return False
    return True


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
        job_id = fields[0].strip()
        if job_id not in job_ids:
            continue
        if job_id in seen:
            observations.pop(job_id, None)
            continue
        seen.add(job_id)
        if len(fields) != 8:
            continue
        job_id, raw_state, submitted, started, ended, nodes, node_list, elapsed = (
            value.strip() for value in fields
        )
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
            allocated_nodes=int(nodes) if re.fullmatch(r"[0-9]+", nodes) else None,
            node_list=node_list,
            elapsed_seconds=int(elapsed) if re.fullmatch(r"[0-9]+", elapsed) else None,
        )
    return observations


def _timestamp(value: str) -> datetime | None:
    """Parse the explicitly requested SLURM UTC presentation format."""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def requeued_job_ended(
    job_id: str,
    submitted_after: datetime,
    *,
    deadline: float,
    executable: Path = Path("/usr/bin/scontrol"),
) -> bool:
    """Bound ambiguous requeue observation; this can justify only UNKNOWN ownership."""
    remaining = deadline - time.monotonic()
    if re.fullmatch(r"[0-9]+", job_id) is None or remaining <= 0:
        return False
    try:
        result = subprocess.run(
            [str(executable), "show", "job", job_id, "--oneliner"],
            check=True,
            capture_output=True,
            text=True,
            timeout=remaining,
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        _requeued_terminal_record(result.stdout, job_id, submitted_after)
        and time.monotonic() < deadline
    )


def _requeued_terminal_record(
    output: str, job_id: str, submitted_after: datetime
) -> bool:
    """Keep current-controller termination separate from accounting/entry closure."""
    lines = output.strip().splitlines()
    if (
        len(output.encode()) > 64 * 1024
        or len(lines) != 1
        or not lines[0].startswith(f"JobId={job_id} ")
    ):
        return False
    pairs = re.findall(r"(?:^| )([A-Za-z][A-Za-z0-9]*)=(\S*)", lines[0])
    fields = dict(pairs)
    if (
        len(fields) != len(pairs)
        or any(key.startswith(("Array", "Het")) for key in fields)
        or fields.get("JobId") != job_id
        or fields.get("BatchFlag") != "1"
        or not re.fullmatch(r"[0-9]+", fields.get("Restarts", ""))
        or int(fields["Restarts"]) < 1
        or re.fullmatch(rf"[^() ]+\({os.getuid()}\)", fields.get("UserId", "")) is None
        or fields.get("JobState") not in _TERMINAL_STATES
    ):
        return False
    submitted, started, ended = (
        _timestamp(fields.get(key, ""))
        for key in ("SubmitTime", "StartTime", "EndTime")
    )
    return (
        submitted is not None
        and started is not None
        and ended is not None
        and submitted_after <= submitted <= started <= ended
    )


def cancel_jobs(
    job_ids: list[str],
    timeout: float = 1.0,
    *,
    executable: Path = Path("/usr/bin/scancel"),
) -> str | None:
    """Request cancellation only for exact numeric job or numeric step IDs."""
    if not job_ids:
        return None
    if not all(_EXACT_ID.fullmatch(value) for value in job_ids):
        return "Refusing cancellation with an invalid exact SLURM identifier"
    try:
        subprocess.run(
            [str(executable), *job_ids],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Exact SLURM cancellation failed: {exc}"
    return None
