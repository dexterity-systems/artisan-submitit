"""Scheduler-facing helpers shared by the SLURM runners."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MAX_ARRAY_SIZE = 1000
MAX_LOG_BYTES = 100_000

logger = logging.getLogger(__name__)


def parse_time_to_minutes(value: str) -> int:
    """Convert an Artisan wall-clock duration to Submitit minutes.

    Seconds are rounded up so the scheduler limit is never shorter than the
    requested duration.

    Args:
        value: Duration as ``HH:MM:SS``, ``MM:SS``, or integer minutes.

    Returns:
        Whole scheduler minutes.

    Raises:
        ValueError: If the value is malformed or negative.
    """
    parts = value.split(":")
    if len(parts) not in {1, 2, 3}:
        msg = f"Invalid wall-clock duration {value!r}"
        raise ValueError(msg)
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        msg = f"Invalid wall-clock duration {value!r}"
        raise ValueError(msg) from exc
    if any(number < 0 for number in numbers):
        msg = f"Wall-clock duration cannot be negative: {value!r}"
        raise ValueError(msg)

    if len(numbers) == 1:
        return numbers[0]
    if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
        msg = f"Invalid wall-clock duration {value!r}"
        raise ValueError(msg)

    if len(numbers) == 2:
        minutes, seconds = numbers
        total_seconds = minutes * 60 + seconds
    else:
        hours, minutes, seconds = numbers
        total_seconds = hours * 3600 + minutes * 60 + seconds
    return math.ceil(total_seconds / 60)


def get_cluster_max_array_size(override: int | None = None) -> int:
    """Read SLURM's ``MaxArraySize``, with a conservative fallback.

    Args:
        override: Explicit positive limit. Skips cluster detection when set.

    Returns:
        The configured, detected, or fallback array size.

    Raises:
        ValueError: If an explicit override is not positive.
    """
    if override is not None:
        if override < 1:
            msg = f"max_array_size must be >= 1, got {override}"
            raise ValueError(msg)
        return override

    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not detect MaxArraySize: %s", exc)
        return DEFAULT_MAX_ARRAY_SIZE

    match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
    if result.returncode == 0 and match is not None:
        return int(match.group(1))
    logger.warning("Using fallback MaxArraySize: %s", DEFAULT_MAX_ARRAY_SIZE)
    return DEFAULT_MAX_ARRAY_SIZE


def chunked[T](items: Sequence[T], size: int) -> list[list[T]]:
    """Split a sequence into ordered chunks no larger than ``size``."""
    if size < 1:
        msg = f"chunk size must be >= 1, got {size}"
        raise ValueError(msg)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def read_job_logs(
    stdout_path: Path,
    stderr_path: Path,
    retries: int = 5,
    delay: float = 1.0,
) -> str | None:
    """Read and combine worker logs after invalidating NFS metadata caches."""
    stdout = ""
    for attempt in range(retries):
        stdout = _read_text_nfs_safe(stdout_path)
        if stdout or attempt == retries - 1:
            break
        time.sleep(delay)
    stderr = _read_text_nfs_safe(stderr_path)

    parts = []
    if stdout:
        parts.append(stdout[-MAX_LOG_BYTES:])
    if stderr:
        parts.append(f"--- stderr ---\n{stderr[-MAX_LOG_BYTES:]}")
    return "\n".join(parts) or None


def _read_text_nfs_safe(path: Path) -> str:
    """Read a text file after forcing an NFS attribute-cache refresh."""
    if not path.exists():
        with contextlib.suppress(OSError):
            list(path.parent.iterdir())
        if not path.exists():
            return ""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fstat(fd)
        finally:
            os.close(fd)
        return path.read_text(errors="replace")
    except OSError:
        return ""
