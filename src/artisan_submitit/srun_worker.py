"""Worker entry point for one serialized Artisan unit batch.

The exit code describes bootstrap health, not operation success. Ordinary
operation failures are already represented by Artisan ``UnitResult`` values.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import cloudpickle


def main(job_folder: str) -> None:
    """Execute ``job.pkl`` and atomically write a result envelope."""
    folder = Path(job_folder)
    with (folder / "job.pkl").open("rb") as stream:
        function = cloudpickle.load(stream)

    try:
        result = function()
        envelope: dict[str, Any] = {"status": "ok", "result": result}
    except Exception as exc:
        envelope = {
            "status": "error",
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    _write_atomic(folder / "result.pkl", envelope)


def _write_atomic(path: Path, value: object) -> None:
    """Write a pickle via same-directory rename for atomic visibility."""
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        Path(temporary_path).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temporary_path).unlink()
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m artisan_submitit.srun_worker <job_folder>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    main(sys.argv[1])
