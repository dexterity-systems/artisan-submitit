"""Tests for the standalone srun result-envelope worker."""

from __future__ import annotations

import pickle
import subprocess
import sys
from functools import partial

import cloudpickle


def _run_worker(folder) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "artisan_submitit.srun_worker", str(folder)],
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_srun_worker_success_is_atomic(tmp_path) -> None:
    with (tmp_path / "job.pkl").open("wb") as stream:
        cloudpickle.dump(lambda: 42, stream)

    completed = _run_worker(tmp_path)

    assert completed.returncode == 0
    with (tmp_path / "result.pkl").open("rb") as stream:
        assert pickle.load(stream) == {"status": "ok", "result": 42}
    assert not list(tmp_path.glob("*.tmp"))


def test_srun_worker_records_callable_error(tmp_path) -> None:
    with (tmp_path / "job.pkl").open("wb") as stream:
        cloudpickle.dump(partial(int, "bad unit"), stream)

    completed = _run_worker(tmp_path)

    assert completed.returncode == 0
    with (tmp_path / "result.pkl").open("rb") as stream:
        envelope = pickle.load(stream)
    assert envelope["status"] == "error"
    assert envelope["type"] == "ValueError"
    assert "bad unit" in envelope["error"]
    assert "Traceback" in envelope["traceback"]


def test_srun_worker_bootstrap_failure_is_nonzero(tmp_path) -> None:
    completed = _run_worker(tmp_path)
    assert completed.returncode != 0
    assert not (tmp_path / "result.pkl").exists()
