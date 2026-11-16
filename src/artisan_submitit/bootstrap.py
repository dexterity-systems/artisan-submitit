"""Verify inert scheduler evidence before Submitit loads its live outer call.

This file is invoked by its qualified installed pathname, so package imports do
not run before the explicit bootstrap checks.
"""

from __future__ import annotations

# SDK/provider imports follow the inert runtime checks.
# ruff: noqa: PLC0415
import argparse
import hashlib
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artisan_submitit.control import PackRequest


def _job_identity() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or os.environ.get("SLURM_ARRAY_JOB_ID"):
        msg = "Provider bootstrap requires one exact non-array SLURM job"
        raise ValueError(msg)
    for key in ("SLURM_PROCID", "SLURM_LOCALID"):
        if os.environ.get(key, "0") != "0":
            msg = "Provider bootstrap requires a single task"
            raise ValueError(msg)
    if os.environ.get("SLURM_NTASKS", "1") != "1":
        msg = "Provider bootstrap requires a single task"
        raise ValueError(msg)
    return job_id


def _verified_request(directory: Path, expected_digest: str) -> PackRequest:
    import json

    from artisan.orchestration.runner_api import (
        LocalWorkerProfile,
        verify_runtime_compatibility,
    )

    encoded = (directory / "pack.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        msg = "Scheduler pack bytes differ from their exact launch binding"
        raise ValueError(msg)
    inert = json.loads(encoded)
    profile = LocalWorkerProfile.from_json(inert["profile"])
    profile.verify()
    assert profile.worker.compatibility is not None
    verify_runtime_compatibility(profile.worker.compatibility)
    from artisan_submitit.control import decode_pack
    from artisan_submitit.software import verify_submitit_plugins

    request = decode_pack(directory, encoded, expected_digest)
    verify_submitit_plugins()
    for call in request.calls:
        call.load(directory)
    return request


def _capture_outer(
    directory: Path,
    sdk_folder: Path,
    job_id: str,
    request: PackRequest,
    expected_digest: str,
) -> bytes:
    from artisan_submitit.control import ReadyRecord

    deadline = time.monotonic() + 60
    while not (directory / "ready.json").exists():
        if time.monotonic() >= deadline:
            msg = "Controller did not publish exact Submitit readiness"
            raise TimeoutError(msg)
        time.sleep(0.05)
    ready = ReadyRecord.model_validate_json((directory / "ready.json").read_bytes())
    if (ready.pack_id, ready.pack_digest, ready.job_id) != (
        request.pack_id,
        request.digest,
        job_id,
    ):
        msg = "Submitit readiness differs from the current job or pack"
        raise ValueError(msg)
    if sdk_folder != directory / "sdk":
        msg = "Submitit folder differs from its controller-owned pack"
        raise ValueError(msg)
    encoded = (directory / "outer.bin").read_bytes()
    if (
        ready.outer_sha256 != expected_digest
        or hashlib.sha256(encoded).hexdigest() != expected_digest
    ):
        msg = "Submitit outer payload differs from the exact controller capture"
        raise ValueError(msg)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--mode", choices=("sbatch", "srun"), required=True)
    parser.add_argument("--outer-sha256")
    args, suffix = parser.parse_known_args()
    directory = args.pack
    if not directory.is_absolute() or "%" in str(directory):
        msg = "Provider control path must be absolute and literal"
        raise ValueError(msg)
    job_id = _job_identity()
    if (directory / "closed").exists():
        msg = "Scheduler pack admission is already closed"
        raise RuntimeError(msg)
    request = _verified_request(directory, args.pack_sha256)
    entry_id = uuid.uuid4().hex
    sdk_private = directory / "sdk-entries" / entry_id
    if args.mode == "sbatch":
        expected = ["-u", "-m", "submitit.core._submit", str(directory / "sdk")]
        if suffix != expected:
            msg = "Unexpected Submitit bootstrap command suffix"
            raise ValueError(msg)
        if args.outer_sha256 is None:
            msg = "Submitit requires its exact pre-submission payload digest"
            raise ValueError(msg)
        encoded = _capture_outer(
            directory, directory / "sdk", job_id, request, args.outer_sha256
        )
        sdk_private.mkdir(parents=True, mode=0o700)
        target = sdk_private / f"{job_id}_submitted.pkl"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
    elif suffix or args.outer_sha256 is not None:
        msg = "Unexpected srun bootstrap arguments"
        raise ValueError(msg)
    os.environ["SUBMITIT_EXECUTOR"] = "slurm"
    os.environ["SUBMITIT_PREEMPT_SIGNAL"] = "USR2"
    os.environ["ARTISAN_SUBMITIT_ENTRY_ID"] = entry_id
    try:
        if args.mode == "sbatch":
            from submitit.core.submission import process_job

            process_job(sdk_private)
        else:
            from artisan_submitit.worker import run_worker_pack

            run_worker_pack(str(directory), args.pack_sha256)
    finally:
        if sdk_private.exists():
            shutil.rmtree(sdk_private)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Inert validation can include runtime values in library exceptions.
        sys.stderr.write(f"Scheduler bootstrap failed: {type(error).__name__}\n")
        raise SystemExit(1) from None
