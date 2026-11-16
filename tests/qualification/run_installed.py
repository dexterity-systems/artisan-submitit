"""Run bounded installed public scheduler tests and preserve exact owner evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from artisan.utils.configuration_secrets import COMMAND_ATTACHMENT_ENV

from artisan_submitit.cancellation import cancel_jobs, query_job_states
from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    ReadyRecord,
    UnitEvidence,
)


def _evidence(
    root: Path, mode: Literal["sbatch", "srun"]
) -> tuple[set[str], list[dict[str, object]], list[str]]:
    jobs: set[str] = set()
    packs = []
    errors = []
    for path in sorted(root.rglob("pack.json")):
        directory = path.parent
        try:
            request = PackRequest.model_validate_json(path.read_bytes())
            assert request.pack_id == directory.name
            if mode == "sbatch":
                ready = ReadyRecord.model_validate_json(
                    (directory / "ready.json").read_bytes()
                )
                assert (ready.pack_id, ready.pack_digest) == (
                    request.pack_id,
                    request.digest,
                )
                jobs.add(ready.job_id)
            entries = sorted((directory / "entries").glob("*/identity.json"))
            assert entries
            identities = []
            entry_errors = []
            for entry_path in entries:
                try:
                    identity = EntryIdentity.model_validate_json(
                        entry_path.read_bytes()
                    )
                    assert identity.pack_id == request.pack_id
                    assert identity.entry_id == entry_path.parent.name
                    if mode == "srun":
                        assert identity.job_id == os.environ["SLURM_JOB_ID"]
                        assert identity.step_id.startswith(identity.job_id + ".")
                        jobs.add(identity.step_id)
                    else:
                        assert identity.job_id == ready.job_id
                    identities.append((entry_path, identity))
                except Exception as error:
                    entry_errors.append(
                        f"{entry_path.parent.name}: {type(error).__name__}"
                    )
            for entry_path, identity in identities:
                try:
                    outcome = EntryEvidence.model_validate_json(
                        (entry_path.parent / "outcome.json").read_bytes()
                    )
                    assert outcome.entry == identity
                    assert outcome.ownership == "closed"
                    for index in range(request.result_count):
                        unit = UnitEvidence.model_validate_json(
                            (entry_path.parent / f"unit-{index}.json").read_bytes()
                        )
                        assert unit.entry == identity
                        assert unit.index == index
                        assert unit.payload_sha256 == request.payload_digest(index)
                        assert unit.ownership == "closed"
                except Exception as error:
                    entry_errors.append(
                        f"{entry_path.parent.name}: {type(error).__name__}"
                    )
            if entry_errors:
                errors.extend(entry_errors)
                continue
            assert (directory / "closed").is_file()
            packs.append(
                {
                    "pack_id": request.pack_id,
                    "units": request.result_count,
                    "closed": True,
                }
            )
        except Exception as error:
            errors.append(f"{directory.name}: {type(error).__name__}")
    return jobs, packs, errors


def _timeout(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt("Installed scheduler suite reached its 540-second budget")


def main() -> None:
    for name in COMMAND_ATTACHMENT_ENV:
        os.environ.pop(name, None)
    import pytest
    from artisan.orchestration.runner_api import verify_wheel_installation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("sbatch", "srun"))
    parser.add_argument("--partition", required=True)
    parser.add_argument(
        "--scenario",
        choices=(
            "pipeline",
            "unit-cancellation",
            "gpu-pipeline",
            "gpu-unit-cancellation",
        ),
        default="pipeline",
    )
    parser.add_argument("--node")
    args = parser.parse_args()
    gpu_scenario = args.scenario.startswith("gpu-")
    if gpu_scenario:
        assert args.node == "artisan-gpu-qualification"
        os.environ["SLURM_TEST_NODE"] = args.node
        os.environ["ARTISAN_GPU_SELECTED_TEST"] = "selected-qualification-value"
        nodes = subprocess.check_output(
            ["/usr/bin/sinfo", "--noheader", "--nodes", args.node, "--format=%N"],
            text=True,
            timeout=5,
        )
        assert nodes.strip() == args.node
    partitions = subprocess.check_output(
        ["/usr/bin/sinfo", "--noheader", "--partition", args.partition, "--format=%P"],
        text=True,
        timeout=5,
    )
    assert args.partition in {
        line.strip().removesuffix("*") for line in partitions.splitlines()
    }
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    profile = args.profile.resolve()
    wheels = tuple(sorted((profile / "wheels").glob("*.whl")))
    verify_wheel_installation(wheels, interpreter=Path(sys.executable))
    os.environ.update(
        ARTISAN_SLURM_MANIFEST=str(profile / "pixi.toml"),
        ARTISAN_SLURM_CACHE=str(profile / "prepared"),
        ARTISAN_SLURM_PIXI="/usr/local/bin/pixi",
        ARTISAN_SLURM_ENVIRONMENT="worker",
        ARTISAN_SLURM_PREPARE="0",
        ARTISAN_SLURM_MODE=args.mode,
        SLURM_TEST_PARTITION=args.partition,
    )
    tests = Path(__file__).resolve().parents[1] / "integration"
    if gpu_scenario:
        module = str(tests / "test_gpu_installed.py")
        selected = (
            [module + "::test_gpu_unit_cancel_and_typed_literal_injections"]
            if args.scenario == "gpu-unit-cancellation"
            else [
                module + "::test_gpu_kernel_activation_and_current_probe_cache",
                module + "::test_gpu_changed_profile_rejects_before_science",
                module + "::test_gpu_worker_cancel_preserves_independent_cpu_worker",
            ]
        )
        expected_cases = 1 if args.scenario == "gpu-unit-cancellation" else 4
    else:
        selected = [
            str(
                tests
                / (
                    "test_unit_cancellation.py"
                    if args.scenario == "unit-cancellation"
                    else "test_slurm.py"
                    if args.mode == "sbatch"
                    else "test_srun.py"
                )
            )
        ]
        expected_cases = (
            1
            if args.scenario == "unit-cancellation"
            else 4
            if args.mode == "sbatch"
            else 2
        )
    report: dict[str, object] = {
        "mode": args.mode,
        "scenario": args.scenario,
        "partition": args.partition,
        "node": args.node,
        "allocation": os.environ.get("SLURM_JOB_ID"),
        "passed": False,
        "controller": sys.executable,
    }
    (root / "allocation.json").write_text(
        json.dumps({"allocation": report["allocation"]}) + "\n"
    )
    started = time.monotonic()
    # pytest-timeout owns SIGALRM separately for each case.
    previous_deadline_handler = signal.signal(signal.SIGUSR2, _timeout)
    deadline_timer = threading.Timer(540, os.kill, args=(os.getpid(), signal.SIGUSR2))
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        status = pytest.main(
            [
                *selected,
                "--run-slurm",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(root / "tests"),
                "--junitxml",
                str(root / "junit.xml"),
                "--timeout=480",
                "--maxfail=1",
            ]
        )
        report["pytest_status"] = int(status)
    except BaseException as error:
        report["error"] = type(error).__name__
    finally:
        report["test_seconds"] = time.monotonic() - started
        deadline_timer.cancel()
        deadline_timer.join(timeout=1)
        report["deadline_timer_closed"] = not deadline_timer.is_alive()
        if report["deadline_timer_closed"]:
            signal.signal(signal.SIGUSR2, previous_deadline_handler)
        report["junit_passed"] = False
        try:
            cases = ET.parse(root / "junit.xml").getroot().findall(".//testcase")
            counts = {
                kind: sum(case.find(kind) is not None for case in cases)
                for kind in ("skipped", "error", "failure")
            }
            report["junit"] = {
                "cases": len(cases),
                "names": [case.attrib["name"] for case in cases],
                **counts,
            }
            report["junit_passed"] = len(cases) == expected_cases and not any(
                counts.values()
            )
        except Exception as error:
            report["junit_error"] = type(error).__name__
        jobs, packs, errors = _evidence(root, args.mode)
        report.update(jobs=sorted(jobs), packs=packs, evidence_errors=errors)
        deadline = time.monotonic() + 25
        failed = (
            report.get("pytest_status") != 0
            or bool(errors)
            or not packs
            or not report["junit_passed"]
            or report["test_seconds"] > 540
            or not report["deadline_timer_closed"]
        )
        if failed and jobs:
            report["cancellation_error"] = cancel_jobs(sorted(jobs), timeout=1.0)
        observations = {}
        while jobs and time.monotonic() < deadline:
            observations = query_job_states(sorted(jobs), timeout=1.0)
            if all(job in observations and observations[job].terminal for job in jobs):
                break
            time.sleep(0.1)
        report["scheduler_terminal"] = bool(jobs) and all(
            job in observations and observations[job].terminal for job in jobs
        )
        report["scheduler_states"] = {
            job: observation.state for job, observation in observations.items()
        }
        try:
            verify_wheel_installation(wheels, interpreter=Path(sys.executable))
            report["installed_bytes_unchanged"] = True
        except Exception as error:
            report["installed_bytes_unchanged"] = False
            report["installed_verification_error"] = type(error).__name__
            failed = True
        report["seconds"] = time.monotonic() - started
        report["passed"] = not failed and bool(report["scheduler_terminal"])
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
