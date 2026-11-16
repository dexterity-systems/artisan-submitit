"""Qualify expected scheduler faults without weakening positive closure checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from artisan.utils.configuration_secrets import COMMAND_ATTACHMENT_ENV

from artisan_submitit.cancellation import cancel_jobs, query_job_states
from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    ReadyRecord,
    UnitEvidence,
)

_CASES = {
    "queued": "test_queued_cancel_requires_positive_never_started_proof",
    "native": "test_expired_native_owner_deadline_retains_unknown_and_fences_retry",
    "requeue": "test_requeue_restores_payload_and_retains_first_entry_unknown",
}


def _discover(
    root: Path,
) -> tuple[set[str], list[tuple[Path, PackRequest, ReadyRecord]], list[str]]:
    """Capture all exact accepted job IDs before interpreting any entry outcome."""
    jobs: set[str] = set()
    packs = []
    errors = []
    for path in sorted(root.rglob("pack.json")):
        try:
            request = PackRequest.model_validate_json(path.read_bytes())
            ready = ReadyRecord.model_validate_json(
                (path.parent / "ready.json").read_bytes()
            )
            assert request.pack_id == path.parent.name == ready.pack_id
            assert request.digest == ready.pack_digest
            jobs.add(ready.job_id)
            packs.append((path.parent, request, ready))
        except Exception as error:
            errors.append(f"{path.parent.name}: {type(error).__name__}")
    return jobs, packs, errors


def _requeued_terminal(job_id: str, deadline: float) -> dict[str, str] | None:
    """Observe current scheduler teardown only; this is not worker closure proof."""
    remaining = deadline - time.monotonic()
    if re.fullmatch(r"[0-9]+", job_id) is None or remaining <= 0:
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/scontrol", "show", "job", job_id, "--oneliner"],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(1.0, remaining),
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if len(result.stdout.encode()) > 64 * 1024:
        return None
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or not lines[0].startswith(f"JobId={job_id} "):
        return None
    pairs = re.findall(r"(?:^| )([A-Za-z][A-Za-z0-9]*)=(\S*)", lines[0])
    fields = dict(pairs)
    required = {"JobId", "JobState", "EndTime", "UserId", "BatchFlag", "Restarts"}
    if (
        len(fields) != len(pairs)
        or not required <= fields.keys()
        or any(key.startswith(("Array", "Het")) for key in fields)
        or fields["JobId"] != job_id
        or fields["BatchFlag"] != "1"
        or not fields["Restarts"].isdigit()
        or int(fields["Restarts"]) < 1
        or re.fullmatch(rf"[^() ]+\({os.getuid()}\)", fields["UserId"]) is None
        or fields["JobState"]
        not in {
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
    ):
        return None
    try:
        datetime.strptime(fields["EndTime"], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    if time.monotonic() >= deadline:
        return None
    return {key: fields[key] for key in sorted(required)}


def _cleanup_states(
    jobs: set[str], deadline: float
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Keep exact requeue controller observations separate from worker evidence."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {}, {}
    observations = query_job_states(sorted(jobs), timeout=min(1.0, remaining))
    if time.monotonic() >= deadline:
        return {}, {}
    states = {job: item.state for job, item in observations.items() if item.terminal}
    controller_records = {}
    for job in sorted(jobs - observations.keys()):
        record = _requeued_terminal(job, deadline)
        if record is not None:
            states[job] = record["JobState"]
            controller_records[job] = record
    return (states, controller_records) if time.monotonic() < deadline else ({}, {})


def cleanup_known_jobs(root: Path, deadline: float) -> dict[str, object]:
    """Attempt only request-bound IDs, including after an interrupted collector."""
    jobs, _packs, errors = _discover(root)
    states, controller_records = _cleanup_states(jobs, deadline)
    active = sorted(jobs - states.keys())
    remaining = deadline - time.monotonic()
    error = (
        cancel_jobs(active, timeout=min(1.0, remaining))
        if active and remaining > 0
        else None
    )
    while jobs and time.monotonic() < deadline and set(states) != jobs:
        states, observed = _cleanup_states(jobs, deadline)
        controller_records.update(observed)
        if set(states) == jobs:
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return {
        "jobs": sorted(jobs),
        "discovery_errors": errors,
        "cancellation_error": error,
        "scheduler_terminal": bool(jobs) and set(states) == jobs,
        "scheduler_states": states,
        "scheduler_controller_records": controller_records,
        "scheduler_observation_scope": "Resource teardown only; worker ownership is unchanged",
    }


def _entries(
    directory: Path, request: PackRequest, ready: ReadyRecord
) -> list[dict[str, object]]:
    result = []
    for entry_path in sorted((directory / "entries").glob("*")):
        assert entry_path.is_dir()
        assert not entry_path.is_symlink()
        path = entry_path / "identity.json"
        identity = EntryIdentity.model_validate_json(path.read_bytes())
        assert identity.pack_id == request.pack_id
        assert identity.entry_id == path.parent.name
        assert identity.job_id == ready.job_id
        outcome_path = path.parent / "outcome.json"
        outcome = (
            EntryEvidence.model_validate_json(outcome_path.read_bytes())
            if outcome_path.exists()
            else None
        )
        if outcome is not None:
            assert outcome.entry == identity
        units = []
        for index in range(request.result_count):
            unit_path = path.parent / f"unit-{index}.json"
            if not unit_path.exists():
                continue
            unit = UnitEvidence.model_validate_json(unit_path.read_bytes())
            assert unit.entry == identity
            assert unit.index == index
            assert unit.payload_sha256 == request.payload_digest(index)
            units.append({"index": index, "ownership": unit.ownership})
        result.append(
            {
                "identity": identity.model_dump(mode="json"),
                "ownership": outcome.ownership if outcome else None,
                "units": units,
            }
        )
    return result


def _proof(
    root: Path, scenario: str, packs: list[tuple[Path, PackRequest, ReadyRecord]]
) -> list[dict[str, object]]:
    (path,) = root.rglob("scenario.json")
    assert path.stat().st_size < 1024 * 1024
    selected = json.loads(path.read_bytes())
    assert selected["scenario"] == scenario
    assert len(packs) == (2 if scenario == "queued" else 1)
    result = []
    target_seen = False
    for directory, request, ready in packs:
        entries = _entries(directory, request, ready)
        target = request.pack_id == selected["pack_id"]
        if target:
            target_seen = True
            assert ready.job_id == selected["job_id"]
            if scenario == "queued":
                assert selected["ownership"] == "closed"
                assert not entries
                assert (directory / "closed").is_file()
                before = json.loads((path.parent / "before-cancel.json").read_bytes())
                assert before == {
                    "job_id": ready.job_id,
                    "state": "PENDING",
                    "entries": [],
                }
                assert query_job_states([ready.job_id])[ready.job_id].never_started
            else:
                assert selected["ownership"] == "unknown"
                assert not (directory / "closed").exists()
                assert len(entries) == (2 if scenario == "requeue" else 1)
                assert selected["retry_fenced"] is True
                assert selected["resume_fenced"] is True
                assert selected["terminal"]["cancellation_status"] == "unknown"
                if scenario == "native":
                    assert selected["deadline_expired_without_closed"] is True
                    assert (
                        selected["deadline_started"]
                        < selected["cleanup_deadline"]
                        <= selected["deadline_observed"]
                    )
                    assert (
                        0.249
                        <= (selected["cleanup_deadline"] - selected["deadline_started"])
                        <= 0.251
                    )
                if scenario == "requeue":
                    assert selected["outer_restored"] is True
                    assert selected["first_entry_retained"] is True
                    assert selected["second_entry_closed"] is True
                    assert selected["second_seal_verified"] is True
                    assert (
                        len(selected["old_deliveries"])
                        == len(selected["new_deliveries"])
                        == 1
                    )
                    assert set(selected["old_deliveries"]).isdisjoint(
                        selected["new_deliveries"]
                    )
                    first_id, second_id = (
                        item["entry_id"] for item in selected["entries"]
                    )
                    assert first_id in Path(selected["first_science"]["cwd"]).parts
                    assert second_id in Path(selected["second_science"]["cwd"]).parts
                    assert (
                        hashlib.sha256(
                            (directory / "outer.bin").read_bytes()
                        ).hexdigest()
                        == selected["outer_sha256"]
                        == ready.outer_sha256
                    )
                    assert {entry["identity"]["entry_id"] for entry in entries} == {
                        entry["entry_id"] for entry in selected["entries"]
                    }
                    by_id = {entry["identity"]["entry_id"]: entry for entry in entries}
                    assert by_id[first_id]["ownership"] is None
                    assert by_id[second_id]["ownership"] == "closed"
                    assert by_id[second_id]["units"] == [
                        {"index": 0, "ownership": "closed"}
                    ]
                    assert (
                        selected["second_receipt"]["delivery"]["delivery_id"]
                        == selected["new_deliveries"][0]
                    )
        else:
            assert (directory / "closed").is_file()
            assert entries
            for entry in entries:
                assert entry["ownership"] == "closed"
                assert len(entry["units"]) == request.result_count
                assert all(unit["ownership"] == "closed" for unit in entry["units"])
        result.append(
            {
                "pack_id": request.pack_id,
                "job_id": ready.job_id,
                "target": target,
                "closed": (directory / "closed").exists(),
                "entries": entries,
            }
        )
    assert target_seen
    return result


def _timeout(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt("Installed scheduler fault reached its 300-second budget")


def main() -> None:
    for name in COMMAND_ATTACHMENT_ENV:
        os.environ.pop(name, None)
    import pytest
    from artisan.orchestration.runner_api import verify_wheel_installation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--scenario", choices=tuple(_CASES), required=True)
    args = parser.parse_args()
    partitions = subprocess.check_output(
        ["/usr/bin/sinfo", "--noheader", "--partition", args.partition, "--format=%P"],
        text=True,
        timeout=5,
    )
    assert args.partition in {
        line.strip().removesuffix("*") for line in partitions.splitlines()
    }
    root, profile = args.root.resolve(), args.profile.resolve()
    root.mkdir(parents=True, exist_ok=False)
    wheels = tuple(sorted((profile / "wheels").glob("*.whl")))
    verify_wheel_installation(wheels, interpreter=Path(sys.executable))
    os.environ.update(
        ARTISAN_SLURM_MANIFEST=str(profile / "pixi.toml"),
        ARTISAN_SLURM_CACHE=str(profile / "prepared"),
        ARTISAN_SLURM_PIXI="/usr/local/bin/pixi",
        ARTISAN_SLURM_ENVIRONMENT="worker",
        ARTISAN_SLURM_PREPARE="0",
        SLURM_TEST_PARTITION=args.partition,
    )
    test = Path(__file__).resolve().parents[1] / "integration/test_ownership_faults.py"
    report: dict[str, object] = {
        "scenario": args.scenario,
        "passed": False,
        "controller": sys.executable,
    }
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, 300)
    try:
        status = pytest.main(
            [
                f"{test}::{_CASES[args.scenario]}",
                "--run-slurm",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(root / "tests"),
                "--junitxml",
                str(root / "junit.xml"),
                "--timeout=280",
            ]
        )
        report["pytest_status"] = int(status)
    except BaseException as error:
        report["error"] = type(error).__name__
    finally:
        report["test_seconds"] = time.monotonic() - started
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["junit_passed"] = False
        try:
            assert (root / "junit.xml").stat().st_size < 1024 * 1024
            cases = ET.parse(root / "junit.xml").getroot().findall(".//testcase")
            report["junit_passed"] = (
                len(cases) == 1
                and cases[0].attrib["name"] == _CASES[args.scenario]
                and not any(
                    cases[0].find(kind) is not None
                    for kind in ("skipped", "error", "failure")
                )
            )
        except Exception as error:
            report["junit_error"] = type(error).__name__
        jobs, packs, errors = _discover(root)
        report.update(jobs=sorted(jobs), evidence_errors=errors)
        report.update(cleanup_known_jobs(root, time.monotonic() + 25))
        report["scenario_proof_passed"] = False
        try:
            report["packs"] = _proof(root, args.scenario, packs)
            report["scenario_proof_passed"] = True
        except Exception as error:
            report["proof_error"] = type(error).__name__
        try:
            verify_wheel_installation(wheels, interpreter=Path(sys.executable))
            report["installed_bytes_unchanged"] = True
        except Exception as error:
            report["installed_bytes_unchanged"] = False
            report["installed_error"] = type(error).__name__
        report["seconds"] = time.monotonic() - started
        report["passed"] = bool(
            report.get("pytest_status") == 0
            and report["junit_passed"]
            and not errors
            and report["scenario_proof_passed"]
            and report["scheduler_terminal"]
            and report["installed_bytes_unchanged"]
            and report["test_seconds"] <= 300
            and report["seconds"] <= 330
        )
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
