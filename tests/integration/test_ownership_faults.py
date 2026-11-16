"""Actual scheduler failures preserve exact owner and controller boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from artisan.errors import PersistenceIntegrityError
from artisan.orchestration import CancellationStatus, PipelineManager, StepStatus
from artisan.orchestration.runner_api import (
    AssignmentReply,
    CommandOwnershipError,
    OwnershipState,
)
from artisan.storage.cache.retained_lookup import lookup_assignment
from artisan.storage.io.retained_seal import capture_retained_delivery

from artisan_submitit.cancellation import cancel_jobs, query_job_states
from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    ReadyRecord,
    UnitEvidence,
)

from ..qualification.fixture_identity import expected_node_python
from ..qualification.run_faults import cleanup_known_jobs
from .helpers import (
    APPLICATIONS,
    RESOURCES,
    assert_closed,
    pipeline,
    runner,
    wait_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(280)]
_UNRESOLVED: list[PipelineManager] = []


def _wait(
    predicate: Callable[[], bool], message: str, *, deadline: float | None = None
) -> None:
    if deadline is None:
        deadline = time.monotonic() + 90
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    assert time.monotonic() < deadline, message


def _pack(root: Path) -> tuple[Path, ReadyRecord]:
    _wait(
        lambda: bool(tuple((root / "scheduler").glob("*/ready.json"))),
        "No scheduler ID",
    )
    (path,) = (root / "scheduler").glob("*/ready.json")
    request = PackRequest.model_validate_json((path.parent / "pack.json").read_bytes())
    ready = ReadyRecord.model_validate_json(path.read_bytes())
    assert (ready.pack_id, ready.pack_digest) == (request.pack_id, request.digest)
    assert ready.pack_id == path.parent.name
    return path.parent, ready


def _close(owner: PipelineManager) -> None:
    if owner.ownership_state() != OwnershipState.CLOSED:
        owner.cancel()
    try:
        owner.close()
    except CommandOwnershipError:
        _UNRESOLVED.append(owner)
        assert owner.ownership_state() == OwnershipState.UNKNOWN


def _resume_rejected(owner: PipelineManager, root: Path) -> None:
    with pytest.raises(PersistenceIntegrityError):
        PipelineManager.resume(
            str(root / "delta"),
            str(root / "staging"),
            working_root=str(root / "working"),
            pipeline_run_id=owner.config.pipeline_run_id,
            default_step_runner=owner.runner,
            application_modules=APPLICATIONS,
        )


def _terminal_cleanup(root: Path, job_id: str, owner: PipelineManager) -> None:
    deadline = time.monotonic() + 25
    assert cancel_jobs([job_id], timeout=1.0) is None
    evidence = cleanup_known_jobs(root, deadline)
    assert evidence["jobs"] == [job_id]
    assert evidence["scheduler_terminal"] is True, evidence
    assert owner.ownership_state() == OwnershipState.UNKNOWN
    (root / "scheduler-cleanup.json").write_text(json.dumps(evidence, indent=2) + "\n")


def test_queued_cancel_requires_positive_never_started_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slurm_qualification_app import ObservableWait

    blocker_root, queued_root = tmp_path / "blocker", tmp_path / "queued"
    marker = tmp_path / "blocker-marker"
    blocker = pipeline(blocker_root, runner(blocker_root, "sbatch"))
    queued = pipeline(queued_root, runner(queued_root, "sbatch"))
    try:
        running = blocker.submit(
            ObservableWait,
            params={"marker_root": str(marker), "duration": 120},
            runner_resources=RESOURCES,
        )
        wait_ready(marker)
        _blocker_pack, blocker_ready = _pack(blocker_root)
        import submitit

        assert submitit.__version__ == "1.5.4"
        target_comment = f"artisan-queue-fault-{blocker_ready.pack_id}"
        original_update = submitit.AutoExecutor.update_parameters
        injected: list[dict[str, object]] = []

        def inject_dependency(
            executor: submitit.AutoExecutor, **parameters: object
        ) -> None:
            if parameters.get("slurm_comment") == target_comment:
                assert "slurm_dependency" not in parameters
                parameters["slurm_dependency"] = f"afterok:{blocker_ready.job_id}"
                injected.append(dict(parameters))
            original_update(executor, **parameters)

        # Explicit scheduling fault injection; the public provider does not accept dependencies.
        monkeypatch.setattr(
            submitit.AutoExecutor, "update_parameters", inject_dependency
        )
        future = queued.submit(
            ObservableWait,
            params={"marker_root": str(tmp_path / "queued-marker"), "duration": 1},
            runner_resources={
                **RESOURCES,
                "extra": {"slurm_comment": target_comment},
            },
        )
        pack, ready = _pack(queued_root)
        assert len(injected) == 1
        (tmp_path / "dependency-injection.json").write_text(
            json.dumps(injected[0], indent=2) + "\n"
        )
        _wait(
            lambda: (
                (observation := query_job_states([ready.job_id]).get(ready.job_id))
                is not None
                and observation.state == "PENDING"
            ),
            "The exact queued job was not observed pending",
        )
        assert not tuple((pack / "entries").glob("*"))
        scheduler_description = subprocess.check_output(
            ["/usr/bin/scontrol", "show", "job", ready.job_id, "--oneliner"],
            text=True,
            timeout=5,
        )
        assert f"Dependency=afterok:{blocker_ready.job_id}" in scheduler_description
        assert "JobState=PENDING" in scheduler_description
        (tmp_path / "dependency-before-cancel.txt").write_text(scheduler_description)
        (tmp_path / "before-cancel.json").write_text(
            json.dumps({"job_id": ready.job_id, "state": "PENDING", "entries": []})
        )
        started = time.monotonic()
        queued.cancel()
        result = future.result(timeout=30)
        assert time.monotonic() - started < 30
        assert result.status == StepStatus.CANCELLED, result
        assert result.cancellation_status == CancellationStatus.CONFIRMED, result
        assert_closed(queued)
        observation = query_job_states([ready.job_id])[ready.job_id]
        assert observation.never_started, observation
        assert not tuple((pack / "entries").glob("*"))
        assert not (tmp_path / "queued-marker" / "ready.json").exists()
        execution = queued.executions[0]
        assert execution.results.outputs(execution.record.step_run_id) == ()
        (marker / "stop").touch()
        survived = running.result(timeout=60)
        assert survived.status == StepStatus.SUCCEEDED, survived.error
        assert_closed(blocker)
        (tmp_path / "scenario.json").write_text(
            json.dumps(
                {
                    "scenario": "queued",
                    "job_id": ready.job_id,
                    "pack_id": ready.pack_id,
                    "ownership": "closed",
                    "never_started": True,
                    "terminal": result.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "stop").touch()
        try:
            _close(queued)
        finally:
            _close(blocker)


def test_expired_native_owner_deadline_retains_unknown_and_fences_retry(
    tmp_path: Path,
) -> None:
    from slurm_qualification_app import ObservableWait

    marker = tmp_path / "marker"
    owner = pipeline(tmp_path, runner(tmp_path, "sbatch"))
    try:
        future = owner.submit(
            ObservableWait,
            params={"marker_root": str(marker), "duration": 120},
            runner_resources=RESOURCES,
        )
        wait_ready(marker)
        pack, ready = _pack(tmp_path)
        entries = tuple((pack / "entries").glob("*/identity.json"))
        assert len(entries) == 1
        (handle,) = owner.executions[0].handles
        worker = handle.worker_owner
        assert worker is not None
        assert worker.ownership_state() == OwnershipState.PENDING
        assert not (marker / "stop").exists()
        started = time.monotonic()
        deadline = started + 0.25
        worker.request_close(deadline)
        owner.cancel()
        while time.monotonic() < deadline:
            assert worker.ownership_state() != OwnershipState.CLOSED
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        expired_at = time.monotonic()
        assert expired_at >= deadline
        assert worker.ownership_state() == OwnershipState.UNKNOWN
        result = future.result(timeout=30)
        elapsed = time.monotonic() - started
        assert elapsed < 30
        assert result.cancellation_status == CancellationStatus.UNKNOWN, result
        assert owner.ownership_state() == OwnershipState.UNKNOWN
        execution = owner.executions[0]
        assert execution.results.outputs(execution.record.step_run_id) == ()
        original = owner.committer.journal.state(
            execution.handles[0].attempt.assignment
        )
        assert original is not None
        assert original.cancelled
        assert original.winner is None
        retried = owner.run(
            ObservableWait,
            params={"marker_root": str(tmp_path / "retry-marker"), "duration": 0},
            runner_resources=RESOURCES,
        )
        assert retried.status != StepStatus.SUCCEEDED
        assert not owner.executions[-1].handles
        assert not owner.executions[-1]._admissions
        assert (
            owner.committer.journal.state(execution.handles[0].attempt.assignment)
            == original
        )
        _resume_rejected(owner, tmp_path)
        with pytest.raises(CommandOwnershipError):
            owner.close()
        assert owner.ownership_state() == OwnershipState.UNKNOWN
        _terminal_cleanup(tmp_path, ready.job_id, owner)
        assert pack.exists()
        assert not (pack / "closed").exists()
        (tmp_path / "scenario.json").write_text(
            json.dumps(
                {
                    "scenario": "native",
                    "job_id": ready.job_id,
                    "pack_id": ready.pack_id,
                    "ownership": "unknown",
                    "cleanup_seconds": elapsed,
                    "deadline_started": started,
                    "cleanup_deadline": deadline,
                    "deadline_observed": expired_at,
                    "deadline_expired_without_closed": True,
                    "terminal": result.model_dump(mode="json"),
                    "retry_fenced": True,
                    "resume_fenced": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        _close(owner)


def _bootstrap_fault(pack: Path | None = None) -> dict[str, object]:
    """Use the fixed node interpreter's native pidfd APIs, never a PID fallback."""
    expected = expected_node_python(
        Path(os.environ["ARTISAN_SLURM_FIXTURE_RECEIPT"]),
        receipt_sha256=os.environ["ARTISAN_SLURM_FIXTURE_SHA256"],
    )
    helper = Path(__file__).resolve().parents[1] / "qualification/kill_bootstrap.py"
    command = ["/usr/bin/python3", "-I", str(helper)]
    if pack is not None:
        command.append(str(pack))
    result = json.loads(subprocess.check_output(command, timeout=5))
    assert result["interpreter_sha256"] == expected
    assert result["helper_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
    return result


def test_requeue_restores_payload_and_retains_first_entry_unknown(
    tmp_path: Path,
) -> None:
    from slurm_qualification_app import ObservableWait

    case_started = time.monotonic()
    capability = _bootstrap_fault()
    assert capability["native_pidfd"] is True
    marker = tmp_path / "marker"
    owner = PipelineManager.create(
        "requeue-fault",
        str(tmp_path / "delta"),
        str(tmp_path / "staging"),
        working_root=str(tmp_path / "working"),
        default_step_runner=runner(tmp_path, "sbatch"),
        application_modules=APPLICATIONS,
        preserve_staging=True,
        preserve_working=True,
    )
    try:
        future = owner.submit(
            ObservableWait,
            params={"marker_root": str(marker), "duration": 120},
            runner_resources=RESOURCES,
        )
        wait_ready(marker)
        pack, ready = _pack(tmp_path)
        (first_path,) = (pack / "entries").glob("*/identity.json")
        first_bytes = first_path.read_bytes()
        first = EntryIdentity.model_validate_json(first_bytes)
        assert first.job_id == ready.job_id
        outer = pack / "outer.bin"
        digest = hashlib.sha256(outer.read_bytes()).hexdigest()
        assert digest == ready.outer_sha256
        request_bytes = (pack / "pack.json").read_bytes()
        mutable = (
            pack / "sdk-entries" / first.entry_id / f"{ready.job_id}_submitted.pkl"
        )
        assert mutable.is_file()
        assert not mutable.is_symlink()
        assert hashlib.sha256(mutable.read_bytes()).hexdigest() == digest
        first_scratch = tmp_path / "working" / pack.name / first.entry_id
        old_deliveries = tuple(
            p.name
            for p in first_scratch.glob("unit-*/*")
            if p.is_dir() and re.fullmatch(r"[0-9a-f]{32}", p.name)
        )
        assert len(old_deliveries) == 1
        first_science = json.loads((marker / "ready.json").read_bytes())
        assert Path(first_science["cwd"]).is_relative_to(first_scratch)
        mutable.write_bytes(b"qualification: stale per-entry SDK checkpoint")
        fault = _bootstrap_fault(pack)
        (tmp_path / "first-entry.json").write_text(
            json.dumps(
                {
                    "entry": first.model_dump(mode="json"),
                    "fault": fault,
                    "outer_sha256": digest,
                    "old_deliveries": old_deliveries,
                },
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            ["/usr/bin/scontrol", "requeue", ready.job_id], check=True, timeout=10
        )
        requeue_record = subprocess.check_output(
            ["/usr/bin/scontrol", "show", "job", ready.job_id, "--oneliner"],
            text=True,
            timeout=1,
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
        )
        assert requeue_record.startswith(f"JobId={ready.job_id} ")
        assert f"UserId=qualification({os.getuid()}) " in requeue_record
        assert " Restarts=1 " in requeue_record
        assert " EligibleTime=" in requeue_record
        wait_evidence = {
            "controller_record": requeue_record,
            "observed_wall_time": time.time(),
            "case_started": case_started,
            "entry_deadline": case_started + 230,
        }
        (tmp_path / "requeue-wait.json").write_text(
            json.dumps(wait_evidence, indent=2) + "\n"
        )
        # Credential revocation can delay re-entry beyond the ordinary 90s wait.
        _wait(
            lambda: len(tuple((pack / "entries").glob("*/identity.json"))) == 2,
            "The exact original job did not enter again",
            deadline=case_started + 230,
        )
        second_path = next(
            p for p in (pack / "entries").glob("*/identity.json") if p != first_path
        )
        second = EntryIdentity.model_validate_json(second_path.read_bytes())
        assert second.job_id == ready.job_id
        assert second.pack_id == first.pack_id
        assert second.entry_id != first.entry_id
        fresh = pack / "sdk-entries" / second.entry_id / f"{ready.job_id}_submitted.pkl"
        _wait(fresh.exists, "Second entry has no isolated SDK checkpoint")
        assert hashlib.sha256(fresh.read_bytes()).hexdigest() == digest
        assert hashlib.sha256(outer.read_bytes()).hexdigest() == digest
        assert (pack / "pack.json").read_bytes() == request_bytes
        assert first_path.read_bytes() == first_bytes
        second_scratch = tmp_path / "working" / pack.name / second.entry_id

        def second_science_ready() -> bool:
            try:
                value = json.loads((marker / "ready.json").read_bytes())
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            return Path(value["cwd"]).is_relative_to(second_scratch)

        _wait(second_science_ready, "No actual science in fresh second-entry scratch")
        second_science = json.loads((marker / "ready.json").read_bytes())
        new_deliveries = tuple(
            p.name
            for p in second_scratch.glob("unit-*/*")
            if p.is_dir() and re.fullmatch(r"[0-9a-f]{32}", p.name)
        )
        assert len(new_deliveries) == 1
        assert set(new_deliveries).isdisjoint(old_deliveries)
        assert first_scratch.is_dir()
        assert second_science["pid"] != first_science["pid"]
        (marker / "stop").touch()
        _wait(
            lambda: (second_path.parent / "outcome.json").exists(),
            "Second actual entry did not publish closure",
        )
        second_outcome = EntryEvidence.model_validate_json(
            (second_path.parent / "outcome.json").read_bytes()
        )
        assert second_outcome.entry == second
        assert second_outcome.ownership == "closed"
        second_unit = UnitEvidence.model_validate_json(
            (second_path.parent / "unit-0.json").read_bytes()
        )
        assert second_unit.entry == second
        assert second_unit.ownership == "closed"
        assert not second_unit.cancelled
        execution = owner.executions[0]
        receipt = AssignmentReply.model_validate_json(second_unit.reply).result(
            execution.handles[0].attempt
        )
        assert receipt.delivery.delivery_id == new_deliveries[0]
        with capture_retained_delivery(
            receipt,
            {},
            expected_recipe=execution.resolve_plan().recipe,
            fs=owner.committer.fs,
        ):
            pass
        result = future.result(timeout=90)
        assert result.status == StepStatus.FAILED, result
        assert result.cancellation_status == CancellationStatus.UNKNOWN, result
        assert owner.ownership_state() == OwnershipState.UNKNOWN
        execution = owner.executions[0]
        assert execution.results.outputs(execution.record.step_run_id) == ()
        state = owner.committer.journal.state(execution.handles[0].attempt.assignment)
        assert state is not None
        winner = state.winner.receipt if state.winner is not None else None
        if winner is not None:
            assert all(
                winner.delivery.delivery_id not in path for path in old_deliveries
            )
        cache_hit = lookup_assignment(execution._admissions[0], reader=execution.reader)
        plans = execution.reader.completed_plans()
        accepted = [plan.receipt.model_dump(mode="json") for plan in plans]
        retried = owner.run(
            ObservableWait,
            params={"marker_root": str(tmp_path / "retry-marker"), "duration": 0},
            runner_resources=RESOURCES,
            skip_cache=True,
        )
        assert retried.status == StepStatus.FAILED
        assert not owner.executions[-1].handles
        assert not owner.executions[-1]._admissions
        assert (
            owner.committer.journal.state(execution.handles[0].attempt.assignment)
            == state
        )
        _resume_rejected(owner, tmp_path)
        with pytest.raises(CommandOwnershipError):
            owner.close()
        assert owner.ownership_state() == OwnershipState.UNKNOWN
        _terminal_cleanup(tmp_path, ready.job_id, owner)
        assert first_path.read_bytes() == first_bytes
        assert not (first_path.parent / "outcome.json").exists()
        assert not (pack / "closed").exists()
        (tmp_path / "scenario.json").write_text(
            json.dumps(
                {
                    "scenario": "requeue",
                    "wait_evidence": wait_evidence,
                    "job_id": ready.job_id,
                    "pack_id": ready.pack_id,
                    "ownership": "unknown",
                    "entries": [
                        first.model_dump(mode="json"),
                        second.model_dump(mode="json"),
                    ],
                    "outer_sha256": digest,
                    "outer_restored": True,
                    "first_entry_retained": True,
                    "old_deliveries": old_deliveries,
                    "new_deliveries": new_deliveries,
                    "first_science": first_science,
                    "second_science": second_science,
                    "second_receipt": receipt.model_dump(mode="json"),
                    "second_entry_closed": True,
                    "second_seal_verified": True,
                    "terminal": result.model_dump(mode="json"),
                    "fault": fault,
                    "winner": winner.model_dump(mode="json") if winner else None,
                    "accepted_receipts": accepted,
                    "current_admission_cache_hit": cache_hit is not None,
                    "operation_cacheable": ObservableWait.cacheable,
                    "retry_fenced": True,
                    "resume_fenced": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        _close(owner)
