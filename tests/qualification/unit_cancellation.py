"""Qualify direct provider unit cancellation independently of manager policy.

The fixture uses the current controller's real admission builder, fences the
cancelled attempt before requesting cancellation, and accepts only the surviving
sealed result. It does not submit this test through the whole-step cancellation
policy, which intentionally stops all units owned by that step.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import CancelledError
from pathlib import Path
from uuid import uuid4

from artisan.orchestration import PipelineManager, StepStatus
from artisan.orchestration.engine.retained_executor import RetainedStepExecution
from artisan.orchestration.engine.retained_inputs import prepare_step_inputs
from artisan.orchestration.engine.retained_step_tracker import RetainedStepTracker
from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AssignmentLaunchError,
    OwnershipState,
    WorkerOwner,
    compile_definition,
)
from artisan.schemas import MetricArtifact
from artisan.schemas.execution.batch_strategy import BatchStrategy
from artisan.schemas.operation_config.runner_resources import RunnerResources
from artisan.schemas.orchestration.step_start_record import StepStartRecord
from artisan.storage.core.retained_reader import RetainedReader
from artisan.storage.io.step_result_plan import StepUnitFailure
from slurm_qualification_app import PackedWait

# Lost closure is still represented by the exact owner, not just its error text.
_UNRESOLVED: list[WorkerOwner] = []


def qualify(pipeline: PipelineManager, marker: Path) -> dict[str, object]:
    """Use two real ordered assignments and accept exactly one surviving receipt."""
    plan = compile_definition(
        PackedWait, params={"marker_root": str(marker), "duration": 120}
    )
    identity = uuid4().hex
    record = StepStartRecord(
        step_run_id=identity,
        step_number=0,
        step_name=plan.recipe.definition.name,
        operation_class=f"{plan.recipe.definition.module}.{plan.recipe.definition.qualname}",
        params_json=plan.recipe.params_json,
        input_refs_json="{}",
        compute_backend=pipeline.runner.name,
        compute_options_json="{}",
        output_roles_json=json.dumps([item.role for item in plan.recipe.outputs]),
        output_types_json=json.dumps(
            {item.role: item.artifact_type for item in plan.recipe.outputs}
        ),
    )
    sources = [
        MetricArtifact.draft({"value": value}, f"input-{value}.json").finalize()
        for value in (0, 1)
    ]
    resources = RunnerResources(cpus=1, memory_gb=1, time_limit="00:05:00")
    execution = RetainedStepExecution(
        record,
        resolve_plan=lambda: plan,
        resolve_inputs=lambda compiled: prepare_step_inputs(
            compiled.recipe,
            {"source": sources},
            step_run_id=identity,
            reader=RetainedReader(pipeline.committer),
        ),
        tracker=RetainedStepTracker(
            pipeline.committer, pipeline.config.pipeline_run_id
        ),
        runner=pipeline.runner,
        dispatch=pipeline.dispatch,
        runner_resources=resources,
        batch_strategy=BatchStrategy(
            artifacts_per_unit=1, units_per_worker=2, max_workers=1
        ),
    )
    execution.tracker.create_attempt(record)
    software = pipeline.runner.resolve(
        plan, resources, application_modules=pipeline.dispatch.application_modules
    )
    execution._publish(plan, execution.resolve_inputs(plan), software)
    execution.tracker.admit(execution._step_admission(plan, software.software_digest))
    owners: tuple[WorkerOwner, ...] = ()
    try:
        try:
            handles = pipeline.runner.launch(
                tuple(
                    AdmittedAssignment(plan, reference)
                    for reference in execution._admissions
                ),
                software=software,
                dispatch=pipeline.dispatch,
            )
        except AssignmentLaunchError as error:
            owners = tuple(
                dict.fromkeys(
                    (
                        *error.worker_owners,
                        *(handle.worker_owner for handle in error.handles),
                    )
                )
            )
            raise
        owners = tuple(dict.fromkeys(handle.worker_owner for handle in handles))
        assert len(handles) == 2
        assert len(owners) == 1
        deadline = time.monotonic() + 90
        while not (marker / "0/heartbeat").exists():
            assert time.monotonic() < deadline, "First unit did not enter authored work"
            time.sleep(0.05)
        pipeline.committer.journal.cancel(handles[0].attempt)
        handles[0].cancel()
        try:
            handles[0].result(timeout=35)
        except CancelledError:
            pass
        else:
            msg = "Cancelled assignment returned a scientific receipt"
            raise AssertionError(msg)
        receipt = handles[1].result(timeout=90)
        deadline = time.monotonic() + 25
        while any(
            owner.ownership_state() == OwnershipState.PENDING for owner in owners
        ):
            assert time.monotonic() < deadline, "Shared owner did not close"
            time.sleep(0.05)
        assert all(owner.ownership_state() == OwnershipState.CLOSED for owner in owners)
        # One explicit acceptance; no fabricated result or acceptance of the fenced unit.
        pipeline.committer.accept(receipt)
        terminal = execution.results.finish(
            identity,
            failures={0: StepUnitFailure(error="Explicit provider unit cancellation")},
        )
        assert terminal.status == StepStatus.PARTIAL, terminal.error
        assert terminal.succeeded_count == terminal.failed_count == 1
        outputs = execution.results.outputs(identity)
        assert sum(len(role.occurrences) for role in outputs) == 1
        return {
            "terminal": terminal.model_dump(mode="json"),
            "surviving_receipt": receipt.model_dump(mode="json"),
            "ownership": [owner.ownership_state().value for owner in owners],
            "scope": "direct runner per-unit cancellation, separate from manager whole-step policy",
        }
    finally:
        _cleanup(owners)


def _cleanup(owners: tuple[WorkerOwner, ...]) -> None:
    """Attempt every retained owner within one deadline; never hide a lost owner."""
    deadline = time.monotonic() + 25
    pending = list(owners)
    for owner in owners:
        try:
            if owner.ownership_state() != OwnershipState.CLOSED:
                owner.request_close(deadline)
        except Exception:
            _UNRESOLVED.append(owner)
    while pending:
        remaining = []
        for owner in pending:
            try:
                state = owner.ownership_state()
            except Exception:
                state = OwnershipState.UNKNOWN
            if state == OwnershipState.UNKNOWN or (
                state != OwnershipState.CLOSED and time.monotonic() >= deadline
            ):
                _UNRESOLVED.append(owner)
            elif state == OwnershipState.PENDING:
                remaining.append(owner)
        pending = remaining
        if pending:
            time.sleep(0.05)


def main() -> None:
    from artisan.schemas import Environment

    from artisan_submitit import SlurmIntraRunner, SlurmRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("sbatch", "srun"), required=True)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--pixi", type=Path, default=Path.home() / ".pixi/bin/pixi")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    kind = SlurmRunner if args.mode == "sbatch" else SlurmIntraRunner
    options = {"slurm_partition": args.partition} if args.mode == "sbatch" else {}
    runner = kind(
        environment=Environment.from_pixi(args.manifest.resolve(), name="worker"),
        cache_root=args.cache_root.resolve(),
        pixi=args.pixi.resolve(),
        log_folder=root / "scheduler",
        **options,
    )
    with PipelineManager.create(
        "direct-unit-cancel",
        str(root / "delta"),
        str(root / "staging"),
        working_root=str(root / "working"),
        default_step_runner=runner,
        application_modules=("slurm_qualification_app",),
    ) as pipeline:
        receipt = qualify(pipeline, root / "markers")
    assert not _UNRESOLVED
    (root / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
