"""Qualify GPU unit cancellation and typed transport injections with real admissions."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import CancelledError
from dataclasses import replace
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
from slurm_qualification_app import CommandGPUIncrement

from .unit_cancellation import _cleanup


def qualify(
    pipeline: PipelineManager,
    marker: Path,
    *,
    manifest: Path,
    literal: str,
    selected: str,
) -> dict[str, object]:
    """Fence one unit, retain its sibling, and accept the surviving receipt once."""
    plan = compile_definition(
        CommandGPUIncrement,
        params={"manifest": str(manifest), "hold": 120, "marker": str(marker)},
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
        for value in (1, 2)
    ]
    resources = RunnerResources(cpus=1, memory_gb=2, gpus=1, time_limit="00:05:00")
    dispatch = replace(
        pipeline.dispatch,
        runtime_env={
            "QUALIFICATION_LITERAL": literal,
            "QUALIFICATION_SELECTED": selected,
        },
    )
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
        dispatch=dispatch,
        runner_resources=resources,
        batch_strategy=BatchStrategy(
            artifacts_per_unit=1, units_per_worker=2, max_workers=1
        ),
    )
    execution.tracker.create_attempt(record)
    owners: tuple[WorkerOwner, ...] = ()
    try:
        try:
            software = pipeline.runner.resolve(
                plan, resources, application_modules=dispatch.application_modules
            )
            execution._publish(plan, execution.resolve_inputs(plan), software)
            execution.tracker.admit(
                execution._step_admission(plan, software.software_digest)
            )
            handles = pipeline.runner.launch(
                tuple(AdmittedAssignment(plan, item) for item in execution._admissions),
                software=software,
                dispatch=dispatch,
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
        # The tool creates the marker before its JSON write finishes.
        while True:
            assert time.monotonic() < deadline, "First GPU tool did not start"
            try:
                pids = json.loads(marker.read_bytes())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            break
        assert set(pids) == {"parent", "child"}
        for pid in pids.values():
            assert Path(f"/proc/{pid}").is_dir()
            assert os.readlink(f"/proc/{pid}/ns/pid") == os.readlink(
                "/proc/self/ns/pid"
            )
        pipeline.committer.journal.cancel(handles[0].attempt)
        handles[0].cancel()
        try:
            handles[0].result(timeout=35)
        except CancelledError:
            pass
        else:
            msg = "Cancelled GPU assignment returned a scientific receipt"
            raise AssertionError(msg)
        receipt = handles[1].result(timeout=90)
        deadline = time.monotonic() + 25
        while any(
            owner.ownership_state() == OwnershipState.PENDING for owner in owners
        ):
            assert time.monotonic() < deadline, "Shared GPU owner did not close"
            time.sleep(0.05)
        assert all(owner.ownership_state() == OwnershipState.CLOSED for owner in owners)
        assert all(not Path(f"/proc/{pid}").exists() for pid in pids.values())
        cancelled = pipeline.committer.journal.state(handles[0].attempt.assignment)
        assert cancelled is not None
        assert cancelled.cancelled
        assert cancelled.winner is None
        pipeline.committer.accept(receipt)
        terminal = execution.results.finish(
            identity,
            failures={
                0: StepUnitFailure(error="Explicit provider GPU unit cancellation")
            },
        )
        assert terminal.status == StepStatus.PARTIAL, terminal.error
        assert terminal.succeeded_count == terminal.failed_count == 1
        (role,) = execution.results.outputs(identity)
        (output,) = role.occurrences
        artifact = output.artifact
        assert isinstance(artifact, MetricArtifact)
        values = artifact.values
        assert values["input_value"] == 2
        assert values["output"] == 42
        assert values["literal"] == literal
        assert values["selected"] == selected
        assert values["activation"] == "prepared-native-gpu"
        assert values["preprocess_pid"] == values["postprocess_pid"]
        assert values["tool_pid"] != values["preprocess_pid"]
        assert values["device_count"] == 1
        assert values["cuda_visible_devices"].startswith("GPU-")
        return {
            "terminal": terminal.model_dump(mode="json"),
            "surviving_receipt": receipt.model_dump(mode="json"),
            "kernel": values,
            "cancelled_pids": pids,
            "ownership": [owner.ownership_state().value for owner in owners],
            "scope": "direct typed AssignmentDispatch injections and per-unit cancellation",
        }
    finally:
        _cleanup(owners)
