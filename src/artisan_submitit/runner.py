"""Reuse the core local resolver while leaving scheduler ownership to providers."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal, cast

from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AdmittedCurator,
    AssignmentDispatch,
    AssignmentHandle,
    AssignmentLaunchError,
    AssignmentSoftware,
    CancellationAcknowledgement,
    CancellationStatus,
    CompiledPlan,
    CuratorHandle,
    CuratorLaunchError,
    Definition,
    LocalRunner,
    OperationDefinition,
    OwnershipState,
    Parameters,
    PreparedLocalBindings,
    RunnerResources,
)

from artisan_submitit.calls import ProbeDispatch
from artisan_submitit.gpu import (
    GPUProbeSpec,
    ScheduledGPUProfile,
    SIFProbeSpec,
    allocated_node,
    concrete_node,
)
from artisan_submitit.software import (
    SchedulerSoftware,
    SlurmSoftware,
    verify_submitit_plugins,
)
from artisan_submitit.transport import (
    PackOwner,
    SchedulerProbeHandle,
    SchedulerStartupError,
)
from artisan_submitit.utils import validate_srun_arguments


@dataclass(frozen=True, kw_only=True)
class SchedulerRunner(LocalRunner):
    """Prepared worker/tool resolution is shared with the core local runner."""

    log_folder: Path = field(default_factory=lambda: Path.cwd() / "slurm_logs")
    scheduler_bin: Path = Path("/usr/bin")
    parameters: Mapping[str, object] = field(default_factory=dict)
    mode: ClassVar[Literal["sbatch", "srun"]]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.environment is None:
            msg = "Scheduler runners require an explicit canonical worker environment"
            raise ValueError(msg)
        if any(
            runtime is not None and runtime.gpu_devices is not None
            for runtime in (self.docker, self.apptainer)
        ):
            msg = "Scheduler container devices must come from the actual allocation"
            raise ValueError(msg)
        if not self.log_folder.is_absolute() or "%" in str(self.log_folder):
            msg = "Scheduler control roots must be absolute literal paths"
            raise ValueError(msg)
        if not self.scheduler_bin.is_absolute():
            msg = "Scheduler binaries require an absolute directory"
            raise ValueError(msg)
        _validate_parameters(self.parameters)
        parameters = {
            key: validate_srun_arguments(value) if key == "srun_args" else value
            for key, value in self.parameters.items()
        }
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def worker_code_roots(self) -> tuple[type, ...]:
        verify_submitit_plugins()
        from submitit import AutoExecutor  # noqa: PLC0415

        return (type(self), AutoExecutor)

    def prepare(
        self,
        definition: Definition,
        *,
        params: Parameters = None,
        runner_resources: RunnerResources | None = None,
        application_modules: tuple[str, ...] = (),
    ) -> None:
        resources = _local_resources(
            runner_resources or RunnerResources(cpus=1, memory_gb=4, gpus=0)
        )
        super().prepare(
            definition,
            params=params,
            runner_resources=resources,
            application_modules=application_modules,
        )

    def prepare_curator(
        self,
        operation: OperationDefinition,
        *,
        runner_resources: RunnerResources | None = None,
        application_modules: tuple[str, ...] = (),
    ) -> None:
        super().prepare_curator(
            operation,
            runner_resources=_local_resources(
                runner_resources or RunnerResources(cpus=1, memory_gb=4, gpus=0)
            ),
            application_modules=application_modules,
        )

    def resolve(
        self,
        plan: CompiledPlan,
        runner_resources: RunnerResources,
        *,
        application_modules: tuple[str, ...] = (),
    ) -> SlurmSoftware:
        _validate_parameters({**self.parameters, **runner_resources.extra})
        local = super().resolve(
            plan,
            _local_resources(runner_resources),
            application_modules=application_modules,
        )
        return self._resolve_software(
            _local_binding(local), runner_resources, curator=False
        )

    def resolve_curator(
        self,
        operation: OperationDefinition,
        runner_resources: RunnerResources,
        *,
        application_modules: tuple[str, ...] = (),
    ) -> SlurmSoftware:
        _validate_parameters({**self.parameters, **runner_resources.extra})
        local = super().resolve_curator(
            operation,
            _local_resources(runner_resources),
            application_modules=application_modules,
        )
        return self._resolve_software(
            _local_binding(local), runner_resources, curator=True
        )

    def _resolve_software(
        self, local: PreparedLocalBindings, resources: RunnerResources, *, curator: bool
    ) -> SlurmSoftware:
        base = SlurmSoftware(
            local,
            SchedulerSoftware.capture(self.mode, self.scheduler_bin),
            resources.model_copy(update={"gpus": 0}).model_dump_json(),
        )
        if not resources.gpus:
            return base
        parameters = {**self.parameters, **resources.extra}
        node = (
            allocated_node()
            if self.mode == "srun"
            else concrete_node(parameters.get("slurm_nodelist"))
        )
        if (
            "slurm_nodelist" in parameters
            and concrete_node(parameters["slurm_nodelist"]) != node
        ):
            msg = "Requested GPU node differs from the actual allocation"
            raise ValueError(msg)
        probe = GPUProbeSpec(
            node=node, device_count=resources.gpus, sif=SIFProbeSpec.from_local(local)
        )
        measured = self._measure_gpu(probe, base, curator=curator)
        return replace(
            base,
            resources_json=resources.model_dump_json(),
            gpu=measured,
        )

    def _measure_gpu(
        self, probe: GPUProbeSpec, software: SlurmSoftware, *, curator: bool
    ) -> ScheduledGPUProfile:
        deadline = time.monotonic() + 120
        owner: PackOwner | None = None
        try:
            owner = self.create_owner(
                probe, software, ProbeDispatch(self.log_folder / "probe-working")
            )
            handle = owner.handles[0]
            assert isinstance(handle, SchedulerProbeHandle)
            result = handle.result(timeout=max(0.0, deadline - time.monotonic() - 10))
            while (
                owner.ownership_state() == OwnershipState.PENDING
                and time.monotonic() < deadline - 10
            ):
                time.sleep(0.05)
            if owner.ownership_state() != OwnershipState.CLOSED:
                msg = "GPU qualification did not confirm scheduler closure"
                raise RuntimeError(msg)
            if (
                result.node != probe.node
                or result.software.device_count != probe.device_count
                or result.sif != probe.sif
            ):
                msg = (
                    "GPU measurement differs from its exact node, count or SIF request"
                )
                raise ValueError(msg)
            return result
        except Exception as error:
            if isinstance(error, SchedulerStartupError):
                owner = error.owner
            if owner is None:
                raise
            cleanup_deadline = min(deadline, time.monotonic() + 10)
            state = OwnershipState.UNKNOWN
            try:
                owner.request_close(cleanup_deadline)
                state = owner.ownership_state()
                while (
                    state == OwnershipState.PENDING
                    and time.monotonic() < cleanup_deadline
                ):
                    time.sleep(0.05)
                    state = owner.ownership_state()
            except Exception:
                state = OwnershipState.UNKNOWN
            launch_error = CuratorLaunchError if curator else AssignmentLaunchError
            msg = f"Scheduled GPU qualification failed: {error}"
            raise launch_error(
                msg,
                handles=(),
                worker_owners=(owner,),
                cancellation=CancellationAcknowledgement(
                    CancellationStatus.CONFIRMED
                    if state == OwnershipState.CLOSED
                    else CancellationStatus.UNKNOWN,
                    "GPU qualification owner closed"
                    if state == OwnershipState.CLOSED
                    else "GPU qualification ownership unresolved",
                ),
            ) from error

    def launch(
        self,
        assignments: Sequence[AdmittedAssignment],
        *,
        software: AssignmentSoftware,
        dispatch: AssignmentDispatch,
    ) -> tuple[AssignmentHandle, ...]:
        selected = self._software(software)
        if not assignments:
            return ()
        for assignment in assignments:
            if (
                assignment.attempt.software_digest != selected.software_digest
                or assignment.attempt.software_cacheable != selected.software_cacheable
            ):
                msg = "Scheduler software differs from the admitted assignment"
                raise ValueError(msg)
            selected.verify_controller(assignment.plan)
        try:
            owner = self.create_owner(assignments, selected, dispatch)
        except SchedulerStartupError as error:
            raise AssignmentLaunchError(
                str(error),
                handles=cast(tuple[AssignmentHandle, ...], error.owner.handles),
                worker_owners=(error.owner,),
                cancellation=CancellationAcknowledgement(
                    CancellationStatus.CONFIRMED, "Scheduler dispatch never started"
                ),
            ) from error
        return cast(tuple[AssignmentHandle, ...], owner.handles)

    def launch_curator(
        self,
        curator: AdmittedCurator,
        *,
        software: AssignmentSoftware,
        dispatch: AssignmentDispatch,
    ) -> CuratorHandle:
        selected = self._software(software)
        selected.verify_controller()
        try:
            owner = self.create_owner(curator, selected, dispatch)
        except SchedulerStartupError as error:
            raise CuratorLaunchError(
                str(error),
                handles=cast(tuple[CuratorHandle, ...], error.owner.handles),
                worker_owners=(error.owner,),
                cancellation=CancellationAcknowledgement(
                    CancellationStatus.CONFIRMED, "Scheduler dispatch never started"
                ),
            ) from error
        return cast(CuratorHandle, owner.handles[0])

    def _software(self, software: AssignmentSoftware) -> SlurmSoftware:
        if (
            not isinstance(software, SlurmSoftware)
            or software.scheduler.transport != self.mode
        ):
            msg = "Scheduler dispatch requires its exact resolved provider software"
            raise ValueError(msg)
        return software

    def create_owner(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
    ) -> PackOwner:
        raise NotImplementedError


def _local_resources(resources: RunnerResources) -> RunnerResources:
    # Canonical preparation never observes devices on the controller/login node.
    return resources.model_copy(update={"gpus": 0, "extra": {}})


def _validate_parameters(parameters: Mapping[str, object]) -> None:
    forbidden = {
        "max_array_size",
        "array_parallelism",
        "slurm_array_parallelism",
        "nodes",
        "slurm_nodes",
        "tasks_per_node",
        "slurm_tasks_per_node",
        "ntasks_per_node",
        "slurm_ntasks_per_node",
        "slurm_python",
        "python",
        "slurm_setup",
        "setup",
        "slurm_signal_delay_s",
        "slurm_max_num_timeout",
    }
    if forbidden & parameters.keys():
        msg = f"Scheduler topology/bootstrap/array controls are owned by Artisan: {sorted(forbidden & parameters.keys())}"
        raise ValueError(msg)
    supported = {
        "slurm_partition",
        "slurm_account",
        "slurm_qos",
        "slurm_constraint",
        "slurm_comment",
        "slurm_nodelist",
        "srun_args",
    }
    if parameters.keys() - supported:
        msg = (
            f"Unsupported scheduler parameters: {sorted(parameters.keys() - supported)}"
        )
        raise ValueError(msg)
    for key, value in parameters.items():
        if key == "slurm_nodelist":
            concrete_node(value)
        elif key == "srun_args":
            validate_srun_arguments(value)
        elif not isinstance(value, str) or "\n" in value:
            msg = f"Scheduler parameter {key} requires a single-line string"
            raise ValueError(msg)


def _local_binding(software: AssignmentSoftware) -> PreparedLocalBindings:
    if not isinstance(software, PreparedLocalBindings):
        msg = "The core local resolver returned an unsupported binding"
        raise TypeError(msg)
    return software
