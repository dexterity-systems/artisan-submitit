"""Measure scheduler clients and compose the qualified core worker binding."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Self

from artisan.orchestration.runner_api import (
    Assignment,
    CompiledPlan,
    ContainerLauncherIdentity,
    ExecuteAdapter,
    NativeGPUAllocation,
    PreparedAssignmentSoftware,
    PreparedCuratorSoftware,
    PreparedLocalBindings,
    PreparedNativeEnvironment,
    RunnerResources,
    RuntimeCompatibility,
    verify_native_gpu_loader,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from artisan_submitit.gpu import (
    ScheduledGPUProfile,
    SIFProbeSpec,
    capture_allocated_gpu,
)
from artisan_submitit.launcher import PRELUDE_SHA256

type ClientName = Literal["sacct", "sbatch", "scancel", "scontrol", "srun"]
_CLIENTS: tuple[ClientName, ...] = ("sacct", "sbatch", "scancel", "scontrol", "srun")


class SchedulerExecutable(BaseModel):
    """One exact client executable used for submission or ownership evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Literal["sacct", "sbatch", "scancel", "scontrol", "srun", "bash"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str

    @field_validator("path")
    @classmethod
    def absolute_path(cls, path: Path) -> Path:
        if not path.is_absolute():
            msg = "Scheduler executables require explicit absolute paths"
            raise ValueError(msg)
        return path

    @classmethod
    def capture(
        cls,
        name: Literal["sacct", "sbatch", "scancel", "scontrol", "srun", "bash"],
        directory: Path,
    ) -> Self:
        path = directory / name
        discovered = shutil.which(name)
        if discovered is None or Path(discovered).resolve() != path.resolve():
            msg = f"Scheduler PATH does not select the qualified client: {name}"
            raise ValueError(msg)
        if not path.is_file() or not os.access(path, os.X_OK):
            msg = f"Scheduler executable is unavailable: {path}"
            raise ValueError(msg)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        version = subprocess.check_output(
            [str(path), "--version"], text=True, timeout=5
        ).strip()
        return cls(name=name, path=path, sha256=digest, version=version)


class SchedulerSoftware(BaseModel):
    """Provider additions remeasured on the node before scientific execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    transport: Literal["sbatch", "srun"]
    submitit_version: Literal["1.5.4"] = "1.5.4"
    executables: tuple[SchedulerExecutable, ...]
    bootstrap_shell: SchedulerExecutable

    @field_validator("bootstrap_shell")
    @classmethod
    def exact_shell(cls, shell: SchedulerExecutable) -> SchedulerExecutable:
        if shell.name != "bash":
            msg = "Scheduler bootstrap requires the measured Bash launcher"
            raise ValueError(msg)
        return shell

    @field_validator("executables")
    @classmethod
    def exact_clients(
        cls, clients: tuple[SchedulerExecutable, ...]
    ) -> tuple[SchedulerExecutable, ...]:
        if tuple(item.name for item in clients) != (
            "sacct",
            "sbatch",
            "scancel",
            "scontrol",
            "srun",
        ):
            msg = "Scheduler software requires the complete ordered client inventory"
            raise ValueError(msg)
        return clients

    @classmethod
    def capture(
        cls, transport: Literal["sbatch", "srun"], directory: Path = Path("/usr/bin")
    ) -> Self:
        verify_submitit_plugins()
        return cls(
            transport=transport,
            bootstrap_shell=SchedulerExecutable.capture("bash", Path("/bin")),
            executables=tuple(
                SchedulerExecutable.capture(name, directory) for name in _CLIENTS
            ),
        )

    def verify(self) -> None:
        verify_submitit_plugins()
        for expected in (*self.executables, self.bootstrap_shell):
            if (
                SchedulerExecutable.capture(expected.name, expected.path.parent)
                != expected
            ):
                msg = f"Qualified scheduler client changed: {expected.name}"
                raise ValueError(msg)

    def executable(self, name: str) -> Path:
        return next(item.path for item in self.executables if item.name == name)


def verify_submitit_plugins() -> None:
    """Close SDK import discovery before executor construction or pickle loading."""
    if importlib.metadata.version("submitit") != "1.5.4":
        msg = "This provider requires qualified Submitit 1.5.4"
        raise ValueError(msg)
    if importlib.metadata.entry_points(group="submitit"):
        msg = "Unqualified Submitit entry-point plugins are unsupported"
        raise ValueError(msg)
    endpoint_overrides = {
        "SLURM_CONF",
        "SLURM_CONF_SERVER",
        "SLURM_CLUSTERS",
    } & os.environ.keys()
    if endpoint_overrides:
        msg = "Qualified default scheduler configuration does not accept endpoint overrides"
        raise ValueError(msg)
    if "_TEST_CLUSTER_" in os.environ:
        msg = "Submitit test-cluster overrides are unsupported"
        raise ValueError(msg)
    if os.environ.get("SUBMITIT_PREEMPT_SIGNAL", "USR2") != "USR2":
        msg = "Submitit requires the qualified USR2 preemption signal"
        raise ValueError(msg)


@dataclass(frozen=True)
class SlurmSoftware(PreparedAssignmentSoftware, PreparedCuratorSoftware):
    """Add scheduler identity without replacing scientific or local-launch evidence."""

    local: PreparedLocalBindings
    scheduler: SchedulerSoftware
    resources_json: str
    gpu: ScheduledGPUProfile | None = None

    def __post_init__(self) -> None:
        if self.local.worker_is_observed:
            msg = "SLURM workers require an explicitly prepared environment"
            raise ValueError(msg)

        if bool(self.resources.gpus) != (self.gpu is not None):
            msg = "Scheduled GPU resources require current measured software"
            raise ValueError(msg)
        if (
            self.gpu is not None
            and self.gpu.software.device_count != self.resources.gpus
        ):
            msg = "Scheduled GPU count differs from current qualification"
            raise ValueError(msg)
        if self.local.native_gpu is not None:
            msg = "Controller bindings cannot contain login-node GPU allocation"
            raise ValueError(msg)
        if self.gpu is not None and self.gpu.sif != SIFProbeSpec.from_local(self.local):
            msg = "Scheduled SIF qualification differs from the selected local targets"
            raise ValueError(msg)

    @property
    def application_modules(self) -> tuple[str, ...]:
        return self.local.application_modules

    @property
    def bootstrap_script(self) -> Path:
        """Locate the installed provider bootstrap inside the prepared worker."""
        worker = self.local.worker
        if not isinstance(worker, PreparedNativeEnvironment):
            msg = "Scheduler bootstrap requires a prepared worker"
            raise ValueError(msg)
        if not any(
            package.name == "artisan-submitit"
            for package in self.compatibility.packages
        ):
            msg = "Prepared worker lacks the provider package"
            raise ValueError(msg)
        major, minor, *_ = self.compatibility.python_version.split(".")
        path = (
            worker.prefix
            / "lib"
            / f"python{major}.{minor}"
            / "site-packages/artisan_submitit/bootstrap.py"
        )
        if not path.is_file() or not path.resolve().is_relative_to(worker.prefix):
            msg = "Prepared worker lacks an in-prefix provider bootstrap"
            raise ValueError(msg)
        return path

    @property
    def resources(self) -> RunnerResources:
        return RunnerResources.model_validate_json(self.resources_json)

    @property
    def software_digest(self) -> str:
        descriptor = {
            "local": self.local.software_digest,
            "scheduler": self.scheduler.model_dump(mode="json"),
            "bootstrap": "inert-submitit-v2",
            "prelude_sha256": PRELUDE_SHA256,
        }
        if self.gpu is not None:
            descriptor["native_gpu"] = self.gpu.software.model_dump(mode="json")
            if self.gpu.container_launcher is not None:
                descriptor["container_gpu"] = self.gpu.container_launcher.model_dump(
                    mode="json"
                )
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"artisan.submitit-software-v1\0" + encoded).hexdigest()

    @property
    def software_cacheable(self) -> bool:
        return self.local.software_cacheable

    @property
    def compatibility(self) -> RuntimeCompatibility:
        return self.local.compatibility

    def verify_controller(self, plan: CompiledPlan | None = None) -> None:
        """Check prepared targets and scheduler identity without controller GPU probes."""
        self.scheduler.verify()
        if plan is None:
            self.local.verify_worker()
        else:
            self.local.verify(plan)

    def _allocation(self) -> NativeGPUAllocation | None:
        if self.gpu is None:
            return None
        allocation = capture_allocated_gpu(
            self.gpu.node, self.gpu.software.device_count, self.gpu.software
        )
        if self.gpu.sif is not None:
            self.gpu.sif.verify(allocation, expected=self.gpu.container_launcher)
        verify_native_gpu_loader(allocation)
        return allocation

    def verify(self, plan: CompiledPlan) -> None:
        self.verify_controller(plan)
        self._allocation()

    def verify_worker(self) -> None:
        self.verify_controller()
        self._allocation()

    def create_adapter(
        self,
        assignment: Assignment,
        *,
        working_root: Path,
        runtime_env: Mapping[str, str],
    ) -> ExecuteAdapter:
        local = _allocated_bindings(
            self.local,
            self._allocation(),
            container_launcher=self.gpu.container_launcher
            if self.gpu is not None
            else None,
        )
        return local.create_adapter(
            assignment, working_root=working_root, runtime_env=runtime_env
        )


def _allocated_bindings(
    local: PreparedLocalBindings,
    allocation: NativeGPUAllocation | None,
    *,
    container_launcher: ContainerLauncherIdentity | None = None,
) -> PreparedLocalBindings:
    """Copy fresh allocation into execution state after node identity checks."""
    if allocation is None or local.container_runtime is None:
        return replace(local, native_gpu=allocation)
    if (
        local.container_launcher is None
        or local.container_launcher.kind != "apptainer"
        or container_launcher is None
        or container_launcher.kind != "apptainer"
    ):
        msg = "Scheduled container GPU binding requires qualified Apptainer software"
        raise ValueError(msg)
    if local.container_runtime.gpu_devices is not None or local.gpu_driver is not None:
        msg = (
            "Controller container bindings cannot select GPU devices or driver evidence"
        )
        raise ValueError(msg)
    return replace(
        local,
        native_gpu=allocation,
        container_runtime=replace(
            local.container_runtime, gpu_devices=allocation.devices
        ),
        gpu_driver=allocation.driver,
        container_launcher=container_launcher,
    )
