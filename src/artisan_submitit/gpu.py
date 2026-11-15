"""Closed scheduled GPU observation, separate from live allocation identities."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from typing import Literal, Self

from artisan.orchestration.runner_api import (
    ApptainerRuntime,
    ContainerLauncherIdentity,
    NativeGPUAllocation,
    NativeGPUSoftware,
    PreparedLocalBindings,
    PreparedSIFEnvironment,
    capture_native_gpu,
    verify_sif_environment,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def concrete_node(value: object) -> str:
    """Require one node, never an unverified range or homogeneous-set label."""
    if not isinstance(value, str) or _NODE.fullmatch(value) is None:
        msg = "Scheduled GPU execution requires one concrete node name"
        raise ValueError(msg)
    return value


class SIFProbeSpec(BaseModel):
    """Exact allocation-free container targets for a fixed scheduled check."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    runtime: ApptainerRuntime
    launcher: ContainerLauncherIdentity
    targets: tuple[PreparedSIFEnvironment, ...]

    @model_validator(mode="after")
    def allocation_free(self) -> Self:
        digests = tuple(target.digest for target in self.targets)
        if (
            self.runtime.gpu_devices is not None
            or self.launcher.kind != "apptainer"
            or not digests
            or digests != tuple(sorted(set(digests)))
        ):
            msg = "Scheduled SIF probes require allocation-free runtime and unique targets"
            raise ValueError(msg)
        return self

    @classmethod
    def from_local(cls, local: PreparedLocalBindings) -> Self | None:
        runtime = local.container_runtime
        if runtime is None:
            return None
        if not isinstance(runtime, ApptainerRuntime) or local.gpu_driver is not None:
            msg = "Scheduled container GPUs require allocation-free Apptainer software"
            raise ValueError(msg)
        assert local.container_launcher is not None
        targets = {
            binding.target.digest: binding.target
            for binding in local.tools
            if isinstance(binding.target, PreparedSIFEnvironment)
        }
        return cls(
            runtime=runtime,
            launcher=local.container_launcher,
            targets=tuple(targets[key] for key in sorted(targets)),
        )

    def verify(
        self,
        allocation: NativeGPUAllocation,
        *,
        expected: ContainerLauncherIdentity | None = None,
    ) -> ContainerLauncherIdentity:
        """Check prepared SIF availability and the allocated GPU launcher."""
        if self.runtime.identity() != self.launcher:
            msg = "Scheduled SIF launcher changed after preparation"
            raise ValueError(msg)
        for target in self.targets:
            verify_sif_environment(target, apptainer=self.runtime)
        measured = replace(self.runtime, gpu_devices=allocation.devices).identity()
        payloads = {item.path: item.sha256 for item in measured.helpers}
        if any(
            payloads.get(f"gpu/rootfs/.singularity.d/libs/{library.name}")
            != library.sha256
            for library in allocation.libraries
        ):
            msg = "Scheduled SIF does not expose the freshly measured CUDA driver libraries"
            raise ValueError(msg)
        if expected is not None and measured != expected:
            msg = "Scheduled SIF GPU software changed after current qualification"
            raise ValueError(msg)
        return measured


class GPUProbeSpec(BaseModel):
    """The fixed read-only measurement has no scientific callable or admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    node: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    device_count: int = Field(gt=0)
    sif: SIFProbeSpec | None = None

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class ScheduledGPUProfile(BaseModel):
    """Bind measured software to a concrete qualified placement, without UUIDs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    node: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    software: NativeGPUSoftware
    sif: SIFProbeSpec | None = None
    container_launcher: ContainerLauncherIdentity | None = None

    @model_validator(mode="after")
    def complete_container_measurement(self) -> Self:
        if (self.sif is not None) != (self.container_launcher is not None) or (
            self.container_launcher is not None
            and self.container_launcher.kind != "apptainer"
        ):
            msg = "Scheduled SIF software requires its measured GPU launcher"
            raise ValueError(msg)
        return self


class GPUProbeReply(BaseModel):
    """Closed measurement evidence; this is never an authored operation result."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["success", "failure"]
    profile: ScheduledGPUProfile | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if (self.kind == "success") != (self.profile is not None) or (
            self.kind == "failure"
        ) != (self.error is not None):
            msg = "GPU measurement result fields are inconsistent"
            raise ValueError(msg)
        return self

    def result(self) -> ScheduledGPUProfile:
        if self.kind == "failure":
            raise RuntimeError(self.error)
        assert self.profile is not None
        return self.profile


def allocated_node() -> str:
    """Validate the scheduler's exact one-node allocation before any GPU access."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1":
        msg = "GPU measurement requires an actual one-node allocation"
        raise ValueError(msg)
    names = {
        concrete_node(os.environ[key])
        for key in ("SLURM_JOB_NODELIST", "SLURM_NODELIST", "SLURMD_NODENAME")
        if key in os.environ
    }
    if len(names) != 1:
        msg = "GPU measurement requires one consistent allocated node"
        raise ValueError(msg)
    return names.pop()


def capture_allocated_gpu(
    node: str, count: int, expected: NativeGPUSoftware | None = None
) -> NativeGPUAllocation:
    """Measure only within the scheduled allocation and reject software drift."""
    if allocated_node() != node:
        msg = "GPU allocation differs from the qualified concrete node"
        raise ValueError(msg)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        msg = "Scheduled GPU visibility was not assigned explicitly"
        raise ValueError(msg)
    allocation = capture_native_gpu(count)
    assert allocation is not None
    if expected is not None and allocation.software_profile() != expected:
        msg = "Scheduled GPU software changed after current qualification"
        raise ValueError(msg)
    return allocation
