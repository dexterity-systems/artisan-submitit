"""Run one prepared process pack within an exact existing SLURM allocation."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AdmittedCurator,
    AssignmentDispatch,
)

from artisan_submitit.calls import ProbeDispatch
from artisan_submitit.control import EntryIdentity, PackRequest
from artisan_submitit.gpu import GPUProbeSpec
from artisan_submitit.launcher import bootstrap_command
from artisan_submitit.runner import SchedulerRunner
from artisan_submitit.software import SlurmSoftware
from artisan_submitit.transport import PackOwner
from artisan_submitit.utils import validate_srun_arguments


@dataclass(frozen=True, kw_only=True)
class SlurmIntraRunner(SchedulerRunner):
    """Target only the owned job.step; sibling allocations and steps remain separate."""

    name: ClassVar[str] = "slurm_intra"
    mode: ClassVar[Literal["sbatch", "srun"]] = "srun"

    def create_owner(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
    ) -> PackOwner:
        allocation = os.environ.get("SLURM_JOB_ID", "")
        if not allocation.isdecimal():
            msg = "SlurmIntraRunner requires an exact existing allocation"
            raise ValueError(msg)
        parameters = {**self.parameters, **software.resources.extra}
        if set(parameters) - {"srun_args", "slurm_nodelist"}:
            msg = "Intra-allocation dispatch accepts only bounded srun_args"
            raise ValueError(msg)
        arguments = list(validate_srun_arguments(parameters.get("srun_args", [])))
        return _SrunPack(
            assignments, software, dispatch, self.log_folder, allocation, arguments
        )


class _SrunPack(PackOwner):
    def __init__(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
        root: Path,
        allocation: str,
        arguments: list[str],
    ) -> None:
        self.allocation = allocation
        self.arguments = arguments
        self.process: subprocess.Popen[bytes] | None = None
        self.step_id: str | None = None
        super().__init__(assignments, software, dispatch, root)

    def submit(self, request: PackRequest) -> None:
        if request.pack_id != self.directory.name:
            msg = "srun request differs from its owned pack"
            raise ValueError(msg)
        resources = self.resources
        command = [
            str(self.software.scheduler.executable("srun")),
            f"--jobid={self.allocation}",
            "--nodes=1",
            "--ntasks=1",
            "--exclusive",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_gb}G",
            f"--time={resources.time_limit}",
            # --gres=none alone does not override inherited SLURM_GPUS.
            *(
                [f"--gpus={resources.gpus}", f"--nodelist={self.gpu_node}"]
                if resources.gpus
                else ["--gpus=0", "--gres=none"]
            ),
            *self.arguments,
            *bootstrap_command(
                self.software,
                self.directory,
                [
                    str(self.software.local.worker.prefix / "bin/python"),
                    "-I",
                    str(self.software.bootstrap_script),
                    "--pack",
                    str(self.directory),
                    "--pack-sha256",
                    request.digest,
                    "--mode",
                    "srun",
                ],
            ),
        ]
        self._job_id = self.allocation
        with (
            (self.directory / "stdout.log").open("wb") as stdout,
            (self.directory / "stderr.log").open("wb") as stderr,
        ):
            self.process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True
            )
        self._submission_started = True

    def observe_identity(self, identity: EntryIdentity) -> None:
        if not identity.step_id.startswith(self.allocation + "."):
            msg = "Worker did not report its exact srun job.step identity"
            raise ValueError(msg)
        if self.step_id is not None and self.step_id != identity.step_id:
            msg = "One synchronous srun launch reported multiple step owners"
            raise ValueError(msg)
        self.step_id = identity.step_id

    def scheduler_id(self) -> str | None:
        return self.step_id

    def transport_closed(self) -> bool:
        return self.process is not None and self.process.poll() is not None

    def transport_failed(self) -> bool:
        return self.process is not None and self.process.poll() is not None

    def close_transport(self, deadline: float) -> None:
        """Terminate and reap only this owned client session, never its allocation."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        for sig, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 1.0)):
            if process.poll() is not None:
                return
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=min(grace, remaining))
            except subprocess.TimeoutExpired:
                continue
            return
