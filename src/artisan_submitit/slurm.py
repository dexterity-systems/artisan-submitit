"""Submit exactly one controller-owned process pack through Submitit."""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AdmittedCurator,
    AssignmentDispatch,
)

from artisan_submitit.calls import ProbeDispatch
from artisan_submitit.control import PackRequest, ReadyRecord, write_once
from artisan_submitit.gpu import GPUProbeSpec
from artisan_submitit.launcher import bootstrap_command
from artisan_submitit.runner import SchedulerRunner
from artisan_submitit.software import SlurmSoftware, verify_submitit_plugins
from artisan_submitit.transport import PackOwner
from artisan_submitit.utils import parse_time_to_minutes

if TYPE_CHECKING:
    from submitit import AutoExecutor

_OUTER_DIGEST_TOKEN = "ARTISAN_CAPTURED_SUBMITIT_SHA256"


@dataclass(frozen=True, kw_only=True)
class SlurmRunner(SchedulerRunner):
    """Use core packing/concurrency and one AutoExecutor submission per pack."""

    slurm_partition: str = "cpu"
    name: ClassVar[str] = "slurm"
    mode: ClassVar[Literal["sbatch", "srun"]] = "sbatch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.slurm_partition or any(
            char in self.slurm_partition for char in "\n\r\x00"
        ):
            msg = "SLURM partition must be a nonempty single-line name"
            raise ValueError(msg)

    def create_owner(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
    ) -> PackOwner:
        parameters = {
            "slurm_partition": self.slurm_partition,
            **self.parameters,
            **software.resources.extra,
        }
        if "srun_args" in parameters:
            parameters["slurm_srun_args"] = parameters.pop("srun_args")
        return _SubmititPack(
            assignments, software, dispatch, self.log_folder, parameters
        )


class _SubmititPack(PackOwner):
    def __init__(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
        root: Path,
        parameters: dict[str, object],
    ) -> None:
        self.parameters = parameters
        super().__init__(assignments, software, dispatch, root)

    def submit(self, request: PackRequest) -> None:
        self.software.scheduler.verify()
        verify_submitit_plugins()
        import submitit  # noqa: PLC0415

        worker_python = self.software.local.worker.prefix / "bin/python"
        command = [
            str(worker_python),
            "-I",
            str(self.software.bootstrap_script),
            "--pack",
            str(self.directory),
            "--pack-sha256",
            request.digest,
            "--mode",
            "sbatch",
            "--outer-sha256",
            _OUTER_DIGEST_TOKEN,
        ]
        executor = submitit.AutoExecutor(
            folder=self.directory / "sdk",
            cluster="slurm",
            slurm_python=shlex.join(
                bootstrap_command(self.software, self.directory, command)
            ),
            slurm_max_num_timeout=0,
        )
        resources = self.resources
        executor.update_parameters(
            timeout_min=parse_time_to_minutes(resources.time_limit),
            cpus_per_task=resources.cpus,
            mem_gb=resources.memory_gb,
            gpus_per_node=resources.gpus,
            nodes=1,
            tasks_per_node=1,
            slurm_signal_delay_s=30,
            **{
                **self.parameters,
                **({"slurm_nodelist": self.gpu_node} if self.gpu_node else {}),
            },
        )
        if (
            executor.parameters.get("nodes", 1) != 1
            or executor.parameters.get("ntasks_per_node", 1) != 1
        ):
            msg = "Submitit changed the admitted single-task topology"
            raise ValueError(msg)
        if self._stop.is_set():
            raise CancelledError()
        from artisan_submitit.worker import run_worker_pack  # noqa: PLC0415

        captured: list[str] = []
        _bind_outer_capture(executor, self, captured)
        job = executor.submit(run_worker_pack, str(self.directory), request.digest)
        self._job_id = str(job.job_id)
        if not self._job_id.isdecimal():
            msg = "Submitit returned an unexpected array or compound job identifier"
            raise ValueError(msg)
        if len(captured) != 1:
            msg = "Submitit did not bind exactly one pre-submission payload"
            raise ValueError(msg)
        self._publish_requests()
        ready = ReadyRecord(
            pack_id=request.pack_id,
            pack_digest=request.digest,
            job_id=self._job_id,
            outer_sha256=captured[0],
        )
        write_once(self.directory / "ready.json", ready.model_dump_json().encode())


def _bind_outer_capture(
    executor: AutoExecutor, owner: _SubmititPack, captured: list[str]
) -> None:
    """Bind the pinned SDK's temporary pickle before it renders an sbatch command."""
    from submitit.slurm.slurm import SlurmExecutor  # noqa: PLC0415

    inner = executor._executor
    if type(inner) is not SlurmExecutor:
        msg = "Unsupported Submitit executor implementation"
        raise ValueError(msg)
    render = inner._make_submission_file_text
    sdk = owner.directory / "sdk"

    def capture(command: str, uid: str) -> str:
        candidates = tuple(sdk.glob("*.pkl"))
        if captured or len(candidates) != 1 or command.count(_OUTER_DIGEST_TOKEN) != 1:
            msg = "Pinned Submitit pre-submission pickle layout changed"
            raise ValueError(msg)
        encoded = candidates[0].read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        script = render(command.replace(_OUTER_DIGEST_TOKEN, digest), uid)
        write_once(owner.directory / "outer.bin", encoded)
        captured.append(digest)
        owner._submission_started = True
        return script

    # The version-pinned SDK exposes no public pre-submission serialization hook.
    inner._make_submission_file_text = capture  # type: ignore[method-assign]
