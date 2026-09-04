"""Intra-allocation Artisan runner using exact ``srun`` processes."""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cloudpickle
from artisan.orchestration.runner_api import (
    BatchStrategy,
    ExecutionUnit,
    LifecycleRouter,
    OrchestratorTraits,
    RunnerBase,
    RunnerResources,
    RuntimeEnvironment,
    UnitResult,
    WorkerTraits,
    execute_unit_batch,
    failure_results_for_units,
    pack_units,
    validate_batch_results,
)

from artisan_submitit.slurm import _attach_worker_log
from artisan_submitit.utils import (
    parse_time_to_minutes,
    read_job_logs,
    reject_managed_scheduler_arguments,
)

logger = logging.getLogger(__name__)

_MIN_LAUNCH_INTERVAL = 0.05
_RESULT_READ_RETRIES = 5
_RESULT_READ_DELAY = 0.1
_SUPPORTED_EXTRAS = frozenset({"slurm_gres", "srun_args"})


@dataclass(frozen=True)
class _SrunProcess:
    """One exact ``srun`` process and the batch it owns."""

    process: subprocess.Popen[bytes]
    units: list[ExecutionUnit]
    folder: Path
    launched_at: float


class SlurmIntraLifecycleRouter(LifecycleRouter):
    """Launch bounded ``srun`` steps inside the current allocation."""

    def __init__(
        self,
        *,
        units_per_worker: int,
        max_workers: int,
        work_folder: Path,
        cpus: int,
        memory_gb: int,
        gpus: int,
        time_limit: str,
        slurm_gres: str | None,
        srun_args: list[str],
        poll_interval: float,
        max_poll_time: float,
        cancel_grace_period: float,
    ) -> None:
        super().__init__()
        self._units_per_worker = units_per_worker
        self._max_workers = max_workers
        self._work_folder = work_folder
        self._cpus = cpus
        self._memory_gb = memory_gb
        self._gpus = gpus
        self._time_limit = time_limit
        self._slurm_gres = slurm_gres
        self._srun_args = srun_args
        self._poll_interval = poll_interval
        self._max_poll_time = max_poll_time
        self._cancel_grace_period = cancel_grace_period
        self._submitted: list[_SrunProcess] = []
        self._cancelling_processes: set[int] = set()
        self._cancelled_processes: set[int] = set()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._dispatch_started = False
        self._last_launch = 0.0

    @property
    def work_folder(self) -> Path:
        """Return this router's collision-resistant work directory."""
        return self._work_folder

    @property
    def submitted_process_ids(self) -> tuple[int, ...]:
        """Return exact local process IDs owned by this router."""
        with self._lock:
            return tuple(item.process.pid for item in self._submitted)

    def _dispatch(
        self,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> None:
        """Start asynchronous, bounded launch and collection."""
        batches = pack_units(units, self._units_per_worker)
        with self._lock:
            self._dispatch_started = True
        if not os.environ.get("SLURM_JOB_ID"):
            msg = (
                "SlurmIntraRunner requires an active SLURM allocation "
                "(SLURM_JOB_ID is not set)"
            )
            raise RuntimeError(msg)
        self._start_background(lambda: self._run_batches(batches, runtime_env))

    def cancel(self) -> None:
        """Terminate only live ``srun`` processes owned by this router."""
        with self._lock:
            if not self._dispatch_started:
                return
            self._cancelled.set()
            alive = [
                item.process
                for item in self._submitted
                if item.process.poll() is None
                and id(item.process) not in self._cancelled_processes
                and id(item.process) not in self._cancelling_processes
            ]
            self._cancelling_processes.update(id(process) for process in alive)
        terminal = _terminate_processes(alive, self._cancel_grace_period)
        with self._lock:
            self._cancelled_processes.update(terminal)
            self._cancelling_processes.difference_update(id(item) for item in alive)

    def _run_batches(
        self,
        batches: list[list[ExecutionUnit]],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Launch every batch under the cap, then collect in input order."""
        self._work_folder.mkdir(parents=True, exist_ok=False)
        launches: list[_SrunProcess | list[UnitResult]] = []
        for index, batch in enumerate(batches):
            if self._cancelled.is_set():
                launches.append(
                    failure_results_for_units(
                        batch, "srun dispatch cancelled before launch"
                    )
                )
                continue
            try:
                self._wait_for_slot()
                launches.append(self._launch(index, batch, runtime_env))
            except Exception as exc:
                launches.append(failure_results_for_units(batch, exc))

        results: list[UnitResult] = []
        for launch in launches:
            if isinstance(launch, list):
                results.extend(launch)
            else:
                results.extend(self._collect_process(launch))
        return results

    def _wait_for_slot(self) -> None:
        """Wait for capacity while enforcing each active process deadline."""
        while True:
            if self._cancelled.is_set():
                msg = "srun dispatch cancelled"
                raise RuntimeError(msg)
            now = time.monotonic()
            with self._lock:
                active = [
                    item for item in self._submitted if item.process.poll() is None
                ]
            expired = [
                item.process
                for item in active
                if now >= item.launched_at + self._max_poll_time
            ]
            if expired:
                _terminate_processes(expired, self._cancel_grace_period)
                time.sleep(self._poll_interval)
                continue
            if len(active) < self._max_workers:
                return
            time.sleep(self._poll_interval)

    def _launch(
        self,
        index: int,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> _SrunProcess:
        """Serialize one batch and launch its exact process handle."""
        elapsed = time.monotonic() - self._last_launch
        if elapsed < _MIN_LAUNCH_INTERVAL:
            time.sleep(_MIN_LAUNCH_INTERVAL - elapsed)

        folder = self._work_folder / f"step_{index:05d}"
        folder.mkdir(parents=True)
        callable_path = folder / "job.pkl"
        call = partial(execute_unit_batch, units, runtime_env)
        with callable_path.open("wb") as stream:
            cloudpickle.dump(call, stream)

        stdout_path = folder / "stdout.log"
        stderr_path = folder / "stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                self._build_command(folder),
                stdout=stdout,
                stderr=stderr,
            )
        submitted = _SrunProcess(
            process=process,
            units=units,
            folder=folder,
            launched_at=time.monotonic(),
        )
        with self._lock:
            self._submitted.append(submitted)
        self._last_launch = submitted.launched_at
        if self._cancelled.is_set():
            self.cancel()
        return submitted

    def _build_command(self, folder: Path) -> list[str]:
        """Map portable resources and explicit srun arguments to a command."""
        command = ["srun", "--exact", "--mpi=none", "-n1"]
        gres = self._slurm_gres or (f"gpu:{self._gpus}" if self._gpus else None)
        if gres:
            command.extend(["--gres", gres])
        command.extend(["--mem", f"{self._memory_gb}G"])
        command.extend(["--cpus-per-task", str(self._cpus)])
        command.extend(["--time", self._time_limit])
        command.extend(self._srun_args)
        command.extend(
            [sys.executable, "-u", "-m", "artisan_submitit.srun_worker", str(folder)]
        )
        return command

    def _collect_process(self, submitted: _SrunProcess) -> list[UnitResult]:
        """Collect one result envelope and attach the step's logs."""
        process = submitted.process
        deadline = submitted.launched_at + self._max_poll_time
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(self._poll_interval)
        if process.poll() is None:
            _terminate_processes([process], self._cancel_grace_period)
            raw_results: object = failure_results_for_units(
                submitted.units,
                f"srun process {process.pid} exceeded {self._max_poll_time:.0f}s",
            )
        elif process.returncode != 0:
            raw_results = failure_results_for_units(
                submitted.units,
                f"srun process {process.pid} exited with code {process.returncode}",
            )
        else:
            raw_results = self._read_result(submitted)
        worker_log = read_job_logs(
            submitted.folder / "stdout.log",
            submitted.folder / "stderr.log",
            retries=5,
            delay=_RESULT_READ_DELAY,
        )
        return _attach_worker_log(
            validate_batch_results(submitted.units, raw_results),
            worker_log,
        )

    def _read_result(self, submitted: _SrunProcess) -> object:
        """Read the atomic worker envelope or return per-unit failures."""
        result_path = submitted.folder / "result.pkl"
        envelope: object | None = None
        read_error: Exception | None = None
        for attempt in range(_RESULT_READ_RETRIES):
            try:
                with result_path.open("rb") as stream:
                    envelope = pickle.load(stream)
                break
            except (FileNotFoundError, EOFError, pickle.UnpicklingError) as exc:
                read_error = exc
                with contextlib.suppress(OSError):
                    list(result_path.parent.iterdir())
            except OSError as exc:
                read_error = exc
            except Exception as exc:
                return failure_results_for_units(submitted.units, exc)
            if attempt < _RESULT_READ_RETRIES - 1:
                time.sleep(_RESULT_READ_DELAY)
        if envelope is None:
            detail = (
                f": {read_error}"
                if read_error and not isinstance(read_error, FileNotFoundError)
                else ""
            )
            return failure_results_for_units(
                submitted.units,
                f"srun process {submitted.process.pid} produced no result envelope{detail}",
            )
        if not isinstance(envelope, dict) or envelope.get("status") not in {
            "ok",
            "error",
        }:
            return failure_results_for_units(
                submitted.units, "srun worker returned a malformed result envelope"
            )
        if envelope["status"] == "error":
            error = (
                f"{envelope.get('type', 'WorkerError')}: {envelope.get('error', '')}"
            )
            return failure_results_for_units(submitted.units, error)
        return envelope.get("result")


class SlurmIntraRunner(RunnerBase):
    """Run Artisan units via ``srun`` inside an existing allocation."""

    name = "slurm_intra"
    worker_traits = WorkerTraits(
        worker_id_env_var="SLURM_STEP_ID",
        shared_filesystem=True,
    )
    orchestrator_traits = OrchestratorTraits(
        shared_filesystem=True,
        staging_verification_timeout=60.0,
    )

    def __init__(
        self,
        *,
        log_folder: str = "slurm_logs",
        srun_launch_concurrency: int | None = None,
        poll_interval: float = 0.5,
        max_poll_time: float | None = None,
        cancel_grace_period: float = 10.0,
    ) -> None:
        if poll_interval <= 0 or cancel_grace_period < 0:
            msg = "poll_interval must be positive and cancel_grace_period non-negative"
            raise ValueError(msg)
        if srun_launch_concurrency is not None and (
            type(srun_launch_concurrency) is not int or srun_launch_concurrency < 1
        ):
            msg = "srun_launch_concurrency must be a positive integer"
            raise ValueError(msg)
        if max_poll_time is not None and max_poll_time <= 0:
            msg = "max_poll_time must be positive when provided"
            raise ValueError(msg)
        self.log_folder = log_folder
        self.srun_launch_concurrency = srun_launch_concurrency
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.cancel_grace_period = cancel_grace_period

    def create_lifecycle_router(
        self,
        runner_resources: RunnerResources,
        batch_strategy: BatchStrategy,
        step_number: int,
        job_name: str,
        log_folder: str | None = None,
        staging_root: str | None = None,
    ) -> LifecycleRouter:
        """Create one isolated srun router for an Artisan step."""
        del staging_root
        extras = dict(runner_resources.extra)
        unsupported = extras.keys() - _SUPPORTED_EXTRAS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            msg = f"Unsupported SlurmIntraRunner resource extras: {names}"
            raise ValueError(msg)
        srun_args = extras.get("srun_args", [])
        if not isinstance(srun_args, list) or not all(
            isinstance(item, str) for item in srun_args
        ):
            msg = "RunnerResources.extra['srun_args'] must be a list of strings"
            raise TypeError(msg)
        reject_managed_scheduler_arguments(
            srun_args, "RunnerResources.extra['srun_args']"
        )
        slurm_gres = extras.get("slurm_gres")
        if slurm_gres is not None and not isinstance(slurm_gres, str):
            msg = "RunnerResources.extra['slurm_gres'] must be a string"
            raise TypeError(msg)

        max_workers = _resolve_srun_concurrency(
            batch_strategy.max_workers,
            self.srun_launch_concurrency,
        )
        timeout_minutes = parse_time_to_minutes(runner_resources.time_limit)
        if timeout_minutes < 1:
            msg = "RunnerResources.time_limit must be at least one second"
            raise ValueError(msg)
        root = Path(log_folder or self.log_folder) / "srun"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_name)
        work_folder = root / f"s{step_number}_{safe_name}_{uuid.uuid4().hex}"
        return SlurmIntraLifecycleRouter(
            units_per_worker=batch_strategy.units_per_worker,
            max_workers=max_workers,
            work_folder=work_folder,
            cpus=runner_resources.cpus,
            memory_gb=runner_resources.memory_gb,
            gpus=runner_resources.gpus,
            time_limit=runner_resources.time_limit,
            slurm_gres=slurm_gres,
            srun_args=srun_args,
            poll_interval=self.poll_interval,
            max_poll_time=(
                self.max_poll_time
                if self.max_poll_time is not None
                else timeout_minutes * 60 * 2
            ),
            cancel_grace_period=self.cancel_grace_period,
        )

    def validate_operation(self, operation: Any) -> None:
        """Warn early when the selected runner has no active allocation."""
        if not os.environ.get("SLURM_JOB_ID"):
            warnings.warn(
                f"Step runner 'slurm_intra' selected for {operation.name!r} but "
                "SLURM_JOB_ID is not set",
                stacklevel=2,
            )


def _resolve_srun_concurrency(
    max_workers: int | None,
    runner_concurrency: int | None,
) -> int:
    """Resolve srun concurrency from Artisan or provider configuration."""
    if max_workers is not None and max_workers < 1:
        msg = f"BatchStrategy.max_workers must be >= 1, got {max_workers}"
        raise ValueError(msg)
    if max_workers is not None and runner_concurrency not in {None, max_workers}:
        msg = (
            f"BatchStrategy.max_workers={max_workers} conflicts with "
            f"SlurmIntraRunner.srun_launch_concurrency={runner_concurrency}"
        )
        raise ValueError(msg)
    return max_workers or runner_concurrency or 128


def _terminate_processes(
    processes: list[subprocess.Popen[bytes]],
    grace_period: float,
) -> set[int]:
    """Terminate exact process handles, escalating to kill after one deadline."""
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        with contextlib.suppress(OSError):
            process.terminate()
    deadline = time.monotonic() + grace_period
    for process in alive:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=5)
    return {id(process) for process in processes if process.poll() is not None}
