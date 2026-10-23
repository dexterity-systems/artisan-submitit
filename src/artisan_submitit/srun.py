"""Intra-allocation Artisan runner using exact ``srun`` processes."""

from __future__ import annotations

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
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import cloudpickle
from artisan.orchestration.runner_api import (
    BatchStrategy,
    CancellationAcknowledgement,
    CancellationStatus,
    ExecutionUnit,
    LifecycleRouter,
    OrchestratorTraits,
    RunnerBase,
    RunnerResources,
    RuntimeEnvironment,
    UnitResult,
    WorkerTraits,
    failure_results_for_units,
    pack_units,
    validate_batch_results,
)

from artisan_submitit.cancellation import (
    CANCELLATION_TIMEOUT,
    COOPERATION_TIMEOUT,
    PROVIDER_CONFIRMATION_TIMEOUT,
    CancellationState,
    cancel_jobs,
    query_job_states,
)
from artisan_submitit.slurm import _attach_worker_log
from artisan_submitit.utils import (
    parse_time_to_minutes,
    read_job_logs,
    reject_managed_scheduler_arguments,
)
from artisan_submitit.worker import (
    WorkerOutcome,
    read_worker_identity,
    read_worker_outcome,
    run_worker_batch,
)

logger = logging.getLogger(__name__)

_MIN_LAUNCH_INTERVAL = 0.05
_RESULT_READ_RETRIES = 5
_RESULT_READ_DELAY = 0.1
_SUPPORTED_EXTRAS = frozenset({"slurm_gres", "srun_args"})


@dataclass
class _SrunProcess:
    """One exact launcher, its batch, and observed cleanup evidence."""

    process: subprocess.Popen[bytes]
    units: list[ExecutionUnit]
    folder: Path
    launched_at: float
    outcome: WorkerOutcome | None = None
    read_error: str | None = None
    timeout_error: str | None = None
    requested_at: float | None = None
    request_written: bool = False
    terminated_at: float | None = None
    kill_sent: bool = False
    terminal_at: float | None = None
    step_id: str | None = None
    scheduler_cancelled: bool = False
    scheduler_terminal: bool = False


class SlurmIntraLifecycleRouter(LifecycleRouter):
    """Launch bounded synchronous steps and retain native ownership evidence."""

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
        self._lock = threading.Lock()
        self._cancellation = CancellationState()
        self._cancelled = self._cancellation.event
        self._dispatch_started = False
        self._allocation_id: str | None = None
        self._last_launch = 0.0
        self._next_scheduler_poll = 0.0
        self._owner_outcomes: list[CancellationAcknowledgement | None] = []
        self._prevented = False

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
        self._allocation_id = os.environ.get("SLURM_JOB_ID")
        if not self._allocation_id:
            msg = (
                "SlurmIntraRunner requires an active SLURM allocation "
                "(SLURM_JOB_ID is not set)"
            )
            raise RuntimeError(msg)
        with self._lock:
            self._dispatch_started = True
        self._start_background(lambda: self._run_batches(batches, runtime_env))

    def cancel(self) -> CancellationAcknowledgement:
        """Close admission promptly; background collection proves cleanup."""
        with self._lock:
            if not self._dispatch_started:
                return CancellationAcknowledgement(
                    CancellationStatus.REJECTED, "srun work has not been dispatched"
                )
        if self.is_done():
            if (
                self._thread is not None
                and self._thread is not threading.current_thread()
            ):
                self._thread.join()
            return self._cancellation.acknowledgement or self._cancellation.finish(
                [], [], False, str(self._error) if self._error else None
            )
        acknowledgement = self._cancellation.request()
        if acknowledgement.status == CancellationStatus.UNKNOWN:
            return acknowledgement
        return CancellationAcknowledgement(
            CancellationStatus.REQUESTED, "Waiting for srun worker and launcher closure"
        )

    def _run_batches(
        self,
        batches: list[list[ExecutionUnit]],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Settle evidence before the lifecycle base marks collection complete."""
        error = None
        try:
            results = self._execute_batches(batches, runtime_env)
        except Exception as exc:
            results = failure_results_for_units(
                [unit for batch in batches for unit in batch], exc
            )
            if self._submitted:
                error = f"srun collection failed: {exc}"
                self._cancellation.request()
                for item in self._submitted:
                    self._request_stop(item)
        acknowledgement = self._cancellation.finish(
            self._owner_outcomes, results, self._prevented, error
        )
        if acknowledgement.status == CancellationStatus.UNKNOWN:
            self._cancellation_acknowledgement = acknowledgement
        return results

    def _execute_batches(
        self,
        batches: list[list[ExecutionUnit]],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Launch under the cap, then collect without delaying sibling cleanup."""
        self._work_folder.mkdir(parents=True, exist_ok=False)
        launches: list[_SrunProcess | list[UnitResult]] = []
        for index, batch in enumerate(batches):
            try:
                self._wait_for_slot()
                launches.append(self._launch(index, batch, runtime_env))
            except Exception as exc:
                self._prevented |= self._cancelled.is_set()
                launches.append(failure_results_for_units(batch, exc))
        results: list[UnitResult] = []
        for launch in launches:
            results.extend(
                launch if isinstance(launch, list) else self._collect_process(launch)
            )
        return results

    def _wait_for_slot(self) -> None:
        """Poll all owned steps so capacity waits cannot starve cancellation."""
        while True:
            self._advance_processes()
            if self._cancelled.is_set():
                msg = "srun dispatch cancelled before launch"
                raise RuntimeError(msg)
            active = [item for item in self._submitted if item.process.poll() is None]
            if any(
                item.requested_at is not None and self._remaining(item) <= 0
                for item in active
            ):
                self._cancellation.request()
                msg = "Timed-out srun launcher could not be reaped; further admission stopped"
                raise RuntimeError(msg)
            if len(active) < self._max_workers:
                return
            self._sleep()

    def _launch(
        self,
        index: int,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> _SrunProcess:
        """Serialize a wrapper and retain the exact launcher even during races."""
        elapsed = time.monotonic() - self._last_launch
        if elapsed < _MIN_LAUNCH_INTERVAL:
            time.sleep(_MIN_LAUNCH_INTERVAL - elapsed)
        folder = self._work_folder / f"step_{index:05d}"
        folder.mkdir(parents=True)
        with (folder / "job.pkl").open("wb") as stream:
            cloudpickle.dump(
                partial(run_worker_batch, units, runtime_env, folder), stream
            )
        if self._cancelled.is_set():
            msg = "srun dispatch cancelled before launch"
            raise RuntimeError(msg)
        with (
            (folder / "stdout.log").open("wb") as stdout,
            (folder / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                self._build_command(folder), stdout=stdout, stderr=stderr
            )
        submitted = _SrunProcess(process, units, folder, time.monotonic())
        with self._lock:
            self._submitted.append(submitted)
        self._last_launch = submitted.launched_at
        if self._cancelled.is_set():
            self._request_stop(submitted)
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

    def _advance_processes(self) -> None:
        """Publish every request before polling outcomes or escalating launchers."""
        now = time.monotonic()
        for item in self._submitted:
            if item.process.poll() is not None:
                item.terminal_at = item.terminal_at or now
            elif now >= item.launched_at + self._max_poll_time:
                item.timeout_error = f"srun process {item.process.pid} exceeded {self._max_poll_time:.0f}s"
            if self._cancelled.is_set() or item.timeout_error:
                self._request_stop(item)
        for item in self._submitted:
            self._observe_outcome(item)
        fallback = [
            item
            for item in self._submitted
            if item.requested_at is not None
            and not self._cooperating(item)
            and (item.process.poll() is None or item.terminated_at is not None)
        ]
        self._fallback_cleanup(fallback, now)

    def _request_stop(self, item: _SrunProcess) -> None:
        """Persist cancellation so a queued or late worker cannot miss it."""
        if item.requested_at is None:
            item.requested_at = time.monotonic()
        if not item.request_written:
            try:
                (item.folder / "cancel").touch(exist_ok=True)
            except OSError as exc:
                item.read_error = f"Could not publish worker cancellation: {exc}"
            else:
                item.request_written = True

    def _observe_outcome(self, item: _SrunProcess) -> None:
        """Read only the provider's atomic outcome, never transport wait APIs."""
        if item.outcome is not None:
            return
        try:
            outcome = read_worker_outcome(item.folder)
            if outcome is not None and len(outcome.results) != len(item.units):
                msg = "Worker outcome cardinality does not match submitted units"
                raise ValueError(msg)
            item.outcome = outcome
            if outcome is not None:
                item.read_error = None
        except (OSError, ValueError) as exc:
            item.read_error = str(exc)

    def _cooperating(self, item: _SrunProcess) -> bool:
        """A later outer cancellation cannot restart an existing cleanup grace."""
        return (
            item.requested_at is not None
            and time.monotonic() - item.requested_at < COOPERATION_TIMEOUT
            and (not self._cancelled.is_set() or self._cancellation.cooperating)
        )

    def _remaining(self, item: _SrunProcess) -> float:
        """Bound all fallback calls by one request deadline."""
        remaining = (
            CANCELLATION_TIMEOUT
            if item.requested_at is None
            else max(0.0, CANCELLATION_TIMEOUT - (time.monotonic() - item.requested_at))
        )
        return (
            min(remaining, self._cancellation.remaining)
            if self._cancelled.is_set()
            else remaining
        )

    def _fallback_cleanup(self, items: list[_SrunProcess], now: float) -> None:
        """Cancel exact validated steps and signal launchers without serial waits."""
        if not items:
            return
        for item in items:
            item.step_id = item.step_id or self._step_identity(item)
            self._signal_launcher(item, now)
        remaining = min(self._remaining(item) for item in items)
        if now < self._next_scheduler_poll or remaining <= 0:
            return
        self._next_scheduler_poll = now + 1.0
        unsent = [
            item for item in items if item.step_id and not item.scheduler_cancelled
        ]
        if (
            unsent
            and cancel_jobs(
                [item.step_id for item in unsent if item.step_id],
                timeout=min(1.0, remaining),
            )
            is None
        ):
            for item in unsent:
                item.scheduler_cancelled = True
        remaining = min(self._remaining(item) for item in items)
        if remaining > 0:
            states = query_job_states(
                [item.step_id for item in items if item.step_id],
                timeout=min(1.0, remaining),
            )
            for item in items:
                state = states.get(item.step_id or "")
                item.scheduler_terminal |= state is not None and state.terminal

    def _step_identity(self, item: _SrunProcess) -> str | None:
        """Accept only a numeric step inside the allocation captured at dispatch."""
        try:
            identity = read_worker_identity(item.folder)
        except (OSError, ValueError) as exc:
            item.read_error = f"Invalid worker identity: {exc}"
            return None
        if identity is None:
            return None
        job_id, step_id = identity.get("job_id"), identity.get("step_id")
        if (
            job_id != self._allocation_id
            or not isinstance(step_id, str)
            or not re.fullmatch(r"[0-9]+", step_id)
        ):
            item.read_error = (
                "Worker identity does not identify an owned numeric allocation step"
            )
            return None
        return f"{job_id}.{step_id}"

    def _signal_launcher(self, item: _SrunProcess, now: float) -> None:
        """Escalate exact Popen handles under a shared deadline without waiting."""
        if item.process.poll() is not None:
            return
        if item.terminated_at is None:
            item.terminated_at = now
            try:
                item.process.terminate()
            except OSError:
                if self._cancel_grace_period > 0 and self._remaining(item) > 0:
                    item.terminated_at = None
                    return
        if not item.kill_sent and (
            now - item.terminated_at >= self._cancel_grace_period
            or self._remaining(item) <= 0
        ):
            try:
                item.process.kill()
            except OSError:
                pass
            else:
                item.kill_sent = True

    def _collect_process(self, item: _SrunProcess) -> list[UnitResult]:
        """Wait for synchronous transport and atomic ownership evidence together."""
        while True:
            self._advance_processes()
            terminal = item.process.poll() is not None
            if (
                terminal
                and (
                    item.outcome is not None
                    or time.monotonic()
                    >= (item.terminal_at or 0)
                    + _RESULT_READ_RETRIES * _RESULT_READ_DELAY
                )
                and (
                    item.terminated_at is None
                    or item.scheduler_terminal
                    or self._remaining(item) <= 0
                )
            ):
                break
            if item.requested_at is not None and self._remaining(item) <= 0:
                break
            self._sleep()
        results, owner = self._process_results(item)
        self._owner_outcomes.append(owner)
        worker_log = read_job_logs(
            item.folder / "stdout.log",
            item.folder / "stderr.log",
            retries=1 if self._cancelled.is_set() or item.requested_at else 5,
            delay=_RESULT_READ_DELAY,
        )
        return _attach_worker_log(results, worker_log)

    def _process_results(
        self, item: _SrunProcess
    ) -> tuple[list[UnitResult], CancellationAcknowledgement | None]:
        """Separate ordinary operation failure from missing ownership proof."""
        outcome = item.outcome
        error = item.read_error or self._bootstrap_error(item)
        if item.process.poll() is None:
            error = (
                f"srun process {item.process.pid} remained alive after cleanup deadline"
            )
        elif item.terminated_at is not None and not item.scheduler_terminal:
            error = f"Could not confirm scheduler step termination for srun process {item.process.pid}"
        elif outcome is None:
            error = (
                error or f"srun process {item.process.pid} produced no worker outcome"
            )
        elif item.process.returncode != 0:
            error = f"srun process {item.process.pid} exited with code {item.process.returncode}"
        if error:
            acknowledgement = CancellationAcknowledgement(
                CancellationStatus.UNKNOWN, error
            )
            results = (
                outcome.results
                if outcome is not None
                else failure_results_for_units(item.units, error)
            )
            return [
                replace(
                    result,
                    success=False,
                    error=result.error or error,
                    cancellation_acknowledgement=(
                        result.cancellation_acknowledgement
                        if result.cancellation_acknowledgement is not None
                        and result.cancellation_acknowledgement.status
                        == CancellationStatus.UNKNOWN
                        else acknowledgement
                    ),
                )
                for result in results
            ], acknowledgement
        assert outcome is not None
        uncertain = outcome.cancellation_acknowledgement
        if (
            item.timeout_error
            and not self._cancelled.is_set()
            and not (
                uncertain is not None and uncertain.status == CancellationStatus.UNKNOWN
            )
            and not any(
                result.cancellation_acknowledgement is not None
                and result.cancellation_acknowledgement.status
                in {CancellationStatus.UNKNOWN, CancellationStatus.REQUESTED}
                for result in outcome.results
            )
        ):
            return failure_results_for_units(item.units, item.timeout_error), None
        return validate_batch_results(
            item.units, outcome.results
        ), outcome.cancellation_acknowledgement

    def _bootstrap_error(self, item: _SrunProcess) -> str | None:
        """Retain outer callable-bootstrap diagnostics without trusting its result."""
        if item.outcome is not None:
            return None
        try:
            with (item.folder / "result.pkl").open("rb") as stream:
                envelope = pickle.load(stream)
        except Exception:
            return None
        if isinstance(envelope, dict) and envelope.get("status") == "error":
            return f"{envelope.get('type', 'WorkerError')}: {envelope.get('error', '')}"
        return None

    def _sleep(self) -> None:
        """Wake for cancellation and retain each launcher's cleanup deadline."""
        if self._cancelled.is_set():
            time.sleep(min(self._poll_interval, 0.1, self._cancellation.remaining))
        else:
            delay = self._poll_interval
            for item in self._submitted:
                if item.requested_at is not None:
                    remaining = self._remaining(item)
                    if remaining > 0:
                        delay = min(delay, remaining, 0.1)
                    continue
                if item.process.poll() is not None:
                    continue
                delay = min(
                    delay,
                    max(0.0, item.launched_at + self._max_poll_time - time.monotonic()),
                )
            self._cancelled.wait(timeout=delay)


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
        cancellation_confirmation_timeout=PROVIDER_CONFIRMATION_TIMEOUT,
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
        _reject_unowned_srun_arguments(srun_args)
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


def _reject_unowned_srun_arguments(arguments: list[str]) -> None:
    """Require synchronous launch inside the dispatch allocation."""
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument[2:].split("=", maxsplit=1)[0]
        if option and any(name.startswith(option) for name in ("async", "jobid")):
            msg = f"srun_args cannot override Artisan-managed synchronous allocation ownership: {argument}"
            raise ValueError(msg)
