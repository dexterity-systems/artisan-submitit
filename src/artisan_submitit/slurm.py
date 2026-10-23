"""Submitit-backed SLURM runner for Artisan execution units."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import submitit
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
    PROVIDER_CONFIRMATION_TIMEOUT,
    CancellationState,
    SchedulerObservation,
    cancel_jobs,
    query_job_states,
)
from artisan_submitit.utils import (
    chunked,
    get_cluster_max_array_size,
    is_managed_scheduler_option,
    parse_time_to_minutes,
    read_job_logs,
    reject_managed_scheduler_arguments,
)
from artisan_submitit.worker import WorkerOutcome, read_worker_outcome, run_worker_batch

logger = logging.getLogger(__name__)

_PORTABLE_PARAMETER_NAMES = frozenset(
    {
        "cpus_per_gpu",
        "cpus_per_task",
        "gpus_per_node",
        "gpus_per_task",
        "mem",
        "mem_gb",
        "mem_per_cpu",
        "mem_per_gpu",
        "num_gpus",
        "slurm_cpus_per_gpu",
        "slurm_cpus_per_task",
        "slurm_gpus_per_node",
        "slurm_gpus_per_task",
        "slurm_mem",
        "slurm_mem_gb",
        "slurm_mem_per_cpu",
        "slurm_mem_per_gpu",
        "slurm_num_gpus",
        "slurm_time",
        "slurm_timeout_min",
        "time",
        "timeout_min",
    }
)
_TOPOLOGY_PARAMETER_NAMES = frozenset(
    {
        "nodes",
        "ntasks_per_node",
        "slurm_nodes",
        "slurm_ntasks_per_node",
        "slurm_tasks_per_node",
        "tasks_per_node",
    }
)
_REMOVED_PARAMETER_NAMES = {
    "additional_parameters": "slurm_additional_parameters",
    "array_parallelism": "slurm_array_parallelism",
    "gres": "slurm_gres",
    "partition": "slurm_partition",
    "srun_args": "slurm_srun_args",
    "time_limit": "RunnerResources.time_limit",
}


@dataclass
class _SubmittedBatch:
    """One exact Submitit job handle and the units it owns."""

    job: submitit.Job[Any]
    units: list[ExecutionUnit]
    control_folder: Path
    submitted_after: datetime
    outcome: WorkerOutcome | None = None
    observation: SchedulerObservation | None = None
    results: list[UnitResult] | None = None
    terminal_since: float | None = None
    request_written: bool = False
    error: str | None = None
    outcome_error: str | None = None


class SlurmLifecycleRouter(LifecycleRouter):
    """Submit ordered unit batches as SLURM arrays through Submitit."""

    def __init__(
        self,
        executor: submitit.AutoExecutor,
        units_per_worker: int,
        max_array_size: int,
        control_folder: Path,
    ) -> None:
        super().__init__()
        self._executor = executor
        self._units_per_worker = units_per_worker
        self._max_array_size = max_array_size
        self._control_folder = control_folder
        self._submitted: list[_SubmittedBatch] = []
        self._lock = threading.Lock()
        self._cancellation = CancellationState()
        self._dispatch_started = False
        self._owners: list[CancellationAcknowledgement | None] = []
        self._prevented = False
        self._closure_error: str | None = None

    @property
    def submitted_job_ids(self) -> tuple[str, ...]:
        """Return the exact scheduler handles owned by this router."""
        with self._lock:
            return tuple(str(item.job.job_id) for item in self._submitted)

    def _dispatch(
        self,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> None:
        """Start asynchronous scheduler submission and collection."""
        with self._lock:
            self._dispatch_started = True
        self._start_background(lambda: self._run_waves(units, runtime_env))

    def cancel(self) -> CancellationAcknowledgement:
        """Request background cleanup without transport I/O or a live-thread join."""
        with self._lock:
            if not self._dispatch_started:
                return CancellationAcknowledgement(
                    CancellationStatus.REJECTED, "No SLURM work was dispatched"
                )
        evidence = self._cancellation.request()
        if self.is_done():
            if self._thread is not None:
                self._thread.join()
            return self._cancellation.current()
        if evidence.status == CancellationStatus.UNKNOWN:
            return evidence
        return CancellationAcknowledgement(
            CancellationStatus.REQUESTED, "Waiting for SLURM ownership closure"
        )

    def _run_waves(
        self,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Submit and collect one scheduler-sized wave at a time."""
        results: list[UnitResult] = []
        try:
            results = self._execute_waves(units, runtime_env)
        except Exception as exc:
            results = failure_results_for_units(units, exc)
            if self._submitted:
                self._closure_error = f"SLURM collection failed: {exc}"
        finally:
            evidence = self._cancellation.finish(
                self._owners, results, self._prevented, self._closure_error
            )
            if (
                self._cancellation.event.is_set()
                or evidence.status == CancellationStatus.UNKNOWN
            ):
                self._cancellation_acknowledgement = evidence
        return results

    def _execute_waves(
        self,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Retain global array concurrency and aligned results for skipped waves."""
        batches = pack_units(units, self._units_per_worker)
        waves = chunked(batches, self._max_array_size)
        results: list[UnitResult] = []
        for index, wave in enumerate(waves):
            if self._cancellation.event.is_set():
                self._prevented = True
                results.extend(
                    _failure_results_for_waves(
                        waves[index:], "SLURM dispatch cancelled before submission"
                    )
                )
                break
            try:
                submitted = self._submit_array(wave, runtime_env)
            except Exception as exc:
                results.extend(_failure_results_for_waves(waves[index:], exc))
                break
            results.extend(self._collect_submitted(submitted))
            submitted_count = sum(len(item.units) for item in submitted)
            skipped = [unit for batch in wave for unit in batch][submitted_count:]
            if skipped:
                self._prevented = True
                results.extend(
                    failure_results_for_units(
                        skipped, "SLURM dispatch cancelled before submission"
                    )
                )
        return results

    def _submit_array(
        self,
        batches: list[list[ExecutionUnit]],
        runtime_env: RuntimeEnvironment,
    ) -> list[_SubmittedBatch]:
        """Submit one scheduler-sized array and retain every exact handle."""
        pending: list[_SubmittedBatch] = []
        submitted_after = datetime.now(UTC).replace(microsecond=0)
        try:
            with self._executor.batch():
                for batch in batches:
                    if self._cancellation.event.is_set():
                        break
                    control_folder = self._control_folder / uuid.uuid4().hex
                    control_folder.mkdir(parents=True)
                    job = self._executor.submit(
                        run_worker_batch, batch, runtime_env, control_folder
                    )
                    pending.append(
                        _SubmittedBatch(job, batch, control_folder, submitted_after)
                    )
        except Exception as exc:
            self._closure_error = f"SLURM submission ownership is unresolved: {exc}"
            self._publish_requests(pending)
            # Default batch contexts forbid implicit submission on attribute reads.
            promoted = [item for item in pending if hasattr(item.job, "job_id")]
            with self._lock:
                self._submitted.extend(promoted)
            if promoted:
                cancel_jobs([str(item.job.job_id) for item in promoted])
            raise
        with self._lock:
            self._submitted.extend(pending)
        if self._cancellation.event.is_set():
            self._publish_requests(pending)
            # Promotion may return after the caller already received UNKNOWN.
            if self._cancellation.expired:
                cancel_jobs([str(item.job.job_id) for item in pending])
        return pending

    def _collect_submitted(
        self,
        submitted: list[_SubmittedBatch],
    ) -> list[UnitResult]:
        """Progress every owner together, then assemble results in input order."""
        if not submitted:
            return []
        next_query = 0.0
        while any(item.results is None for item in submitted):
            pending = [item for item in submitted if item.results is None]
            if self._cancellation.event.is_set():
                self._publish_requests(pending)
            if self._cancellation.expired:
                for item in pending:
                    self._fail_unresolved(
                        item, "SLURM cancellation closure is unresolved"
                    )
                break
            if time.monotonic() >= next_query:
                self._observe_jobs(pending)
                next_query = time.monotonic() + 1.0
            for item in pending:
                self._read_outcome(item)
                self._settle_job(item)
            time.sleep(0.05)
        with ThreadPoolExecutor(max_workers=min(32, len(submitted))) as pool:
            retries = 1 if self._cancellation.event.is_set() else 5
            logs = pool.map(
                lambda item: _read_submitit_job_log(item.job, retries), submitted
            )
            return [
                result
                for item, worker_log in zip(submitted, logs, strict=True)
                for result in _attach_worker_log(item.results or [], worker_log)
            ]

    def _publish_requests(self, submitted: list[_SubmittedBatch]) -> None:
        """Notify all admitted batches before waiting on any one outcome."""
        for item in submitted:
            if not item.request_written:
                try:
                    (item.control_folder / "cancel").touch(exist_ok=True)
                    item.request_written = True
                except OSError as exc:
                    item.error = f"Cannot publish worker cancellation: {exc}"

    def _observe_jobs(self, pending: list[_SubmittedBatch]) -> None:
        """Poll accounting and retry exact cancellation within one shared budget."""
        timeout = max(0.001, min(1.0, self._cancellation.remaining))
        states = query_job_states(
            [str(item.job.job_id) for item in pending],
            submitted_after=min(item.submitted_after for item in pending),
            timeout=timeout,
        )
        for item in pending:
            item.observation = states.get(str(item.job.job_id))
        if self._cancellation.event.is_set() and not self._cancellation.cooperating:
            ids = [
                str(item.job.job_id)
                for item in pending
                if item.observation is None or not item.observation.terminal
            ]
            timeout = max(0.001, min(1.0, self._cancellation.remaining))
            error = cancel_jobs(ids, timeout=timeout) if ids else None
            if error:
                for item in pending:
                    item.error = error

    def _read_outcome(self, item: _SubmittedBatch) -> None:
        """Read only the provider's atomic protocol, never Submitit's watcher."""
        if item.outcome is not None or item.outcome_error is not None:
            return
        try:
            item.outcome = read_worker_outcome(item.control_folder)
        except (OSError, ValueError) as exc:
            item.outcome_error = f"Invalid worker outcome: {exc}"

    def _settle_job(self, item: _SubmittedBatch) -> None:
        """Require transport closure as well as native or queued non-start proof."""
        state = item.observation
        if state is None or not state.terminal:
            return
        if item.outcome_error is not None:
            self._fail_unresolved(item, item.outcome_error)
            return
        if item.outcome is not None:
            if len(item.outcome.results) != len(item.units):
                self._fail_unresolved(item, "Invalid worker batch result cardinality")
                return
            item.results = validate_batch_results(item.units, item.outcome.results)
            self._owners.append(item.outcome.cancellation_acknowledgement)
            return
        if self._cancellation.event.is_set() and state.never_started:
            self._prevented = True
            item.results = failure_results_for_units(
                item.units, "SLURM job cancelled before worker startup"
            )
            return
        if item.terminal_since is None:
            item.terminal_since = time.monotonic()
        if time.monotonic() - item.terminal_since >= 1.0:
            self._fail_unresolved(
                item,
                item.error
                or f"SLURM job {item.job.job_id} ended as {state.state} without worker ownership evidence",
            )

    def _fail_unresolved(self, item: _SubmittedBatch, error: str) -> None:
        """Keep missing worker proof visible to both core and direct callers."""
        acknowledgement = CancellationAcknowledgement(CancellationStatus.UNKNOWN, error)
        self._owners.append(acknowledgement)
        results = (
            item.outcome.results
            if item.outcome is not None and len(item.outcome.results) == len(item.units)
            else failure_results_for_units(item.units, error)
        )
        item.results = [
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
        ]


class SlurmRunner(RunnerBase):
    """Run Artisan execution units as Submitit-managed SLURM arrays.

    Constructor values are provider defaults. Portable per-operation resources
    still come from Artisan's ``RunnerResources``. Extra Submitit parameters use
    Submitit's native ``slurm_*`` vocabulary.
    """

    name = "slurm"
    worker_traits = WorkerTraits(
        worker_id_env_var="SLURM_ARRAY_TASK_ID",
        shared_filesystem=True,
    )
    orchestrator_traits = OrchestratorTraits(
        shared_filesystem=True,
        staging_verification_timeout=60.0,
        cancellation_confirmation_timeout=PROVIDER_CONFIRMATION_TIMEOUT,
    )

    def __init__(
        self,
        slurm_partition: str = "cpu",
        *,
        log_folder: str = "slurm_logs",
        max_array_size: int | None = None,
        slurm_array_parallelism: int | None = None,
        **submitit_parameters: Any,
    ) -> None:
        self.slurm_partition = slurm_partition
        self.log_folder = log_folder
        self.max_array_size = max_array_size
        self.slurm_array_parallelism = slurm_array_parallelism
        self.submitit_parameters = submitit_parameters
        self._cached_max_array_size: int | None = None
        _validate_provider_parameters(submitit_parameters)

    def create_lifecycle_router(
        self,
        runner_resources: RunnerResources,
        batch_strategy: BatchStrategy,
        step_number: int,
        job_name: str,
        log_folder: str | None = None,
        staging_root: str | None = None,
    ) -> LifecycleRouter:
        """Create and configure the router for one Artisan step."""
        del staging_root
        parameters = self._executor_parameters(
            runner_resources,
            batch_strategy,
            step_number,
            job_name,
        )
        folder = Path(log_folder or self.log_folder) / uuid.uuid4().hex / "%j"
        executor = submitit.AutoExecutor(folder=folder, cluster="slurm")
        executor.update_parameters(**parameters)
        _validate_effective_executor(executor, parameters)
        return SlurmLifecycleRouter(
            executor=executor,
            units_per_worker=batch_strategy.units_per_worker,
            max_array_size=self._get_max_array_size(),
            control_folder=folder.parent / "control",
        )

    def _executor_parameters(
        self,
        resources: RunnerResources,
        batch_strategy: BatchStrategy,
        step_number: int,
        job_name: str,
    ) -> dict[str, Any]:
        """Translate portable Artisan resources to native Submitit names."""
        extras = {**self.submitit_parameters, **resources.extra}
        _validate_provider_parameters(extras)
        parallelism = _resolve_array_parallelism(
            batch_strategy.max_workers,
            self.slurm_array_parallelism,
            extras.pop("slurm_array_parallelism", None),
        )
        partition = extras.pop("slurm_partition", self.slurm_partition)
        parameters: dict[str, Any] = {
            "slurm_partition": partition,
            "timeout_min": parse_time_to_minutes(resources.time_limit),
            "mem_gb": resources.memory_gb,
            "cpus_per_task": resources.cpus,
            "slurm_array_parallelism": parallelism,
            "slurm_job_name": f"s{step_number}_{job_name}",
            "slurm_srun_args": ["--export=ALL"],
        }
        slurm_gres = extras.pop("slurm_gres", None)
        if slurm_gres:
            parameters["slurm_gres"] = slurm_gres
            if resources.gpus:
                logger.warning(
                    "Both RunnerResources.gpus=%s and slurm_gres=%s were set; "
                    "using slurm_gres",
                    resources.gpus,
                    slurm_gres,
                )
        elif resources.gpus:
            parameters["gpus_per_node"] = resources.gpus
        parameters.update(extras)
        if parameters["timeout_min"] < 1:
            msg = "RunnerResources.time_limit must be at least one second"
            raise ValueError(msg)
        return parameters

    def _get_max_array_size(self) -> int:
        """Detect the scheduler limit once per reusable runner instance."""
        if self._cached_max_array_size is None:
            self._cached_max_array_size = get_cluster_max_array_size(
                self.max_array_size
            )
        return self._cached_max_array_size


def _resolve_array_parallelism(
    max_workers: int | None,
    runner_parallelism: int | None,
    resource_parallelism: object,
) -> int:
    """Resolve Artisan concurrency without silently overriding provider input."""
    explicit = resource_parallelism
    if explicit is None:
        explicit = runner_parallelism
    if explicit is not None and (
        isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1
    ):
        msg = f"slurm_array_parallelism must be a positive integer, got {explicit!r}"
        raise ValueError(msg)
    if max_workers is not None and max_workers < 1:
        msg = f"BatchStrategy.max_workers must be >= 1, got {max_workers}"
        raise ValueError(msg)
    if max_workers is not None and explicit not in {None, max_workers}:
        msg = (
            f"BatchStrategy.max_workers={max_workers} conflicts with "
            f"slurm_array_parallelism={explicit}"
        )
        raise ValueError(msg)
    return max_workers or explicit or 1000


def _validate_provider_parameters(parameters: dict[str, Any]) -> None:
    """Reject aliases and duplicate portable-resource controls early."""
    removed = _REMOVED_PARAMETER_NAMES.keys() & parameters.keys()
    if removed:
        key = next(iter(removed))
        replacement = _REMOVED_PARAMETER_NAMES[key]
        msg = f"{key!r} is not a Submitit-native parameter; use {replacement!r}"
        raise ValueError(msg)
    duplicate = _PORTABLE_PARAMETER_NAMES & parameters.keys()
    if duplicate:
        names = ", ".join(sorted(duplicate))
        msg = f"Configure portable resources through RunnerResources, not: {names}"
        raise ValueError(msg)
    topology = _TOPOLOGY_PARAMETER_NAMES & parameters.keys()
    if topology:
        names = ", ".join(sorted(topology))
        msg = f"Artisan Submitit jobs require one task; unsupported options: {names}"
        raise ValueError(msg)
    slurm_gres = parameters.get("slurm_gres")
    if slurm_gres is not None and not isinstance(slurm_gres, str):
        msg = "slurm_gres must be a string"
        raise TypeError(msg)
    srun_arguments = parameters.get("slurm_srun_args")
    if srun_arguments is not None:
        reject_managed_scheduler_arguments(srun_arguments, "slurm_srun_args")
    _validate_additional_parameters(parameters.get("slurm_additional_parameters"))


def _validate_additional_parameters(value: object) -> None:
    """Reject hidden topology and portable-resource overrides."""
    if value is None:
        return
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = "slurm_additional_parameters must be a dict with string keys"
        raise TypeError(msg)
    managed = {key for key in value if is_managed_scheduler_option(key)}
    if managed:
        names = ", ".join(sorted(managed))
        msg = (
            "slurm_additional_parameters cannot override Artisan-managed "
            f"options: {names}"
        )
        raise ValueError(msg)


def _validate_effective_executor(
    executor: submitit.AutoExecutor,
    expected: dict[str, Any],
) -> None:
    """Ensure Submitit's canonical parameters preserve Artisan's invariants."""
    parameters = executor.parameters
    _validate_effective_topology(parameters)
    _validate_effective_resources(parameters, expected)
    reject_managed_scheduler_arguments(
        parameters.get("srun_args", []), "effective slurm_srun_args"
    )
    _validate_additional_parameters(parameters.get("additional_parameters"))


def _validate_effective_topology(parameters: Mapping[str, Any]) -> None:
    """Reject topology changes after Submitit has canonicalized aliases."""
    if parameters.get("nodes", 1) != 1 or parameters.get("ntasks_per_node") not in {
        None,
        1,
    }:
        msg = "Submitit canonicalization changed Artisan's single-task topology"
        raise ValueError(msg)


def _validate_effective_resources(
    parameters: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Verify canonical portable values and array concurrency were not replaced."""
    comparisons = {
        "time": expected["timeout_min"],
        "cpus_per_task": expected["cpus_per_task"],
        "array_parallelism": expected["slurm_array_parallelism"],
    }
    for name, value in comparisons.items():
        if parameters.get(name) != value:
            msg = f"Submitit canonicalization changed Artisan-managed {name!r}"
            raise ValueError(msg)
    forbidden = (
        "cpus_per_gpu",
        "gpus_per_task",
        "mem_per_cpu",
        "mem_per_gpu",
        "num_gpus",
    )
    if any(parameters.get(name) is not None for name in forbidden):
        msg = "Submitit canonicalization introduced conflicting portable resources"
        raise ValueError(msg)
    expected_memory = f"{expected['mem_gb']}GB"
    if parameters.get("mem") != expected_memory:
        msg = "Submitit canonicalization changed Artisan-managed memory"
        raise ValueError(msg)
    expected_gres = expected.get("slurm_gres")
    expected_gpus = None if expected_gres else expected.get("gpus_per_node")
    if parameters.get("gres") != expected_gres:
        msg = "Submitit canonicalization changed Artisan-managed GPU resources"
        raise ValueError(msg)
    if parameters.get("gpus_per_node") != expected_gpus:
        msg = "Submitit canonicalization changed Artisan-managed GPU count"
        raise ValueError(msg)


def _failure_results_for_waves(
    waves: list[list[list[ExecutionUnit]]],
    error: BaseException | str,
) -> list[UnitResult]:
    """Return aligned failures for scheduler waves not successfully submitted."""
    units = [unit for wave in waves for batch in wave for unit in batch]
    return failure_results_for_units(units, error)


def _read_submitit_job_log(job: submitit.Job[Any], retries: int = 5) -> str | None:
    """Read logs from paths belonging to the exact Submitit handle."""
    try:
        return read_job_logs(
            Path(job.paths.stdout), Path(job.paths.stderr), retries=retries
        )
    except Exception:
        logger.debug("Failed to read logs for SLURM job %s", job.job_id, exc_info=True)
        return None


def _attach_worker_log(
    results: list[UnitResult],
    worker_log: str | None,
) -> list[UnitResult]:
    """Copy one scheduler job's log onto each result produced by that job."""
    if not worker_log:
        return results
    return [replace(result, worker_log=worker_log) for result in results]
