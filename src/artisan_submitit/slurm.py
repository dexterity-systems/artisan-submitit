"""Submitit-backed SLURM runner for Artisan execution units."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import submitit
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

from artisan_submitit.utils import (
    chunked,
    get_cluster_max_array_size,
    is_managed_scheduler_option,
    parse_time_to_minutes,
    read_job_logs,
    reject_managed_scheduler_arguments,
)

logger = logging.getLogger(__name__)

_PORTABLE_PARAMETER_NAMES = frozenset(
    {
        "cpus_per_task",
        "gpus_per_node",
        "mem_gb",
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
        "timeout_min",
    }
)
_TOPOLOGY_PARAMETER_NAMES = frozenset(
    {
        "nodes",
        "slurm_nodes",
        "slurm_ntasks_per_node",
        "slurm_tasks_per_node",
        "tasks_per_node",
    }
)
_REMOVED_PARAMETER_NAMES = {
    "partition": "slurm_partition",
    "time_limit": "RunnerResources.time_limit",
}


@dataclass(frozen=True)
class _SubmittedBatch:
    """One exact Submitit job handle and the units it owns."""

    job: submitit.Job[Any]
    units: list[ExecutionUnit]


class SlurmLifecycleRouter(LifecycleRouter):
    """Submit ordered unit batches as SLURM arrays through Submitit."""

    def __init__(
        self,
        executor: submitit.AutoExecutor,
        units_per_worker: int,
        max_array_size: int,
    ) -> None:
        super().__init__()
        self._executor = executor
        self._units_per_worker = units_per_worker
        self._max_array_size = max_array_size
        self._submitted: list[_SubmittedBatch] = []
        self._cancelled_handles: set[int] = set()
        self._completed_handles: set[int] = set()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

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
        self._start_background(lambda: self._run_waves(units, runtime_env))

    def cancel(self) -> None:
        """Cancel only jobs submitted by this router; safe to call repeatedly."""
        self._cancelled.set()
        with self._lock:
            jobs = [
                item.job
                for item in self._submitted
                if id(item.job) not in self._cancelled_handles
                and id(item.job) not in self._completed_handles
            ]
            self._cancelled_handles.update(id(job) for job in jobs)
        for job in jobs:
            try:
                job.cancel(check=False)
            except Exception:
                with self._lock:
                    self._cancelled_handles.discard(id(job))
                logger.debug("Failed to cancel an owned SLURM job", exc_info=True)

    def _run_waves(
        self,
        units: list[ExecutionUnit],
        runtime_env: RuntimeEnvironment,
    ) -> list[UnitResult]:
        """Submit and collect one scheduler-sized wave at a time."""
        try:
            batches = pack_units(units, self._units_per_worker)
            waves = chunked(batches, self._max_array_size)
        except Exception as exc:
            return failure_results_for_units(units, exc)

        results: list[UnitResult] = []
        for index, wave in enumerate(waves):
            if self._cancelled.is_set():
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
        return results

    def _submit_array(
        self,
        batches: list[list[ExecutionUnit]],
        runtime_env: RuntimeEnvironment,
    ) -> list[_SubmittedBatch]:
        """Submit one scheduler-sized array and retain every exact handle."""
        pending: list[_SubmittedBatch] = []
        with self._executor.batch():
            for batch in batches:
                job = self._executor.submit(execute_unit_batch, batch, runtime_env)
                pending.append(_SubmittedBatch(job=job, units=batch))
        with self._lock:
            self._submitted.extend(pending)
        if self._cancelled.is_set():
            self.cancel()
        return pending

    def _collect_submitted(
        self,
        submitted: list[_SubmittedBatch],
    ) -> list[UnitResult]:
        """Collect batches in submission order and attach their exact job logs."""
        results: list[UnitResult] = []
        for item in submitted:
            try:
                batch_results = validate_batch_results(item.units, item.job.result())
            except Exception as exc:
                batch_results = failure_results_for_units(item.units, exc)
            with self._lock:
                self._completed_handles.add(id(item.job))
            worker_log = _read_submitit_job_log(item.job)
            results.extend(_attach_worker_log(batch_results, worker_log))
        return results


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
        folder = Path(log_folder or self.log_folder) / "%j"
        executor = submitit.AutoExecutor(folder=folder)
        executor.update_parameters(**parameters)
        return SlurmLifecycleRouter(
            executor=executor,
            units_per_worker=batch_strategy.units_per_worker,
            max_array_size=self._get_max_array_size(),
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
        if "slurm_gres" in extras:
            if resources.gpus:
                logger.warning(
                    "Both RunnerResources.gpus=%s and slurm_gres=%s were set; "
                    "using slurm_gres",
                    resources.gpus,
                    extras["slurm_gres"],
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


def _failure_results_for_waves(
    waves: list[list[list[ExecutionUnit]]],
    error: BaseException | str,
) -> list[UnitResult]:
    """Return aligned failures for scheduler waves not successfully submitted."""
    units = [unit for wave in waves for batch in wave for unit in batch]
    return failure_results_for_units(units, error)


def _read_submitit_job_log(job: submitit.Job[Any]) -> str | None:
    """Read logs from paths belonging to the exact Submitit handle."""
    try:
        return read_job_logs(Path(job.paths.stdout), Path(job.paths.stderr))
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
