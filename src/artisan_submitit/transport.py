"""Retain exact scheduler ownership independently of per-unit result readiness."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AdmittedCurator,
    AssignmentAttempt,
    AssignmentDispatch,
    AssignmentReply,
    CancellationAcknowledgement,
    CancellationStatus,
    CommandOwnershipError,
    CuratorAdmissionReference,
    CuratorReply,
    OwnershipState,
    RunnerResources,
    SealedDelivery,
    StagingResult,
    WorkerOwner,
)

from artisan_submitit.calls import ProbeDispatch, capture_pack, capture_probe
from artisan_submitit.cancellation import (
    cancel_jobs,
    query_job_states,
    requeued_job_ended,
)
from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    UnitEvidence,
    write_once,
)
from artisan_submitit.gpu import GPUProbeReply, GPUProbeSpec, ScheduledGPUProfile
from artisan_submitit.software import SlurmSoftware


def _transport_failure(
    error: BaseException, dispatch: AssignmentDispatch | ProbeDispatch
) -> Exception:
    """Pre-entry transport errors never acquire scientific failure authority."""
    message = str(error)
    for value in sorted(
        set(dispatch.runtime_env.values()) - {""}, key=len, reverse=True
    ):
        message = message.replace(value, "[REDACTED]")
    if isinstance(error, CommandOwnershipError):
        return CommandOwnershipError(message)
    if isinstance(error, CancelledError):
        return CancelledError()
    return RuntimeError(message or type(error).__name__)


class SchedulerStartupError(RuntimeError):
    def __init__(self, owner: PackOwner, error: BaseException) -> None:
        super().__init__("Scheduler ownership thread could not start")
        self.owner = owner
        self.error = error


class SchedulerHandle[ResultT]:
    """One admitted unit, with its result and whole-worker ownership separate."""

    def __init__(self, owner: PackOwner, index: int) -> None:
        self.owner = owner
        self.index = index
        self.future: Future[ResultT] = Future()
        self.cancelled = threading.Event()
        self._cancel_deadline: float | None = None

    @property
    def worker_owner(self) -> WorkerOwner:
        return self.owner

    def result(self, timeout: float | None = None) -> ResultT:
        return self.future.result(timeout)

    def done(self) -> bool:
        return self.future.done()

    def ownership_state(self) -> OwnershipState:
        return self.owner.ownership_state()

    def cancel(self) -> CancellationAcknowledgement:
        state = self.ownership_state()
        if state == OwnershipState.UNKNOWN:
            return CancellationAcknowledgement(
                CancellationStatus.UNKNOWN, "Scheduler ownership remains unresolved"
            )
        if self.done() and not self.cancelled.is_set():
            return CancellationAcknowledgement(
                CancellationStatus.REJECTED, "Unit already completed"
            )
        self.cancelled.set()
        self._cancel_deadline = self._cancel_deadline or time.monotonic() + 8
        if not self.done() and time.monotonic() >= self._cancel_deadline:
            self.owner._unknown = True
            return CancellationAcknowledgement(
                CancellationStatus.UNKNOWN, "Unit cancellation has not settled"
            )
        return CancellationAcknowledgement(
            CancellationStatus.CONFIRMED
            if state == OwnershipState.CLOSED
            else CancellationStatus.REQUESTED,
            "Unit cancellation requested",
        )

    def receive(self, encoded: str) -> None:
        result = self.decode(encoded)
        if not self.future.done():
            self.future.set_result(result)

    @abstractmethod
    def decode(self, encoded: str) -> ResultT: ...


class SchedulerAssignmentHandle(SchedulerHandle[SealedDelivery]):
    def __init__(
        self, owner: PackOwner, index: int, attempt: AssignmentAttempt
    ) -> None:
        super().__init__(owner, index)
        self._attempt = attempt

    @property
    def attempt(self) -> AssignmentAttempt:
        return self._attempt

    def decode(self, encoded: str) -> SealedDelivery:
        return AssignmentReply.model_validate_json(encoded).result(
            self.attempt,
            sensitive_values=tuple(self.owner.dispatch.runtime_env.values()),
        )


class SchedulerCuratorHandle(SchedulerHandle[StagingResult]):
    def __init__(self, owner: PackOwner, admission: CuratorAdmissionReference) -> None:
        super().__init__(owner, 0)
        self._admission = admission

    @property
    def admission(self) -> CuratorAdmissionReference:
        return self._admission

    def decode(self, encoded: str) -> StagingResult:
        return CuratorReply.model_validate_json(encoded).result(
            self.admission,
            sensitive_values=tuple(self.owner.dispatch.runtime_env.values()),
        )


class SchedulerProbeHandle(SchedulerHandle[ScheduledGPUProfile]):
    def decode(self, encoded: str) -> ScheduledGPUProfile:
        return GPUProbeReply.model_validate_json(encoded).result()


class PackOwner(ABC):
    """One submission thread owns every actual entry, including partial startup."""

    def __init__(
        self,
        assignments: Sequence[AdmittedAssignment] | AdmittedCurator | GPUProbeSpec,
        software: SlurmSoftware,
        dispatch: AssignmentDispatch | ProbeDispatch,
        control_root: Path,
    ) -> None:
        self.assignments = assignments
        self.software = software
        self.dispatch = dispatch
        self.directory = control_root / uuid.uuid4().hex
        self.handles = (
            (SchedulerProbeHandle(self, 0),)
            if isinstance(assignments, GPUProbeSpec)
            else (SchedulerCuratorHandle(self, assignments.admission),)
            if isinstance(assignments, AdmittedCurator)
            else tuple(
                SchedulerAssignmentHandle(self, index, item.attempt)
                for index, item in enumerate(assignments)
            )
        )
        self._lock = threading.Lock()
        self._state = OwnershipState.PENDING
        self._stop = threading.Event()
        self._deadline: float | None = None
        self._unknown = False
        self._submission_started = False
        self._job_id: str | None = None
        self._submitted_after = datetime.now(UTC).replace(microsecond=0)
        self._cleanup_thread: threading.Thread | None = None
        self._thread = threading.Thread(target=self._run, name="artisan-slurm-pack")
        try:
            self._thread.start()
        except BaseException as error:
            self._state = OwnershipState.CLOSED
            sanitized = _transport_failure(error, self.dispatch)
            for handle in self.handles:
                handle.future.set_exception(sanitized)
            raise SchedulerStartupError(self, sanitized) from None

    def ownership_state(self) -> OwnershipState:
        with self._lock:
            if self._unknown or (
                self._deadline is not None
                and time.monotonic() >= self._deadline
                and self._state != OwnershipState.CLOSED
            ):
                self._unknown = True
                return OwnershipState.UNKNOWN
            return self._state

    def request_close(self, deadline: float) -> None:
        with self._lock:
            self._deadline = (
                min(self._deadline, deadline)
                if self._deadline is not None
                else deadline
            )
            self._stop.set()
            if (
                self._unknown
                and not self._thread.is_alive()
                and (
                    self._cleanup_thread is None or not self._cleanup_thread.is_alive()
                )
            ):
                self._cleanup_thread = threading.Thread(
                    target=self._cleanup_only,
                    args=(deadline,),
                    name="artisan-slurm-cleanup",
                )
                self._cleanup_thread.start()

    @property
    def resources(self) -> RunnerResources:
        if isinstance(self.assignments, GPUProbeSpec):
            return RunnerResources(
                cpus=1,
                memory_gb=2,
                gpus=self.assignments.device_count,
                time_limit="00:02:00",
            )
        return self.software.resources

    @property
    def gpu_node(self) -> str | None:
        if isinstance(self.assignments, GPUProbeSpec):
            return self.assignments.node
        return self.software.gpu.node if self.software.gpu is not None else None

    @abstractmethod
    def submit(self, request: PackRequest) -> None:
        """Set exact scheduler identity, retaining ambiguity after acceptance."""

    def _cleanup_only(self, deadline: float) -> None:
        """Clean exact known owners without reading damaged scientific evidence."""
        try:
            try:
                self._publish_requests()
            except OSError:
                self._unknown = True
            scheduler_id = self.scheduler_id()
            if scheduler_id is not None:
                cancel_jobs(
                    [scheduler_id],
                    timeout=max(0.01, min(1.0, deadline - time.monotonic())),
                    executable=self.software.scheduler.executable("scancel"),
                )
            self.close_transport(deadline)
            while scheduler_id is not None and time.monotonic() < deadline:
                observation = query_job_states(
                    [scheduler_id],
                    self._submitted_after,
                    timeout=max(0.01, min(1.0, deadline - time.monotonic())),
                    executable=self.software.scheduler.executable("sacct"),
                    scontrol=self.software.scheduler.executable("scontrol"),
                ).get(scheduler_id)
                if (
                    observation is not None
                    and observation.terminal
                    and self.transport_closed()
                ):
                    return
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        except Exception:
            self._unknown = True

    def close_transport(self, _deadline: float) -> None:
        """Providers with an owned local client close it without widening remote scope."""
        return

    def _publish_requests(self) -> None:
        if self._stop.is_set() and not (self.directory / "cancel").exists():
            write_once(self.directory / "cancel", b"")
        for handle in self.handles:
            path = self.directory / f"cancel-{handle.index}"
            if handle.cancelled.is_set() and not path.exists():
                write_once(path, b"")

    def _collect_entry(self, entry_dir: Path, request: PackRequest) -> bool:
        identity_path = entry_dir / "identity.json"
        if not identity_path.exists():
            return False
        identity = EntryIdentity.model_validate_json(identity_path.read_bytes())
        assert self._job_id is not None
        if (
            identity.entry_id != entry_dir.name
            or identity.pack_id != request.pack_id
            or identity.job_id != self._job_id.split(".")[0]
        ):
            msg = "Worker entry differs from its exact scheduler owner"
            raise ValueError(msg)
        self.observe_identity(identity)
        for handle in self.handles:
            path = entry_dir / f"unit-{handle.index}.json"
            if not path.exists():
                continue
            evidence = UnitEvidence.model_validate_json(path.read_bytes())
            if (
                evidence.entry != identity
                or evidence.index != handle.index
                or evidence.payload_sha256 != request.payload_digest(handle.index)
            ):
                msg = "Worker result differs from its exact submitted call"
                raise ValueError(msg)
            self._unknown |= evidence.ownership == "unknown"
            try:
                if evidence.ownership == "unknown":
                    msg = "Worker unit cleanup remains unknown"
                    raise CommandOwnershipError(msg)
                if evidence.cancelled:
                    if not (handle.cancelled.is_set() or self._stop.is_set()):
                        msg = "Worker cancelled without a controller request"
                        raise RuntimeError(msg)
                    raise CancelledError()
                handle.receive(evidence.reply)
            except Exception as error:
                self._unknown |= isinstance(error, CommandOwnershipError)
                if not handle.done():
                    handle.future.set_exception(error)
        outcome_path = entry_dir / "outcome.json"
        if not outcome_path.exists():
            return False
        outcome = EntryEvidence.model_validate_json(outcome_path.read_bytes())
        if outcome.entry != identity:
            msg = "Worker closure belongs to another entry"
            raise ValueError(msg)
        self._unknown |= outcome.ownership == "unknown"
        return outcome.ownership == "closed"

    def observe_identity(self, identity: EntryIdentity) -> None:
        """Require batch entries to remain within their exact submitted job."""
        if identity.job_id != self._job_id:
            msg = "Worker entry has a foreign job owner"
            raise ValueError(msg)

    def scheduler_id(self) -> str | None:
        return self._job_id

    def transport_closed(self) -> bool:
        return True

    def transport_failed(self) -> bool:
        return False

    def _observe(self, request: PackRequest) -> None:
        terminal_since = None
        requeue_terminal_since = None
        transport_stopped_since = None
        cancel_sent = False
        while True:
            self._publish_requests()
            entries = sorted((self.directory / "entries").glob("*"))
            closed = [self._collect_entry(path, request) for path in entries]
            scheduler_id = self.scheduler_id()
            query_deadline = time.monotonic() + 1
            if self._deadline is not None:
                query_deadline = min(query_deadline, self._deadline)
            observation = (
                query_job_states(
                    [scheduler_id],
                    self._submitted_after,
                    timeout=max(0.0, query_deadline - time.monotonic()),
                    executable=self.software.scheduler.executable("sacct"),
                    scontrol=self.software.scheduler.executable("scontrol"),
                ).get(scheduler_id)
                if scheduler_id
                else None
            )
            if (
                observation is None
                and scheduler_id is not None
                and requeued_job_ended(
                    scheduler_id,
                    self._submitted_after,
                    deadline=query_deadline,
                    executable=self.software.scheduler.executable("scontrol"),
                )
            ):
                if requeue_terminal_since is None:
                    requeue_terminal_since = time.monotonic()
                if time.monotonic() - requeue_terminal_since >= 5:
                    for path in sorted((self.directory / "entries").glob("*")):
                        self._collect_entry(path, request)
                    self._unknown = True
                    return
            else:
                requeue_terminal_since = None
            if observation is not None and observation.terminal:
                terminal_since = terminal_since or time.monotonic()
                if not entries and observation.never_started and self._stop.is_set():
                    if (
                        self._deadline is not None
                        and time.monotonic() >= self._deadline
                    ) or any((self.directory / "entries").glob("*")):
                        self._unknown = True
                        return
                    if self.transport_closed():
                        return
                if (
                    entries
                    and all(closed)
                    and all(handle.done() for handle in self.handles)
                    and self.transport_closed()
                ):
                    return
                if time.monotonic() - terminal_since >= 5:
                    self._unknown = True
                    return
            if (
                self._stop.is_set()
                and self._deadline is not None
                and (
                    (
                        not entries
                        and observation is not None
                        and observation.state == "PENDING"
                        and not any((self.directory / "entries").glob("*"))
                    )
                    or time.monotonic() >= self._deadline - 2
                )
            ):
                if scheduler_id is not None and not cancel_sent:
                    error = cancel_jobs(
                        [scheduler_id],
                        timeout=max(0.01, min(1.0, self._deadline - time.monotonic())),
                        executable=self.software.scheduler.executable("scancel"),
                    )
                    cancel_sent = error is None
                self.close_transport(self._deadline)
            if self.transport_failed() and not entries:
                transport_stopped_since = transport_stopped_since or time.monotonic()
                if time.monotonic() - transport_stopped_since >= 5:
                    self._unknown = True
                    return
            if self._deadline is not None and time.monotonic() >= self._deadline:
                self._unknown = True
                return
            time.sleep(0.1)

    def _run(self) -> None:
        error: BaseException | None = None
        try:
            self.directory.mkdir(parents=True, mode=0o700)
            if isinstance(self.assignments, GPUProbeSpec):
                assert isinstance(self.dispatch, ProbeDispatch)
                request = capture_probe(
                    self.assignments, self.software, self.dispatch, self.directory
                )
            else:
                request = capture_pack(
                    self.assignments,
                    self.software,
                    cast(AssignmentDispatch, self.dispatch),
                    self.directory,
                )
            if self._stop.is_set():
                raise CancelledError()
            self.submit(request)
            self._observe(request)
        except BaseException as failure:
            error = failure
            if self._submission_started:
                self._unknown = True
                deadline = self._deadline or time.monotonic() + 8
                self.request_close(deadline)
                self._cleanup_only(deadline)
        finally:
            if not self._unknown and self.directory.exists():
                try:
                    bootstrap_scratch = self.directory / "bootstrap"
                    if bootstrap_scratch.exists():
                        shutil.rmtree(bootstrap_scratch)
                    write_once(self.directory / "closed", b"")
                except OSError:
                    self._unknown = True
            with self._lock:
                self._state = (
                    OwnershipState.UNKNOWN if self._unknown else OwnershipState.CLOSED
                )
            for handle in self.handles:
                if not handle.done():
                    terminal_error = (
                        CommandOwnershipError(
                            "Scheduler ownership could not be confirmed"
                        )
                        if self._unknown
                        else (error or CancelledError())
                    )
                    handle.future.set_exception(
                        _transport_failure(terminal_error, self.dispatch)
                    )
