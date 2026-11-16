"""Run inert packs under core ownership and publish only closed JSON evidence."""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from artisan.orchestration.runner_api import (
    AssignmentReply,
    AssignmentReplyContext,
    CancellationAcknowledgement,
    CommandOwnershipError,
    CuratorReply,
    CuratorReplyContext,
    LocalPackStartupError,
    LocalWorkerProfile,
    OwnershipState,
    WorkerOwner,
    launch_inert_calls,
)

from artisan_submitit.control import (
    EntryEvidence,
    EntryIdentity,
    PackRequest,
    UnitEvidence,
    read_pack,
    write_once,
)
from artisan_submitit.gpu import (
    GPUProbeReply,
    ScheduledGPUProfile,
    capture_allocated_gpu,
)
from artisan_submitit.launcher import ALLOCATION_VARIABLES

_CLEANUP_SECONDS = 8.0
_TERMINATION_RESERVE_SECONDS = 2.0


class _CallHandle(Protocol):
    @property
    def worker_owner(self) -> WorkerOwner | None: ...
    def result(self, timeout: float | None = None) -> bytes: ...
    def done(self) -> bool: ...
    def cancel(self) -> CancellationAcknowledgement: ...
    def ownership_state(self) -> OwnershipState: ...


def _failure_reply(request: PackRequest, index: int, error: Exception) -> str:
    if request.probe is not None:
        return GPUProbeReply(
            kind="failure", error=str(error) or type(error).__name__
        ).model_dump_json()
    encoded = request.calls[index].reply_context
    if json.loads(encoded)["domain"] == "assignment":
        context = AssignmentReplyContext.model_validate_json(encoded)
        return AssignmentReply.failed(
            context.attempt, error, sensitive_values=context.sensitive_values
        ).model_dump_json()
    curator_context = CuratorReplyContext.model_validate_json(encoded)
    return CuratorReply.failed(curator_context, error).model_dump_json()


def _publish_result(
    directory: Path,
    identity: EntryIdentity,
    request: PackRequest,
    index: int,
    handle: _CallHandle | None,
    error: Exception | None = None,
) -> bool:
    cancelled = False
    unknown = isinstance(error, CommandOwnershipError)
    try:
        if error is not None:
            raise error
        assert handle is not None
        reply = handle.result(timeout=0).decode()
    except Exception as failure:
        cancelled = isinstance(failure, CancelledError)
        unknown = isinstance(failure, CommandOwnershipError)
        reply = _failure_reply(request, index, failure)
    evidence = UnitEvidence(
        entry=identity,
        index=index,
        payload_sha256=request.payload_digest(index),
        reply=reply,
        cancelled=cancelled,
        ownership="unknown" if unknown else "closed",
    )
    write_once(directory / f"unit-{index}.json", evidence.model_dump_json().encode())
    return not unknown


def _write_result(
    directory: Path,
    identity: EntryIdentity,
    request: PackRequest,
    index: int,
    handle: _CallHandle | None,
    error: Exception | None = None,
) -> bool:
    """Evidence failure cannot skip cleanup or authorize closed ownership."""
    try:
        return _publish_result(directory, identity, request, index, handle, error)
    except Exception:
        return False


def _collect(
    directory: Path,
    entry_dir: Path,
    identity: EntryIdentity,
    request: PackRequest,
    handles: tuple[_CallHandle, ...],
    owners: tuple[WorkerOwner, ...],
) -> bool:
    observed: set[int] = set()
    deadline = None
    unknown = False
    while True:
        cancel_pack = (directory / "cancel").exists()
        for index, handle in enumerate(handles):
            if cancel_pack or unknown or (directory / f"cancel-{index}").exists():
                try:
                    handle.cancel()
                except Exception:
                    unknown = True
        states = []
        resources: tuple[WorkerOwner | _CallHandle, ...] = (*owners, *handles)
        for resource in resources:
            try:
                states.append(resource.ownership_state())
            except Exception:
                states.append(OwnershipState.UNKNOWN)
        unknown |= OwnershipState.UNKNOWN in states
        for index, handle in enumerate(handles):
            if index in observed:
                continue
            try:
                ready = handle.done()
            except Exception:
                ready = False
                unknown = True
            if ready or unknown:
                error = (
                    CommandOwnershipError("Scheduler pack ownership is unresolved")
                    if unknown
                    else None
                )
                closed = _write_result(
                    entry_dir, identity, request, index, handle, error
                )
                unknown |= not closed
                observed.add(index)
        if len(observed) == len(handles) and all(
            state == OwnershipState.CLOSED for state in states
        ):
            return not unknown
        if cancel_pack or unknown:
            deadline = deadline or time.monotonic() + _CLEANUP_SECONDS
            if len(observed) == len(handles) or (
                time.monotonic() >= deadline - _TERMINATION_RESERVE_SECONDS
            ):
                for owner in owners:
                    try:
                        owner.request_close(deadline)
                    except Exception:
                        unknown = True
        if deadline is not None and time.monotonic() >= deadline:
            for index in range(len(handles)):
                if index not in observed:
                    _write_result(
                        entry_dir,
                        identity,
                        request,
                        index,
                        None,
                        CommandOwnershipError("Scheduler pack cleanup is unresolved"),
                    )
            return False
        time.sleep(0.02)


def run_worker_pack(control_directory: str, expected_digest: str) -> None:
    """Enter afresh on every SDK redelivery; prior UNKNOWN evidence is immutable."""
    directory = Path(control_directory)
    request = read_pack(directory, expected_digest)
    entry_id = os.environ["ARTISAN_SUBMITIT_ENTRY_ID"]
    job_id = os.environ["SLURM_JOB_ID"]
    step = os.environ.get("SLURM_STEP_ID", "batch")
    step_id = f"{job_id}.{step}" if step.isdecimal() else job_id
    identity = EntryIdentity(
        pack_id=request.pack_id, entry_id=entry_id, job_id=job_id, step_id=step_id
    )
    entry_dir = directory / "entries" / entry_id
    entry_dir.mkdir(parents=True, mode=0o700)
    write_once(entry_dir / "identity.json", identity.model_dump_json().encode())
    scratch = request.working_root / request.pack_id / entry_id
    scratch.mkdir(parents=True, mode=0o700)
    if (directory / "cancel").exists():
        closed = True
        for index in range(request.result_count):
            closed &= _write_result(
                entry_dir, identity, request, index, None, CancelledError()
            )
    elif request.probe is not None:
        closed = _run_probe(directory, entry_dir, identity, request)
    else:
        injections = {
            **request.runtime_env,
            **{
                key: os.environ[key]
                for key in ALLOCATION_VARIABLES
                if key in os.environ
            },
            "ARTISAN_SUBMITIT_ENTRY_ID": entry_id,
        }
        handles: tuple[_CallHandle, ...]
        try:
            profile = LocalWorkerProfile.from_json(request.profile)
            if request.gpu is not None:
                allocation = capture_allocated_gpu(
                    request.gpu.node,
                    request.gpu.software.device_count,
                    request.gpu.software,
                )
                if request.gpu.sif is not None:
                    request.gpu.sif.verify(
                        allocation, expected=request.gpu.container_launcher
                    )
                profile = replace(profile, native_gpu=allocation)
                visible = ",".join(allocation.devices)
                injections.update(
                    CUDA_VISIBLE_DEVICES=visible, NVIDIA_VISIBLE_DEVICES=visible
                )
            handles = launch_inert_calls(
                tuple(call.load(directory) for call in request.calls),
                profile,
                working_root=scratch / "transport",
                runtime_env=injections,
                preserve_working=request.preserve_working,
            )
        except LocalPackStartupError as error:
            handles = error.handles
            owners = error.worker_owners
            startup_unknown = any(
                outcome.state != "closed" for outcome in error.transport_outcomes
            )
            failure = (
                CommandOwnershipError("Scheduler pack startup cleanup is unresolved")
                if startup_unknown
                else error
            )
            published = True
            for index in range(len(handles), len(request.calls)):
                published &= _write_result(
                    entry_dir, identity, request, index, None, failure
                )
            closed = _collect(directory, entry_dir, identity, request, handles, owners)
            closed &= not startup_unknown and published
        except Exception as error:
            closed = True
            for index in range(request.result_count):
                closed &= _write_result(
                    entry_dir, identity, request, index, None, error
                )
        else:
            owners = tuple(
                dict.fromkeys(
                    handle.worker_owner
                    for handle in handles
                    if handle.worker_owner is not None
                )
            )
            closed = _collect(directory, entry_dir, identity, request, handles, owners)
    if closed and not request.preserve_working:
        try:
            shutil.rmtree(scratch)
        except OSError:
            closed = False
    evidence = EntryEvidence(
        entry=identity,
        ownership="closed" if closed else "unknown",
        message=None if closed else "Scheduler entry cleanup could not be confirmed",
    )
    write_once(entry_dir / "outcome.json", evidence.model_dump_json().encode())


def _run_probe(
    directory: Path, entry_dir: Path, identity: EntryIdentity, request: PackRequest
) -> bool:
    """Fixed bounded hardware reads run under the existing exact scheduler owner."""
    assert request.probe is not None
    try:
        allocation = capture_allocated_gpu(
            request.probe.node, request.probe.device_count
        )
        if (directory / "cancel").exists():
            raise CancelledError()
        container_launcher = (
            request.probe.sif.verify(allocation)
            if request.probe.sif is not None
            else None
        )
        reply = GPUProbeReply(
            kind="success",
            profile=ScheduledGPUProfile(
                node=request.probe.node,
                software=allocation.software_profile(),
                sif=request.probe.sif,
                container_launcher=container_launcher,
            ),
        )
        evidence = UnitEvidence(
            entry=identity,
            index=0,
            payload_sha256=request.probe.digest,
            reply=reply.model_dump_json(),
            ownership="closed",
        )
        write_once(entry_dir / "unit-0.json", evidence.model_dump_json().encode())
        return True
    except Exception as error:
        return _write_result(entry_dir, identity, request, 0, None, error)
