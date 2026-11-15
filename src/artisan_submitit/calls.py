"""Capture trusted core calls without transporting controller-hydrated artifacts."""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from artisan.orchestration.runner_api import (
    AdmittedAssignment,
    AdmittedCurator,
    AssignmentDispatch,
    AssignmentReplyContext,
    CuratorReplyContext,
    InertWorkerCall,
    SealedDelivery,
    StagingResult,
    execute_admitted_assignment,
    execute_admitted_curator,
    serialize_worker_call,
)

from artisan_submitit.control import CallRecord, PackRequest, write_once
from artisan_submitit.gpu import GPUProbeSpec
from artisan_submitit.launcher import ALLOCATION_VARIABLES, PRELUDE_SHA256
from artisan_submitit.software import SlurmSoftware


def capture_pack(
    assignments: Sequence[AdmittedAssignment] | AdmittedCurator,
    software: SlurmSoftware,
    dispatch: AssignmentDispatch,
    directory: Path,
) -> PackRequest:
    """Capture the closed entrypoint and exact admission before scheduler submit."""
    if dispatch.application_modules != software.application_modules:
        msg = "Application selection differs from resolved scheduler software"
        raise ValueError(msg)
    calls = []
    items: Sequence[AdmittedAssignment | AdmittedCurator] = (
        (assignments,) if isinstance(assignments, AdmittedCurator) else assignments
    )
    for index, admitted in enumerate(items):
        function = (
            execute_curator_call
            if isinstance(admitted, AdmittedCurator)
            else execute_assignment_call
        )
        call = serialize_worker_call(
            function,
            compatibility=software.compatibility,
            application_modules=software.application_modules,
            args=(admitted, software, dispatch, directory.name, index),
        )
        sensitive = tuple(dispatch.runtime_env.values())
        context = (
            CuratorReplyContext(
                admission=admitted.admission,
                payload_sha256=call.header.payload_sha256,
                sensitive_values=sensitive,
            )
            if isinstance(admitted, AdmittedCurator)
            else AssignmentReplyContext(
                attempt=admitted.attempt,
                payload_sha256=call.header.payload_sha256,
                sensitive_values=sensitive,
            )
        )
        inert = InertWorkerCall(
            call.header, call.payload, context.model_dump_json().encode()
        )
        write_once(directory / f"call-{index}.bin", inert.payload)
        calls.append(
            CallRecord(
                index=index,
                header=inert.header,
                reply_context=inert.reply_context.decode(),
            )
        )
    request = PackRequest(
        pack_id=directory.name,
        prelude_sha256=PRELUDE_SHA256,
        profile=software.local.worker_profile.to_json().decode(),
        calls=tuple(calls),
        gpu=software.gpu,
        working_root=dispatch.working_root,
        runtime_env=dict(dispatch.runtime_env),
        preserve_working=dispatch.preserve_working,
    )
    write_once(directory / "pack.json", request.model_dump_json().encode())
    return request


def _working_root(dispatch: AssignmentDispatch, pack_id: str, index: int) -> Path:
    entry = os.environ["ARTISAN_SUBMITIT_ENTRY_ID"]
    if uuid.UUID(hex=entry).hex != entry:
        msg = "Provider entry identity is invalid"
        raise ValueError(msg)
    return dispatch.working_root / pack_id / entry / f"unit-{index}"


def execute_assignment_call(
    admitted: AdmittedAssignment,
    software: SlurmSoftware,
    dispatch: AssignmentDispatch,
    pack_id: str,
    index: int,
) -> SealedDelivery:
    return execute_admitted_assignment(
        admitted,
        software,
        _allocated_dispatch(dispatch),
        working_root=_working_root(dispatch, pack_id, index),
    )


def execute_curator_call(
    admitted: AdmittedCurator,
    software: SlurmSoftware,
    dispatch: AssignmentDispatch,
    pack_id: str,
    index: int,
) -> StagingResult:
    return execute_admitted_curator(
        admitted,
        software,
        _allocated_dispatch(dispatch),
        working_root=_working_root(dispatch, pack_id, index),
    )


def _allocated_dispatch(dispatch: AssignmentDispatch) -> AssignmentDispatch:
    values = {key: os.environ[key] for key in ALLOCATION_VARIABLES if key in os.environ}
    return replace(dispatch, runtime_env={**dispatch.runtime_env, **values})


@dataclass(frozen=True)
class ProbeDispatch:
    """Only scratch is needed for fixed qualification; no store is invented."""

    working_root: Path

    @property
    def runtime_env(self) -> dict[str, str]:
        return {}


def capture_probe(
    probe: GPUProbeSpec,
    software: SlurmSoftware,
    dispatch: ProbeDispatch,
    directory: Path,
) -> PackRequest:
    request = PackRequest(
        pack_id=directory.name,
        prelude_sha256=PRELUDE_SHA256,
        profile=software.local.worker_profile.to_json().decode(),
        calls=(),
        probe=probe,
        working_root=dispatch.working_root,
        runtime_env={},
    )
    write_once(directory / "pack.json", request.model_dump_json().encode())
    return request
