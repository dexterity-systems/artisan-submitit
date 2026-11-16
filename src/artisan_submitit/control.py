"""Closed shared-filesystem evidence for exact scheduler packs and redeliveries."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal, Self

from artisan.orchestration.runner_api import (
    InertWorkerCall,
    LocalWorkerProfile,
    WorkerTransportHeader,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from artisan_submitit.gpu import GPUProbeSpec, ScheduledGPUProfile
from artisan_submitit.launcher import ALLOCATION_VARIABLES, PRELUDE_SHA256


class ControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CallRecord(ControlModel):
    """Fixed filenames prevent control records from selecting external payloads."""

    index: int = Field(ge=0)
    header: WorkerTransportHeader
    reply_context: str

    def load(self, directory: Path) -> InertWorkerCall:
        return InertWorkerCall(
            self.header,
            (directory / f"call-{self.index}.bin").read_bytes(),
            self.reply_context.encode(),
        )


class PackRequest(ControlModel):
    domain: Literal["artisan.submitit-pack"] = "artisan.submitit-pack"
    version: Literal[2] = 2
    pack_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    profile: str = Field(repr=False)
    prelude_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calls: tuple[CallRecord, ...]
    probe: GPUProbeSpec | None = None
    gpu: ScheduledGPUProfile | None = None
    working_root: Path
    runtime_env: dict[str, str] = Field(repr=False)
    preserve_working: bool = False

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            msg = "Scheduler pack version must be an integer"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_pack(self) -> Self:
        if self.prelude_sha256 != PRELUDE_SHA256:
            msg = "Scheduler pack differs from the fixed bootstrap prelude"
            raise ValueError(msg)
        if not self.working_root.is_absolute() or bool(self.calls) == (
            self.probe is not None
        ):
            msg = "A scheduler pack needs absolute scratch and either calls or a fixed probe"
            raise ValueError(msg)
        if tuple(call.index for call in self.calls) != tuple(range(len(self.calls))):
            msg = "Scheduler pack call membership is incomplete or reordered"
            raise ValueError(msg)
        profile = LocalWorkerProfile.from_json(self.profile)
        if self.probe is not None and (self.gpu is not None or self.runtime_env):
            msg = "A fixed GPU probe cannot carry scientific GPU software or injections"
            raise ValueError(msg)
        if profile.native_gpu is not None:
            msg = "Scheduler profiles cannot carry controller allocation identities"
            raise ValueError(msg)
        if profile.observed:
            msg = "Remote scheduler packs require a prepared worker"
            raise ValueError(msg)
        if any(
            call.header.compatibility != profile.worker.compatibility
            for call in self.calls
        ):
            msg = "Scheduler pack calls differ from their exact worker profile"
            raise ValueError(msg)
        if set(ALLOCATION_VARIABLES) & self.runtime_env.keys() or any(
            key.startswith(("ARTISAN_SUBMITIT_", "SLURM_", "SUBMITIT_"))
            for key in self.runtime_env
        ):
            msg = "Runtime injections cannot override provider allocation controls"
            raise ValueError(msg)
        return self

    @property
    def result_count(self) -> int:
        return 1 if self.probe is not None else len(self.calls)

    def payload_digest(self, index: int) -> str:
        if self.probe is not None:
            if index != 0:
                msg = "A fixed probe has exactly one result"
                raise ValueError(msg)
            return self.probe.digest
        return self.calls[index].header.payload_sha256

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class ReadyRecord(ControlModel):
    """Bind SDK bytes to the exact accepted scheduler job and submitted pack."""

    pack_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    pack_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    job_id: str = Field(pattern=r"^[0-9]+$")
    outer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EntryIdentity(ControlModel):
    """A fresh actual provider entry, independent of admitted scientific identity."""

    pack_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    entry_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    job_id: str = Field(pattern=r"^[0-9]+$")
    step_id: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)?$")


class UnitEvidence(ControlModel):
    entry: EntryIdentity
    index: int = Field(ge=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reply: str
    cancelled: bool = False
    ownership: Literal["closed", "unknown"]


class EntryEvidence(ControlModel):
    entry: EntryIdentity
    ownership: Literal["closed", "unknown"]
    message: str | None = None


def write_once(path: Path, encoded: bytes) -> None:
    """Publish complete private control evidence without replacing prior authority."""
    temporary = path.with_suffix(path.suffix + ".pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def decode_pack(directory: Path, encoded: bytes, expected_digest: str) -> PackRequest:
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        msg = "Scheduler pack bytes differ from their exact launch binding"
        raise ValueError(msg)
    request = PackRequest.model_validate_json(encoded)
    if directory.name != request.pack_id:
        msg = "Scheduler control directory differs from its pack identity"
        raise ValueError(msg)
    return request


def read_pack(directory: Path, expected_digest: str) -> PackRequest:
    return decode_pack(
        directory, (directory / "pack.json").read_bytes(), expected_digest
    )
