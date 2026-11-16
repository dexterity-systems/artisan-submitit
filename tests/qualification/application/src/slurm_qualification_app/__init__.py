"""Installed operations for scheduler ownership and failure qualification."""

from __future__ import annotations

import json
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from artisan.operations.base import OperationDefinition, PerArtifact
from artisan.schemas import (
    ArtifactResult,
    CSVArtifact,
    ExecuteInput,
    InputSpec,
    MetricArtifact,
    OutputSpec,
    PostprocessInput,
    PreprocessInput,
)
from pydantic import BaseModel, Field

from slurm_qualification_app.file_tool import FileToolIncrement as FileToolIncrement
from slurm_qualification_app.gpu_operations import (
    CommandGPUIncrement as CommandGPUIncrement,
)
from slurm_qualification_app.gpu_operations import (
    GPUMerge as GPUMerge,
)
from slurm_qualification_app.gpu_operations import (
    NativeGPUIncrement as NativeGPUIncrement,
)


class ObservableWait(OperationDefinition):
    """Publish readiness from actual authored execution, then wait boundedly."""

    name = "slurm_qualification_wait"
    cacheable: ClassVar[bool] = False

    inputs: ClassVar = {}

    class OutputRole(StrEnum):
        output = "output"

    outputs: ClassVar = {
        "output": OutputSpec(artifact_type="csv", derives_from={"inputs": []})
    }

    class Params(BaseModel):
        """Owned observation paths and a bounded authored failure switch."""

        duration: float = Field(default=1, ge=0, description="Bounded wait in seconds.")
        marker_root: str = Field(description="Absolute shared test observation path.")
        fail: bool = Field(
            default=False, description="Raise after recording readiness."
        )

    params: Params

    def execute_function(self, inputs: ExecuteInput) -> None:
        """Record live execution identity without borrowing scheduler authority."""
        root = Path(self.params.marker_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "ready.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cwd": str(Path.cwd()),
                    "job": os.environ.get("SLURM_JOB_ID"),
                    "step": os.environ.get("SLURM_STEP_ID"),
                }
            )
        )
        if self.params.fail:
            msg = "intentional scheduler operation failure"
            raise RuntimeError(msg)
        started = time.monotonic()
        while (
            time.monotonic() - started < self.params.duration
            and not (root / "stop").exists()
        ):
            (root / "heartbeat").write_text(str(time.monotonic_ns()))
            time.sleep(0.05)
        elapsed = time.monotonic() - started
        (Path(inputs.execute_dir) / "wait_marker.csv").write_text(
            f"requested,actual\n{self.params.duration},{elapsed}\n"
        )

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        """Publish the file produced by both native and command execution."""
        path = next(
            Path(value)
            for value in inputs.file_outputs
            if Path(value).name == "wait_marker.csv"
        )
        return ArtifactResult(
            artifacts={"output": [CSVArtifact.draft(path.read_bytes(), path.name)]},
            lineage={"output": []},
        )


class ManagedWait(ObservableWait):
    """Run the same bounded authored body through the inherited command target."""

    name = "slurm_qualification_managed_wait"
    execute_as_tool: ClassVar[bool] = True


class PackedWait(OperationDefinition):
    """One metric input per admitted unit, with independently observable commands."""

    name = "slurm_qualification_packed_wait"
    execute_as_tool: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False

    class Params(BaseModel):
        """Shared observation root and bounded first-unit delay."""

        marker_root: str = Field(description="Absolute shared test observation path.")
        duration: float = Field(default=120, ge=0, description="First-unit wait limit.")

    params: Params

    class InputRole(StrEnum):
        source = "source"

    class OutputRole(StrEnum):
        result = "result"

    inputs: ClassVar = {"source": InputSpec(artifact_type="metric", materialize=False)}
    outputs: ClassVar = {
        "result": OutputSpec(
            artifact_type="metric", derives_from={"inputs": ["source"]}
        )
    }

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        """Enforce this fixture's one-item unit and keep command inputs file-shaped."""
        (source,) = inputs.input_artifacts["source"]
        path = Path(inputs.preprocess_dir) / "input.json"
        assert isinstance(source, MetricArtifact)
        assert source.content is not None
        path.write_bytes(source.content)
        return {"source": PerArtifact([str(path)])}

    def execute_function(self, inputs: ExecuteInput) -> None:
        """Wait for unit zero while the second unit can later finish normally."""
        (path,) = inputs.inputs["source"]
        value = json.loads(Path(path).read_text())["value"]
        root = Path(self.params.marker_root) / str(value)
        root.mkdir(parents=True, exist_ok=True)
        (root / "ready.json").write_text(
            json.dumps({"pid": os.getpid(), "value": value})
        )
        deadline = time.monotonic() + (self.params.duration if value == 0 else 0)
        while time.monotonic() < deadline and not (root / "stop").exists():
            (root / "heartbeat").write_text(str(time.monotonic_ns()))
            time.sleep(0.05)
        (Path(inputs.execute_dir) / "result.json").write_text(
            json.dumps({"value": value + 1, "pid": os.getpid()})
        )

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        path = next(
            Path(value)
            for value in inputs.file_outputs
            if Path(value).name == "result.json"
        )
        result = ArtifactResult()
        result.add_artifact(
            "result",
            MetricArtifact.draft(json.loads(path.read_text()), "result.json"),
            sources={"source": [inputs.ref("source", index=0)]},
        )
        return result
