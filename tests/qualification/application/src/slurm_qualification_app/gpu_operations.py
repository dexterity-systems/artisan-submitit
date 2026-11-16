"""Native, command and curator GPU fixtures using the current public API."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from artisan.operations.base import OperationDefinition, PerArtifact
from artisan.operations.curator import Merge
from artisan.schemas import (
    ArtifactResult,
    CuratorInput,
    Environment,
    ExecuteInput,
    InputSpec,
    MetricArtifact,
    OutputSpec,
    PassthroughResult,
    PostprocessInput,
    PreprocessInput,
    ToolSpec,
)
from artisan.storage import ArtifactStore
from pydantic import BaseModel, Field

from slurm_qualification_app.gpu_kernel import run_kernel


def _preprocess(inputs: PreprocessInput) -> dict[str, Any]:
    (artifact,) = inputs.input_artifacts["source"]
    assert isinstance(artifact, MetricArtifact)
    source = Path(inputs.preprocess_dir) / "source.json"
    source.write_text(
        json.dumps(
            {"input_value": artifact.values["value"], "preprocess_pid": os.getpid()}
        )
    )
    return {"source": PerArtifact([str(source)])}


def _postprocess(inputs: PostprocessInput) -> ArtifactResult:
    path = next(
        Path(value) for value in inputs.file_outputs if Path(value).name == "gpu.json"
    )
    values = json.loads(path.read_bytes())
    values["postprocess_pid"] = os.getpid()
    result = ArtifactResult()
    result.add_artifact(
        "metrics",
        MetricArtifact.draft(values, "gpu.json"),
        sources={"source": [inputs.ref("source", index=0)]},
    )
    return result


class NativeGPUIncrement(OperationDefinition):
    """One input per unit; the allocated worker runs the bounded41→42 kernel."""

    name = "slurm_qualification_native_gpu"

    class InputRole(StrEnum):
        source = "source"

    class OutputRole(StrEnum):
        metrics = "metrics"

    inputs: ClassVar = {"source": InputSpec(artifact_type="metric", materialize=False)}
    outputs: ClassVar = {
        "metrics": OutputSpec(
            artifact_type="metric", derives_from={"inputs": ["source"]}
        )
    }

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        return _preprocess(inputs)

    def execute_function(self, inputs: ExecuteInput) -> None:
        (source,) = inputs.inputs["source"]
        values = json.loads(Path(source).read_bytes())
        values.update(run_kernel())
        values["worker_pid"] = os.getpid()
        (Path(inputs.execute_dir) / "gpu.json").write_text(json.dumps(values))

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        return _postprocess(inputs)


class CommandGPUIncrement(OperationDefinition):
    """Run copied measured resources in a separately prepared minimal Python tool."""

    name = "slurm_qualification_command_gpu"
    tool: ToolSpec = ToolSpec(executable="python")

    class InputRole(StrEnum):
        source = "source"

    class OutputRole(StrEnum):
        metrics = "metrics"

    inputs: ClassVar = {"source": InputSpec(artifact_type="metric", materialize=False)}
    outputs: ClassVar = {
        "metrics": OutputSpec(
            artifact_type="metric", derives_from={"inputs": ["source"]}
        )
    }

    class Params(BaseModel):
        """Canonical tool manifest and bounded child ownership evidence."""

        manifest: str = Field(description="Canonical multi-environment Pixi manifest.")
        hold: float = Field(
            default=0, ge=0, le=120, description="First input's pre-kernel delay."
        )
        marker: str = Field(default="", description="Owned process observation file.")

    params: Params

    def model_post_init(self, _context: Any) -> None:
        self.environment = Environment.from_pixi(
            Path(self.params.manifest), name="gpu-tool"
        )

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        prepared = _preprocess(inputs)
        for name in ("gpu_kernel.py", "gpu_program.py"):
            target = Path(inputs.preprocess_dir) / name
            target.write_bytes(Path(__file__).with_name(name).read_bytes())
            prepared[name] = str(target)
        return prepared

    def execute_command(self, inputs: dict[str, Any]) -> list[str]:
        command = [
            *self.tool.parts(),
            inputs["gpu_program.py"],
            inputs["source"],
            inputs["gpu_kernel.py"],
            "gpu.json",
            "--hold",
            str(self.params.hold),
        ]
        if self.params.marker:
            command += ["--marker", self.params.marker]
        return command

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        return _postprocess(inputs)


class GPUMerge(Merge):
    """Exercise the allocated capability in an ordinary top-level curator."""

    name = "slurm_qualification_gpu_merge"

    def execute_curator(
        self, inputs: CuratorInput, step_number: int, artifact_store: ArtifactStore
    ) -> PassthroughResult:
        measured = run_kernel()
        result = super().execute_curator(inputs, step_number, artifact_store)
        result.metadata["gpu_kernel"] = measured
        return result
