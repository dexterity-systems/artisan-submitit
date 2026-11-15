"""Explicit CPU command fixture for a separately prepared tool environment."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from artisan.operations.base import OperationDefinition, PerArtifact
from artisan.schemas import (
    ArtifactResult,
    Environment,
    InputSpec,
    MetricArtifact,
    OutputSpec,
    PostprocessInput,
    PreprocessInput,
    ToolSpec,
)
from pydantic import BaseModel, Field


class FileToolIncrement(OperationDefinition):
    """Increment one metric through file-only CPU command execution."""

    name = "slurm_qualification_file_tool"
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
        """Existing minimal tool recipe and bounded cancellation observations."""

        manifest: str = Field(description="Canonical multi-environment Pixi manifest.")
        hold: float = Field(
            default=0, ge=0, le=120, description="First input's bounded tool delay."
        )
        marker: str = Field(default="", description="Owned process observation file.")

    params: Params

    def model_post_init(self, _context: Any) -> None:
        self.environment = Environment.from_pixi(
            Path(self.params.manifest), name="gpu-tool"
        )

    def preprocess(self, inputs: PreprocessInput) -> dict[str, Any]:
        (artifact,) = inputs.input_artifacts["source"]
        assert isinstance(artifact, MetricArtifact)
        source = Path(inputs.preprocess_dir) / "source.json"
        source.write_text(
            json.dumps(
                {"input_value": artifact.values["value"], "preprocess_pid": os.getpid()}
            )
        )
        program = Path(inputs.preprocess_dir) / "file_program.py"
        program.write_bytes(Path(__file__).with_name(program.name).read_bytes())
        return {"source": PerArtifact([str(source)]), "program": str(program)}

    def execute_command(self, inputs: dict[str, Any]) -> list[str]:
        command = [
            *self.tool.parts(),
            inputs["program"],
            inputs["source"],
            "file-tool.json",
            "--hold",
            str(self.params.hold),
        ]
        if self.params.marker:
            command += ["--marker", self.params.marker]
        return command

    def postprocess(self, inputs: PostprocessInput) -> ArtifactResult:
        path = next(
            Path(value)
            for value in inputs.file_outputs
            if Path(value).name == "file-tool.json"
        )
        values = json.loads(path.read_bytes())
        values["postprocess_pid"] = os.getpid()
        result = ArtifactResult()
        result.add_artifact(
            "metrics",
            MetricArtifact.draft(values, path.name),
            sources={"source": [inputs.ref("source", index=0)]},
        )
        return result
