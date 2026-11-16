"""Source-only lifecycle tests; the CUDA kernel is qualified separately on hardware."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import compile_definition
from artisan.schemas import (
    ExecuteInput,
    FrozenInputContext,
    MetricArtifact,
    PostprocessInput,
    PreprocessInput,
)


@pytest.fixture
def gpu_application(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).parent / "qualification/application/src")
    )
    return importlib.import_module("slurm_qualification_app.gpu_operations")


def test_native_gpu_fixture_preserves_file_result_and_exact_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu_application: ModuleType,
) -> None:
    source = MetricArtifact.draft({"value": 41}, "input.json").finalize()
    operation = gpu_application.NativeGPUIncrement()
    monkeypatch.setattr(
        gpu_application, "run_kernel", lambda: {"output": 42, "device_count": 1}
    )
    pre, execute, post = (tmp_path / name for name in ("pre", "execute", "post"))
    for path in (pre, execute, post):
        path.mkdir()
    prepared = operation.preprocess(
        PreprocessInput(input_artifacts={"source": [source]}, preprocess_dir=str(pre))
    )
    assert (
        operation.execute_function(
            ExecuteInput(inputs=prepared, execute_dir=str(execute))
        )
        is None
    )
    result = operation.postprocess(
        PostprocessInput(
            memory_outputs=None,
            file_outputs=[str(execute / "gpu.json")],
            input_artifacts={"source": [source]},
            step_number=0,
            postprocess_dir=str(post),
            _context=FrozenInputContext("gpu-unit", {"source": [source]}),
        )
    )
    assert result.artifacts["metrics"][0].values["output"] == 42
    (parent,) = result.lineage["metrics"]
    assert parent.source_artifact_id == source.artifact_id
    assert parent.source_input_index == 0
    assert (
        compile_definition(gpu_application.NativeGPUIncrement).recipe.execution.mode
        == "native"
    )


def test_gpu_fixture_rejects_multiple_inputs_within_one_unit(
    tmp_path: Path,
    gpu_application: ModuleType,
) -> None:
    source = MetricArtifact.draft({"value": 41}, "input.json").finalize()
    with pytest.raises(ValueError, match="too many values"):
        gpu_application.NativeGPUIncrement().preprocess(
            PreprocessInput(
                input_artifacts={"source": [source, source]},
                preprocess_dir=str(tmp_path),
            )
        )


def test_gpu_command_copies_exact_resources_and_builds_literal_argv(
    tmp_path: Path,
    gpu_application: ModuleType,
) -> None:
    source = MetricArtifact.draft({"value": 1}, "input.json").finalize()
    operation = gpu_application.CommandGPUIncrement(
        params={
            "manifest": str(tmp_path / "pixi.toml"),
            "hold": 8,
            "marker": str(tmp_path / "marker with spaces.json"),
        }
    )
    prepared = operation.preprocess(
        PreprocessInput(
            input_artifacts={"source": [source]}, preprocess_dir=str(tmp_path)
        )
    )
    arguments = dict(prepared, source=prepared["source"][0])
    command = operation.execute_command(arguments)
    assert command[0] == "python"
    assert command[-1] == str(tmp_path / "marker with spaces.json")
    assert operation.environment.name == "gpu-tool"
    package = Path(gpu_application.__file__).parent
    for name in ("gpu_kernel.py", "gpu_program.py"):
        assert Path(prepared[name]).read_bytes() == (package / name).read_bytes()
    assert json.loads(Path(arguments["source"]).read_bytes())["input_value"] == 1


def test_surviving_gpu_tool_does_not_replace_cancelled_unit_pid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu_application: ModuleType,
) -> None:
    program = importlib.import_module("slurm_qualification_app.gpu_program")
    source, kernel, output, marker = (
        tmp_path / name for name in ("source.json", "kernel.py", "out.json", "pid.json")
    )
    source.write_text(json.dumps({"input_value": 2}))
    kernel.write_text("def run_kernel(): return {'output': 42}\n")
    original = json.dumps({"parent": 101, "child": 102})
    marker.write_text(original)
    spawn = Mock(side_effect=AssertionError("Survivor must not create another child"))
    monkeypatch.setattr(program.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_program", str(source), str(kernel), str(output), "--marker", str(marker)],
    )
    program.main()
    assert marker.read_text() == original
    assert json.loads(output.read_bytes())["output"] == 42
    spawn.assert_not_called()
