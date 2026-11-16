"""Source-only CPU file-tool checks; scheduled SIF execution is a separate gate."""

from __future__ import annotations

import errno
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from artisan.schemas import (
    FrozenInputContext,
    MetricArtifact,
    PostprocessInput,
    PreprocessInput,
)


@pytest.fixture
def file_application(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).parent / "qualification/application/src")
    )
    return importlib.import_module("slurm_qualification_app.file_tool")


def test_file_tool_runs_literal_cli_and_preserves_file_result_and_exact_parent(
    tmp_path: Path, file_application: ModuleType
) -> None:
    source = MetricArtifact.draft({"value": 41}, "input.json").finalize()
    operation = file_application.FileToolIncrement(
        params={"manifest": str(tmp_path / "pixi.toml")}
    )
    pre, execute, post = (tmp_path / name for name in ("pre", "execute", "post"))
    for path in (pre, execute, post):
        path.mkdir()
    prepared = operation.preprocess(
        PreprocessInput(input_artifacts={"source": [source]}, preprocess_dir=str(pre))
    )
    assert operation.environment.name == "gpu-tool"
    assert (
        Path(prepared["program"]).read_bytes()
        == Path(file_application.__file__).with_name("file_program.py").read_bytes()
    )
    arguments = dict(prepared, source=prepared["source"][0])
    original_input = Path(arguments["source"]).read_bytes()
    command = operation.execute_command(arguments)
    assert command[0] == "python"
    literal = "literal:$(touch should-not-exist)"
    completed = subprocess.run(
        [sys.executable, *command[1:]],
        cwd=execute,
        env=dict(
            os.environ,
            GPU_TOOL_ACTIVATION="prepared-native-gpu",
            QUALIFICATION_LITERAL=literal,
            QUALIFICATION_SELECTED="selected value",
        ),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (execute / "should-not-exist").exists()
    assert Path(arguments["source"]).read_bytes() == original_input
    result = operation.postprocess(
        PostprocessInput(
            memory_outputs=None,
            file_outputs=[str(execute / "file-tool.json")],
            input_artifacts={"source": [source]},
            step_number=0,
            postprocess_dir=str(post),
            _context=FrozenInputContext("file-unit", {"source": [source]}),
        )
    )
    values = result.artifacts["metrics"][0].values
    assert values["output"] == 42
    assert values["literal"] == literal
    assert values["selected"] == "selected value"
    assert values["activation"] == "prepared-native-gpu"
    assert values["preprocess_pid"] == values["postprocess_pid"] == os.getpid()
    assert values["tool_pid"] != os.getpid()
    assert values["executable"] == sys.executable
    assert set(values["namespaces"]) == {"mnt", "pid", "user", "ipc", "uts", "net"}
    assert int(values["effective_capabilities"], 16) >= 0
    (parent,) = result.lineage["metrics"]
    assert parent.source_artifact_id == source.artifact_id
    assert parent.source_input_index == 0


def test_file_tool_rejects_multiple_inputs_in_one_unit(
    tmp_path: Path, file_application: ModuleType
) -> None:
    source = MetricArtifact.draft({"value": 41}, "input.json").finalize()
    operation = file_application.FileToolIncrement(
        params={"manifest": str(tmp_path / "pixi.toml")}
    )
    with pytest.raises(ValueError, match="too many values"):
        operation.preprocess(
            PreprocessInput(
                input_artifacts={"source": [source, source]},
                preprocess_dir=str(tmp_path),
            )
        )


def test_file_tool_survivor_preserves_detached_child_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_application: ModuleType
) -> None:
    program = importlib.import_module("slurm_qualification_app.file_program")
    source, output, marker = (
        tmp_path / name
        for name in ("source.json", "output.json", "pid with spaces.json")
    )
    source.write_text(json.dumps({"input_value": 2}))
    original = json.dumps({"parent": 101, "child": 102})
    marker.write_text(original)
    spawn = Mock(side_effect=AssertionError("Survivor must not create another child"))
    monkeypatch.setattr(program.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        sys,
        "argv",
        ["file_program", str(source), str(output), "--marker", str(marker)],
    )
    program.main()
    assert marker.read_text() == original
    assert json.loads(output.read_bytes())["output"] == 3
    spawn.assert_not_called()


def test_file_tool_records_readonly_open_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_application: ModuleType
) -> None:
    program = importlib.import_module("slurm_qualification_app.file_program")
    source = tmp_path / "source.json"
    source.write_text('{"input_value": 41}')
    before = source.read_bytes()
    denied = Mock(side_effect=OSError(errno.EROFS, "Read-only file system"))
    monkeypatch.setattr(program.os, "open", denied)
    assert program._writable(source) is False
    assert source.read_bytes() == before
    denied.assert_called_once_with(source, os.O_WRONLY)


@pytest.mark.parametrize("hold", ["-1", "121", "nan"])
def test_file_tool_cli_rejects_unbounded_hold_before_output(
    tmp_path: Path, file_application: ModuleType, hold: str
) -> None:
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    source.write_text('{"input_value": 41}')
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(file_application.__file__).with_name("file_program.py")),
            str(source),
            str(output),
            "--hold",
            hold,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert "between zero and 120 seconds" in completed.stderr
    assert not output.exists()
