"""Local scientific checks for the packaged live-scheduler fixtures."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from artisan.orchestration.runner_api import compile_definition


@pytest.fixture
def application(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Source-only fixture test; installed scheduler tests reject this overlay."""
    monkeypatch.syspath_prepend(
        str(Path(__file__).parent / "qualification/application/src")
    )
    return importlib.import_module("slurm_qualification_app")


def test_wait_modes_keep_one_generative_invocation(application: ModuleType) -> None:
    for operation, mode in (
        (application.ObservableWait, "native"),
        (application.ManagedWait, "python_adapter"),
    ):
        plan = compile_definition(operation, params={"marker_root": "/tmp/probe"})
        assert plan.recipe.execution.mode == mode
        assert not plan.definition.inputs


def test_native_wait_records_real_readiness_and_output(
    application: ModuleType, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    operation = application.ObservableWait(
        params={"marker_root": str(markers), "duration": 0}
    )
    result = operation.execute_function(SimpleNamespace(execute_dir=tmp_path))
    assert result is None
    ready = json.loads((markers / "ready.json").read_text())
    assert ready["pid"] > 0
    assert (tmp_path / "wait_marker.csv").read_text().startswith("requested,actual\n")


def test_authored_failure_follows_readiness(
    application: ModuleType, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    operation = application.ObservableWait(
        params={"marker_root": str(markers), "fail": True}
    )
    with pytest.raises(RuntimeError, match="intentional scheduler operation failure"):
        operation.execute_function(SimpleNamespace(execute_dir=tmp_path))
    assert (markers / "ready.json").exists()
    assert not (tmp_path / "wait_marker.csv").exists()


def test_packed_wait_uses_file_shaped_inputs_and_exact_declared_parent(
    application: ModuleType, tmp_path: Path
) -> None:
    from artisan.schemas import MetricArtifact

    operation = application.PackedWait(
        params={"marker_root": str(tmp_path / "markers"), "duration": 0}
    )
    source = MetricArtifact.draft({"value": 1}, "input.json").finalize()
    prepared = operation.preprocess(
        SimpleNamespace(input_artifacts={"source": [source]}, preprocess_dir=tmp_path)
    )
    operation.execute_function(SimpleNamespace(inputs=prepared, execute_dir=tmp_path))
    values = json.loads((tmp_path / "result.json").read_text())
    assert values["value"] == 2
    assert (tmp_path / "markers/1/ready.json").is_file()
    plan = compile_definition(
        application.PackedWait,
        params={"marker_root": str(tmp_path / "markers"), "duration": 0},
    )
    assert plan.recipe.execution.mode == "python_adapter"
    assert plan.definition.outputs["result"].derives_from == {"inputs": ["source"]}


def test_managed_wait_satisfies_actual_file_only_cli_contract(
    application: ModuleType, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    app_source = Path(application.__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(app_source), environment.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "artisan.cli",
            "op",
            "run",
            "slurm_qualification_app:ManagedWait",
            "--params",
            json.dumps({"marker_root": str(markers), "duration": 0}),
            "--inputs",
            "{}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (markers / "ready.json").is_file()
    assert (tmp_path / "wait_marker.csv").read_text().startswith("requested,actual\n")
