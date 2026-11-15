"""Hostile ambient startup controls do not reach measured worker Python."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from artisan_submitit.launcher import bootstrap_command


@pytest.mark.parametrize("gpu", [None, "", "0"])
def test_fixed_prelude_preserves_literal_allocation_and_scrubs_ambient(
    tmp_path: Path, gpu: str | None
) -> None:
    marker = tmp_path / "unexpected-startup"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"touch {marker}\n")
    selected = SimpleNamespace(
        local=SimpleNamespace(system_env=Path("/usr/bin/env")),
        scheduler=SimpleNamespace(
            executables=(SimpleNamespace(path=Path("/usr/bin/srun")),),
            bootstrap_shell=SimpleNamespace(path=Path("/bin/bash")),
        ),
    )
    command = bootstrap_command(
        selected,
        tmp_path,
        [
            sys.executable,
            "-I",
            "-c",
            "import json,os;print(json.dumps(dict(os.environ)))",
        ],
    )
    literal = f"literal $(touch {marker}) $HOME with spaces"
    environment = {
        **os.environ,
        "BASH_ENV": str(startup),
        "ENV": str(startup),
        "BASH_FUNC_exec%%": f"() {{ touch {marker}; }}",
        "SHELLOPTS": "xtrace",
        "PYTHONPATH": "/unqualified",
        "PYTHONHOME": "/unqualified",
        "LD_LIBRARY_PATH": "/unqualified",
        "LD_AUDIT": "",
        "AMBIENT_CREDENTIAL": "not-forwarded",
        "SLURM_JOB_ID": "41",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODEID": "0",
        "SLURM_RESTART_COUNT": "2",
        "SLURM_JOB_NODELIST": literal,
    }
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=True
    )
    captured = json.loads(result.stdout)
    assert captured["SLURM_JOB_NODELIST"] == literal
    assert captured["SLURM_RESTART_COUNT"] == "2"
    assert captured["SLURM_JOB_NUM_NODES"] == "1"
    assert captured.get("CUDA_VISIBLE_DEVICES") == gpu
    assert ("CUDA_VISIBLE_DEVICES" in captured) is (gpu is not None)
    assert captured["HOME"] == str(tmp_path / "bootstrap/home")
    assert not any(name.startswith(("PYTHON", "LD_", "BASH_")) for name in captured)
    assert "AMBIENT_CREDENTIAL" not in captured
    assert "ENV" not in captured
    assert not marker.exists()
