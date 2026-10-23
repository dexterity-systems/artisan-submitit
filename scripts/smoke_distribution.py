"""Qualify installed wheels with real workers and a local srun transport shim."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _check_installed() -> None:
    """Exercise the installed provider without importing either source checkout."""
    from importlib.metadata import version

    import artisan
    from artisan.operations.examples import CsvHead, DataGenerator
    from artisan.orchestration import PipelineManager
    from artisan.visualization import PipelineTimings

    import artisan_submitit
    from artisan_submitit import SlurmIntraRunner, SlurmRunner

    for module in (artisan, artisan_submitit):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix))
    assert artisan.__version__ == version("dexterity-artisan") == "0.2.0"
    assert artisan_submitit.__version__ == version("artisan-submitit")
    assert SlurmRunner().name == "slurm"
    _install_local_srun(Path.cwd())
    pipeline = PipelineManager.create(
        name="installed-provider-smoke",
        delta_root=str(Path("delta").resolve()),
        staging_root=str(Path("staging").resolve()),
        working_root=str(Path("working").resolve()),
        default_step_runner=SlurmIntraRunner(poll_interval=0.05),
    )
    try:
        generated = pipeline.run(DataGenerator, name="generate", params={"count": 2})
        transformed = pipeline.run(
            CsvHead,
            inputs={"dataset": pipeline.output("generate", "datasets")},
            params={"rows": 2},
            batch_strategy={"artifacts_per_unit": 1, "units_per_worker": 2},
        )
    finally:
        summary = pipeline.finalize()
    assert summary["overall_success"], summary
    assert generated.succeeded_count == 1
    assert transformed.succeeded_count == 2
    timings = PipelineTimings.from_delta(str(Path("delta").resolve()))
    assert timings.command_timings(1).height == 2
    sys.stdout.write(
        json.dumps(
            {
                "core": artisan.__version__,
                "provider": artisan_submitit.__version__,
                "summary": summary,
            },
            default=str,
        )
        + "\n"
    )


def _install_local_srun(root: Path) -> None:
    """Replace scheduler admission only; execute the installed worker normally."""
    binary = root / "bin"
    binary.mkdir()
    shim = binary / "srun"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "start = sys.argv.index('-u') - 1\n"
        "os.environ['SLURM_STEP_ID'] = '1'\n"
        "os.execv(sys.argv[start], sys.argv[start:])\n"
    )
    shim.chmod(0o755)
    os.environ["PATH"] = str(binary) + os.pathsep + os.environ["PATH"]
    os.environ["SLURM_JOB_ID"] = "123"
    os.environ["ARTISAN_LOAD_MODULES"] = ""


def _qualify(provider_wheel: Path, core_wheel: Path | None) -> None:
    """Install a candidate pair or the actual published core in a fresh venv."""
    with tempfile.TemporaryDirectory(prefix="artisan-submitit-dist-") as folder:
        root = Path(folder)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = str(root / "venv/bin/python")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        core = str(core_wheel.resolve()) if core_wheel else "dexterity-artisan==0.2.0"
        subprocess.run(
            [
                python,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--index-url",
                "https://pypi.org/simple",
                core,
                str(provider_wheel.resolve()),
            ],
            check=True,
            cwd=root,
            env=environment,
        )
        subprocess.run(
            [python, "-I", "-m", "pip", "check"], check=True, cwd=root, env=environment
        )
        script = root / "smoke_distribution.py"
        shutil.copyfile(__file__, script)
        subprocess.run(
            [python, "-I", str(script), "--check-installed"],
            check=True,
            cwd=root,
            env=environment,
            timeout=120,
        )


def main() -> None:
    """Use a core wheel only for candidate CI; releases require published 0.2.0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider_wheel", type=Path, nargs="?")
    parser.add_argument("--core-wheel", type=Path)
    parser.add_argument(
        "--check-installed", action="store_true", help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.check_installed:
        _check_installed()
    elif arguments.provider_wheel is None:
        parser.error("provider_wheel is required")
    else:
        _qualify(arguments.provider_wheel, arguments.core_wheel)


if __name__ == "__main__":
    main()
