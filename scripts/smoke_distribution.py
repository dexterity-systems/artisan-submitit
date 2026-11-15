"""Check paired ordinary-wheel metadata and imports without emulating a scheduler."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from importlib.metadata import requires, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _check_core_requirement() -> str:
    """Require the installed core to satisfy the provider's declared dependency."""
    requirements = []
    for value in requires("artisan-submitit") or []:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) == "dexterity-artisan" and (
            requirement.marker is None or requirement.marker.evaluate()
        ):
            requirements.append(requirement)
    if len(requirements) != 1:
        message = (
            "Expected one active dexterity-artisan dependency in provider metadata"
        )
        raise ValueError(message)
    requirement = requirements[0]
    core_version = version("dexterity-artisan")
    if core_version not in requirement.specifier:
        message = (
            f"Installed dexterity-artisan {core_version} does not satisfy {requirement}"
        )
        raise ValueError(message)
    return core_version


def _check_installed() -> None:
    """Check ordinary wheel imports; live scheduler qualification is separate."""
    import artisan
    from artisan.orchestration import PipelineManager
    from artisan.orchestration.runner_api import LocalRunner
    from artisan.schemas import Environment

    import artisan_submitit
    from artisan_submitit import SlurmIntraRunner, SlurmRunner

    for module in (artisan, artisan_submitit):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix))
    assert artisan.__version__ == _check_core_requirement()
    assert artisan_submitit.__version__ == version("artisan-submitit")
    assert version("submitit") == "1.5.4"
    assert issubclass(SlurmRunner, LocalRunner)
    assert issubclass(SlurmIntraRunner, LocalRunner)
    assert callable(PipelineManager.create)
    source = Environment.from_pixi(Path.cwd() / "pixi.toml")
    assert SlurmRunner(environment=source).name == "slurm"
    assert SlurmIntraRunner(environment=source).name == "slurm_intra"
    sys.stdout.write(
        json.dumps(
            {
                "core": artisan.__version__,
                "provider": artisan_submitit.__version__,
                "submitit": version("submitit"),
                "scope": "ordinary-wheel imports and metadata only",
            }
        )
        + "\n"
    )


def _qualify(provider_wheel: Path, core_wheel: Path | None) -> None:
    """Install a candidate pair or the actual published core in a fresh venv."""
    with tempfile.TemporaryDirectory(prefix="artisan-submitit-dist-") as folder:
        root = Path(folder)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = str(root / "venv/bin/python")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        candidates = [str(provider_wheel.resolve())]
        if core_wheel is not None:
            candidates.append(str(core_wheel.resolve()))
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
                *candidates,
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
    """Use a core wheel for candidate CI; otherwise resolve the published dependency."""
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
