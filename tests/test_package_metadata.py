from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import artisan_submitit


def test_artisan_dependency_requires_runner_api_release() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    artisan_requirements = [
        requirement
        for requirement in project["dependencies"]
        if requirement.startswith("dexterity-artisan")
    ]

    assert artisan_requirements == ["dexterity-artisan>=0.2.0"]


def test_version_matches_declared_and_installed_metadata() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    assert artisan_submitit.__version__ == version("artisan-submitit")
    assert artisan_submitit.__version__ == project["version"]
    assert "version" not in project.get("dynamic", [])
