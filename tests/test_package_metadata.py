from __future__ import annotations

import tomllib
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

    assert artisan_requirements == ["dexterity-artisan>=0.1.2a6"]


def test_version_is_generated_from_vcs() -> None:
    assert artisan_submitit.__version__
    assert artisan_submitit.__version_tuple__
