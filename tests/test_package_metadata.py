from __future__ import annotations

import tomllib
from pathlib import Path


def test_artisan_dependency_requires_runner_api_release() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    artisan_requirements = [
        requirement
        for requirement in project["dependencies"]
        if requirement.startswith("dexterity-artisan")
    ]

    assert artisan_requirements == ["dexterity-artisan>=0.1.2a6"]
