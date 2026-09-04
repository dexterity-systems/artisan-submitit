"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the opt-in switch for real scheduler tests."""
    parser.addoption(
        "--run-slurm",
        action="store_true",
        default=False,
        help="run tests that submit real SLURM jobs",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip scheduler tests unless explicitly enabled."""
    if config.getoption("--run-slurm"):
        return
    skip = pytest.mark.skip(reason="requires --run-slurm")
    for item in items:
        if "slurm" in item.keywords:
            item.add_marker(skip)
