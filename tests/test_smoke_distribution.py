"""Installed distribution smoke follows the provider's declared core contract."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest
from packaging.requirements import InvalidRequirement

from scripts import smoke_distribution


def _metadata(
    monkeypatch: pytest.MonkeyPatch,
    requirements: list[str] | None,
    core_version: str = "0.2.0.dev0",
) -> None:
    def installed_requires(name: str) -> list[str] | None:
        assert name == "artisan-submitit"
        return requirements

    def installed_version(name: str) -> str:
        assert name == "dexterity-artisan"
        return core_version

    monkeypatch.setattr(smoke_distribution, "requires", installed_requires)
    monkeypatch.setattr(smoke_distribution, "version", installed_version)


@pytest.mark.parametrize(
    "core_version", ["0.2.0.dev0", "0.2.0.dev1", "0.2.0rc1", "0.2.0", "0.3.0"]
)
def test_supported_core_versions(
    monkeypatch: pytest.MonkeyPatch, core_version: str
) -> None:
    _metadata(monkeypatch, ["dexterity-artisan>=0.2.0.dev0"], core_version)

    assert smoke_distribution._check_core_requirement() == core_version


@pytest.mark.parametrize("core_version", ["0.1.2a5", "0.1.9"])
def test_older_core_is_rejected(
    monkeypatch: pytest.MonkeyPatch, core_version: str
) -> None:
    _metadata(monkeypatch, ["dexterity-artisan>=0.2.0.dev0"], core_version)

    with pytest.raises(ValueError, match="does not satisfy"):
        smoke_distribution._check_core_requirement()


def test_declared_upper_bound_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _metadata(monkeypatch, ["dexterity-artisan>=0.2.0.dev0,<0.3"], "0.3.0")

    with pytest.raises(ValueError, match="does not satisfy"):
        smoke_distribution._check_core_requirement()


@pytest.mark.parametrize(
    "requirements",
    [
        None,
        [],
        ["another-artisan>=0.2.0.dev0"],
        ['dexterity-artisan>=0.2.0.dev0; python_version < "2"'],
        ["dexterity-artisan>=0.2.0.dev0", "dexterity-artisan<1"],
    ],
)
def test_missing_or_ambiguous_core_requirement_is_rejected(
    monkeypatch: pytest.MonkeyPatch, requirements: list[str] | None
) -> None:
    _metadata(monkeypatch, requirements)

    with pytest.raises(ValueError, match="Expected one active dexterity-artisan"):
        smoke_distribution._check_core_requirement()


def test_normalized_core_name_and_active_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(
        monkeypatch,
        ["cloudpickle>=3", 'Dexterity_Artisan>=0.2.0.dev0; python_version >= "3.12"'],
    )

    assert smoke_distribution._check_core_requirement() == "0.2.0.dev0"


def test_malformed_requirement_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _metadata(monkeypatch, ["dexterity-artisan=>0.2.0.dev0"])

    with pytest.raises(InvalidRequirement):
        smoke_distribution._check_core_requirement()


def test_malformed_installed_version_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata(monkeypatch, ["dexterity-artisan>=0.2.0.dev0"], "not-a-version")

    with pytest.raises(ValueError, match="not-a-version"):
        smoke_distribution._check_core_requirement()


def test_missing_installed_core_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _metadata(monkeypatch, ["dexterity-artisan>=0.2.0.dev0"])

    def missing_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(smoke_distribution, "version", missing_version)

    with pytest.raises(PackageNotFoundError):
        smoke_distribution._check_core_requirement()
