"""Verify exact shared-wheel reuse before any profile preparation or installation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .qualification.prepare_profile import _copy_verified_wheel, _shared_core_wheel


def test_shared_core_wheel_checks_distribution_version_and_exact_bytes(
    tmp_path: Path,
) -> None:
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="example-core"\nversion="1.0.0"\n'
    )
    source = tmp_path / "example_core-1.0.0-py3-none-any.whl"
    source.write_bytes(b"synthetic wheel copy fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "wheels"
    destination.mkdir()
    assert _shared_core_wheel(source, digest, core, destination) == {
        source.name: digest
    }
    assert (destination / source.name).read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="SHA256"):
        _shared_core_wheel(source, "0" * 64, core, destination)
    foreign = tmp_path / "other_core-1.0.0-py3-none-any.whl"
    foreign.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="distribution and version"):
        _shared_core_wheel(foreign, digest, core, destination)


def test_changed_copy_is_rejected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.whl", tmp_path / "copied.whl"
    source.write_bytes(b"original wheel")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "tests.qualification.prepare_profile.shutil.copyfile",
        lambda _source, target: target.write_bytes(b"changed copy"),
    )
    with pytest.raises(ValueError, match="Copied wheel differs"):
        _copy_verified_wheel(source, destination, digest)
    assert destination.read_bytes() == b"changed copy"
