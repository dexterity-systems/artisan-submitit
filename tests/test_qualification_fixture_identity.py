"""The trusted fixture rejects independent software and identity drift."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.qualification import fixture_identity as fixture


def _receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, int | str], dict[str, bytes]]:
    python, config = b"measured node interpreter", b"measured scheduler configuration"
    document = {
        "schema_version": 1,
        "image_id": "sha256:" + "a" * 64,
        "user": "qualification",
        "uid": 1001,
        "gid": 1001,
        "node_python_sha256": hashlib.sha256(python).hexdigest(),
        "slurm_conf_sha256": hashlib.sha256(config).hexdigest(),
        "slurmdbd_conf_sha256": "b" * 64,
    }
    receipt = tmp_path / "fixture.json"
    receipt.write_text(json.dumps(document))
    monkeypatch.setattr(fixture.os, "getuid", lambda: 1001)
    monkeypatch.setattr(fixture.os, "getgid", lambda: 1001)
    monkeypatch.setattr(
        fixture.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="qualification")
    )
    files = {"/usr/bin/python3": python, "/etc/slurm/slurm.conf": config}
    original = Path.open
    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *a, **kw: io.BytesIO(files[str(self)])
        if str(self) in files
        else original(self, *a, **kw),
    )
    return receipt, document, files


def test_prepared_identity_matches_without_invoking_node_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, document, _ = _receipt(tmp_path, monkeypatch)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert (
        fixture.expected_node_python(receipt, receipt_sha256=digest)
        == document["node_python_sha256"]
    )


@pytest.mark.parametrize("path", ["/usr/bin/python3", "/etc/slurm/slurm.conf"])
def test_drift_rejected_before_fault_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    receipt, _, files = _receipt(tmp_path, monkeypatch)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    files[path] = b"changed after fixture preparation"
    with pytest.raises(ValueError, match="fixture changed"):
        fixture.expected_node_python(receipt, receipt_sha256=digest)


def test_receipt_changed_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _, _ = _receipt(tmp_path, monkeypatch)
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        fixture.expected_node_python(receipt, receipt_sha256=digest)


@pytest.mark.parametrize(("field", "value"), [("uid", 1002), ("gid", 1002)])
def test_account_mismatch_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: int
) -> None:
    receipt, document, _ = _receipt(tmp_path, monkeypatch)
    document[field] = value
    receipt.write_text(json.dumps(document))
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="qualification account"):
        fixture.expected_node_python(receipt, receipt_sha256=digest)


def test_malformed_receipt_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _, _ = _receipt(tmp_path, monkeypatch)
    receipt.write_text('{"schema_version": 1}')
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpected fields"):
        fixture.expected_node_python(receipt, receipt_sha256=digest)


def test_missing_receipt_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fixture.expected_node_python(tmp_path / "absent.json", receipt_sha256="a" * 64)
