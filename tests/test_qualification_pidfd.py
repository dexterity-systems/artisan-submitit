"""The fixed fault helper signals only the exact owned throwaway bootstrap."""

from __future__ import annotations

import json
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

HELPER = Path(__file__).parent / "qualification/kill_bootstrap.py"


@contextmanager
def _child(root: Path, pack: Path) -> Iterator[subprocess.Popen[bytes]]:
    module = root / "artisan_submitit/bootstrap.py"
    module.parent.mkdir(parents=True)
    module.write_text("import time; time.sleep(15)\n")
    process = subprocess.Popen(
        ["/usr/bin/python3", "-I", str(module), "--pack", str(pack)]
    )
    try:
        deadline = time.monotonic() + 3
        while (
            str(module).encode()
            not in Path(f"/proc/{process.pid}/cmdline").read_bytes()
        ):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)


def test_pidfd_helper_checks_native_apis_without_signalling() -> None:
    evidence = json.loads(
        subprocess.check_output(["/usr/bin/python3", "-I", str(HELPER)], timeout=5)
    )
    assert evidence["native_pidfd"] is True
    assert len(evidence["interpreter_sha256"]) == len(evidence["helper_sha256"]) == 64
    assert "pid" not in evidence


def test_pidfd_helper_signals_exact_owned_child(tmp_path: Path) -> None:
    pack = tmp_path / "owned-pack"
    with _child(tmp_path / "child", pack) as child:
        evidence = json.loads(
            subprocess.check_output(
                ["/usr/bin/python3", "-I", str(HELPER), str(pack)], timeout=5
            )
        )
        assert evidence["pid"] == child.pid
        assert evidence["signal"] == "SIGKILL via pidfd"
        assert evidence["start_ticks"].isdecimal()
        assert child.wait(timeout=3) == -signal.SIGKILL


@pytest.mark.parametrize("duplicate", [False, True])
def test_pidfd_helper_rejects_foreign_or_ambiguous_pack(
    tmp_path: Path, duplicate: bool
) -> None:
    pack = tmp_path / "owned-pack"
    with (
        _child(tmp_path / "first", pack) as first,
        _child(
            tmp_path / "second", pack if duplicate else tmp_path / "other-pack"
        ) as second,
    ):
        target = pack if duplicate else tmp_path / "unowned-pack"
        result = subprocess.run(
            ["/usr/bin/python3", "-I", str(HELPER), str(target)],
            capture_output=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert b"exactly one" in result.stderr
        assert first.poll() is None
        assert second.poll() is None
