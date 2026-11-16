"""Signal one exact owned test bootstrap through the node Python's native pidfd API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kill(pack: Path) -> dict[str, object]:
    if not pack.is_absolute() or pack.resolve() != pack:
        raise ValueError("Fault requires an exact absolute owned pack path")
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            command = (process / "cmdline").read_bytes().split(b"\0")
            start = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (
            len(command) > 5
            and command[1] == b"-I"
            and command[2].endswith(b"/artisan_submitit/bootstrap.py")
            and command[3:5] == [b"--pack", os.fsencode(pack)]
        ):
            matches.append((process, command, start))
    if len(matches) != 1:
        raise ValueError("Fault requires exactly one matching bootstrap")
    process, expected, start = matches[0]
    pid = int(process.name)
    descriptor = os.pidfd_open(pid)
    try:
        assert process.stat().st_uid == os.getuid()
        assert (process / "cmdline").read_bytes().split(b"\0") == expected
        assert (process / "stat").read_text().rsplit(")", 1)[1].split()[19] == start
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
    finally:
        os.close(descriptor)
    return {
        "pid": pid,
        "start_ticks": start,
        "signal": "SIGKILL via pidfd",
        "command_sha256": hashlib.sha256(b"\0".join(expected)).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, nargs="?")
    args = parser.parse_args()
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Native pidfd APIs are required for this fault fixture")
    evidence = {
        "native_pidfd": True,
        "interpreter_sha256": _digest(Path(sys.executable)),
        "helper_sha256": _digest(Path(__file__)),
    }
    if args.pack is not None:
        evidence.update(_kill(args.pack))
    sys.stdout.write(json.dumps(evidence) + "\n")


if __name__ == "__main__":
    main()
