"""Run a bounded CPU calculation and record the actual command environment."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _writable(source: Path) -> bool:
    """Check the kernel's write-open decision without changing input bytes."""
    try:
        descriptor = os.open(source, os.O_WRONLY)
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise
        return False
    os.close(descriptor)
    return True


def main() -> None:
    import packaging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hold", type=float, default=0)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()
    if not 0 <= args.hold <= 120:
        raise ValueError("Qualification hold must be between zero and 120 seconds")
    payload = json.loads(args.source.read_bytes())
    if args.marker is not None and payload["input_value"] == 1:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(json.dumps({"parent": os.getpid(), "child": child.pid}))
    time.sleep(args.hold if payload["input_value"] == 1 else 0)
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
    )
    payload.update(
        output=payload["input_value"] + 1,
        tool_pid=os.getpid(),
        executable=sys.executable,
        uid=os.getuid(),
        namespaces={
            name: os.readlink(f"/proc/self/ns/{name}")
            for name in ("mnt", "pid", "user", "ipc", "uts", "net")
        },
        effective_capabilities=status["CapEff"].strip(),
        source_writable=_writable(args.source),
        packaging=packaging.__version__,
        activation=os.environ.get("GPU_TOOL_ACTIVATION"),
        literal=os.environ.get("QUALIFICATION_LITERAL"),
        selected=os.environ.get("QUALIFICATION_SELECTED"),
        cwd=str(Path.cwd()),
    )
    args.destination.write_text(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
