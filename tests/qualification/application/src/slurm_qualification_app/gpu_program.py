"""Run the captured GPU kernel in an isolated selected tool environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    import packaging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("kernel", type=Path)
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
    payload.update(runpy.run_path(str(args.kernel))["run_kernel"]())
    payload.update(
        tool_pid=os.getpid(),
        packaging=packaging.__version__,
        activation=os.environ.get("GPU_TOOL_ACTIVATION"),
        literal=os.environ.get("QUALIFICATION_LITERAL"),
        selected=os.environ.get("QUALIFICATION_SELECTED"),
        cwd=str(Path.cwd()),
    )
    paths = {
        fields[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(fields := line.split(maxsplit=5)) == 6
        and Path(fields[5]).name.startswith(
            ("libcuda.so", "libnvidia-ptxjitcompiler.so")
        )
    }
    payload["driver_payloads"] = {
        Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    args.destination.write_text(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
