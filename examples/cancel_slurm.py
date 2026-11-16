#!/usr/bin/env python
"""Submit an owned scheduler worker and request cancellation with Ctrl+C."""

from __future__ import annotations

import argparse
from pathlib import Path

from artisan.operations.examples import Wait
from artisan.orchestration import PipelineManager
from artisan.schemas import Environment

from artisan_submitit import SlurmRunner


def main() -> None:
    """Use the pipeline signal owner and retain any unresolved cleanup evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", default="worker")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    runner = SlurmRunner(
        environment=Environment.from_pixi(
            args.manifest.resolve(), name=args.environment
        ),
        cache_root=args.cache_root.resolve(),
        log_folder=root / "scheduler",
        slurm_partition=args.partition,
    )
    if args.prepare:
        runner.prepare(Wait)
    with PipelineManager.create(
        "cancel_demo",
        str(root / "delta"),
        str(root / "staging"),
        working_root=str(root / "working"),
        default_step_runner=runner,
    ) as pipeline:
        pipeline.submit(
            Wait,
            params={"duration": 60},
            runner_resources={"cpus": 1, "memory_gb": 1, "time_limit": "00:02:00"},
        )
        print(
            "Press Ctrl+C to request cancellation; cleanup uncertainty remains UNKNOWN."
        )
        print(pipeline.finalize())


if __name__ == "__main__":
    main()
