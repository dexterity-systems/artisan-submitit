#!/usr/bin/env python
"""Cancel a SLURM-backed Artisan pipeline with Ctrl+C."""

from __future__ import annotations

import os
from pathlib import Path

from artisan.operations.examples import Wait
from artisan.orchestration import PipelineManager

from artisan_submitit import SlurmRunner


def main() -> None:
    """Submit slow work and rely on exact-handle cancellation."""
    root = Path(".artisan-slurm-cancel-demo").resolve()
    pipeline = PipelineManager.create(
        name="cancel_demo_slurm",
        delta_root=str(root / "delta"),
        staging_root=str(root / "staging"),
        working_root=str(root / "working"),
    )
    runner = SlurmRunner(slurm_partition=os.environ.get("SLURM_TEST_PARTITION", "cpu"))
    for index in range(6):
        pipeline.submit(
            Wait,
            name=f"wait_{index}",
            params={"duration": 60},
            step_runner=runner,
        )
    print("Press Ctrl+C to request pipeline cancellation.")
    print(pipeline.finalize())


if __name__ == "__main__":
    main()
