#!/usr/bin/env python
"""Run Artisan through ``srun`` inside an existing allocation."""

from __future__ import annotations

from pathlib import Path

from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import PipelineManager

from artisan_submitit import SlurmIntraRunner


def main() -> None:
    """Generate locally, then transform via bounded srun processes."""
    root = Path(".artisan-srun-demo").resolve()
    pipeline = PipelineManager.create(
        name="srun_demo",
        delta_root=str(root / "delta"),
        staging_root=str(root / "staging"),
        working_root=str(root / "working"),
    )
    output = pipeline.output
    pipeline.run(DataGenerator, name="generate", params={"count": 8, "seed": 42})
    pipeline.run(
        DataTransformer,
        name="transform",
        inputs={"dataset": output("generate", "datasets")},
        params={"scale_factor": 0.5, "variants": 1, "seed": 100},
        step_runner=SlurmIntraRunner(srun_launch_concurrency=4),
        runner_resources={"cpus": 1, "memory_gb": 2, "time_limit": "00:10:00"},
    )
    summary = pipeline.finalize()
    print(summary)


if __name__ == "__main__":
    main()
