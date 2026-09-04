#!/usr/bin/env python
"""Run a mixed local/SLURM Artisan pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import PipelineManager

from artisan_submitit import SlurmRunner


def main() -> None:
    """Generate locally, then transform through Submitit-managed arrays."""
    root = Path(".artisan-slurm-demo").resolve()
    pipeline = PipelineManager.create(
        name="slurm_demo",
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
        step_runner=SlurmRunner(
            slurm_partition=os.environ.get("SLURM_TEST_PARTITION", "cpu")
        ),
        batch_strategy={"artifacts_per_unit": 1, "units_per_worker": 2},
    )
    summary = pipeline.finalize()
    print(summary)


if __name__ == "__main__":
    main()
