#!/usr/bin/env python
"""Run a prepared, package-owned Artisan pipeline through sbatch."""

from __future__ import annotations

import argparse
from pathlib import Path

from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import PipelineManager, StepStatus
from artisan.schemas import Environment

from artisan_submitit import SlurmRunner


def main() -> None:
    """Prepare only when requested, then verify public pipeline outputs."""
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
        runner.prepare(DataGenerator)
        runner.prepare(DataTransformer)
    with PipelineManager.create(
        "sbatch_demo",
        str(root / "delta"),
        str(root / "staging"),
        working_root=str(root / "working"),
        default_step_runner=runner,
    ) as pipeline:
        source = pipeline.run(
            DataGenerator, name="generate", params={"count": 4, "seed": 42}
        )
        assert source.status == StepStatus.SUCCEEDED, source.error
        result = pipeline.run(
            DataTransformer,
            name="transform",
            inputs={"dataset": source.output("datasets")},
            params={"scale_factor": 0.5, "variants": 1, "seed": 100},
            batch_strategy={
                "artifacts_per_unit": 1,
                "units_per_worker": 2,
                "max_workers": 2,
            },
            runner_resources={"cpus": 1, "memory_gb": 2, "time_limit": "00:05:00"},
        )
        assert result.status == StepStatus.SUCCEEDED, result.error
        assert result.succeeded_count == 4
    print(pipeline.finalize())


if __name__ == "__main__":
    main()
