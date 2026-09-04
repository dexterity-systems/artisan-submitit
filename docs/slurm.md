# Queue-based SLURM execution

Use `SlurmRunner` when the Artisan manager runs on a login or workflow node and
SLURM should allocate resources for each unit batch.

```python
from artisan_submitit import SlurmRunner

runner = SlurmRunner(
    slurm_partition="cpu",
    slurm_account="my-account",
)

step = pipeline.run(
    MyOperation,
    step_runner=runner,
    runner_resources={
        "cpus": 4,
        "memory_gb": 16,
        "gpus": 0,
        "time_limit": "02:00:00",
    },
    batch_strategy={
        "artifacts_per_unit": 4,
        "units_per_worker": 2,
        "max_workers": 32,
    },
)
```

The router packs units using Artisan's shared helper, submits each
scheduler-sized chunk within a Submitit `batch()` context, and retains the
returned job objects. It never reconstructs handles from job names.

Use `squeue` for live status and `sacct -j <job-id>` for terminal state. Logs
live beneath the pipeline-provided log directory when one is configured, or
`slurm_logs/` otherwise.
