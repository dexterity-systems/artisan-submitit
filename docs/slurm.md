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

## Resource ownership

Each array entry launches one Artisan worker as a single scheduler task and
rank. Use `BatchStrategy` to control how execution units are packed and how
many workers run at once. Multi-node and multi-rank worker topologies are not
supported.

Set portable CPU, memory, GPU, and time requirements through
`RunnerResources`. The runner rejects equivalent low-level Submitit parameters
and managed flags hidden in `slurm_srun_args` or
`slurm_additional_parameters`, including node and task counts. Genuine
scheduler policy such as partition, account, constraint, and QoS remains
provider configuration.

`slurm_gres` is the supported exception for clusters that need a custom GPU
resource string. When set on the runner or in `RunnerResources.extra`, it
overrides `RunnerResources.gpus`.

## Array waves

The router packs units using Artisan's shared helper. It detects the cluster's
`MaxArraySize` and submits at most that many packed batches in one Submitit
array. If more batches remain, it collects the current array before submitting
the next scheduler-sized wave. Because waves do not overlap,
`BatchStrategy.max_workers` caps the whole step rather than each array
independently.

The router retains every returned job object and never reconstructs handles
from job names.

Use `squeue` for live status and `sacct -j <job-id>` for terminal state. Logs
live beneath the pipeline-provided log directory when one is configured, or
`slurm_logs/` otherwise.
