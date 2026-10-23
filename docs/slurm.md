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

Each array entry runs a provider wrapper, one Artisan local worker, and its
command supervisor inside a single scheduler task and rank. The extra processes
share the allocated resources; a packed batch still executes sequentially.
Use `BatchStrategy` to control how execution units are packed and how
many workers run at once. Multi-node and multi-rank worker topologies are not
supported.

Set portable CPU, memory, GPU, and time requirements through
`RunnerResources`. The runner rejects equivalent low-level Submitit parameters
and managed flags hidden in `slurm_srun_args` or
`slurm_additional_parameters`, including node and task counts. Genuine
scheduler policy such as partition, account, constraint, and QoS remains
provider configuration.

`SlurmRunner` forces Submitit's SLURM executor. It fails immediately when
SLURM commands are unavailable rather than falling back to local execution.
Submitit's deprecated unprefixed resource, topology, srun, additional
parameter, and array-parallelism aliases are rejected.

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

## Cancellation and evidence

The shared control directory carries a persistent cancellation request to each
worker. Artisan's public local runner stops the Python worker and its managed
commands, then publishes ownership evidence. Collection requires that evidence
and a terminal exact `sacct` record. Keep SLURM accounting available to the
manager. Records must contain untruncated IDs and timestamps in the requested
format; stale records from recycled job IDs are ignored.

A queued job can also be confirmed cancelled when accounting proves that it
never started. `CANCELLED` alone is insufficient. Missing or ambiguous evidence
produces `UNKNOWN`, even if `scancel` succeeded. Nested compute cancellation
uncertainty is preserved.

Some clusters assign a start timestamp to jobs cancelled while held, or retain
only an aggregate accounting row for an unstarted array. Those records cannot
prove exact-task non-start, so this fallback returns `UNKNOWN`. Running workers
can still confirm cancellation through their native ownership outcome.

Cancellation closes future array admission, gives native owners up to 10 seconds
to cooperate, and retries cancellation of exact owned jobs within a shared
25-second budget. Artisan allows 30 seconds for confirmation. A stuck submission
or filesystem call may continue after `UNKNOWN`; late promoted jobs retain the
cancellation request and receive best-effort exact cleanup.

Use [intra-allocation execution](srun.md) when you already own an allocation.
