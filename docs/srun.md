# Intra-allocation execution

Use `SlurmIntraRunner` when the Artisan manager itself is already running in a
SLURM allocation:

```bash
salloc -n8 --mem=16G --time=00:30:00
python examples/srun_pipeline.py
```

```python
from artisan_submitit import SlurmIntraRunner

runner = SlurmIntraRunner()
pipeline.run(
    MyOperation,
    step_runner=runner,
    runner_resources={"cpus": 1, "memory_gb": 2, "time_limit": "00:10:00"},
    batch_strategy={"max_workers": 8},
)
```

The runner maps resources as follows:

| Artisan value | srun flag |
| --- | --- |
| `cpus` | `--cpus-per-task` |
| `memory_gb` | `--mem <N>G` |
| `gpus` | `--gres gpu:<N>` |
| `time_limit` | `--time` |

Each `srun` command launches a provider wrapper with one task and rank. The
wrapper uses Artisan's public local runner to own its Python worker and managed
commands within that allocation. Use
`BatchStrategy.units_per_worker` to pack execution units and
`BatchStrategy.max_workers` to bound concurrent processes. Node and task-count
flags are not supported.

`RunnerResources.extra["slurm_gres"]` is the supported override for a
cluster-specific GPU request. When set, it takes precedence over
`RunnerResources.gpus`. `RunnerResources.extra["srun_args"]` appends explicit
flags before the Python worker command, but it cannot override nodes, tasks,
CPU, memory, GPU, or time settings. Use it only for unmanaged `srun` behavior,
such as `--exclusive`. Asynchronous launch and allocation override options
(`--async`, `--jobid`, including abbreviations) are unsupported because the
provider must own a synchronous launcher in the original allocation.
Conflicting flags and unsupported extras fail fast.

Each process executes a cloudpickle-serialized callable and atomically publishes
its results and separate ownership evidence. A nonzero worker exit or missing
outcome becomes an ordered failure result. Process deadlines remain enforced
while launch concurrency is saturated.

## Cancellation inside an allocation

Cancellation first signals every owned worker through its shared control
directory. The wrapper asks Artisan to stop its Python worker and managed
commands. Confirmation requires valid ownership evidence and a reaped synchronous
launcher. An exited launcher alone is insufficient; missing or nested uncertain
evidence produces `UNKNOWN`.

After up to 10 seconds of cooperation, fallback termination targets only the
worker's validated numeric `job.step` receipt and its exact local launcher.
`cancel_grace_period` controls the subsequent TERM-to-KILL interval, capped by
the remaining shared 25-second cleanup budget. It neither shortens native
cooperation nor extends Artisan's 30-second confirmation allowance.

A missing receipt never permits cancellation of the allocation. Unrelated steps
remain untouched. If a pending launcher cannot provide ownership evidence before
the deadline, cleanup stops that launcher and reports `UNKNOWN`.

See [queue execution](slurm.md) for jobs that need their own allocation.
