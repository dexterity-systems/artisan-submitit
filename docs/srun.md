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

Each `srun` command launches one Artisan worker with one task and rank. Use
`BatchStrategy.units_per_worker` to pack execution units and
`BatchStrategy.max_workers` to bound concurrent processes. Node and task-count
flags are not supported.

`RunnerResources.extra["slurm_gres"]` is the supported override for a
cluster-specific GPU request. When set, it takes precedence over
`RunnerResources.gpus`. `RunnerResources.extra["srun_args"]` appends explicit
flags before the Python worker command, but it cannot override nodes, tasks,
CPU, memory, GPU, or time settings. Use it only for unmanaged `srun` behavior,
such as `--exclusive`. Conflicting flags and unsupported extras fail fast.

Each process executes a cloudpickle-serialized callable and atomically renames
its completed result envelope into place. A nonzero worker exit or missing
envelope becomes an ordered failure result. Cancellation acts only on processes
created by that router. Process deadlines remain enforced while launch
concurrency is saturated, and result/log reads use bounded retries for shared
filesystem visibility delays.
