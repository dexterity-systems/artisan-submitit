# Intra-allocation execution

Use `SlurmIntraRunner` when the Artisan manager itself is already running in a
SLURM allocation:

```bash
salloc -n8 --mem=16G --time=00:30:00
python examples/srun_pipeline.py
```

```python
from artisan_submitit import SlurmIntraRunner

runner = SlurmIntraRunner(srun_launch_concurrency=8)
pipeline.run(
    MyOperation,
    step_runner=runner,
    runner_resources={"cpus": 1, "memory_gb": 2, "time_limit": "00:10:00"},
)
```

The runner maps resources as follows:

| Artisan value | srun flag |
| --- | --- |
| `cpus` | `--cpus-per-task` |
| `memory_gb` | `--mem <N>G` |
| `gpus` | `--gres gpu:<N>` |
| `time_limit` | `--time` |

`RunnerResources.extra["slurm_gres"]` overrides the generated GPU request.
`RunnerResources.extra["srun_args"]` appends a list of explicit flags before
the Python worker command. Other extras fail fast because `srun` does not use
Submitit's executor parameter translation.

Each process executes a cloudpickle-serialized callable and atomically renames
its completed result envelope into place. A nonzero worker exit or missing
envelope becomes an ordered failure result. Cancellation acts only on processes
created by that router.
