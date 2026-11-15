# Queue-based SLURM execution

Use `SlurmRunner` when each worker pack needs a scheduler allocation. Follow the
[complete preparation example](../README.md#prepare-and-run) first. The selected
environment must contain the installed application, core, provider and pinned
Submitit packages. Execution never installs missing software.

```python
# Continue with the prepared runner and pipeline from the README.
step = pipeline.run(
    DataTransformer,
    inputs={"dataset": generated.output("datasets")},
    step_runner=runner,
    runner_resources={"cpus": 4, "memory_gb": 16, "gpus": 0, "time_limit": "02:00:00"},
    batch_strategy={"artifacts_per_unit": 4, "units_per_worker": 2, "max_workers": 8},
)
assert step.status == StepStatus.SUCCEEDED, step.error
```

Each pack is one `AutoExecutor.submit`, one node and one task. The provider uses
Submitit's Slurm executor explicitly. `max_workers` bounds concurrent packs;
there are no arrays, scheduler-sized waves or `MaxArraySize` lookups.

Use `slurm_partition` for the default partition. The `parameters` mapping accepts
`slurm_partition`, `slurm_account`, `slurm_qos`, `slurm_constraint`,
`slurm_comment`, `slurm_nodelist` and checked `srun_args`. Step-local values may
use `RunnerResources.extra`. Resource/topology/bootstrap overrides and array
settings are rejected; `slurm_gres` cannot override the typed GPU count.

GPU work requires `parameters={'slurm_nodelist': 'node-name'}` with one concrete
node, not a range or an assumed homogeneous partition. A scheduled probe measures
the current driver/launcher software before cache lookup. A probe failure retains
its owner and cannot authorize scientific admission. UUIDs belong to the fresh
allocation, not reusable software identity. Native/Pixi execution and prepared
Apptainer SIF tools use this same allocation check. For SIF tools, the probe also
verifies image bytes, launcher configuration and the GPU libraries selected for
injection. Worker entry repeats these checks before loading scientific calls.
Leave the controller runtime's `gpu_devices` unset so only the fresh allocation
supplies devices. Docker GPU tools remain rejected. Qualify actual SIF mounts
and device visibility on your site before relying on this runtime combination.

## Ownership evidence

The provider binds the exact Submitit payload digest before `sbatch`; bootstrap
verifies it before SDK deserialization. Full scientific software is verified
again inside the worker. The scheduler's initial shell is a trusted site
boundary; the fixed prelude removes loader and startup injections before measured
Python starts.

Cancellation targets only retained numeric job identities. Unit replies and
whole-worker closure are separate facts. A valid early receipt does not prove a
later worker closed. Cleanup uses one absolute deadline for command draining,
worker termination and scheduler cancellation. Missing, malformed or late
closure evidence remains `UNKNOWN` and blocks current retry/resume.

Queued non-start proof requires exact fresh accounting evidence; a `CANCELLED`
state alone is insufficient. Provider redelivery creates a fresh entry and
scratch directory. Success of that entry never proves a previous unresolved
entry closed.
