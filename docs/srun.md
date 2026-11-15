# Intra-allocation execution

Start the controller inside the allocation that owns its workers:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=6G --time=00:10:00
python examples/srun_pipeline.py --manifest /shared/software/pixi.toml \
    --environment worker --cache-root /shared/artisan/prepared \
    --root /shared/artisan/srun-example --prepare
```

The runner requires an explicit canonical environment, as in the
[queue example](../README.md#prepare-and-run). Preparation is separate from
submission and must finish before work starts. All control, data and software
paths must be visible at the same absolute locations on the allocated node.

Each pack launches a synchronous `srun` with one task and one rank. CPU, memory
and time use `--cpus-per-task`, `--mem` and `--time`. CPU-only work adds
`--gres=none` so it does not inherit an unrelated GPU request. GPU work uses
`--gpus` and the exact allocated node; it does not add the CPU gres override.

`units_per_worker` packs sequential units and `max_workers` limits concurrent
steps. Request enough allocation resources for their simultaneous use. Supported
extras are bounded `srun_args` and a concrete `slurm_nodelist` matching the
allocation. Placement policy belongs to `salloc`/`sbatch`. Arguments that replace
resources, topology, allocation identity or synchronous ownership are rejected.

Cancellation requests per-unit cleanup before closing the shared worker.
Fallback signals target only an observed numeric `job.step` and the retained
local launcher. Without a step identity, the provider can stop its own launcher
but cannot cancel the allocation. Missing remote closure remains `UNKNOWN`.
Sibling jobs and steps are never cleanup substitutes.

Native/Pixi GPU work and prepared Apptainer SIF tools use a fresh probe inside
the same concrete allocation node before cache lookup. SIF checks include image
bytes, launcher configuration and selected GPU libraries. Worker entry rechecks
the software and allocated devices before loading scientific calls. Leave the
controller runtime's `gpu_devices` unset; execution binds the fresh allocation.
Docker GPU tools remain rejected. Qualify actual SIF mounts and device visibility
for intra-allocation execution at your site before relying on this combination.
