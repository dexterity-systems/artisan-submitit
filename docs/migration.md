# Migrating to prepared worker execution

Use matching current Artisan and `artisan-submitit` packages and a fresh store.
This release has no old-store conversion or compatibility runner aliases.

| Removed contract | Current contract |
| --- | --- |
| Runner enums or provider strings | Import and pass `SlurmRunner` or `SlurmIntraRunner` instances |
| Implicit host interpreter/operation endpoint | Explicit `Environment.from_pixi(...)`, then `runner.prepare(...)` |
| Lifecycle router construction | The pipeline owns admitted handles and worker owners |
| Submitit arrays, array limits and waves | One submission per worker pack; core `units_per_worker` and `max_workers` |
| Free-form runner keyword options | Validated `parameters` mapping and typed `runner_resources` |
| `slurm_gres` overriding GPU counts | Typed `gpus` plus current scheduled GPU measurement |
| Worker log fields on execution rows | Public failure/log and command inspection readers |

Prepared environment records now identify declared software and completed
preparation rather than an inventory of all installed files. Prepare existing
environments once with the matching new core/provider versions; old prepared
records are not reused. This change does not require another fresh store if you
already use the retained store format. Operations and pipelines share the new
prepared targets, and normal execution does not audit arbitrary in-place edits
to installed packages.

Operations and composites share `pipeline.run(operation, ...)` and
`pipeline.submit(operation, ...)`. Top-level curators run in the selected worker.
When preparing through a curator, pass its actual instance to `prepare_curator`.
Preserve trusted explicit application registration modules through preparation
and pipeline creation.

Resuming requires the configured provider instance. Stored names do not authorize
imports or reconstruct scheduler settings. The run appends new steps and keeps
its previous failed attempts visible. Unresolved ownership blocks resume.

`UNKNOWN` is an ownership result, not permission to retry. An accepted unit
receipt, a finished scheduler job and a closed shared worker are distinct facts.
See [queue ownership](slurm.md#ownership-evidence) and [intra-allocation
cleanup](srun.md) before adapting cancellation code.
