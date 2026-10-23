# Migration from Prefect-backed SLURM execution

`artisan-submitit` preserves scheduler behavior but does not preserve Prefect's
API. Migrate concepts rather than constructor calls:

| Previous concept | Native Artisan replacement |
| --- | --- |
| `SlurmTaskRunner` on a Prefect flow | `SlurmRunner` passed to Artisan |
| Prefect `.map()` | Artisan execution units and `BatchStrategy` |
| `units_per_worker` on the task runner | `BatchStrategy.units_per_worker` |
| Prefect task states | `UnitResult` and Artisan execution records |
| Prefect server/UI | Artisan run history, inspection, and failure logs |
| `execution_mode="srun"` | `SlurmIntraRunner` |

The new runner uses Submitit-native parameter names. Configure portable CPU,
memory, GPU, and time requirements through `RunnerResources`; keep only genuine
scheduler options in `SlurmRunner(...)` or `RunnerResources.extra`.

The old implementation cancelled by job name in Artisan core. The provider
stores exact Submitit jobs and `Popen` values, so cancellation cannot affect an
unrelated run with the same operation or step name.

## Artisan 0.2.0 stores and APIs

Start with a fresh format-7 store and use current Artisan 0.2.0 APIs. This release
does not provide old-store migration or backward-compatibility shims. Import
`SlurmRunner` or `SlurmIntraRunner` directly from `artisan_submitit` and pass an
instance to `PipelineManager` or an individual step.

Cancellation now returns typed evidence. `CONFIRMED` means required worker and
transport ownership closed; `UNKNOWN` means cleanup could not be proved and the
step fails. Read worker logs through Artisan's public
`artisan.visualization.inspect_worker_log` rather than an execution-row
`worker_log` field. See [queue execution](slurm.md) and
[intra-allocation execution](srun.md) for current cancellation behavior.
