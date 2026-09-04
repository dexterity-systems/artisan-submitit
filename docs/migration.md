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
