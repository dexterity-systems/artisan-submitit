# artisan-submitit

`artisan-submitit` provides optional SLURM runners for
[Artisan](https://github.com/dexterity-systems/artisan). Artisan retains
pipeline orchestration, batching, records, caching, and artifact commits; this
package transports Artisan execution units through `sbatch` or `srun`.

The package contains no Prefect dependency or compatibility layer. It exports:

- `SlurmRunner`, which submits packed execution units as Submitit job arrays;
- `SlurmIntraRunner`, which launches bounded `srun` processes inside an
  existing allocation.

## Installation

```bash
pip install artisan-submitit
```

## Usage

```python
from artisan.orchestration import PipelineManager
from artisan_submitit import SlurmRunner

pipeline = PipelineManager.create(
    name="training",
    delta_root="/shared/artisan/delta",
    staging_root="/shared/artisan/staging",
    default_step_runner=SlurmRunner(),
)
```

Use `SlurmIntraRunner()` instead when the pipeline manager is already running
inside an allocation created by `salloc` or `sbatch`.

Runner instances may also be supplied on individual `pipeline.run()` and
`pipeline.submit()` calls. Artisan core intentionally resolves strings only for
its built-in local runner.

## Requirements

- Python 3.12 or newer;
- `dexterity-artisan` 0.1.2a6.dev0 or newer, which provides
  `artisan.orchestration.runner_api`;
- Submitit 1.5 or newer;
- a shared filesystem visible at the same paths from the pipeline manager and
  workers; and
- SLURM command access (`sbatch` for `SlurmRunner`, an active allocation for
  `SlurmIntraRunner`).

## Configuration

Portable operation resources stay in Artisan:

| `RunnerResources` field | `SlurmRunner` | `SlurmIntraRunner` |
| --- | --- | --- |
| `cpus` | `cpus_per_task` | `--cpus-per-task` |
| `memory_gb` | `mem_gb` | `--mem` |
| `gpus` | `gpus_per_node` | `--gres=gpu:N` |
| `time_limit` | converted to `timeout_min` | `--time` |

Provider-specific configuration uses Submitit's native names:

```python
runner = SlurmRunner(
    slurm_partition="gpu",
    max_array_size=1000,
    slurm_constraint="a100",
    slurm_account="research",
)
```

Operation-specific values can go in `RunnerResources.extra`, for example
`{"slurm_partition": "gpu", "slurm_constraint": "a100"}`. Portable values
such as memory and CPU count must use their typed `RunnerResources` fields.
The old aliases `partition` and `time_limit` are rejected instead of silently
translated.

`BatchStrategy.max_workers` controls array parallelism for `SlurmRunner` and
simultaneous processes for `SlurmIntraRunner`. A conflicting low-level
`slurm_array_parallelism` or `srun_launch_concurrency` fails fast.

## Batching and array limits

Artisan owns both batching levels:

```text
input artifacts
  / artifacts_per_unit  -> ExecutionUnit values
  / units_per_worker    -> scheduler jobs or srun processes
```

`SlurmRunner` detects the cluster's `MaxArraySize` with `scontrol` and splits
larger submissions into ordered arrays. Set `max_array_size` only when cluster
configuration cannot be queried. Results are validated against their submitted
batch and returned in original unit order.

## Failure, logs, and cancellation

Ordinary operation failures remain Artisan `UnitResult` values. Scheduler,
bootstrap, and malformed-result failures become one failed result per affected
unit, preserving positional alignment.

Each scheduler handle's stdout and stderr are attached to the corresponding
results. Reads include NFS metadata-cache invalidation and a short retry for
late log flushing.

Cancellation targets exact handles only:

- `SlurmRunner` calls `cancel(check=False)` on jobs it submitted;
- `SlurmIntraRunner` terminates its own `Popen` handles, waits for one grace
  period, then kills survivors.

No job-name-wide `scancel` query is used.

## Intra-allocation execution

Start an allocation before using `SlurmIntraRunner`:

```bash
salloc --cpus-per-node=8 --mem=16G --time=00:30:00
python examples/srun_pipeline.py
```

`RunnerResources.extra` supports `slurm_gres` and an explicit `srun_args` list
for this runner. Scheduler placement such as partition and account belongs to
the command that created the allocation. Every router uses a unique work
directory so concurrent pipelines cannot collide on step indexes.

## Development and Docker

The Pixi environment expects Artisan in a real sibling `../artisan` checkout:

```bash
pixi install --locked -e dev
pixi run --locked -e dev fmt
pixi run --locked -e dev typecheck
pixi run --locked -e dev test
```

That sibling is the conventional standalone development layout. Docker does
not assume it: set `ARTISAN_WORKTREE` explicitly when Artisan uses a nested or
otherwise non-sibling worktree layout.

Use an actual checkout rather than a symlink for the host Pixi environment.
Pixi records the canonical path of editable dependencies in `pixi.lock`, so a
symlink to a differently named worktree makes the otherwise portable lock look
stale. `ARTISAN_WORKTREE` has no such restriction because Compose always mounts
the selected source at `/workspace/artisan`.

The Docker cluster is adapted from the `prefect-submitit` v0.1.8 development
image and mounts both editable repositories:

```bash
export ARTISAN_WORKTREE=/absolute/path/to/prefect-free-artisan
export ARTISAN_DOCKER_UID=$(id -u)
export ARTISAN_DOCKER_GID=$(id -g)
pixi run slurm-build
pixi run slurm-up
pixi run slurm-shell
```

Then, inside the container:

```bash
pixi install --locked -e dev
pixi run --locked -e dev test-sbatch-docker
pixi run --locked -e dev test-srun-docker
```

Artisan's source-build fallback allows its editable install even when a mounted
worktree's `.git` pointer refers to metadata outside the container. Both mounted
source trees are also placed on `PYTHONPATH` for submitted jobs.

See [docker/README.md](docker/README.md) for environment details and
[examples/](examples/) for runnable pipelines.

## Migrating from built-in Artisan SLURM runners

Import provider runners directly and pass instances:

```python
from artisan_submitit import SlurmIntraRunner, SlurmRunner

pipeline.run(MyOperation, step_runner=SlurmRunner())
pipeline.run(MyOperation, step_runner=SlurmIntraRunner())
```

There are deliberately no `Runner.SLURM`, `Runner.SLURM_INTRA`, or string-name
compatibility shims. Historical runner names remain in persisted provenance;
resuming such a pipeline requires supplying the matching provider instance.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
