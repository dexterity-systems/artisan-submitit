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
- `dexterity-artisan` 0.2.0 or newer, which provides
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

Both runners use a single-rank topology: one scheduler task launches one
Artisan worker process. Scale out with execution units, `units_per_worker`, and
`max_workers`, not with scheduler node or task-count options.

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
must use their typed `RunnerResources` fields. Low-level Submitit parameters,
`slurm_srun_args`, `slurm_additional_parameters`, and `srun_args` cannot
override task topology or Artisan's CPU, memory, GPU, and time settings; the
runners reject those conflicts. The supported exception is `slurm_gres`, which
overrides `RunnerResources.gpus` when you need a cluster-specific GPU request.
The old aliases `partition` and `time_limit` are also rejected instead of
silently translated. This includes Submitit's unprefixed deprecated aliases
such as `time`, `mem`, `ntasks_per_node`, `srun_args`,
`additional_parameters`, and `array_parallelism`.

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

`SlurmRunner` detects the cluster's `MaxArraySize` with `scontrol`. When a step
exceeds that limit, the runner submits scheduler-sized waves sequentially and
collects each wave before submitting the next. Waves never overlap, so
`BatchStrategy.max_workers` remains a global cap for the step rather than a
per-array cap. Set `max_array_size` only when cluster configuration cannot be
queried. Results are validated against their submitted batch and returned in
original unit order.

## Failure, logs, and cancellation

Ordinary operation failures remain Artisan `UnitResult` values. Scheduler,
bootstrap, and malformed-result failures become one failed result per affected
unit, preserving positional alignment.

Each scheduler handle's stdout and stderr are attached to the corresponding
results. Reads include NFS metadata-cache invalidation and a short retry for
late log flushing.

Cancellation targets exact handles only:

- `SlurmRunner` checks terminal state and requires successful `scancel` for
  jobs it submitted;
- `SlurmIntraRunner` terminates its own `Popen` handles, waits for one grace
  period, then kills survivors.

No job-name-wide `scancel` query is used.
Cancellation before dispatch is an idle no-op, and a failed scheduler or
process cancellation remains retryable.

## Intra-allocation execution

Start an allocation before using `SlurmIntraRunner`:

```bash
salloc --cpus-per-node=8 --mem=16G --time=00:30:00
python examples/srun_pipeline.py
```

`RunnerResources.extra` supports `slurm_gres` and an explicit `srun_args` list
for this runner. Use `srun_args` only for flags that do not replace the
single-rank topology or portable resources. Scheduler placement such as
partition and account belongs to the command that created the allocation.
Every router uses a unique work directory so concurrent pipelines cannot
collide on step indexes.

## Development and Docker

The Pixi environment expects Artisan in a real sibling `../artisan` checkout:

```bash
pixi install --locked -e dev
pixi run --locked -e dev setup
pixi run --locked -e dev fmt
pixi run --locked -e dev typecheck
pixi run --locked -e dev test
pixi run --locked -e dev pre-commit run --all-files
pixi run --locked -e dev build-dist
pixi run --locked -e dev check-dist
pixi run --locked -e docs docs-build
```

That sibling is the conventional standalone development layout. Docker does
not assume it: set `ARTISAN_WORKTREE` explicitly when Artisan uses a nested or
otherwise non-sibling worktree layout.

Use an actual checkout rather than a symlink for the host Pixi environment.
Pixi records the canonical path of editable dependencies in `pixi.lock`, so a
symlink to a differently named worktree makes the otherwise portable lock look
stale. `ARTISAN_WORKTREE` has no such restriction because Compose always mounts
the selected source at `/workspace/artisan`.

Until `dexterity-artisan` 0.2.0 is published, Pixi explicitly overrides the
provider's release dependency with the sibling checkout. CI creates the same
sibling layout by checking out Artisan beside this repository. Merge and
publish the compatible Artisan release before publishing `artisan-submitit`,
then remove the temporary dependency override and regenerate `pixi.lock`.

The focused guides in `docs/` build as a small Jupyter Book 2 site and are
validated in CI.

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

See the [Docker environment](https://github.com/dexterity-systems/artisan-submitit/blob/main/docker/README.md)
and [runnable examples](https://github.com/dexterity-systems/artisan-submitit/tree/main/examples).

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

BSD 3-Clause. See the [license](https://github.com/dexterity-systems/artisan-submitit/blob/main/LICENSE).
