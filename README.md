# artisan-submitit

`artisan-submitit` supplies `SlurmRunner` and `SlurmIntraRunner` for Artisan.
Artisan owns scientific partitions, worker packing, accepted results and cache
selection. This package owns exact scheduler jobs, steps and their cleanup.

## Prepare and run

Install matching Artisan and provider packages. Submitit is pinned to 1.5.4
because the provider binds its serialized payload before scheduler submission.
Use a locked Pixi manifest containing the same installed core, provider and
application packages as your controller. The manifest, prepared cache, data and
control directories must be visible at the same absolute paths on compute nodes.

```python
from pathlib import Path

from artisan.schemas import Environment
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import PipelineManager, StepStatus
from artisan_submitit import SlurmRunner

root = Path("/shared/artisan/example")
runner = SlurmRunner(
    environment=Environment.from_pixi(
        Path("/shared/software/pixi.toml"), name="worker"
    ),
    cache_root=Path("/shared/artisan/prepared"),
    log_folder=root / "scheduler",
    slurm_partition="cpu",
    parameters={"slurm_account": "research"},
)

# Explicit preparation may install software. Pipeline submission only looks it up.
runner.prepare(DataGenerator)
runner.prepare(DataTransformer)
with PipelineManager.create(
    "example",
    str(root / "delta"),
    str(root / "staging"),
    working_root=str(root / "working"),
    default_step_runner=runner,
) as pipeline:
    generated = pipeline.run(DataGenerator, params={"count": 4, "seed": 42})
    assert generated.status == StepStatus.SUCCEEDED, generated.error
    transformed = pipeline.run(
        DataTransformer,
        inputs={"dataset": generated.output("datasets")},
        params={"scale_factor": 2.0, "variants": 1},
        batch_strategy={
            "artifacts_per_unit": 1,
            "units_per_worker": 2,
            "max_workers": 2,
        },
        runner_resources={"cpus": 1, "memory_gb": 2, "time_limit": "00:05:00"},
    )
    assert transformed.status == StepStatus.SUCCEEDED, transformed.error
    assert transformed.succeeded_count == 4
```

Replace the example paths and account with your site settings. Application
definitions must come from installed importable packages. Prepare curators with
`runner.prepare_curator(operation_instance)`. Pass the same explicit
`application_modules` to preparation and pipeline creation when your application
registers artifact types.

`SlurmIntraRunner` accepts the same canonical environment and cache configuration
inside an existing allocation. Supply an actual configured runner when resuming
a run; its recorded name does not reconstruct software or scheduler settings.

## Scheduling and cleanup

One worker pack uses one `AutoExecutor.submit` call or one synchronous `srun`
step, with one node and one rank. `units_per_worker` determines pack size and
`max_workers` bounds concurrent packs. Units within a pack execute sequentially.
Array-only settings such as `max_array_size` and `slurm_array_parallelism` are
rejected because there is no array submission.

Portable CPU, memory, GPU and time requests use `runner_resources`. Scheduler
policy uses the runner's `parameters` mapping. See [queue execution](docs/slurm.md)
and [intra-allocation execution](docs/srun.md) for the supported boundaries.

Cancellation drains owned command scopes before closing a shared worker. A
successful `scancel` or an exited launcher does not establish cleanup: the exact
worker and transport must both be closed. Missing evidence remains `UNKNOWN`;
the owner and diagnostic files remain reachable. Cancelling an intra-allocation
step never authorizes cancelling its containing allocation or sibling steps.

GPU work supports native/Pixi execution and prepared Apptainer SIF tools on an
exact concrete node. A fresh scheduled probe checks software before cache lookup;
worker entry rechecks the allocated devices before loading scientific calls. SIF
checks also cover the image, launcher configuration and selected GPU libraries.
The controller's Apptainer runtime must leave `gpu_devices` unset: execution uses
the scheduler's fresh allocation. Docker GPU tools remain rejected. Container
use requires a prepared and verified site capability; qualify actual SIF mounts
and device visibility before relying on this combination at your site.

## Tutorials and development

The [tutorials](docs/tutorials/index.md) use installed operations and explicit
canonical preparation. [Examples](examples) provide command-line entry points.
Live scheduler checks are separate from source-only unit tests and installed
wheel import checks.

Development uses a real sibling Artisan checkout at `../artisan`, Pixi 0.83.0,
and this checkout's own environment:

```bash
pixi install -e dev
pixi run -e dev fmt
pixi run -e dev typecheck
pixi run -e dev test
pixi run -e dev build-dist
pixi run -e dev check-dist
```

Agent and CI checks add `--locked`. Dependency changes require an intentional
lock refresh. Do not share installed worktree environments. Release validation
must install matching ordinary core and provider wheels without a source
overlay, then qualify the actual scheduler and selected command backends.
The [Docker fixture](docker/README.md) is development infrastructure; its source
mounts do not establish installed-worker qualification.

See [migration](docs/migration.md) for removed contracts. Follow the
[committed-content policy](AGENTS.md#committed-content).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
