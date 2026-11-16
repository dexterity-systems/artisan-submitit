# Docker SLURM environment

This is the single-node SLURM environment carried forward from
`prefect-submitit` v0.1.8. Its Prefect server, PostgreSQL process, port, and
environment variables have been removed. MariaDB and `slurmdbd` remain because
the integration suite checks real scheduler state through `sacct`.

## Start the cluster

Use real sibling checkouts when refreshing the host Pixi lock. CI qualifies
[Artisan `main`](https://github.com/andrewcolinhunt/artisan-dev/tree/main).
Use that branch in the sibling checkout and point `ARTISAN_WORKTREE` at it with
an absolute path; Compose resolves relative paths from this `docker/` directory:

```bash
export ARTISAN_WORKTREE=/absolute/path/to/artisan
export ARTISAN_DOCKER_UID=$(id -u)
export ARTISAN_DOCKER_GID=$(id -g)
export COMPOSE_PROJECT_NAME=artisan-submitit-release-check
docker compose -f docker/docker-compose.yml config
pixi run slurm-build
pixi run slurm-up
pixi run slurm-shell
```

`slurm-up` waits until SLURM responds and the container has made its Pixi
environment volume writable by the development user. Startup fails if ownership
setup fails or the service does not become healthy within 180 seconds.

The UID/GID variables let the container's unprivileged `docker` user write
generated editable-install metadata and pytest state without changing bind-
mounted repository ownership.

Inside the container, install both editable projects for development:

```bash
pixi install --locked -e dev
pixi run --locked -e dev python scripts/verify_sources.py \
    /workspace/artisan /workspace/artisan-submitit \
    main
```

The Compose environment mounts:

- this repository at `/workspace/artisan-submitit`;
- the selected Artisan worktree at `/workspace/artisan`; and
- a named Pixi environment volume at `/pixi-env`.

Both source trees are on `PYTHONPATH`, including inside submitted SLURM jobs.
The environment installs the provider and mounted Artisan worktree as editable
projects. Both packages build without Git metadata, so worktree mounts whose
`.git` pointers refer to host-only metadata are supported. Source edits in either
repository are visible without rebuilding the image. Ordinary and locked Pixi
commands refresh version-only metadata without changing the lock. Regenerate
the lock deliberately on the host when the dependency set changes.

Use Pixi 0.83.0. Each host worktree owns its `.pixi/envs`; the container deliberately
uses a detached environment volume at `/pixi-env`. Keep a unique Compose project
name and volume for each checkout, and never share this installed environment
with another worktree. Package caches may be shared.

For scheduler qualification, use the [SLURM Docker workflow](../.github/workflows/slurm-docker.yaml)
with the `test-slurm` pull-request label or manual dispatch. It builds ordinary
core, provider and application wheels, locks an explicit controller/worker profile,
and prepares the targets before running the CPU pipeline, cancellation, ownership
and tutorial checks. Its fixture receipt binds the immutable image, qualification
account, node interpreter and scheduler configuration. Every controller launch
checks that receipt independently. The generated manifest, lock and reports are
retained as workflow artifacts.

The qualification fixture mounts its receipt and configuration read-only and uses
a separate writable job directory. It does not mount host home directories or
package caches. The development Compose file above remains useful for source
inspection and editing; its editable environment alone does not supply the
installed profile required by the integration tests.

If your account already has noninteractive sudo access to Docker, invoke Compose
through `sudo -n env ARTISAN_WORKTREE=... ARTISAN_DOCKER_UID=... ARTISAN_DOCKER_GID=...
docker compose -p artisan-submitit-release-check -f docker/docker-compose.yml ...`.
Pass the absolute checkout path and your actual UID/GID; run host Pixi commands
as your normal user. Keep the same project name through teardown so only this
qualification cluster and its environment volume are removed.

## Execute the tutorials

The [tutorial notebooks](../docs/tutorials/index.md) run in the installed
controller profile declared by the qualification workflow. Notebook dependencies
are included before the profile is locked and verified. The workflow supplies
`ARTISAN_SLURM_MANIFEST`, `ARTISAN_SLURM_CACHE`, `ARTISAN_SLURM_ENVIRONMENT`,
`ARTISAN_SLURM_PIXI`, and the test partition, and runs the allocation notebook
inside a separately owned allocation. Executed notebook copies and JUnit reports
are retained with the other qualification evidence.

## Useful commands

```bash
sinfo
squeue
sacct -j <job-id>
srun hostname
scancel <job-id>
```

Stop and remove the cluster and environment volume with:

```bash
pixi run slurm-down
```
