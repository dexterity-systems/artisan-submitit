# Docker SLURM environment

This is the single-node SLURM environment carried forward from
`prefect-submitit` v0.1.8. Its Prefect server, PostgreSQL process, port, and
environment variables have been removed. MariaDB and `slurmdbd` remain because
the integration suite checks real scheduler state through `sacct`.

## Start the cluster

Use real sibling checkouts when refreshing the host Pixi lock. CI currently
qualifies Artisan's `main` branch from
`andrewcolinhunt/artisan-dev`. Point `ARTISAN_WORKTREE` at that checkout with an
absolute path; Compose resolves relative paths from this `docker/` directory:

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

The UID/GID variables let the container's unprivileged `docker` user write
generated editable-install metadata and pytest state without changing bind-
mounted repository ownership.

Inside the container, install both editable projects and run the tests:

```bash
pixi install --locked -e dev
pixi run --locked -e dev python scripts/verify_sources.py /workspace/artisan /workspace/artisan-submitit main
pixi run --locked -e dev test-sbatch-docker
pixi run --locked -e dev test-srun-docker
```

The Compose environment mounts:

- this repository at `/workspace/artisan-submitit`;
- the selected Artisan worktree at `/workspace/artisan`; and
- a named Pixi environment volume at `/pixi-env`.

Both source trees are on `PYTHONPATH`, including inside submitted SLURM jobs.
The environment installs the provider and mounted Artisan worktree as editable
projects. Artisan's packaging fallback supports worktree mounts whose `.git`
pointer refers to host-only metadata. Source edits in either repository are
visible without rebuilding the image. Re-run the locked install after source
changes that affect dynamic version metadata, and regenerate the lock on the
host when the dependency set changes.

For release qualification, record both Git revisions, imported module paths,
the `debug` partition/allocation, and each suite's pass/fail/skip counts. The
source verification command resolves the requested branch or commit, prints the
revisions, and rejects a different core or imports outside the mounted source
trees. Use independent clones to keep your development checkouts unchanged.

If your account already has noninteractive sudo access to Docker, invoke Compose
through `sudo -n env ARTISAN_WORKTREE=... ARTISAN_DOCKER_UID=... ARTISAN_DOCKER_GID=...
docker compose -p artisan-submitit-release-check -f docker/docker-compose.yml ...`.
Pass the absolute checkout path and your actual UID/GID; run host Pixi commands
as your normal user. Keep the same project name through teardown so only this
qualification cluster and its environment volume are removed.

## Execute the tutorials

The [tutorial notebooks](../docs/tutorials/index.md) run against this cluster
with the same built-in Artisan operations used on a real site. Inside the
container, install the notebook environment and select the test partition:

```bash
/usr/local/bin/pixi install --locked -e tutorials
export ARTISAN_SLURM_PARTITION=debug
export ARTISAN_TUTORIAL_ROOT=/workspace/artisan-submitit/tutorial-runs
/usr/local/bin/pixi run --locked -e tutorials test-tutorial-sbatch
/usr/local/bin/pixi run --locked -e tutorials test-tutorial-srun
```

The first task executes the queue and batching notebooks. The second creates an
allocation and executes the allocation notebook inside it. Each test uses a
separate directory beneath the shared tutorial root. Executed notebook copies,
including outputs from failed cells, are retained under `slurm_logs/tutorials/`.
The source notebooks keep their outputs empty.

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
