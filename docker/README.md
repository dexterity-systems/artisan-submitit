# Docker SLURM environment

This is the single-node SLURM environment carried forward from
`prefect-submitit` v0.1.8. Its Prefect server, PostgreSQL process, port, and
environment variables have been removed. MariaDB and `slurmdbd` remain because
the integration suite checks real scheduler state through `sacct`.

## Start the cluster

Place the Artisan and `artisan-submitit` repositories next to each other, or
point `ARTISAN_WORKTREE` at the Prefect-free Artisan checkout:

```bash
export ARTISAN_WORKTREE=/absolute/path/to/artisan
export ARTISAN_DOCKER_UID=$(id -u)
export ARTISAN_DOCKER_GID=$(id -g)
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
