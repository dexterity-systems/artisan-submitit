# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SlurmRunner` for Submitit-managed SLURM job arrays.
- `SlurmIntraRunner` for bounded `srun` execution inside an allocation.
- Unit, real-SLURM Docker, packaging, documentation, and release workflows.
- Runnable tutorials using Artisan's built-in operations for queue execution,
  existing allocations, batching, and managed commands, with a Pixi notebook
  environment and real-SLURM notebook tests.

### Changed

- SLURM execution moved out of Artisan core into this optional provider.
- Versions are derived from Git tags with hatch-vcs.
- Scheduler arguments reject abbreviated forms of Artisan-managed resource and
  topology options. Empty GRES overrides preserve portable GPU requests, and
  array/intra-allocation concurrency limits require positive integers.
- Cancellation implements Artisan 0.2.0's typed evidence contract. Workers
  compose the public local runner for Python and managed-command cleanup;
  confirmation requires native ownership evidence and terminal transport.
  Missing or nested uncertain evidence remains UNKNOWN.
- Cancellation uses one bounded background cleanup budget, persistent per-batch
  requests, and exact job or validated step IDs. Collection reads atomic provider
  outcomes without Submitit's unbounded result or watcher waits.
- Intra-allocation execution rejects asynchronous launch and allocation override
  options. Current APIs and fresh format-7 stores are required.
- CI pins the selected core candidate, verifies mounted source revisions, and
  smoke-tests clean built distributions. Publication requires the actual
  published Artisan 0.2.0 package to pass the installed-provider smoke.
- Docker integration modules collect under pytest 9; real `sbatch` and `srun`
  suites exercise the package against the inherited Prefect Submitit container.
- Submitit result/log directories are unique per router invocation, preventing
  stale envelopes when a recreated scheduler reuses numeric job IDs.
