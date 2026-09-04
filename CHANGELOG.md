# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SlurmRunner` for Submitit-managed SLURM job arrays.
- `SlurmIntraRunner` for bounded `srun` execution inside an allocation.
- Unit, real-SLURM Docker, packaging, documentation, and release workflows.

### Changed

- SLURM execution moved out of Artisan core into this optional provider.
- Versions are derived from Git tags with hatch-vcs.
