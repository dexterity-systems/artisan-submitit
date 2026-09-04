"""SLURM execution providers for Artisan."""

from __future__ import annotations

from artisan_submitit._version import __version__, __version_tuple__
from artisan_submitit.slurm import SlurmRunner
from artisan_submitit.srun import SlurmIntraRunner

__all__ = ["SlurmIntraRunner", "SlurmRunner", "__version__", "__version_tuple__"]
