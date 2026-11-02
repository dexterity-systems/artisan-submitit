"""SLURM execution providers for Artisan."""

from __future__ import annotations

from importlib.metadata import version

from artisan_submitit.slurm import SlurmRunner
from artisan_submitit.srun import SlurmIntraRunner

__version__ = version("artisan-submitit")

__all__ = ["SlurmIntraRunner", "SlurmRunner", "__version__"]
