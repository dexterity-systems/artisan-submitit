"""Fixed scheduler-shell prelude that starts measured Python with a clean environment.

The scheduler's existing shell is trusted launch infrastructure. This prelude
makes no claim to retroactively isolate that shell; it prevents its ambient
Python, loader, shell-startup and credential settings from reaching Python.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artisan_submitit.software import SlurmSoftware

ALLOCATION_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NODEID",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_RESTART_COUNT",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
)
_PRELUDE = (
    r"""set -eu
env_program=$1
scratch=$2
command_path=$3
shift 3
values=("PATH=$command_path" "HOME=$scratch/home" "TMPDIR=$scratch/tmp" "LANG=C.UTF-8")
for name in """
    + " ".join(ALLOCATION_VARIABLES)
    + r"""; do
    if [[ -v $name ]]; then values+=("$name=${!name}"); fi
done
exec "$env_program" -i "${values[@]}" "$@"
"""
)
PRELUDE_SHA256 = hashlib.sha256(_PRELUDE.encode()).hexdigest()


def bootstrap_command(
    software: SlurmSoftware, directory: Path, command: list[str]
) -> list[str]:
    """Use measured shell/env binaries and literal allocation values for both transports."""
    scratch = directory / "bootstrap"
    for name in ("home", "tmp"):
        (scratch / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    client_directories = dict.fromkeys(
        str(item.path.parent) for item in software.scheduler.executables
    )
    command_path = ":".join((*client_directories, "/usr/bin", "/bin"))
    system_env = str(software.local.system_env)
    return [
        system_env,
        "-u",
        "BASH_ENV",
        "-u",
        "ENV",
        "-u",
        "LD_PRELOAD",
        "-u",
        "LD_LIBRARY_PATH",
        "-u",
        "LD_AUDIT",
        str(software.scheduler.bootstrap_shell.path),
        "--noprofile",
        "--norc",
        "-p",
        "-c",
        _PRELUDE,
        "artisan-scheduler-bootstrap",
        system_env,
        str(scratch),
        command_path,
        *command,
    ]
