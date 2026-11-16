"""Prepare and resolve installed targets without allocating GPUs on the actual scheduler node.

The supervising caller owns this process subtree. Clearing its inherited command
attachment gives the controller its ordinary top-level runtime scope; all child
processes remain under the caller's operating-system ownership.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from artisan.utils.configuration_secrets import COMMAND_ATTACHMENT_ENV


def main() -> None:
    for name in COMMAND_ATTACHMENT_ENV:
        os.environ.pop(name, None)
    from artisan.operations.curator import Merge
    from artisan.operations.examples import DataGenerator, DataTransformer
    from artisan.orchestration.runner_api import (
        RunnerResources,
        compile_definition,
        verify_wheel_installation,
    )
    from artisan.schemas import Environment
    from slurm_qualification_app import ManagedWait, ObservableWait, PackedWait

    from artisan_submitit import SlurmIntraRunner, SlurmRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--pixi", required=True, type=Path)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    profile = args.profile.resolve()
    control = args.control_root.resolve()
    wheels = tuple(sorted((profile / "wheels").glob("*.whl")))
    verify_wheel_installation(wheels, interpreter=Path(sys.executable))
    cache = profile / "prepared"
    cache.mkdir(exist_ok=True)
    resources = RunnerResources(cpus=1, memory_gb=1, gpus=0, time_limit="00:05:00")
    applications = ("slurm_qualification_app",)
    runner = SlurmRunner(
        environment=Environment.from_pixi(profile / "pixi.toml", name="worker"),
        cache_root=cache,
        pixi=args.pixi.resolve(),
        log_folder=control / "scheduler",
        slurm_partition=args.partition,
    )
    declarations = [
        (DataGenerator, {}),
        (DataTransformer, {}),
        *(
            (operation, {"marker_root": str(control / "markers"), "duration": 1})
            for operation in (ObservableWait, ManagedWait, PackedWait)
        ),
    ]
    if args.gpu:
        from slurm_qualification_app import (
            CommandGPUIncrement,
            FileToolIncrement,
            NativeGPUIncrement,
        )

        declarations.extend(
            (
                (NativeGPUIncrement, {}),
                (CommandGPUIncrement, {"manifest": str(profile / "pixi.toml")}),
                (FileToolIncrement, {"manifest": str(profile / "pixi.toml")}),
            )
        )
    resolved = {}
    for operation, params in declarations:
        runner.prepare(
            operation,
            params=params,
            runner_resources=resources,
            application_modules=applications,
        )
        software = runner.resolve(
            compile_definition(operation, params=params),
            resources,
            application_modules=applications,
        )
        assert software.software_cacheable
        resolved[operation.__name__] = software.software_digest
    runner.prepare_curator(Merge(), application_modules=applications)
    curator = runner.resolve_curator(
        Merge(), resources, application_modules=applications
    )
    assert curator.software_cacheable
    if args.gpu:
        from slurm_qualification_app import GPUMerge

        runner.prepare_curator(GPUMerge(), application_modules=applications)
        gpu_curator = runner.resolve_curator(
            GPUMerge(), resources, application_modules=applications
        )
        assert gpu_curator.software_cacheable
        resolved[GPUMerge.__name__] = gpu_curator.software_digest
    intra = SlurmIntraRunner(
        environment=runner.environment,
        cache_root=cache,
        pixi=runner.pixi,
        log_folder=control / "scheduler-intra",
    )
    intra_software = intra.resolve(
        compile_definition(DataGenerator), resources, application_modules=applications
    )
    assert intra_software.software_cacheable
    verify_wheel_installation(wheels, interpreter=Path(sys.executable))
    (control / "targets.json").write_text(
        json.dumps(
            {
                "scope": "Canonical CPU lookup only; GPU allocation and science are not run",
                "gpu_definitions_prepared": args.gpu,
                "controller": sys.executable,
                "pixi": str(runner.pixi),
                "prepared": resolved,
                "curator": curator.software_digest,
                "intra": intra_software.software_digest,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
