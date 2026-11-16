"""Shared assertions for real scheduler tests using installed packages."""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Literal

from artisan.operations.curator import Merge
from artisan.operations.examples import DataGenerator, DataTransformer
from artisan.orchestration import CancellationStatus, PipelineManager, StepStatus
from artisan.orchestration.runner_api import OwnershipState
from artisan.schemas import Environment

from artisan_submitit import SlurmIntraRunner, SlurmRunner

APPLICATIONS = ("slurm_qualification_app",)
RESOURCES = {"cpus": 1, "memory_gb": 1, "time_limit": "00:05:00"}


def runner(
    root: Path, mode: Literal["sbatch", "srun"]
) -> SlurmRunner | SlurmIntraRunner:
    """Require installed app and canonical profile; prepare only by explicit opt-in."""
    app = importlib.import_module(APPLICATIONS[0])
    assert Path(app.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    manifest = Path(os.environ["ARTISAN_SLURM_MANIFEST"]).resolve()
    cache = Path(os.environ["ARTISAN_SLURM_CACHE"]).resolve()
    options = (
        {"slurm_partition": os.environ.get("SLURM_TEST_PARTITION", "cpu")}
        if mode == "sbatch"
        else {}
    )
    kind = SlurmRunner if mode == "sbatch" else SlurmIntraRunner
    selected = kind(
        environment=Environment.from_pixi(
            manifest, name=os.environ.get("ARTISAN_SLURM_ENVIRONMENT", "worker")
        ),
        cache_root=cache,
        pixi=Path(
            os.environ.get("ARTISAN_SLURM_PIXI", str(Path.home() / ".pixi/bin/pixi"))
        ),
        log_folder=root / "scheduler",
        **options,
    )
    if os.environ.get("ARTISAN_SLURM_PREPARE") == "1":
        for operation in (DataGenerator, DataTransformer):
            selected.prepare(operation, application_modules=APPLICATIONS)
        for operation in (app.ObservableWait, app.ManagedWait, app.PackedWait):
            selected.prepare(
                operation,
                params={"marker_root": str(root / "prepare-marker"), "duration": 1},
                application_modules=APPLICATIONS,
            )
        selected.prepare_curator(Merge(), application_modules=APPLICATIONS)
    return selected


def pipeline(
    root: Path, selected: SlurmRunner | SlurmIntraRunner, name: str = "scheduler-test"
) -> PipelineManager:
    return PipelineManager.create(
        name,
        str(root / "delta"),
        str(root / "staging"),
        working_root=str(root / "working"),
        default_step_runner=selected,
        application_modules=APPLICATIONS,
    )


def wait_ready(root: Path) -> None:
    """Wait for actual science, not just scheduler admission or bootstrap."""
    deadline = time.monotonic() + 90
    while not (root / "heartbeat").exists():
        assert time.monotonic() < deadline, f"Worker did not start: {root}"
        time.sleep(0.05)


def assert_closed(owner: PipelineManager) -> None:
    assert owner.ownership_state() == OwnershipState.CLOSED
    for execution in owner.executions:
        assert all(
            handle.ownership_state() == OwnershipState.CLOSED
            for handle in execution.handles
        )


def cancelled_worker_keeps_sibling(root: Path, mode: Literal["sbatch", "srun"]) -> None:
    """Cancel one managed command while another exact worker remains live."""
    from slurm_qualification_app import ManagedWait

    first_root, second_root = root / "first", root / "second"
    first_marker, second_marker = root / "marker-first", root / "marker-second"
    first = pipeline(first_root, runner(first_root, mode))
    second = pipeline(second_root, runner(second_root, mode))
    with first, second:
        cancelled = first.submit(
            ManagedWait,
            params={"marker_root": str(first_marker), "duration": 120},
            runner_resources=RESOURCES,
        )
        wait_ready(first_marker)
        surviving = second.submit(
            ManagedWait,
            params={"marker_root": str(second_marker), "duration": 120},
            runner_resources=RESOURCES,
        )
        try:
            wait_ready(second_marker)
            first.cancel()
            result = cancelled.result(timeout=35)
            assert result.status == StepStatus.CANCELLED, result.error
            assert result.cancellation_status == CancellationStatus.CONFIRMED
            before = (second_marker / "heartbeat").read_bytes()
            time.sleep(0.15)
            assert (second_marker / "heartbeat").read_bytes() != before
            settled = (first_marker / "heartbeat").read_bytes()
            time.sleep(0.15)
            assert (first_marker / "heartbeat").read_bytes() == settled
        finally:
            second_marker.mkdir(parents=True, exist_ok=True)
            (second_marker / "stop").touch()
        result = surviving.result(timeout=60)
        assert result.status == StepStatus.SUCCEEDED, result.error
    assert_closed(first)
    assert_closed(second)
