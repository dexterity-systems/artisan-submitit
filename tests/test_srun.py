"""Tests for intra-allocation srun routing."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from artisan.orchestration.runner_api import BatchStrategy, RunnerResources

from artisan_submitit.srun import SlurmIntraLifecycleRouter, SlurmIntraRunner


def _router(tmp_path, **overrides) -> SlurmIntraLifecycleRouter:
    values = {
        "units_per_worker": 1,
        "max_workers": 2,
        "work_folder": tmp_path / "work",
        "cpus": 2,
        "memory_gb": 8,
        "gpus": 1,
        "time_limit": "00:30:00",
        "slurm_gres": None,
        "srun_args": [],
        "poll_interval": 0.01,
        "max_poll_time": 60.0,
        "cancel_grace_period": 0.0,
    }
    values.update(overrides)
    return SlurmIntraLifecycleRouter(**values)


class TestSlurmIntraRunner:
    def test_traits(self) -> None:
        assert SlurmIntraRunner.name == "slurm_intra"
        assert SlurmIntraRunner.worker_traits.worker_id_env_var == "SLURM_STEP_ID"
        assert SlurmIntraRunner.worker_traits.shared_filesystem is True

    def test_create_router_maps_resources_and_uses_unique_folder(
        self, tmp_path
    ) -> None:
        runner = SlurmIntraRunner(log_folder=str(tmp_path))
        resources = RunnerResources(
            cpus=4,
            memory_gb=16,
            gpus=2,
            time_limit="02:00:00",
            extra={"slurm_gres": "gpu:a100:1", "srun_args": ["--exclusive"]},
        )
        first = runner.create_lifecycle_router(
            resources, BatchStrategy(max_workers=3), 2, "train"
        )
        second = runner.create_lifecycle_router(
            resources, BatchStrategy(max_workers=3), 2, "train"
        )

        assert isinstance(first, SlurmIntraLifecycleRouter)
        assert first.work_folder != second.work_folder
        assert first._max_workers == 3
        assert first._cpus == 4
        assert first._slurm_gres == "gpu:a100:1"
        assert first._srun_args == ["--exclusive"]

    def test_concurrency_conflict_fails_fast(self) -> None:
        runner = SlurmIntraRunner(srun_launch_concurrency=4)
        with pytest.raises(ValueError, match="conflicts"):
            runner.create_lifecycle_router(
                RunnerResources(), BatchStrategy(max_workers=2), 0, "op"
            )

    def test_non_positive_max_workers_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(ValueError, match="max_workers"):
            runner.create_lifecycle_router(
                RunnerResources(), BatchStrategy(max_workers=0), 0, "op"
            )

    def test_non_positive_poll_timeout_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="max_poll_time"):
            SlurmIntraRunner(max_poll_time=0)

    def test_unsupported_resource_extra_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(ValueError, match="Unsupported"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"slurm_constraint": "a100"}),
                BatchStrategy(),
                0,
                "op",
            )

    def test_non_string_gres_fails_fast(self) -> None:
        runner = SlurmIntraRunner()
        with pytest.raises(TypeError, match="slurm_gres"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"slurm_gres": 1}),
                BatchStrategy(),
                0,
                "op",
            )

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--ntasks=2"],
            ["--nodes", "2"],
            ["-n2"],
            ["-N2"],
            ["--cpus-per-task", "8"],
            ["-c8"],
            ["--mem=8G"],
            ["--gres", "gpu:2"],
            ["-G2"],
            ["--time=01:00:00"],
            ["-t60"],
        ],
    )
    def test_managed_srun_argument_fails_fast(self, arguments: list[str]) -> None:
        runner = SlurmIntraRunner()

        with pytest.raises(ValueError, match="Artisan-managed"):
            runner.create_lifecycle_router(
                RunnerResources(extra={"srun_args": arguments}),
                BatchStrategy(),
                0,
                "op",
            )

    @patch.dict("os.environ", {}, clear=True)
    def test_validate_operation_warns_outside_allocation(self) -> None:
        operation = MagicMock(name="operation")
        operation.name = "train"
        with pytest.warns(UserWarning, match="SLURM_JOB_ID"):
            SlurmIntraRunner().validate_operation(operation)


class TestSlurmIntraLifecycleRouter:
    def test_command_maps_resources(self, tmp_path) -> None:
        router = _router(
            tmp_path,
            slurm_gres="gpu:a100:1",
            srun_args=["--exclusive"],
        )
        command = router._build_command(tmp_path / "step")
        assert command[:4] == ["srun", "--exact", "--mpi=none", "-n1"]
        assert command[4:6] == ["--gres", "gpu:a100:1"]
        assert "8G" in command
        assert "2" in command
        assert "00:30:00" in command
        assert "--exclusive" in command
        assert "artisan_submitit.srun_worker" in command

    @patch.dict("os.environ", {}, clear=True)
    def test_dispatch_fails_without_allocation(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
            _router(tmp_path).dispatch([], MagicMock())

    def test_cancel_terminates_only_owned_processes(self, tmp_path) -> None:
        router = _router(tmp_path)
        running = MagicMock(pid=10)
        running.poll.return_value = None
        running.wait.side_effect = subprocess.TimeoutExpired("srun", 0)
        complete = MagicMock(pid=11)
        complete.poll.return_value = 0
        router._submitted = [
            MagicMock(process=running),
            MagicMock(process=complete),
        ]

        router.cancel()
        router.cancel()

        running.terminate.assert_called_once()
        running.kill.assert_called_once()
        complete.terminate.assert_not_called()

    @patch("artisan_submitit.srun._terminate_processes")
    @patch("artisan_submitit.srun.subprocess.Popen")
    def test_process_launched_during_cancel_is_not_leaked(
        self,
        popen: MagicMock,
        terminate: MagicMock,
        tmp_path,
    ) -> None:
        router = _router(tmp_path)
        process = MagicMock(pid=12)
        process.poll.return_value = None
        popen.return_value = process
        router.cancel()
        terminate.reset_mock()

        router._launch(0, [MagicMock()], MagicMock())

        terminate.assert_called_once_with([process], 0.0)

    @patch("artisan_submitit.srun.time.sleep")
    def test_wait_for_slot_honors_process_cap(
        self,
        sleep: MagicMock,
        tmp_path,
    ) -> None:
        router = _router(tmp_path, max_workers=1)
        process = MagicMock(pid=10)
        process.poll.side_effect = [None, 0]
        router._submitted = [MagicMock(process=process)]

        router._wait_for_slot()

        sleep.assert_called_once_with(0.01)
