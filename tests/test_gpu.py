"""Scheduled GPU qualification is current, owned, and allocation-independent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
from artisan.orchestration.runner_api import (
    ApptainerRuntime,
    AssignmentLaunchError,
    ContainerLauncherIdentity,
    CuratorLaunchError,
    LocalLauncherIdentity,
    LocalRunner,
    NativeGPUAllocation,
    OwnershipState,
    PreparedLocalBindings,
    PreparedNativeEnvironment,
    PreparedSIFEnvironment,
    PreparedToolBinding,
    RunnerResources,
)
from artisan.orchestration.runners import DockerRuntime

from artisan_submitit import gpu
from artisan_submitit.control import PackRequest, UnitEvidence
from artisan_submitit.gpu import (
    GPUProbeReply,
    GPUProbeSpec,
    ScheduledGPUProfile,
    SIFProbeSpec,
)
from artisan_submitit.runner import SchedulerRunner
from artisan_submitit.slurm import SlurmRunner
from artisan_submitit.software import SlurmSoftware, _allocated_bindings
from artisan_submitit.transport import SchedulerProbeHandle, SchedulerStartupError
from tests.test_runner import _runner
from tests.test_worker import _request


@pytest.fixture
def allocation() -> NativeGPUAllocation:
    return NativeGPUAllocation.model_validate_json(
        json.dumps(
            {
                "devices": ["GPU-11111111-1111-1111-1111-111111111111"],
                "driver": {
                    "device_models": ["NVIDIA A10"],
                    "driver_version": "570.195.03",
                    "nvidia_smi_sha256": "a" * 64,
                    "libcuda_sha256": "b" * 64,
                },
                "libraries": [
                    {
                        "name": "libcuda.so.1",
                        "path": "/driver/libcuda.so.1",
                        "sha256": "b" * 64,
                    },
                    {
                        "name": "libnvidia-ptxjitcompiler.so.1",
                        "path": "/driver/libnvidia-ptxjitcompiler.so.1",
                        "sha256": "c" * 64,
                    },
                ],
                "shell_sha256": "d" * 64,
            }
        )
    )


@pytest.fixture
def local() -> PreparedLocalBindings:
    value = Mock(spec=PreparedLocalBindings)
    value.worker_is_observed = False
    value.native_gpu = None
    value.container_runtime = None
    value.software_digest = "f" * 64
    value.application_modules = ()
    return value


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> Mock:
    value = Mock()
    value.model_dump.return_value = {"transport": "sbatch"}
    monkeypatch.setattr(
        "artisan_submitit.runner.SchedulerSoftware.capture", lambda *_args: value
    )
    return value


def _allocated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-node")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("SLURM_NODELIST", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)


@pytest.mark.parametrize("curator", [False, True])
def test_resolution_never_measures_login_gpu(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    local: PreparedLocalBindings,
    scheduler: Mock,
    curator: bool,
) -> None:
    method = "resolve_curator" if curator else "resolve"
    resolve = Mock(return_value=local)
    monkeypatch.setattr(LocalRunner, method, resolve)
    monkeypatch.setattr(
        gpu, "capture_native_gpu", lambda _count: pytest.fail("Login GPU capture")
    )
    measured = Mock(
        return_value=ScheduledGPUProfile(
            node="gpu-node", software=allocation.software_profile()
        )
    )
    monkeypatch.setattr(SchedulerRunner, "_measure_gpu", measured)
    runner = _runner()
    object.__setattr__(runner, "parameters", {"slurm_nodelist": "gpu-node"})
    result = getattr(runner, method)(Mock(), RunnerResources(gpus=1))
    assert resolve.call_args.args[1].gpus == 0
    assert measured.call_args.args[0] == GPUProbeSpec(node="gpu-node", device_count=1)
    assert measured.call_args.kwargs == {"curator": curator}
    assert result.gpu == ScheduledGPUProfile(
        node="gpu-node", software=allocation.software_profile()
    )


@pytest.mark.parametrize("node", [None, "gpu-[1-2]", "gpu1,gpu2", "gpu\nnode"])
def test_gpu_resolution_rejects_unproven_placement(
    monkeypatch: pytest.MonkeyPatch,
    local: PreparedLocalBindings,
    scheduler: Mock,
    node: str | None,
) -> None:
    monkeypatch.setattr(LocalRunner, "resolve", Mock(return_value=local))
    probe = Mock()
    monkeypatch.setattr(SchedulerRunner, "_measure_gpu", probe)
    runner = _runner()
    object.__setattr__(runner, "parameters", {"slurm_nodelist": node})
    with pytest.raises(ValueError, match="concrete"):
        runner.resolve(Mock(), RunnerResources(gpus=1))
    probe.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SLURM_JOB_NODELIST", "other"),
        ("SLURM_JOB_NUM_NODES", "2"),
        ("SLURM_JOB_ID", ""),
        ("CUDA_VISIBLE_DEVICES", ""),
        ("SLURMD_NODENAME", "other"),
    ],
)
def test_allocated_capture_rejects_unqualified_context_before_hardware(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    _allocated(monkeypatch)
    monkeypatch.setenv(field, value)
    capture = Mock()
    monkeypatch.setattr(gpu, "capture_native_gpu", capture)
    with pytest.raises(ValueError, match="GPU"):
        gpu.capture_allocated_gpu("gpu-node", 1)
    capture.assert_not_called()


def test_worker_remeasures_profile_and_allows_uuid_remapping(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
) -> None:
    _allocated(monkeypatch)
    remapped = allocation.model_copy(
        update={"devices": ("GPU-22222222-2222-2222-2222-222222222222",)}
    )
    monkeypatch.setattr(gpu, "capture_native_gpu", lambda _count: remapped)
    assert (
        gpu.capture_allocated_gpu("gpu-node", 1, allocation.software_profile())
        == remapped
    )
    changed = allocation.software_profile().model_copy(
        update={"shell_sha256": "e" * 64}
    )
    with pytest.raises(ValueError, match="changed"):
        gpu.capture_allocated_gpu("gpu-node", 1, changed)


def test_software_digest_excludes_placement_but_binds_driver_payload(
    allocation: NativeGPUAllocation,
    local: PreparedLocalBindings,
    scheduler: Mock,
) -> None:
    software = SlurmSoftware(
        local,
        scheduler,
        RunnerResources(gpus=1).model_dump_json(),
        ScheduledGPUProfile(node="node-a", software=allocation.software_profile()),
    )
    moved = replace(software, gpu=software.gpu.model_copy(update={"node": "node-b"}))
    assert moved.software_digest == software.software_digest
    changed = replace(
        software,
        gpu=ScheduledGPUProfile(
            node="node-a",
            software=allocation.software_profile().model_copy(
                update={"shell_sha256": "e" * 64}
            ),
        ),
    )
    assert changed.software_digest != software.software_digest


@pytest.mark.parametrize("curator", [False, True])
@pytest.mark.parametrize("state", [OwnershipState.CLOSED, OwnershipState.UNKNOWN])
def test_probe_failure_retains_exact_owner_without_scientific_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    curator: bool,
    state: OwnershipState,
) -> None:
    owner = Mock(ownership_state=lambda: state)
    handle = SchedulerProbeHandle(owner, 0)
    handle.future.set_exception(TimeoutError("probe timed out"))
    owner.handles = (handle,)
    runner = _runner()
    object.__setattr__(runner, "log_folder", tmp_path)
    monkeypatch.setattr(type(runner), "create_owner", lambda *_args: owner)
    error_type = CuratorLaunchError if curator else AssignmentLaunchError
    with pytest.raises(error_type) as caught:
        runner._measure_gpu(
            GPUProbeSpec(node="gpu-node", device_count=1), Mock(), curator=curator
        )
    assert caught.value.handles == ()
    assert caught.value.worker_owners == (owner,)
    owner.request_close.assert_called_once()
    assert caught.value.cancellation.status.value == (
        "confirmed" if state == OwnershipState.CLOSED else "unknown"
    )


def test_probe_success_requires_closed_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allocation: NativeGPUAllocation,
) -> None:
    states = iter(
        (OwnershipState.PENDING, OwnershipState.CLOSED, OwnershipState.CLOSED)
    )
    owner = Mock(ownership_state=lambda: next(states))
    handle = SchedulerProbeHandle(owner, 0)
    handle.future.set_result(
        ScheduledGPUProfile(node="gpu-node", software=allocation.software_profile())
    )
    owner.handles = (handle,)
    runner = _runner()
    object.__setattr__(runner, "log_folder", tmp_path)
    monkeypatch.setattr(type(runner), "create_owner", lambda *_args: owner)
    monkeypatch.setattr("artisan_submitit.runner.time.sleep", lambda _seconds: None)
    assert runner._measure_gpu(
        GPUProbeSpec(node="gpu-node", device_count=1), Mock(), curator=False
    ) == ScheduledGPUProfile(node="gpu-node", software=allocation.software_profile())
    owner.request_close.assert_not_called()


def test_probe_startup_error_retains_zero_handle_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owner = Mock(ownership_state=lambda: OwnershipState.UNKNOWN)
    runner = _runner()
    object.__setattr__(runner, "log_folder", tmp_path)
    monkeypatch.setattr(
        type(runner),
        "create_owner",
        Mock(side_effect=SchedulerStartupError(owner, RuntimeError("thread"))),
    )
    with pytest.raises(AssignmentLaunchError) as caught:
        runner._measure_gpu(
            GPUProbeSpec(node="gpu-node", device_count=1), Mock(), curator=False
        )
    assert caught.value.worker_owners == (owner,)
    assert caught.value.handles == ()


def test_fixed_worker_probe_does_not_launch_scientific_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
) -> None:
    from artisan_submitit.worker import run_worker_pack

    request = _request(tmp_path)
    request.calls = ()
    request.probe = GPUProbeSpec(node="gpu-node", device_count=1)
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.capture_allocated_gpu", lambda *_args: allocation
    )
    launch = Mock()
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    run_worker_pack(str(tmp_path), "c" * 64)
    launch.assert_not_called()
    evidence = UnitEvidence.model_validate_json(
        (tmp_path / "entries" / ("b" * 32) / "unit-0.json").read_bytes()
    )
    assert evidence.payload_sha256 == request.probe.digest
    assert GPUProbeReply.model_validate_json(
        evidence.reply
    ).result() == ScheduledGPUProfile(
        node="gpu-node", software=allocation.software_profile()
    )
    assert evidence.ownership == "closed"


def test_probe_request_cannot_contain_live_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from artisan.orchestration.runner_api import LocalWorkerProfile

    from artisan_submitit.launcher import PRELUDE_SHA256

    monkeypatch.setattr(
        LocalWorkerProfile,
        "from_json",
        lambda _value: SimpleNamespace(
            observed=False, native_gpu=None, worker=SimpleNamespace(compatibility=None)
        ),
    )
    request = PackRequest(
        pack_id="a" * 32,
        profile="opaque",
        prelude_sha256=PRELUDE_SHA256,
        calls=(),
        probe=GPUProbeSpec(node="gpu-node", device_count=1),
        working_root=tmp_path,
        runtime_env={},
    )
    assert request.result_count == 1
    assert request.payload_digest(0) == request.probe.digest
    with pytest.raises(ValueError, match="either"):
        PackRequest.model_validate({**request.model_dump(), "probe": None})
    with pytest.raises(ValueError, match="integer"):
        PackRequest.model_validate({**request.model_dump(), "version": True})
    with pytest.raises(ValueError, match="fixed GPU probe"):
        PackRequest.model_validate(
            {**request.model_dump(), "runtime_env": {"TOKEN": "synthetic"}}
        )


def test_gpu_drift_fails_before_loading_scientific_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
) -> None:
    from artisan_submitit.worker import run_worker_pack

    request = _request(tmp_path)
    request.gpu = ScheduledGPUProfile(
        node="gpu-node", software=allocation.software_profile()
    )
    request.calls[0].load = Mock()
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.LocalWorkerProfile.from_json", lambda _encoded: None
    )
    monkeypatch.setattr(
        "artisan_submitit.worker.capture_allocated_gpu",
        Mock(side_effect=ValueError("driver changed")),
    )
    launch = Mock()
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    run_worker_pack(str(tmp_path), "c" * 64)
    launch.assert_not_called()
    request.calls[0].load.assert_not_called()
    evidence = UnitEvidence.model_validate_json(
        (tmp_path / "entries" / ("b" * 32) / "unit-0.json").read_bytes()
    )
    assert "driver changed" in evidence.reply
    assert evidence.ownership == "closed"


def test_probe_timeout_bounds_cleanup_with_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = [0.0]
    monkeypatch.setattr("artisan_submitit.runner.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "artisan_submitit.runner.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    owner = Mock(ownership_state=lambda: OwnershipState.PENDING)
    handle = SchedulerProbeHandle(owner, 0)

    def timeout(
        _self: SchedulerProbeHandle, timeout: float | None = None
    ) -> ScheduledGPUProfile:
        assert timeout == 110
        now[0] += timeout
        raise TimeoutError("bounded probe timeout")

    monkeypatch.setattr(SchedulerProbeHandle, "result", timeout)
    owner.handles = (handle,)
    runner = _runner()
    object.__setattr__(runner, "log_folder", tmp_path)
    monkeypatch.setattr(type(runner), "create_owner", lambda *_args: owner)
    with pytest.raises(AssignmentLaunchError) as caught:
        runner._measure_gpu(
            GPUProbeSpec(node="gpu-node", device_count=1), Mock(), curator=False
        )
    owner.request_close.assert_called_once_with(120)
    assert 120 <= now[0] < 120.1
    assert caught.value.cancellation.status.value == "unknown"


def _runtime(
    root: Path,
    kind: Literal["apptainer", "docker"] = "apptainer",
    devices: tuple[str, ...] | None = None,
) -> ApptainerRuntime | DockerRuntime:
    executable = root / kind
    executable.write_text("#!/bin/sh\nexit 1\n")
    executable.chmod(0o700)
    if kind == "docker":
        return DockerRuntime(executable, gpu_devices=devices)
    return ApptainerRuntime(executable, gpu_devices=devices)


def _bindings(
    root: Path,
    runtime: ApptainerRuntime | DockerRuntime | None,
    allocation: NativeGPUAllocation | None = None,
) -> PreparedLocalBindings:
    worker = Mock(spec=PreparedNativeEnvironment)
    worker.compatibility = Mock()
    launcher = ContainerLauncherIdentity(
        kind="docker" if isinstance(runtime, DockerRuntime) else "apptainer",
        version="qualified",
        executable_sha256="f" * 64,
    )
    return PreparedLocalBindings(
        worker=worker,
        tools=(PreparedToolBinding(("tool",), Mock()),) if runtime else (),
        pixi=root / "pixi",
        system_env=Path("/usr/bin/env"),
        launcher=LocalLauncherIdentity(
            "pixi 0.83.0", "a" * 64, "b" * 64, ("glibc", "2.35")
        ),
        container_runtime=runtime,
        container_launcher=launcher if runtime else None,
        gpu_driver=allocation.driver
        if allocation and runtime and runtime.gpu_devices
        else None,
    )


@pytest.mark.parametrize("kind", ["apptainer", "docker"])
@pytest.mark.parametrize("devices", [(), ("0",)])
def test_scheduler_rejects_controller_devices_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["apptainer", "docker"],
    devices: tuple[str, ...],
) -> None:
    runtime = _runtime(tmp_path, kind, devices)
    monkeypatch.setattr(
        type(runtime), "identity", Mock(side_effect=AssertionError("GPU lookup"))
    )
    with pytest.raises(ValueError, match="actual allocation"):
        SlurmRunner(environment=Mock(), log_folder=tmp_path, **{kind: runtime})


def test_scheduler_accepts_allocation_free_apptainer(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runner = SlurmRunner(environment=Mock(), log_folder=tmp_path, apptainer=runtime)
    assert runner.apptainer is runtime
    assert runtime.gpu_devices is None


def test_execution_copy_uses_only_fresh_uuid_and_driver(
    tmp_path: Path, allocation: NativeGPUAllocation
) -> None:
    runtime = _runtime(tmp_path)
    local = _bindings(tmp_path, runtime)
    fresh = allocation.model_copy(
        update={"devices": ("GPU-22222222-2222-2222-2222-222222222222",)}
    )
    bound = _allocated_bindings(
        local, fresh, container_launcher=local.container_launcher
    )
    assert bound is not local
    assert bound.native_gpu == fresh
    assert bound.container_runtime is not runtime
    assert bound.container_runtime.gpu_devices == fresh.devices
    assert bound.gpu_driver == fresh.driver
    assert bound.tools is local.tools
    assert bound.container_launcher is local.container_launcher
    assert local.native_gpu is None
    assert local.gpu_driver is None
    assert runtime.gpu_devices is None


def test_execution_copy_preserves_native_allocation_behavior(
    tmp_path: Path, allocation: NativeGPUAllocation
) -> None:
    local = _bindings(tmp_path, None)
    bound = _allocated_bindings(local, allocation)
    assert bound.native_gpu == allocation
    assert bound.container_runtime is None
    assert bound.gpu_driver is None
    assert local.native_gpu is None


def test_execution_copy_rejects_explicit_container_device_selection(
    tmp_path: Path, allocation: NativeGPUAllocation
) -> None:
    runtime = _runtime(tmp_path, devices=("0",))
    local = _bindings(tmp_path, runtime, allocation)
    with pytest.raises(ValueError, match="Controller container bindings"):
        _allocated_bindings(
            local, allocation, container_launcher=local.container_launcher
        )
    assert runtime.gpu_devices == ("0",)


def test_execution_copy_rejects_unqualified_docker_gpu(
    tmp_path: Path, allocation: NativeGPUAllocation
) -> None:
    local = _bindings(tmp_path, _runtime(tmp_path, "docker"))
    with pytest.raises(ValueError, match="qualified Apptainer"):
        _allocated_bindings(local, allocation)


def test_container_gpu_requires_measured_sif_targets(
    tmp_path: Path, allocation: NativeGPUAllocation
) -> None:
    local = _bindings(tmp_path, _runtime(tmp_path))
    gpu = ScheduledGPUProfile(node="gpu-node", software=allocation.software_profile())
    with pytest.raises(ValueError, match="unique targets"):
        SlurmSoftware(local, Mock(), RunnerResources(gpus=1).model_dump_json(), gpu)


def test_cpu_container_binding_stays_allocation_free(tmp_path: Path) -> None:
    local = _bindings(tmp_path, _runtime(tmp_path))
    bound = _allocated_bindings(local, None)
    assert bound.native_gpu is None
    assert bound.container_runtime is local.container_runtime
    assert bound.gpu_driver is None


@pytest.fixture
def sif(tmp_path: Path) -> SIFProbeSpec:
    runtime = _runtime(tmp_path)
    assert isinstance(runtime, ApptainerRuntime)
    launcher = ContainerLauncherIdentity(
        kind="apptainer", version="apptainer version 1.5.4", executable_sha256="a" * 64
    )
    image = tmp_path / "tool.sif"
    image.write_bytes(b"prepared tool image")
    target = PreparedSIFEnvironment.model_validate(
        {
            "sif": image,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "build_runtime": launcher,
            "derivation": {
                "source_image_id": "sha256:" + "c" * 64,
                "source_config_sha256": "d" * 64,
                "rootfs_sha256": "e" * 64,
                "flattened_image_id": "sha256:" + "f" * 64,
                "archive_sha256": "1" * 64,
            },
            "oci": {
                "image_id": "sha256:" + "c" * 64,
                "base_image_id": "sha256:" + "2" * 64,
                "recipe_sha256": "3" * 64,
                "pixi_sha256": "4" * 64,
                "build_runtime": {
                    "kind": "docker",
                    "version": "28.5.1/28.5.1",
                    "executable_sha256": "5" * 64,
                    "engine_id": "synthetic-engine",
                },
                "native": {
                    "source": {
                        "manifest_name": "pixi.toml",
                        "name": "tool",
                        "files": [
                            {"path": "pixi.lock", "sha256": "6" * 64},
                            {"path": "pixi.toml", "sha256": "7" * 64},
                        ],
                    },
                    "platform": "linux-64",
                    "workspace": "/opt/artisan/environment",
                    "prefix": "/opt/artisan/environment/.pixi/envs/tool",
                },
            },
        }
    )
    (image.parent / "binding.json").write_text(target.model_dump_json())
    return SIFProbeSpec(runtime=runtime, launcher=launcher, targets=(target,))


@pytest.fixture
def gpu_launcher(sif: SIFProbeSpec) -> ContainerLauncherIdentity:
    values = sif.launcher.model_dump()
    values["helpers"] = [
        {"path": "gpu/rootfs/.singularity.d/libs/libcuda.so.1", "sha256": "b" * 64},
        {
            "path": "gpu/rootfs/.singularity.d/libs/libnvidia-ptxjitcompiler.so.1",
            "sha256": "c" * 64,
        },
    ]
    return ContainerLauncherIdentity.model_validate(values)


def _sif_bindings(tmp_path: Path, sif: SIFProbeSpec) -> PreparedLocalBindings:
    return replace(
        _bindings(tmp_path, sif.runtime),
        tools=(PreparedToolBinding(("tool",), sif.targets[0]),),
        container_launcher=sif.launcher,
    )


def _mock_sif_host(
    monkeypatch: pytest.MonkeyPatch,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
) -> None:
    target = sif.targets[0]
    labels = {
        "org.artisan.prepared.payload": target.oci.payload_digest,
        "org.artisan.oci.source": target.derivation.source_image_id,
        "org.artisan.oci.config": target.derivation.source_config_sha256,
        "org.artisan.oci.rootfs": target.derivation.rootfs_sha256,
    }
    monkeypatch.setattr(
        "subprocess.check_output",
        Mock(return_value=json.dumps({"data": {"attributes": {"labels": labels}}})),
    )
    monkeypatch.setattr(
        ApptainerRuntime,
        "identity",
        lambda runtime: gpu_launcher if runtime.gpu_devices else sif.launcher,
    )


def test_sif_probe_roundtrip_preserves_exact_targets(sif: SIFProbeSpec) -> None:
    probe = GPUProbeSpec(node="gpu-node", device_count=1, sif=sif)
    restored = GPUProbeSpec.model_validate_json(probe.model_dump_json())
    assert restored == probe
    assert restored.sif.runtime.gpu_devices is None


@pytest.mark.parametrize("field", ["runtime", "targets"])
def test_sif_probe_rejects_allocation_or_incomplete_targets(
    sif: SIFProbeSpec, field: str
) -> None:
    values = sif.model_dump()
    values["runtime"] = sif.runtime
    values[field] = (
        replace(sif.runtime, gpu_devices=("0",)) if field == "runtime" else ()
    )
    with pytest.raises(ValueError, match="allocation-free"):
        SIFProbeSpec.model_validate(values)


def test_sif_measurement_checks_launcher_image_and_selected_gpu_payload(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
) -> None:
    seen = []
    monkeypatch.setattr(
        ApptainerRuntime,
        "identity",
        lambda runtime: seen.append(runtime.gpu_devices)
        or (gpu_launcher if runtime.gpu_devices else sif.launcher),
    )
    verify = Mock()
    monkeypatch.setattr(gpu, "verify_sif_environment", verify)
    assert sif.verify(allocation, expected=gpu_launcher) == gpu_launcher
    assert seen == [None, allocation.devices]
    verify.assert_called_once_with(sif.targets[0], apptainer=sif.runtime)
    assert sif.runtime.gpu_devices is None


def test_sif_probe_reuses_prepared_target_without_reading_image_bytes(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
) -> None:
    image = sif.targets[0].sif
    image.write_bytes(b"installed image bytes are not audited")
    _mock_sif_host(monkeypatch, sif, gpu_launcher)
    original = Path.read_bytes

    def read_metadata(path: Path) -> bytes:
        if path == image:
            pytest.fail("Prepared image payload was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_metadata)
    monkeypatch.setattr(
        "artisan.execution.environments.prepared_sif.file_digest",
        Mock(side_effect=AssertionError("Prepared image payload was hashed")),
    )
    assert sif.verify(allocation, expected=gpu_launcher) == gpu_launcher


def test_sif_launcher_drift_rejects_before_image_access(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
) -> None:
    monkeypatch.setattr(ApptainerRuntime, "identity", lambda _runtime: gpu_launcher)
    verify = Mock()
    monkeypatch.setattr(gpu, "verify_sif_environment", verify)
    with pytest.raises(ValueError, match="launcher changed"):
        sif.verify(allocation)
    verify.assert_not_called()


def test_sif_gpu_payload_drift_rejects_current_measurement(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
) -> None:
    changed = gpu_launcher.model_copy(
        update={
            "helpers": (
                *gpu_launcher.helpers,
                gpu_launcher.helpers[0].model_copy(
                    update={"path": "gpu/config/nvliblist.conf", "sha256": "f" * 64}
                ),
            )
        }
    )
    monkeypatch.setattr(
        ApptainerRuntime,
        "identity",
        lambda runtime: changed if runtime.gpu_devices else sif.launcher,
    )
    monkeypatch.setattr(gpu, "verify_sif_environment", Mock())
    with pytest.raises(ValueError, match="GPU software changed"):
        sif.verify(allocation, expected=gpu_launcher)


def test_sif_resolution_uses_closed_scheduled_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    scheduler: Mock,
) -> None:
    local = _sif_bindings(tmp_path, sif)
    monkeypatch.setattr(LocalRunner, "resolve", Mock(return_value=local))
    monkeypatch.setattr(
        gpu, "capture_native_gpu", Mock(side_effect=AssertionError("login GPU"))
    )
    measured = ScheduledGPUProfile(
        node="gpu-node",
        software=allocation.software_profile(),
        sif=sif,
        container_launcher=gpu_launcher,
    )
    measure = Mock(return_value=measured)
    monkeypatch.setattr(SchedulerRunner, "_measure_gpu", measure)
    runner = _runner()
    object.__setattr__(runner, "parameters", {"slurm_nodelist": "gpu-node"})
    software = runner.resolve(Mock(), RunnerResources(gpus=1))
    assert software.gpu == measured
    assert measure.call_args.args[0].sif == sif
    assert local.native_gpu is None
    assert local.container_runtime.gpu_devices is None


def test_sif_gpu_digest_binds_selected_payloads_without_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    scheduler: Mock,
) -> None:
    local = _sif_bindings(tmp_path, sif)
    monkeypatch.setattr(
        PreparedLocalBindings, "software_digest", property(lambda _self: "e" * 64)
    )
    measured = ScheduledGPUProfile(
        node="gpu-node",
        software=allocation.software_profile(),
        sif=sif,
        container_launcher=gpu_launcher,
    )
    software = SlurmSoftware(
        local, scheduler, RunnerResources(gpus=1).model_dump_json(), measured
    )
    moved = replace(software, gpu=measured.model_copy(update={"node": "another-node"}))
    assert moved.software_digest == software.software_digest
    changed = replace(
        software,
        gpu=measured.model_copy(
            update={
                "container_launcher": gpu_launcher.model_copy(update={"helpers": ()})
            }
        ),
    )
    assert changed.software_digest != software.software_digest
    assert local.container_runtime.gpu_devices is None


@pytest.mark.parametrize("drift", ["missing_image", "selected_payload"])
def test_sif_entry_drift_prevents_scientific_payload_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    drift: str,
) -> None:
    from artisan_submitit.worker import run_worker_pack

    measured = gpu_launcher
    if drift == "missing_image":
        sif.targets[0].sif.unlink()
        error = "Prepared SIF is unavailable"
    else:
        measured = gpu_launcher.model_copy(
            update={
                "helpers": (
                    *gpu_launcher.helpers,
                    gpu_launcher.helpers[0].model_copy(
                        update={"path": "gpu/config/nvliblist.conf", "sha256": "f" * 64}
                    ),
                )
            }
        )
        error = "Scheduled SIF GPU software changed"
    _mock_sif_host(monkeypatch, sif, measured)
    request = _request(tmp_path)
    request.gpu = ScheduledGPUProfile(
        node="gpu-node",
        software=allocation.software_profile(),
        sif=sif,
        container_launcher=gpu_launcher,
    )
    request.calls[0].load = Mock()
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.LocalWorkerProfile.from_json", lambda _encoded: None
    )
    monkeypatch.setattr(
        "artisan_submitit.worker.capture_allocated_gpu", lambda *_args: allocation
    )
    launch = Mock()
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    run_worker_pack(str(tmp_path), "c" * 64)
    launch.assert_not_called()
    request.calls[0].load.assert_not_called()
    evidence = UnitEvidence.model_validate_json(
        (tmp_path / "entries" / ("b" * 32) / "unit-0.json").read_bytes()
    )
    assert error in evidence.reply
    assert evidence.ownership == "closed"


def test_sif_adapter_installs_measured_launcher_only_in_execution_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    scheduler: Mock,
) -> None:
    local = _sif_bindings(tmp_path, sif)
    measured = ScheduledGPUProfile(
        node="gpu-node",
        software=allocation.software_profile(),
        sif=sif,
        container_launcher=gpu_launcher,
    )
    software = SlurmSoftware(
        local, scheduler, RunnerResources(gpus=1).model_dump_json(), measured
    )
    monkeypatch.setattr(
        "artisan_submitit.software.capture_allocated_gpu", lambda *_args: allocation
    )
    monkeypatch.setattr("artisan_submitit.software.verify_native_gpu_loader", Mock())
    verify = Mock(return_value=gpu_launcher)
    monkeypatch.setattr(SIFProbeSpec, "verify", verify)
    selected = []
    monkeypatch.setattr(
        PreparedLocalBindings,
        "create_adapter",
        lambda binding, *_args, **_kwargs: selected.append(binding),
    )
    software.create_adapter(Mock(), working_root=tmp_path, runtime_env={})
    assert len(selected) == 1
    assert selected[0].container_launcher == gpu_launcher
    assert selected[0].container_runtime.gpu_devices == allocation.devices
    assert selected[0].gpu_driver == allocation.driver
    assert local.container_launcher == sif.launcher
    assert local.container_runtime.gpu_devices is None
    verify.assert_called_once_with(allocation, expected=gpu_launcher)


@pytest.mark.parametrize("missing", [False, True])
def test_fixed_sif_probe_checks_container_without_scientific_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    missing: bool,
) -> None:
    from artisan_submitit.worker import run_worker_pack

    request = _request(tmp_path)
    request.calls = ()
    request.probe = GPUProbeSpec(node="gpu-node", device_count=1, sif=sif)
    monkeypatch.setenv("ARTISAN_SUBMITIT_ENTRY_ID", "b" * 32)
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    monkeypatch.setattr("artisan_submitit.worker.read_pack", lambda *_args: request)
    monkeypatch.setattr(
        "artisan_submitit.worker.capture_allocated_gpu", lambda *_args: allocation
    )
    _mock_sif_host(monkeypatch, sif, gpu_launcher)
    if missing:
        sif.targets[0].sif.unlink()
    launch = Mock()
    monkeypatch.setattr("artisan_submitit.worker.launch_inert_calls", launch)
    run_worker_pack(str(tmp_path), "c" * 64)
    launch.assert_not_called()
    evidence = UnitEvidence.model_validate_json(
        (tmp_path / "entries" / ("b" * 32) / "unit-0.json").read_bytes()
    )
    assert evidence.payload_sha256 == request.probe.digest
    assert evidence.ownership == "closed"
    reply = GPUProbeReply.model_validate_json(evidence.reply)
    if missing:
        with pytest.raises(RuntimeError, match="Prepared SIF is unavailable"):
            reply.result()
    else:
        profile = reply.result()
        assert profile.sif == sif
        assert profile.container_launcher == gpu_launcher
        assert profile.software == allocation.software_profile()


@pytest.mark.parametrize("missing", [False, True])
def test_sif_selected_libraries_must_match_fresh_native_driver(
    monkeypatch: pytest.MonkeyPatch,
    allocation: NativeGPUAllocation,
    sif: SIFProbeSpec,
    gpu_launcher: ContainerLauncherIdentity,
    missing: bool,
) -> None:
    values = gpu_launcher.model_dump()
    if missing:
        values["helpers"] = values["helpers"][:-1]
    else:
        values["helpers"][0]["sha256"] = "f" * 64
    changed = ContainerLauncherIdentity.model_validate(values)
    monkeypatch.setattr(
        ApptainerRuntime,
        "identity",
        lambda runtime: changed if runtime.gpu_devices else sif.launcher,
    )
    monkeypatch.setattr(gpu, "verify_sif_environment", Mock())
    with pytest.raises(ValueError, match="freshly measured CUDA"):
        sif.verify(allocation)
