"""Prepare three ordinary wheels and a locked final-path scheduler test profile.

Run from the provider's own locked development interpreter. The supplied builder
binding must name an already prepared static Hatch environment. A supervising
caller owns the overall deadline and storage limits; this script never starts
scheduler work and never copies an installed environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

from artisan.orchestration.runner_api import (
    PreparedNativeEnvironment,
    verify_native_environment,
    verify_wheel_installation,
)
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    """Record package inputs conservatively, including canonical resource files."""
    paths = set((root / "src").rglob("*"))
    paths.update((root / "skills").rglob("*"))
    paths.update(
        root / name
        for name in (
            "pyproject.toml",
            "pixi.lock",
            "README.md",
            "LICENSE",
            ".gitignore",
        )
    )
    inventory = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(paths)
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    inventory.update(
        {
            f"@root-member:{path.name}": "symlink"
            if path.is_symlink()
            else "directory"
            if path.is_dir()
            else "file"
            for path in root.iterdir()
            if path.name not in {".git", ".pixi", "__pycache__"}
        }
    )
    return inventory


def _versions(
    requirements: Iterable[Requirement], local: dict[str, str]
) -> dict[str, str]:
    pending = list(requirements)
    seen: set[tuple[str, tuple[str, ...]]] = set()
    versions: dict[str, str] = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        version = local.get(name) or importlib.metadata.version(name)
        if version not in requirement.specifier:
            msg = (
                f"Controller dependency {name} {version} does not satisfy {requirement}"
            )
            raise ValueError(msg)
        key = (name, tuple(sorted(requirement.extras)))
        if name in local or key in seen:
            continue
        seen.add(key)
        versions[name] = version
        for raw in importlib.metadata.requires(name) or ():
            child = Requirement(raw)
            if child.marker is None or any(
                child.marker.evaluate({"extra": extra})
                for extra in ("", *requirement.extras)
            ):
                child.marker = None
                pending.append(child)
    return versions


def _run(
    command: list[str], *, cwd: Path, environment: dict[str, str], timeout: int = 600
) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=timeout)


def _copy_verified_wheel(source: Path, destination: Path, digest: str) -> None:
    """Copy only the caller's exact ordinary-wheel bytes, verifying both sides."""
    if _sha(source) != digest:
        msg = "Selected wheel differs from its required SHA256"
        raise ValueError(msg)
    shutil.copyfile(source, destination)
    if _sha(destination) != digest:
        msg = "Copied wheel differs from the exact selected wheel bytes"
        raise ValueError(msg)


def _shared_core_wheel(
    source: Path, digest: str, core: Path, destination: Path
) -> dict[str, str]:
    """Consume a separately measured core build without rebuilding its bytes."""
    project = tomllib.loads((core / "pyproject.toml").read_text())["project"]
    name, version, _build, _tags = parse_wheel_filename(source.name)
    if name != canonicalize_name(project["name"]) or version != Version(
        project["version"]
    ):
        msg = "Selected shared wheel is not this core distribution and version"
        raise ValueError(msg)
    _copy_verified_wheel(source, destination / source.name, digest)
    return {source.name: digest}


def _reuse_wheels(
    profile: Path, roots: tuple[Path, ...], destination: Path
) -> dict[str, str]:
    """Reuse only unchanged source-bound core/provider wheel bytes."""
    receipt = json.loads((profile / "installed-receipt.json").read_bytes())
    if receipt.get("installed") is not True:
        msg = "Wheel reuse requires a verified installed profile receipt"
        raise ValueError(msg)
    reused: dict[str, str] = {}
    for root in roots[:2]:
        if receipt["source_inventory"].get(str(root)) != _inventory(root):
            msg = "Reused wheel source differs from the exact installed profile"
            raise ValueError(msg)
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        name = canonicalize_name(project["name"])
        selected = [
            filename
            for filename in receipt["wheel_sha256"]
            if canonicalize_name(filename.split("-")[0]) == name
        ]
        if len(selected) != 1:
            msg = "Reused profile must name exactly one wheel per distribution"
            raise ValueError(msg)
        filename = selected[0]
        source = profile / "wheels" / filename
        digest = receipt["wheel_sha256"][filename]
        if source.name != filename:
            msg = "Reused wheel name differs from the installed profile receipt"
            raise ValueError(msg)
        _copy_verified_wheel(source, destination / filename, digest)
        reused[filename] = digest
    return reused


def prepare(args: argparse.Namespace) -> dict[str, object]:
    provider = Path(__file__).resolve().parents[2]
    roots = (
        args.core.resolve(),
        provider,
        provider / "tests/qualification/application",
    )
    documents = [
        tomllib.loads((root / "pyproject.toml").read_text())["project"]
        for root in roots
    ]
    local = {canonicalize_name(doc["name"]): doc["version"] for doc in documents}
    runtime = _versions(
        (Requirement(value) for doc in documents for value in doc["dependencies"]),
        local,
    )
    controller = _versions(
        (Requirement("pytest"), Requirement("pytest-timeout")), local
    )
    builder = PreparedNativeEnvironment.model_validate_json(
        args.builder_binding.read_bytes()
    )
    verify_native_environment(builder)
    builder_python = builder.prefix / "bin/python"
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    wheels = destination / "wheels"
    wheels.mkdir()
    scratch = destination / "tmp"
    scratch.mkdir()
    package_cache = (args.package_cache or destination / "cache").resolve()
    environment = {
        **os.environ,
        "TMPDIR": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIXI_CACHE_DIR": str(package_cache / "pixi"),
        "UV_CACHE_DIR": str(package_cache / "uv"),
    }
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    before = {str(root): _inventory(root) for root in roots}
    _run(
        [
            str(builder_python),
            "-I",
            "-c",
            "from importlib.metadata import version; assert version('hatchling') == '1.31.0'",
        ],
        cwd=destination,
        environment=environment,
    )
    reused: dict[str, str] = {}
    build_roots = roots
    if args.reuse_profile is not None:
        reused = _reuse_wheels(args.reuse_profile.resolve(), roots, wheels)
        build_roots = roots[2:]
    elif args.core_wheel is not None:
        reused = _shared_core_wheel(
            args.core_wheel.resolve(), args.core_wheel_sha256, roots[0], wheels
        )
        build_roots = roots[1:]
    for root in build_roots:
        _run(
            [
                str(builder_python),
                "-I",
                "-m",
                "hatchling",
                "build",
                "--target",
                "wheel",
                "--directory",
                str(wheels),
            ],
            cwd=root,
            environment=environment,
            timeout=180,
        )
    if before != {str(root): _inventory(root) for root in roots}:
        msg = "Package source changed during ordinary-wheel construction"
        raise ValueError(msg)
    verify_native_environment(builder)
    built = tuple(sorted(wheels.glob("*.whl")))
    if len(built) != 3 or {
        canonicalize_name(path.name.split("-")[0]) for path in built
    } != set(local):
        msg = "Expected exactly the core, provider and qualification application wheels"
        raise ValueError(msg)
    lines = [
        "[workspace]",
        'name = "installed-slurm-qualification"',
        'channels = ["conda-forge"]',
        'platforms = ["linux-64"]',
        'requires-pixi = "==0.83.0"',
        "",
        "[environments]",
        'controller = {features = ["worker", "controller"], no-default-feature = true}',
        'worker = {features = ["worker"], no-default-feature = true}',
        "",
        "[feature.worker.dependencies]",
        f'python = "=={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"',
        "",
        "[feature.worker.pypi-dependencies]",
    ]
    if args.gpu_tool:
        index = lines.index("[feature.worker.dependencies]") - 1
        lines.insert(
            index, 'gpu-tool = {features = ["gpu-tool"], no-default-feature = true}'
        )
    lines += [
        f"{json.dumps(canonicalize_name(path.name.split('-')[0]))} = {{path = {json.dumps('wheels/' + path.name)}}}"
        for path in built
    ]
    lines += [
        f'{json.dumps(name)} = "=={version}"'
        for name, version in sorted(runtime.items())
    ]
    lines += ["", "[feature.controller.pypi-dependencies]"]
    lines += [
        f'{json.dumps(name)} = "=={version}"'
        for name, version in sorted(controller.items())
        if name not in runtime
    ]
    if args.gpu_tool:
        lines += [
            "",
            "[feature.gpu-tool.dependencies]",
            f'python = "=={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"',
            "",
            "[feature.gpu-tool.pypi-dependencies]",
            f'packaging = "=={runtime["packaging"]}"',
            "",
            "[feature.gpu-tool.activation.env]",
            'GPU_TOOL_ACTIVATION = "prepared-native-gpu"',
        ]
    manifest = destination / "pixi.toml"
    manifest.write_text("\n".join(lines) + "\n")
    receipt = {
        "source_inventory": before,
        "builder_binding": str(args.builder_binding),
        "builder_digest": builder.digest,
        "wheel_sha256": {path.name: _sha(path) for path in built},
        "runtime_versions": runtime,
        "controller_versions": controller,
        "launcher_sha256": _sha(Path(__file__)),
        "scope": "ordinary installed profile; no scheduler work",
        "reused_wheel_sha256": reused,
        "gpu_tool": args.gpu_tool,
        "package_cache": str(package_cache),
    }
    (destination / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    _run(
        [str(args.pixi), "lock", "--no-config", "--manifest-path", str(manifest)],
        cwd=destination,
        environment=environment,
    )
    for name in ("controller", "worker"):
        _run(
            [
                str(args.pixi),
                "install",
                "--locked",
                "--no-config",
                "--manifest-path",
                str(manifest),
                "-e",
                name,
            ],
            cwd=destination,
            environment=environment,
        )
        interpreter = destination / ".pixi/envs" / name / "bin/python"
        verify_wheel_installation(built, interpreter=interpreter)
        _run(
            [
                str(interpreter),
                "-I",
                "-c",
                "import artisan, artisan_submitit, slurm_qualification_app, sys; from pathlib import Path; assert all(Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) for module in (artisan, artisan_submitit, slurm_qualification_app))",
            ],
            cwd=destination,
            environment=environment,
        )
    if before != {str(root): _inventory(root) for root in roots}:
        msg = "Package source changed during profile installation"
        raise ValueError(msg)
    return {
        **receipt,
        "manifest_sha256": _sha(manifest),
        "lock_sha256": _sha(destination / "pixi.lock"),
        "installed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--builder-binding", type=Path, required=True)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-profile", type=Path, default=None)
    reuse.add_argument("--core-wheel", type=Path, default=None)
    parser.add_argument("--core-wheel-sha256", default=None)
    parser.add_argument("--gpu-tool", action="store_true")
    parser.add_argument("--package-cache", type=Path, default=None)
    parser.add_argument("--pixi", type=Path, default=Path.home() / ".pixi/bin/pixi")
    args = parser.parse_args()
    if (args.core_wheel is None) != (args.core_wheel_sha256 is None):
        parser.error("--core-wheel and --core-wheel-sha256 must be supplied together")
    receipt = prepare(args)
    (args.destination / "installed-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
