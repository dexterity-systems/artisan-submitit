"""Verify one prepared Docker scheduler fixture before unprivileged execution."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

DOCKER = ("/usr/bin/docker", "--context", "default")
RECEIPT_PATH = "/qualification-config/fixture.json"


@dataclass(frozen=True)
class FixtureIdentity:
    """Strict host copy of the independently prepared fixture contract."""

    schema_version: int
    image_id: str
    user: str
    uid: int
    gid: int
    node_python_sha256: str
    slurm_conf_sha256: str
    slurmdbd_conf_sha256: str

    @classmethod
    def read(cls, path: Path, expected_sha256: str) -> FixtureIdentity:
        """Verify exact receipt bytes before interpreting their fields."""
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected_sha256:
            msg = "The prepared scheduler fixture receipt changed"
            raise ValueError(msg)
        value = json.loads(body)
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            msg = "The scheduler fixture receipt has unexpected fields"
            raise ValueError(msg)
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            msg = "The scheduler fixture receipt version is unsupported"
            raise ValueError(msg)
        if value["user"] != "qualification" or any(
            type(value[key]) is not int or value[key] < 1 for key in ("uid", "gid")
        ):
            msg = "The fixture requires an unprivileged qualification user"
            raise ValueError(msg)
        for key in ("node_python_sha256", "slurm_conf_sha256", "slurmdbd_conf_sha256"):
            if (
                not isinstance(value[key], str)
                or re.fullmatch(r"[a-f0-9]{64}", value[key]) is None
            ):
                msg = "The fixture requires exact software SHA256 values"
                raise ValueError(msg)
        if (
            not isinstance(value["image_id"], str)
            or re.fullmatch(r"sha256:[a-f0-9]{64}", value["image_id"]) is None
        ):
            msg = "The fixture requires an immutable Docker image ID"
            raise ValueError(msg)
        return cls(**value)


def expected_node_python(receipt: Path, *, receipt_sha256: str) -> str:
    """Reject receipt, account, or software drift before running a fault helper.

    The host launcher supplies the receipt hash and checks the immutable image ID
    and root-owned accounting configuration before launching the controller.
    """
    identity = FixtureIdentity.read(receipt, receipt_sha256)
    if (
        os.getuid() != identity.uid
        or os.getgid() != identity.gid
        or pwd.getpwuid(os.getuid()).pw_name != identity.user
    ):
        raise ValueError("The controller is not the prepared qualification account")
    for path, expected in (
        (Path("/usr/bin/python3"), identity.node_python_sha256),
        (Path("/etc/slurm/slurm.conf"), identity.slurm_conf_sha256),
    ):
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            msg = f"The prepared scheduler fixture changed: {path}"
            raise ValueError(msg)
    return identity.node_python_sha256


def _docker_output(arguments: Sequence[str]) -> str:
    """Run only a bounded read of the selected local Docker fixture."""
    return subprocess.run(
        [*DOCKER, *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def _container_id(
    container: str, identity: FixtureIdentity, expected_container_id: str
) -> str:
    observed = json.loads(_docker_output(["container", "inspect", container]))
    if not isinstance(observed, list) or len(observed) != 1:
        msg = "Expected exactly one named qualification container"
        raise ValueError(msg)
    selected = observed[0]
    if (
        not isinstance(selected, dict)
        or selected.get("Name") != "/" + container
        or selected.get("Image") != identity.image_id
        or selected.get("Id") != expected_container_id
        or not isinstance(selected.get("Id"), str)
        or re.fullmatch(r"[a-f0-9]{64}", selected["Id"]) is None
    ):
        msg = "The named qualification container or immutable image changed"
        raise ValueError(msg)
    return str(selected["Id"])


def launch(
    command: Sequence[str],
    *,
    root: Path,
    container: str,
    receipt_sha256: str,
    expected_container_id: str,
) -> None:
    """Reject fixture drift before replacing this process with unprivileged exec."""
    if not command or not command[0]:
        msg = "An explicit fixture command is required"
        raise ValueError(msg)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container) is None:
        msg = "QUALIFICATION_CONTAINER must be one literal container name"
        raise ValueError(msg)
    if re.fullmatch(r"[a-f0-9]{64}", expected_container_id) is None:
        msg = "A prepared immutable qualification container ID is required"
        raise ValueError(msg)
    receipt = root / "config/fixture.json"
    if not root.is_absolute() or receipt.resolve(strict=True) != receipt:
        msg = "QUALIFICATION_ROOT must identify the real absolute fixture root"
        raise ValueError(msg)
    identity = FixtureIdentity.read(receipt, receipt_sha256)
    container_id = _container_id(container, identity, expected_container_id)
    paths = (
        "/usr/bin/python3",
        "/etc/slurm/slurm.conf",
        "/etc/slurm/slurmdbd.conf",
    )
    actual = _docker_output(
        ["exec", "--user", "0", "--", container_id, "/usr/bin/sha256sum", "--", *paths]
    ).splitlines()
    expected = [
        f"{digest}  {path}"
        for digest, path in zip(
            (
                identity.node_python_sha256,
                identity.slurm_conf_sha256,
                identity.slurmdbd_conf_sha256,
            ),
            paths,
            strict=True,
        )
    ]
    if actual != expected:
        msg = "The node Python or scheduler configuration changed"
        raise ValueError(msg)
    argv = [
        *DOCKER,
        "exec",
        "--user",
        f"{identity.uid}:{identity.gid}",
        "--env",
        f"ARTISAN_SLURM_FIXTURE_RECEIPT={RECEIPT_PATH}",
        "--env",
        f"ARTISAN_SLURM_FIXTURE_SHA256={receipt_sha256}",
        "--",
        container_id,
        *command,
    ]
    os.execv(DOCKER[0], argv)


def main() -> None:
    """Require the workflow's explicit root, container name, and receipt digest."""
    launch(
        sys.argv[1:],
        root=Path(os.environ["QUALIFICATION_ROOT"]),
        container=os.environ["QUALIFICATION_CONTAINER"],
        receipt_sha256=os.environ["QUALIFICATION_FIXTURE_SHA256"],
        expected_container_id=os.environ["QUALIFICATION_CONTAINER_ID"],
    )


if __name__ == "__main__":
    main()
