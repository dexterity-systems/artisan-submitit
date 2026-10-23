"""Operations used by real-scheduler provider tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, NoReturn

from artisan.operations.examples import Wait
from artisan.schemas.operation_config.environment_spec import LocalEnvironmentSpec
from artisan.schemas.specs.input_models import ExecuteInput
from artisan.utils.external_tools import run_command
from pydantic import Field


class FailingWait(Wait):
    """Emit a worker log marker, then fail inside the operation lifecycle."""

    name = "failing_wait"

    def execute_function(self, inputs: ExecuteInput) -> NoReturn:
        """Fail after proving scheduler stdout belongs to this unit."""
        del inputs
        print("artisan-submitit worker log marker", flush=True)  # noqa: T201
        msg = "intentional scheduler operation failure"
        raise RuntimeError(msg)


class ObservableWait(Wait):
    """Publish real Python execution readiness before waiting."""

    name = "observable_wait"

    class Params(Wait.Params):
        """Expose a shared observation directory to scheduler tests."""

        marker_root: str = Field(
            description="Shared directory for scheduler test observations."
        )

    params: Params

    def execute_function(self, inputs: ExecuteInput) -> dict[str, Any]:
        """Publish startup from the actual worker, then execute Wait normally."""
        root = Path(self.params.marker_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "ready").write_text(str(os.getpid()))
        return super().execute_function(inputs)


_MANAGED_WAIT_SCRIPT = """
import os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
duration = float(sys.argv[2])
end = time.monotonic() + duration
(root / 'ready').write_text(str(os.getpid()))
while time.monotonic() < end and not (root / 'stop').exists():
    (root / 'heartbeat').write_text(str(time.monotonic_ns()))
    time.sleep(0.05)
pathlib.Path('wait_marker.csv').write_text('requested,actual\\n' + str(duration) + ',' + str(duration) + '\\n')
"""


class ManagedWait(ObservableWait):
    """Run an observable managed command under the scheduler worker owner."""

    name = "managed_wait"

    def execute_function(self, inputs: ExecuteInput) -> dict[str, Any]:
        """Publish readiness and heartbeat from a real managed child command."""
        Path(self.params.marker_root).mkdir(parents=True, exist_ok=True)
        run_command(
            LocalEnvironmentSpec(),
            [
                sys.executable,
                "-c",
                _MANAGED_WAIT_SCRIPT,
                self.params.marker_root,
                str(self.params.duration),
            ],
            cwd=inputs.execute_dir,
            log_path=inputs.log_path,
        )
        return {"elapsed": self.params.duration}
