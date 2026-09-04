"""Operations used by real-scheduler provider tests."""

from __future__ import annotations

from typing import NoReturn

from artisan.operations.examples import Wait
from artisan.schemas.specs.input_models import ExecuteInput


class FailingWait(Wait):
    """Emit a worker log marker, then fail inside the operation lifecycle."""

    name = "failing_wait"

    def execute_function(self, inputs: ExecuteInput) -> NoReturn:
        """Fail after proving scheduler stdout belongs to this unit."""
        del inputs
        print("artisan-submitit worker log marker", flush=True)  # noqa: T201
        msg = "intentional scheduler operation failure"
        raise RuntimeError(msg)
