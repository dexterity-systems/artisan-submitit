"""Direct provider unit cancellation through real current admission and seals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, cast

import pytest

from .helpers import assert_closed, pipeline, runner

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(300)]


def test_unit_cancellation_keeps_shared_worker_and_surviving_receipt(
    tmp_path: Path,
) -> None:
    from ..qualification.unit_cancellation import qualify

    mode = os.environ["ARTISAN_SLURM_MODE"]
    assert mode in {"sbatch", "srun"}
    with pipeline(
        tmp_path, runner(tmp_path, cast(Literal["sbatch", "srun"], mode))
    ) as owner:
        receipt = qualify(owner, tmp_path / "markers")
    assert_closed(owner)
    assert receipt["ownership"] == ["closed"]
    (tmp_path / "unit-cancellation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
