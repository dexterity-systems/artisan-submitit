"""Execute the published tutorials against a real scheduler."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slurm, pytest.mark.timeout(600)]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "notebook_name",
    [
        pytest.param("01-first-slurm-pipeline.ipynb", id="sbatch-first-pipeline"),
        pytest.param("02-inside-an-allocation.ipynb", id="srun-allocation"),
        pytest.param("03-batching-and-commands.ipynb", id="sbatch-batching-commands"),
    ],
)
def test_tutorial(notebook_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run notebook assertions and retain executed copies for diagnosis."""
    import nbformat
    from nbclient import NotebookClient

    shared_root = Path(
        os.environ.get("ARTISAN_TUTORIAL_ROOT", _PROJECT_ROOT / "tutorial-runs")
    ).expanduser()
    if not shared_root.is_absolute():
        raise ValueError("ARTISAN_TUTORIAL_ROOT must be an absolute shared path")
    run_name = f"{Path(notebook_name).stem}-{uuid.uuid4().hex[:8]}"
    run_root = shared_root / run_name
    run_root.mkdir(parents=True)
    monkeypatch.setenv("ARTISAN_TUTORIAL_ROOT", str(run_root))
    output_root = _PROJECT_ROOT / "slurm_logs" / "tutorials" / run_name
    output_root.mkdir(parents=True)
    notebook = nbformat.read(
        _PROJECT_ROOT / "docs" / "tutorials" / notebook_name, as_version=4
    )
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=180,
        resources={"metadata": {"path": str(_PROJECT_ROOT)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output_root / notebook_name)
