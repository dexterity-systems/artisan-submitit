"""Tests for scheduler utility helpers."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from artisan_submitit.utils import (
    DEFAULT_MAX_ARRAY_SIZE,
    chunked,
    get_cluster_max_array_size,
    parse_time_to_minutes,
    read_job_logs,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("60", 60),
        ("30:00", 30),
        ("10:30", 11),
        ("01:30:00", 90),
        ("00:00:59", 1),
    ],
)
def test_parse_time_to_minutes(value: str, expected: int) -> None:
    assert parse_time_to_minutes(value) == expected


@pytest.mark.parametrize("value", ["", "1:2:3:4", "one", "01:60:00", "1:-1"])
def test_parse_time_to_minutes_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match=r"duration|negative"):
        parse_time_to_minutes(value)


@patch("artisan_submitit.utils.subprocess.run")
def test_get_cluster_max_array_size_detects_limit(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="MaxArraySize = 2048\nOther = value",
    )
    assert get_cluster_max_array_size() == 2048


@patch("artisan_submitit.utils.subprocess.run")
def test_get_cluster_max_array_size_falls_back(mock_run: MagicMock) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired("scontrol", 10)
    assert get_cluster_max_array_size() == DEFAULT_MAX_ARRAY_SIZE


def test_get_cluster_max_array_size_uses_override() -> None:
    assert get_cluster_max_array_size(17) == 17
    with pytest.raises(ValueError, match="must be >= 1"):
        get_cluster_max_array_size(0)


def test_chunked_preserves_order_and_remainder() -> None:
    assert chunked(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_read_job_logs_combines_and_truncates(tmp_path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("hello")
    stderr_path.write_text("problem")

    assert read_job_logs(stdout_path, stderr_path, retries=1) == (
        "hello\n--- stderr ---\nproblem"
    )


@patch("artisan_submitit.utils.time.sleep")
def test_read_job_logs_accepts_empty_existing_files_without_retry(
    sleep: MagicMock,
    tmp_path,
) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.touch()
    stderr_path.write_text("problem")

    assert read_job_logs(stdout_path, stderr_path) == "--- stderr ---\nproblem"
    sleep.assert_not_called()


@patch("artisan_submitit.utils.time.sleep")
def test_read_job_logs_retries_when_either_path_is_missing(
    sleep: MagicMock,
    tmp_path,
) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.touch()

    assert read_job_logs(stdout_path, stderr_path, retries=3, delay=0.1) is None
    assert sleep.call_count == 2


def test_read_job_logs_retries_delayed_stderr_visibility(tmp_path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.touch()

    def publish_stderr(_: float) -> None:
        stderr_path.write_text("late warning")

    with patch(
        "artisan_submitit.utils.time.sleep", side_effect=publish_stderr
    ) as sleep:
        assert read_job_logs(stdout_path, stderr_path) == (
            "--- stderr ---\nlate warning"
        )
    sleep.assert_called_once()
