from pathlib import Path

import pytest

from src.ds_agent_tools import LOG_FILE, preview_dataset, profile_dataset_tool


def _read_log_lines() -> list[str]:
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text(encoding="utf-8").splitlines()


def _lines_since(before: int) -> list[str]:
    lines = _read_log_lines()
    return lines[before:]


def test_preview_dataset_logs_usage(sample_context):
    start_lines = len(_read_log_lines())

    preview_dataset.invoke({"dataset_name": "left_dataset", "limit": 2})

    new_lines = _lines_since(start_lines)
    assert any("tool=preview_dataset" in line for line in new_lines)
    assert any("dataset=left_dataset" in line for line in new_lines)


@pytest.mark.usefixtures("sample_context")
def test_profile_dataset_logs_duration():
    cache_path = Path("data/profiles/left_dataset.json")
    try:
        if cache_path.exists():
            cache_path.unlink()

        start_lines = len(_read_log_lines())

        profile_dataset_tool.invoke({"dataset_name": "left_dataset", "sample_size": 3})

        new_lines = _lines_since(start_lines)
        assert any("tool=profile_dataset" in line for line in new_lines)
        assert any("duration=" in line for line in new_lines)
    finally:
        if cache_path.exists():
            cache_path.unlink()

