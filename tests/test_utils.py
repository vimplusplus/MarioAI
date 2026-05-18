"""
Tests for utility helpers — checkpoint discovery and display formatting.
These tests use only the stdlib (no gym, torch, etc.).
"""

import os
import tempfile
from pathlib import Path

import pytest

from mario_ai.utils import (
    _extract_step,
    find_latest_checkpoint,
    format_config_table,
    list_checkpoints,
)


class TestExtractStep:
    def test_normal_filename(self):
        p = Path("best_model_70000.zip")
        assert _extract_step(p) == 70000

    def test_zero_step(self):
        p = Path("best_model_0.zip")
        assert _extract_step(p) == 0

    def test_unrecognised_filename(self):
        p = Path("some_other_file.zip")
        assert _extract_step(p) == -1


class TestFindLatestCheckpoint:
    def test_returns_none_when_dir_missing(self):
        assert find_latest_checkpoint("/nonexistent/path/xyz") is None

    def test_returns_none_when_dir_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert find_latest_checkpoint(tmp) is None

    def test_finds_highest_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            for step in (10000, 30000, 20000):
                Path(tmp, f"best_model_{step}.zip").touch()
            latest = find_latest_checkpoint(tmp)
            assert latest is not None
            assert "30000" in latest

    def test_single_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "best_model_5000.zip").touch()
            latest = find_latest_checkpoint(tmp)
            assert latest is not None
            assert "5000" in latest


class TestListCheckpoints:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert list_checkpoints(tmp) == []

    def test_missing_dir(self):
        assert list_checkpoints("/nonexistent/abc") == []

    def test_sorted_ascending(self):
        with tempfile.TemporaryDirectory() as tmp:
            for step in (30000, 10000, 20000):
                Path(tmp, f"best_model_{step}.zip").touch()
            result = list_checkpoints(tmp)
            steps = [s for s, _ in result]
            assert steps == sorted(steps)

    def test_returns_correct_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "best_model_1000.zip").touch()
            result = list_checkpoints(tmp)
            assert len(result) == 1
            step, path = result[0]
            assert isinstance(step, int)
            assert isinstance(path, Path)


class TestFormatConfigTable:
    def test_returns_string(self):
        table = format_config_table({"key": "value"})
        assert isinstance(table, str)

    def test_contains_key(self):
        table = format_config_table({"learning_rate": 1e-6})
        assert "learning_rate" in table

    def test_contains_value(self):
        table = format_config_table({"n_steps": 512})
        assert "512" in table
