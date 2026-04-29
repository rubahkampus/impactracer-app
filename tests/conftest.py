"""Pytest fixtures shared across unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_repo_path() -> Path:
    """Path to the tests/fixtures/sample_repo fixture."""
    return Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def sample_gt_path() -> Path:
    """Path to the tests/fixtures/sample_gt.json fixture."""
    return Path(__file__).parent / "fixtures" / "sample_gt.json"
