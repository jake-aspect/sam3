"""Shared fixtures for the SAM3 test suite."""

import torch
import pytest


@pytest.fixture()
def device() -> str:
    """Return the best available device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"
