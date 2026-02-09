"""
Shared fixtures for the SAM3 test suite.
"""

import pytest


@pytest.fixture()
def device() -> str:
    """Return the best available device string."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
