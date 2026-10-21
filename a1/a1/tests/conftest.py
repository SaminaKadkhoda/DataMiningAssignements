"""Shared pytest configuration."""
import torch
import pytest

torch.set_num_threads(max(1, torch.get_num_threads() // 2))
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
