"""Pytest configuration for correctness tests."""

import pytest
import mlx.core as mx


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    mx.random.seed(42)
    yield


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return {"atol": 1e-5, "rtol": 1e-5}


@pytest.fixture
def loose_tolerance():
    """Looser tolerance for approximate algorithms."""
    return {"atol": 1e-3, "rtol": 1e-3}


@pytest.fixture
def sample_input_1d():
    """Sample 1D input tensor."""
    return mx.random.normal((2, 32, 64))


@pytest.fixture
def sample_input_2d():
    """Sample 2D input tensor (images)."""
    return mx.random.normal((2, 16, 16, 64))


@pytest.fixture
def sample_attention_input():
    """Sample input for attention layers."""
    batch_size = 2
    seq_len = 32
    dims = 128
    return mx.random.normal((batch_size, seq_len, dims))
