"""Pytest configuration and fixtures for MLX Primitives tests."""

import pytest
import mlx.core as mx


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    mx.random.seed(42)
    yield


@pytest.fixture
def small_batch():
    """Small batch size for quick tests."""
    return 2


@pytest.fixture
def medium_batch():
    """Medium batch size for typical tests."""
    return 8


@pytest.fixture
def large_batch():
    """Large batch size for stress tests."""
    return 32
