"""Pytest configuration for MLX Primitives tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "stress: marks tests as stress tests (long running, resource intensive)")
    config.addinivalue_line(
        "markers",
        "cross_validation: marks tests that require PyTorch for cross-validation"
    )
    config.addinivalue_line(
        "markers",
        "cross_validation_jax: marks tests that require JAX for cross-validation"
    )


@pytest.fixture
def pytorch_available() -> bool:
    """Check if PyTorch is available for cross-validation."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_without_pytorch(pytorch_available: bool) -> None:
    """Skip test if PyTorch is not available."""
    if not pytorch_available:
        pytest.skip("PyTorch not available for cross-validation")


@pytest.fixture
def jax_available() -> bool:
    """Check if JAX is available for cross-validation."""
    try:
        import jax
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_without_jax(jax_available: bool) -> None:
    """Skip test if JAX is not available."""
    if not jax_available:
        pytest.skip("JAX not available for cross-validation")
