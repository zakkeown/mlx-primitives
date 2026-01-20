"""Pytest configuration for correctness tests.

This module provides:
- Dual-mode testing: golden files (no PyTorch) vs live comparison (with PyTorch)
- Golden file loading utilities
- Numerical comparison utilities with configurable tolerances
- Standard fixtures for test inputs
"""

import sys
from pathlib import Path

# Add tests/correctness to path so golden_utils can be imported
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

import mlx.core as mx

# Try to import PyTorch for live comparison mode
try:
    import torch

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


# =============================================================================
# Paths
# =============================================================================

GOLDEN_DIR = Path(__file__).parent.parent / "golden"


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--comparison-mode",
        action="store",
        default="golden",
        choices=["golden", "live"],
        help="Comparison mode: 'golden' uses pre-generated files, 'live' compares against PyTorch",
    )


@pytest.fixture
def comparison_mode(request):
    """Get the comparison mode from command line option."""
    mode = request.config.getoption("--comparison-mode", default="golden")
    if mode == "live" and not HAS_PYTORCH:
        pytest.skip("Live comparison requires PyTorch: pip install torch")
    return mode


# =============================================================================
# Golden File Utilities
# =============================================================================


def golden_exists(category: str, name: str) -> bool:
    """Check if a golden file exists."""
    return (GOLDEN_DIR / category / f"{name}.npz").exists()


def load_golden(category: str, name: str) -> Dict[str, Any]:
    """Load golden file and parse metadata.

    Args:
        category: Category subdirectory (e.g., 'attention', 'activations')
        name: Name of the golden file (without .npz extension)

    Returns:
        Dict containing:
        - Input arrays (original keys)
        - Expected outputs (keys prefixed with 'expected_')
        - __metadata__: parsed metadata dict with tolerance info

    Raises:
        pytest.skip: If golden file not found
    """
    path = GOLDEN_DIR / category / f"{name}.npz"
    if not path.exists():
        pytest.skip(
            f"Golden file not found: {path}\n"
            f"Run: python scripts/validation/generate_all.py --category {category}"
        )

    data = np.load(path, allow_pickle=True)
    result = {}

    for key in data.files:
        if key == "__metadata__":
            # Parse metadata JSON
            metadata_str = str(data[key][0])
            result["__metadata__"] = json.loads(metadata_str)
        else:
            result[key] = data[key]

    return result


def get_tolerance_from_golden(
    golden: Dict[str, Any], dtype: str = "fp32"
) -> Tuple[float, float]:
    """Extract rtol/atol tolerance from golden file metadata.

    Args:
        golden: Loaded golden file dict
        dtype: One of 'fp32', 'fp16', 'bf16'

    Returns:
        Tuple of (rtol, atol)
    """
    metadata = golden.get("__metadata__", {})
    tolerance = metadata.get("tolerance", {})

    rtol_key = f"rtol_{dtype}"
    atol_key = f"atol_{dtype}"

    rtol = tolerance.get(rtol_key, 1e-5)
    atol = tolerance.get(atol_key, 1e-6)

    return rtol, atol


def get_max_mean_tolerance(
    golden: Dict[str, Any], dtype: str = "fp32"
) -> Tuple[Optional[float], Optional[float]]:
    """Extract max_diff/mean_diff tolerance from golden file metadata.

    Args:
        golden: Loaded golden file dict
        dtype: One of 'fp32', 'fp16', 'bf16'

    Returns:
        Tuple of (max_diff, mean_diff), either can be None
    """
    metadata = golden.get("__metadata__", {})
    tolerance = metadata.get("tolerance", {})

    max_diff = tolerance.get(f"max_diff_{dtype}")
    mean_diff = tolerance.get(f"mean_diff_{dtype}")

    return max_diff, mean_diff


# =============================================================================
# Numerical Comparison Utilities
# =============================================================================


def assert_close(
    actual: mx.array,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_diff: Optional[float] = None,
    mean_diff: Optional[float] = None,
    msg: str = "",
) -> None:
    """Assert arrays are close with multiple tolerance checks.

    Args:
        actual: MLX array from implementation under test
        expected: NumPy array with expected values (from golden file or PyTorch)
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose
        max_diff: Optional maximum absolute difference threshold
        mean_diff: Optional mean absolute difference threshold
        msg: Optional message prefix for assertion errors

    Raises:
        AssertionError: If any tolerance check fails
    """
    # Convert MLX array to NumPy
    actual_np = np.array(actual)

    # Check shapes match
    assert actual_np.shape == expected.shape, (
        f"{msg}Shape mismatch: got {actual_np.shape}, expected {expected.shape}"
    )

    # Check for NaN/Inf
    assert not np.any(np.isnan(actual_np)), f"{msg}Output contains NaN values"
    assert not np.any(np.isinf(actual_np)), f"{msg}Output contains Inf values"

    # Standard relative/absolute tolerance check
    np.testing.assert_allclose(
        actual_np,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{msg}Arrays not close within rtol={rtol}, atol={atol}",
    )

    # Additional max diff check
    if max_diff is not None:
        actual_max_diff = float(np.max(np.abs(actual_np - expected)))
        assert actual_max_diff < max_diff, (
            f"{msg}Max diff {actual_max_diff:.6e} exceeds threshold {max_diff:.6e}"
        )

    # Additional mean diff check
    if mean_diff is not None:
        actual_mean_diff = float(np.mean(np.abs(actual_np - expected)))
        assert actual_mean_diff < mean_diff, (
            f"{msg}Mean diff {actual_mean_diff:.6e} exceeds threshold {mean_diff:.6e}"
        )


def assert_close_golden(
    actual: mx.array,
    golden: Dict[str, Any],
    output_key: str = "out",
    dtype: str = "fp32",
    msg: str = "",
) -> None:
    """Assert MLX output matches golden file expected output.

    Convenience function that extracts expected output and tolerances
    from a loaded golden file.

    Args:
        actual: MLX array from implementation under test
        golden: Loaded golden file dict
        output_key: Key for expected output (will be prefixed with 'expected_')
        dtype: Dtype for tolerance lookup ('fp32', 'fp16', 'bf16')
        msg: Optional message prefix for assertion errors
    """
    expected_key = f"expected_{output_key}"
    assert expected_key in golden, f"Golden file missing expected output: {expected_key}"

    expected = golden[expected_key]
    rtol, atol = get_tolerance_from_golden(golden, dtype)
    max_diff, mean_diff = get_max_mean_tolerance(golden, dtype)

    assert_close(
        actual=actual,
        expected=expected,
        rtol=rtol,
        atol=atol,
        max_diff=max_diff,
        mean_diff=mean_diff,
        msg=msg,
    )


# =============================================================================
# Standard Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    mx.random.seed(42)
    np.random.seed(42)
    if HAS_PYTORCH:
        torch.manual_seed(42)
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
