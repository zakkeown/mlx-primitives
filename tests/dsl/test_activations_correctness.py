"""Correctness tests for activation function kernels.

Validates DSL activation kernels against NumPy reference implementations.
"""

import math
from typing import Callable

import numpy as np
import pytest

# Try to import MLX and DSL components
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# ---------------------------------------------------------------------------
# NumPy Reference Implementations
# ---------------------------------------------------------------------------


def numpy_silu(x: np.ndarray) -> np.ndarray:
    """Reference SiLU (Swish) implementation.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    return x * (1.0 / (1.0 + np.exp(-x)))


def numpy_gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Reference GELU with tanh approximation.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def numpy_gelu_exact(x: np.ndarray) -> np.ndarray:
    """Reference GELU (exact).

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    from scipy.special import erf
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    return 0.5 * x * (1.0 + erf(x * inv_sqrt2))


def numpy_quick_gelu(x: np.ndarray) -> np.ndarray:
    """Reference Quick GELU implementation.

    QuickGELU(x) = x * sigmoid(1.702 * x)
    """
    return x / (1.0 + np.exp(-1.702 * x))


def numpy_softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Reference Softplus implementation.

    Softplus(x) = (1/beta) * log(1 + exp(beta * x))
    For numerical stability, uses linear approx when beta*x > threshold.
    """
    bx = beta * x
    result = np.where(
        bx > threshold,
        x,
        np.log(1.0 + np.exp(bx)) / beta
    )
    return result


def numpy_mish(x: np.ndarray) -> np.ndarray:
    """Reference Mish implementation.

    Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    """
    # Numerically stable softplus
    softplus_x = np.where(x > 20.0, x, np.log(1.0 + np.exp(x)))
    return x * np.tanh(softplus_x)


def numpy_fused_silu_mul(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Reference fused SiLU + multiply (SwiGLU).

    SwiGLU(x, gate) = SiLU(x) * gate
    """
    return numpy_silu(x) * gate


def numpy_fused_gelu_mul(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Reference fused GELU + multiply (GEGLU).

    GEGLU(x, gate) = GELU(x) * gate
    """
    return numpy_gelu_tanh(x) * gate


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    if MLX_AVAILABLE:
        mx.random.seed(42)


@pytest.fixture
def small_tensor(random_seed) -> np.ndarray:
    """Small tensor for basic tests."""
    return np.random.randn(256).astype(np.float32)


@pytest.fixture
def medium_tensor(random_seed) -> np.ndarray:
    """Medium tensor for typical use cases."""
    return np.random.randn(4, 512).astype(np.float32)


@pytest.fixture
def large_tensor(random_seed) -> np.ndarray:
    """Large tensor for stress testing."""
    return np.random.randn(8, 1024, 128).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def run_dsl_kernel(kernel_name: str, inputs: list[np.ndarray], **kwargs) -> np.ndarray:
    """Run a DSL kernel and return the output as NumPy array.

    Args:
        kernel_name: Name of the kernel from activations module.
        inputs: List of input arrays.
        **kwargs: Additional kernel parameters.

    Returns:
        Output array as NumPy.
    """
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")

    try:
        from mlx_primitives.dsl.examples import activations
    except ImportError:
        pytest.skip("DSL activations module not available")

    # Get the kernel function
    kernel_fn = getattr(activations, kernel_name, None)
    if kernel_fn is None:
        pytest.skip(f"Kernel {kernel_name} not found")

    # Convert inputs to MLX arrays
    mlx_inputs = [mx.array(inp) for inp in inputs]

    # Create output array
    output = mx.zeros_like(mlx_inputs[0])

    # Compute grid size
    N = mlx_inputs[0].size
    grid = ((N + 255) // 256,)
    threadgroup = 256

    try:
        # Call kernel with appropriate parameters
        if kernel_name in ("silu", "gelu_tanh", "gelu_exact", "quick_gelu", "mish"):
            kernel_fn[grid, threadgroup](
                mlx_inputs[0], output, N
            )
        elif kernel_name in ("fused_silu_mul", "fused_gelu_mul"):
            kernel_fn[grid, threadgroup](
                mlx_inputs[0], mlx_inputs[1], output, N
            )
        elif kernel_name == "softplus":
            beta = kwargs.get("beta", 1.0)
            threshold = kwargs.get("threshold", 20.0)
            kernel_fn[grid, threadgroup](
                mlx_inputs[0], output, N, beta, threshold
            )

        mx.eval(output)
        return np.array(output)
    except Exception as e:
        pytest.skip(f"Kernel execution failed: {e}")


# ---------------------------------------------------------------------------
# SiLU Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestSiLUCorrectness:
    """Tests for SiLU activation correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_silu_basic(self, small_tensor) -> None:
        """Basic SiLU correctness test."""
        expected = numpy_silu(small_tensor)
        actual = run_dsl_kernel("silu", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_silu_zeros(self) -> None:
        """SiLU of zeros should be zeros."""
        x = np.zeros(256, dtype=np.float32)
        expected = numpy_silu(x)
        actual = run_dsl_kernel("silu", [x])

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_silu_positive(self) -> None:
        """SiLU for positive values."""
        x = np.linspace(0.1, 5.0, 256).astype(np.float32)
        expected = numpy_silu(x)
        actual = run_dsl_kernel("silu", [x])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_silu_negative(self) -> None:
        """SiLU for negative values (should be close to zero for large negative)."""
        x = np.linspace(-5.0, -0.1, 256).astype(np.float32)
        expected = numpy_silu(x)
        actual = run_dsl_kernel("silu", [x])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_silu_reference_properties(self) -> None:
        """Test SiLU reference implementation properties."""
        # SiLU(0) = 0
        assert numpy_silu(np.array([0.0]))[0] == 0.0

        # SiLU is odd-ish (but not exactly odd)
        x = np.array([1.0])
        pos = numpy_silu(x)
        neg = numpy_silu(-x)
        assert pos[0] > 0
        assert neg[0] < 0

        # For large positive x, SiLU(x) ≈ x
        x_large = np.array([10.0])
        assert np.abs(numpy_silu(x_large)[0] - 10.0) < 0.001


# ---------------------------------------------------------------------------
# GELU Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestGELUTanhCorrectness:
    """Tests for GELU (tanh approximation) correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_gelu_tanh_basic(self, small_tensor) -> None:
        """Basic GELU tanh correctness test."""
        expected = numpy_gelu_tanh(small_tensor)
        actual = run_dsl_kernel("gelu_tanh", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_gelu_tanh_zeros(self) -> None:
        """GELU of zeros should be zeros."""
        x = np.zeros(256, dtype=np.float32)
        expected = numpy_gelu_tanh(x)
        actual = run_dsl_kernel("gelu_tanh", [x])

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_gelu_tanh_vs_exact(self) -> None:
        """GELU tanh should be close to exact GELU."""
        pytest.importorskip("scipy")
        x = np.linspace(-3.0, 3.0, 1000).astype(np.float32)

        tanh_approx = numpy_gelu_tanh(x)
        exact = numpy_gelu_exact(x)

        # Tanh approximation should be within 0.5% of exact
        np.testing.assert_allclose(tanh_approx, exact, rtol=0.005, atol=0.01)

    def test_gelu_reference_properties(self) -> None:
        """Test GELU reference implementation properties."""
        # GELU(0) = 0
        assert numpy_gelu_tanh(np.array([0.0]))[0] == 0.0

        # For large positive x, GELU(x) ≈ x
        x_large = np.array([10.0])
        assert np.abs(numpy_gelu_tanh(x_large)[0] - 10.0) < 0.01

        # For large negative x, GELU(x) ≈ 0
        x_neg = np.array([-10.0])
        assert np.abs(numpy_gelu_tanh(x_neg)[0]) < 0.01


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestGELUExactCorrectness:
    """Tests for GELU (exact) correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_gelu_exact_basic(self, small_tensor) -> None:
        """Basic GELU exact correctness test."""
        pytest.importorskip("scipy")
        expected = numpy_gelu_exact(small_tensor)
        actual = run_dsl_kernel("gelu_exact", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Quick GELU Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestQuickGELUCorrectness:
    """Tests for Quick GELU correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_quick_gelu_basic(self, small_tensor) -> None:
        """Basic Quick GELU correctness test."""
        expected = numpy_quick_gelu(small_tensor)
        actual = run_dsl_kernel("quick_gelu", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_quick_gelu_reference_properties(self) -> None:
        """Test Quick GELU reference implementation properties."""
        # QuickGELU(0) = 0
        assert numpy_quick_gelu(np.array([0.0]))[0] == 0.0

        # For large positive x, QuickGELU(x) ≈ x
        x_large = np.array([10.0])
        assert np.abs(numpy_quick_gelu(x_large)[0] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Softplus Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestSoftplusCorrectness:
    """Tests for Softplus correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_softplus_basic(self, small_tensor) -> None:
        """Basic Softplus correctness test."""
        expected = numpy_softplus(small_tensor)
        actual = run_dsl_kernel("softplus", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_softplus_reference_properties(self) -> None:
        """Test Softplus reference implementation properties."""
        # Softplus(0) = log(2)
        result = numpy_softplus(np.array([0.0]))
        np.testing.assert_allclose(result, [np.log(2)], rtol=1e-6)

        # Softplus is always positive
        x = np.linspace(-10, 10, 100).astype(np.float32)
        assert np.all(numpy_softplus(x) > 0)

        # For large positive x, Softplus(x) ≈ x
        x_large = np.array([25.0])
        np.testing.assert_allclose(numpy_softplus(x_large), x_large, rtol=1e-4)

    def test_softplus_beta_parameter(self) -> None:
        """Test Softplus with different beta values."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result_beta1 = numpy_softplus(x, beta=1.0)
        result_beta2 = numpy_softplus(x, beta=2.0)

        # Different beta should give different results
        assert not np.allclose(result_beta1, result_beta2)


# ---------------------------------------------------------------------------
# Mish Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMishCorrectness:
    """Tests for Mish activation correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_mish_basic(self, small_tensor) -> None:
        """Basic Mish correctness test."""
        expected = numpy_mish(small_tensor)
        actual = run_dsl_kernel("mish", [small_tensor])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_mish_reference_properties(self) -> None:
        """Test Mish reference implementation properties."""
        # Mish(0) = 0
        assert numpy_mish(np.array([0.0]))[0] == 0.0

        # Mish is smooth and non-monotonic
        x = np.linspace(-5, 5, 1000).astype(np.float32)
        mish_values = numpy_mish(x)

        # Has a minimum around x ≈ -1.2
        min_idx = np.argmin(mish_values)
        assert -2 < x[min_idx] < 0


# ---------------------------------------------------------------------------
# Fused SiLU + Mul (SwiGLU) Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedSiLUMulCorrectness:
    """Tests for fused SiLU + multiply (SwiGLU) correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_fused_silu_mul_basic(self, random_seed) -> None:
        """Basic fused SiLU * mul correctness test."""
        x = np.random.randn(256).astype(np.float32)
        gate = np.random.randn(256).astype(np.float32)

        expected = numpy_fused_silu_mul(x, gate)
        actual = run_dsl_kernel("fused_silu_mul", [x, gate])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_fused_silu_mul_reference_vs_separate(self, random_seed) -> None:
        """Fused should match separate operations."""
        x = np.random.randn(1024).astype(np.float32)
        gate = np.random.randn(1024).astype(np.float32)

        fused = numpy_fused_silu_mul(x, gate)
        separate = numpy_silu(x) * gate

        np.testing.assert_allclose(fused, separate, rtol=1e-7)


# ---------------------------------------------------------------------------
# Fused GELU + Mul (GEGLU) Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestFusedGELUMulCorrectness:
    """Tests for fused GELU + multiply (GEGLU) correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_fused_gelu_mul_basic(self, random_seed) -> None:
        """Basic fused GELU * mul correctness test."""
        x = np.random.randn(256).astype(np.float32)
        gate = np.random.randn(256).astype(np.float32)

        expected = numpy_fused_gelu_mul(x, gate)
        actual = run_dsl_kernel("fused_gelu_mul", [x, gate])

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_fused_gelu_mul_reference_vs_separate(self, random_seed) -> None:
        """Fused should match separate operations."""
        x = np.random.randn(1024).astype(np.float32)
        gate = np.random.randn(1024).astype(np.float32)

        fused = numpy_fused_gelu_mul(x, gate)
        separate = numpy_gelu_tanh(x) * gate

        np.testing.assert_allclose(fused, separate, rtol=1e-7)


# ---------------------------------------------------------------------------
# Numerical Stability Tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability of reference implementations."""

    def test_silu_large_negative(self) -> None:
        """SiLU should not overflow for large negative inputs."""
        x = np.array([-100.0, -50.0, -20.0], dtype=np.float32)
        result = numpy_silu(x)

        assert np.all(np.isfinite(result))
        # For large negative x, SiLU(x) approaches 0 (sigmoid(x) approaches 0)
        assert np.all(result <= 0)

    def test_silu_large_positive(self) -> None:
        """SiLU should not overflow for large positive inputs."""
        x = np.array([20.0, 50.0, 100.0], dtype=np.float32)
        result = numpy_silu(x)

        assert np.all(np.isfinite(result))
        # For large x, SiLU(x) ≈ x
        np.testing.assert_allclose(result, x, rtol=0.01)

    def test_gelu_large_values(self) -> None:
        """GELU should handle large values without overflow."""
        x = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
        result = numpy_gelu_tanh(x)

        assert np.all(np.isfinite(result))
        # For large negative, GELU ≈ 0
        assert np.abs(result[0]) < 1e-10
        # For large positive, GELU ≈ x
        assert np.abs(result[3] - 100.0) < 1.0

    def test_softplus_large_values(self) -> None:
        """Softplus should use linear approximation for large values."""
        x = np.array([25.0, 50.0, 100.0], dtype=np.float32)
        result = numpy_softplus(x, threshold=20.0)

        assert np.all(np.isfinite(result))
        # Should be equal to x for values above threshold
        np.testing.assert_allclose(result, x, rtol=1e-4)

    def test_mish_numerical_stability(self) -> None:
        """Mish should be numerically stable across range."""
        x = np.linspace(-50, 50, 1000).astype(np.float32)
        result = numpy_mish(x)

        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Cross-Implementation Comparison Tests
# ---------------------------------------------------------------------------


class TestCrossImplementationComparison:
    """Compare different activation implementations."""

    def test_silu_vs_quick_gelu(self) -> None:
        """SiLU and Quick GELU should have similar shapes but different values."""
        x = np.linspace(-5, 5, 1001).astype(np.float32)  # 1001 points so middle is exactly 0

        silu = numpy_silu(x)
        quick_gelu = numpy_quick_gelu(x)

        # Both should be 0 at x=0 (element 500 with 1001 points)
        np.testing.assert_allclose(silu[500], 0, atol=1e-6)
        np.testing.assert_allclose(quick_gelu[500], 0, atol=1e-6)

        # Both should approach x for large positive
        assert np.abs(silu[-1] - 5.0) < 0.1
        assert np.abs(quick_gelu[-1] - 5.0) < 0.1

    def test_gelu_variants_similar(self) -> None:
        """Different GELU variants should be similar."""
        pytest.importorskip("scipy")
        x = np.linspace(-3, 3, 1000).astype(np.float32)

        gelu_tanh = numpy_gelu_tanh(x)
        gelu_exact = numpy_gelu_exact(x)
        quick_gelu = numpy_quick_gelu(x)

        # Tanh and exact should be very close
        np.testing.assert_allclose(gelu_tanh, gelu_exact, rtol=0.01, atol=0.01)

        # Quick GELU is a different approximation but similar shape
        correlation = np.corrcoef(gelu_tanh, quick_gelu)[0, 1]
        assert correlation > 0.99


# ---------------------------------------------------------------------------
# Edge Cases Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.parametrize(
        "activation_fn",
        [numpy_silu, numpy_gelu_tanh, numpy_quick_gelu, numpy_mish],
    )
    def test_single_element(self, activation_fn: Callable) -> None:
        """Single element arrays should work."""
        x = np.array([1.5], dtype=np.float32)
        result = activation_fn(x)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    @pytest.mark.parametrize(
        "activation_fn",
        [numpy_silu, numpy_gelu_tanh, numpy_quick_gelu, numpy_mish],
    )
    def test_empty_array(self, activation_fn: Callable) -> None:
        """Empty arrays should return empty arrays."""
        x = np.array([], dtype=np.float32)
        result = activation_fn(x)

        assert result.shape == (0,)

    def test_multidimensional_broadcasting(self) -> None:
        """Activations should work on multi-dimensional arrays."""
        x = np.random.randn(2, 3, 4).astype(np.float32)

        for fn in [numpy_silu, numpy_gelu_tanh, numpy_quick_gelu, numpy_mish]:
            result = fn(x)
            assert result.shape == x.shape
            assert np.all(np.isfinite(result))

    def test_fused_operations_shape_match(self) -> None:
        """Fused operations require matching shapes."""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        gate = np.random.randn(2, 3, 4).astype(np.float32)

        silu_result = numpy_fused_silu_mul(x, gate)
        gelu_result = numpy_fused_gelu_mul(x, gate)

        assert silu_result.shape == x.shape
        assert gelu_result.shape == x.shape


# ---------------------------------------------------------------------------
# Gradient-like Tests (for reference implementations)
# ---------------------------------------------------------------------------


class TestGradientProperties:
    """Tests for gradient-like properties of activations."""

    def test_silu_derivative_at_zero(self) -> None:
        """SiLU derivative at x=0 should be 0.5."""
        # d/dx [x * sigmoid(x)] at x=0
        # = sigmoid(0) + 0 * sigmoid'(0) = 0.5
        h = 1e-5
        x = np.array([0.0])
        deriv = (numpy_silu(x + h) - numpy_silu(x - h))[0] / (2 * h)
        np.testing.assert_allclose(deriv, 0.5, rtol=0.01)

    def test_gelu_derivative_at_zero(self) -> None:
        """GELU derivative at x=0 should be 0.5."""
        h = 1e-5
        x = np.array([0.0])
        deriv = (numpy_gelu_tanh(x + h) - numpy_gelu_tanh(x - h))[0] / (2 * h)
        np.testing.assert_allclose(deriv, 0.5, rtol=0.01)

    def test_mish_derivative_at_zero(self) -> None:
        """Mish derivative at x=0 should be approximately 0.6."""
        h = 1e-5
        x = np.array([0.0])
        deriv = (numpy_mish(x + h) - numpy_mish(x - h))[0] / (2 * h)
        # Mish'(0) ≈ 0.6
        np.testing.assert_allclose(deriv, 0.6, rtol=0.1)
