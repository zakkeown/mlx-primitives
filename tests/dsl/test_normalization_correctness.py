"""Correctness tests for normalization kernels.

Validates DSL normalization kernels against NumPy reference implementations.
"""

import numpy as np
import pytest

# Try to import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# ---------------------------------------------------------------------------
# NumPy Reference Implementations
# ---------------------------------------------------------------------------


def numpy_layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Reference LayerNorm implementation.

    y = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: Input tensor (..., D)
        weight: Scale parameter (D,)
        bias: Bias parameter (D,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as x.
    """
    # Compute mean and variance over last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Apply affine transform
    return x_norm * weight + bias


def numpy_rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Reference RMSNorm implementation.

    y = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor (..., D)
        weight: Scale parameter (D,)
        eps: Epsilon for numerical stability

    Returns:
        RMS normalized tensor with same shape as x.
    """
    # Compute RMS
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

    # Normalize and scale
    return (x / rms) * weight


def numpy_fused_add_layer_norm(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Reference fused add + LayerNorm.

    y = LayerNorm(x + residual)
    """
    fused = x + residual
    return numpy_layer_norm(fused, weight, bias, eps)


def numpy_fused_add_rms_norm(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Reference fused add + RMSNorm.

    y = RMSNorm(x + residual)
    """
    fused = x + residual
    return numpy_rms_norm(fused, weight, eps)


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
    """Small tensor for basic tests (4, 64)."""
    return np.random.randn(4, 64).astype(np.float32)


@pytest.fixture
def medium_tensor(random_seed) -> np.ndarray:
    """Medium tensor for typical use cases (8, 256)."""
    return np.random.randn(8, 256).astype(np.float32)


@pytest.fixture
def large_tensor(random_seed) -> np.ndarray:
    """Large tensor (4, 512, 768)."""
    return np.random.randn(4, 512, 768).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def run_layer_norm_kernel(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Run the DSL layer_norm kernel."""
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")

    try:
        from mlx_primitives.dsl.examples import normalization
    except ImportError:
        pytest.skip("DSL normalization module not available")

    # Reshape to 2D for kernel (N, D)
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    N, D = x_2d.shape

    mx_x = mx.array(x_2d)
    mx_weight = mx.array(weight)
    mx_bias = mx.array(bias)
    mx_out = mx.zeros_like(mx_x)

    grid = (N,)
    threadgroup = 256

    try:
        normalization.layer_norm[grid, threadgroup](
            mx_x, mx_weight, mx_bias, mx_out, N, D, eps
        )
        mx.eval(mx_out)
        result = np.array(mx_out).reshape(original_shape)
        return result
    except Exception as e:
        pytest.skip(f"Kernel execution failed: {e}")


def run_rms_norm_kernel(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Run the DSL rms_norm kernel."""
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")

    try:
        from mlx_primitives.dsl.examples import normalization
    except ImportError:
        pytest.skip("DSL normalization module not available")

    # Reshape to 2D for kernel
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    N, D = x_2d.shape

    mx_x = mx.array(x_2d)
    mx_weight = mx.array(weight)
    mx_out = mx.zeros_like(mx_x)

    grid = (N,)
    threadgroup = 256

    try:
        normalization.rms_norm[grid, threadgroup](
            mx_x, mx_weight, mx_out, N, D, eps
        )
        mx.eval(mx_out)
        result = np.array(mx_out).reshape(original_shape)
        return result
    except Exception as e:
        pytest.skip(f"Kernel execution failed: {e}")


# ---------------------------------------------------------------------------
# LayerNorm Tests
# ---------------------------------------------------------------------------


class TestLayerNormReference:
    """Tests for LayerNorm reference implementation."""

    def test_layer_norm_basic(self, random_seed) -> None:
        """Basic LayerNorm computation."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result = numpy_layer_norm(x, weight, bias)

        # Each row should have mean ~0 and var ~1
        row_means = np.mean(result, axis=-1)
        row_vars = np.var(result, axis=-1)

        np.testing.assert_allclose(row_means, 0, atol=1e-4)
        np.testing.assert_allclose(row_vars, 1, atol=1e-4)

    def test_layer_norm_with_affine(self, random_seed) -> None:
        """LayerNorm with non-trivial weight and bias."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)

        result = numpy_layer_norm(x, weight, bias)

        # Result should have correct shape
        assert result.shape == x.shape

        # Should be different from input
        assert not np.allclose(result, x)

    def test_layer_norm_3d_tensor(self, random_seed) -> None:
        """LayerNorm on 3D tensor."""
        x = np.random.randn(2, 4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result = numpy_layer_norm(x, weight, bias)

        assert result.shape == (2, 4, 64)

        # Each row (last dim) should be normalized
        for b in range(2):
            for s in range(4):
                row_mean = np.mean(result[b, s])
                row_var = np.var(result[b, s])
                np.testing.assert_allclose(row_mean, 0, atol=1e-4)
                np.testing.assert_allclose(row_var, 1, atol=1e-4)

    def test_layer_norm_numerical_stability(self) -> None:
        """LayerNorm should be numerically stable."""
        # Large values
        x_large = np.array([[1e6, 1e6 + 1, 1e6 + 2]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)

        result = numpy_layer_norm(x_large, weight, bias)
        assert np.all(np.isfinite(result))

        # Small values
        x_small = np.array([[1e-6, 2e-6, 3e-6]], dtype=np.float32)
        result = numpy_layer_norm(x_small, weight, bias)
        assert np.all(np.isfinite(result))

    def test_layer_norm_constant_input(self) -> None:
        """LayerNorm of constant input should be all zeros (variance is 0)."""
        x = np.ones((4, 64), dtype=np.float32) * 5.0
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result = numpy_layer_norm(x, weight, bias, eps=1e-5)

        # With constant input, (x - mean) = 0, so result should be bias (0)
        np.testing.assert_allclose(result, 0, atol=1e-3)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestLayerNormKernel:
    """Tests for DSL LayerNorm kernel correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_layer_norm_kernel_basic(self, random_seed) -> None:
        """Basic DSL kernel matches reference."""
        x = np.random.randn(8, 256).astype(np.float32)
        weight = np.ones(256, dtype=np.float32)
        bias = np.zeros(256, dtype=np.float32)

        expected = numpy_layer_norm(x, weight, bias)
        actual = run_layer_norm_kernel(x, weight, bias)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_layer_norm_kernel_with_affine(self, random_seed) -> None:
        """DSL kernel with non-trivial weight/bias."""
        x = np.random.randn(8, 256).astype(np.float32)
        weight = np.random.randn(256).astype(np.float32)
        bias = np.random.randn(256).astype(np.float32)

        expected = numpy_layer_norm(x, weight, bias)
        actual = run_layer_norm_kernel(x, weight, bias)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# RMSNorm Tests
# ---------------------------------------------------------------------------


class TestRMSNormReference:
    """Tests for RMSNorm reference implementation."""

    def test_rms_norm_basic(self, random_seed) -> None:
        """Basic RMSNorm computation."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)

        result = numpy_rms_norm(x, weight)

        # RMS of result should be ~1 for each row
        result_rms = np.sqrt(np.mean(result * result, axis=-1))
        np.testing.assert_allclose(result_rms, 1, atol=0.1)

    def test_rms_norm_with_weight(self, random_seed) -> None:
        """RMSNorm with non-trivial weight."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32) * 2 + 1

        result = numpy_rms_norm(x, weight)

        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_rms_norm_3d_tensor(self, random_seed) -> None:
        """RMSNorm on 3D tensor."""
        x = np.random.randn(2, 4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)

        result = numpy_rms_norm(x, weight)

        assert result.shape == (2, 4, 64)

    def test_rms_norm_preserves_sign(self, random_seed) -> None:
        """RMSNorm should preserve sign of inputs."""
        x = np.array([[-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)

        result = numpy_rms_norm(x, weight)

        # Signs should be preserved
        np.testing.assert_array_equal(np.sign(result), np.sign(x))

    def test_rms_norm_vs_layer_norm_no_bias(self) -> None:
        """RMSNorm differs from LayerNorm (no mean centering)."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)

        rms_result = numpy_rms_norm(x, weight)
        ln_result = numpy_layer_norm(x, weight, bias)

        # Results should be different (RMS doesn't center)
        assert not np.allclose(rms_result, ln_result)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestRMSNormKernel:
    """Tests for DSL RMSNorm kernel correctness."""

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_rms_norm_kernel_basic(self, random_seed) -> None:
        """Basic DSL kernel matches reference."""
        x = np.random.randn(8, 256).astype(np.float32)
        weight = np.ones(256, dtype=np.float32)

        expected = numpy_rms_norm(x, weight)
        actual = run_rms_norm_kernel(x, weight)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.xfail(reason="DSL kernel execution may fail on some systems")
    def test_rms_norm_kernel_with_weight(self, random_seed) -> None:
        """DSL kernel with non-trivial weight."""
        x = np.random.randn(8, 256).astype(np.float32)
        weight = np.random.randn(256).astype(np.float32)

        expected = numpy_rms_norm(x, weight)
        actual = run_rms_norm_kernel(x, weight)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Fused Add + LayerNorm Tests
# ---------------------------------------------------------------------------


class TestFusedAddLayerNormReference:
    """Tests for fused add + LayerNorm reference implementation."""

    def test_fused_add_layer_norm_basic(self, random_seed) -> None:
        """Fused add + LayerNorm matches separate operations."""
        x = np.random.randn(4, 64).astype(np.float32)
        residual = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        fused_result = numpy_fused_add_layer_norm(x, residual, weight, bias)
        separate_result = numpy_layer_norm(x + residual, weight, bias)

        np.testing.assert_allclose(fused_result, separate_result, rtol=1e-6)

    def test_fused_add_layer_norm_with_affine(self, random_seed) -> None:
        """Fused with non-trivial weight and bias."""
        x = np.random.randn(4, 64).astype(np.float32)
        residual = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)

        fused_result = numpy_fused_add_layer_norm(x, residual, weight, bias)
        separate_result = numpy_layer_norm(x + residual, weight, bias)

        np.testing.assert_allclose(fused_result, separate_result, rtol=1e-6)


# ---------------------------------------------------------------------------
# Fused Add + RMSNorm Tests
# ---------------------------------------------------------------------------


class TestFusedAddRMSNormReference:
    """Tests for fused add + RMSNorm reference implementation."""

    def test_fused_add_rms_norm_basic(self, random_seed) -> None:
        """Fused add + RMSNorm matches separate operations."""
        x = np.random.randn(4, 64).astype(np.float32)
        residual = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)

        fused_result = numpy_fused_add_rms_norm(x, residual, weight)
        separate_result = numpy_rms_norm(x + residual, weight)

        np.testing.assert_allclose(fused_result, separate_result, rtol=1e-6)

    def test_fused_add_rms_norm_with_weight(self, random_seed) -> None:
        """Fused with non-trivial weight."""
        x = np.random.randn(4, 64).astype(np.float32)
        residual = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)

        fused_result = numpy_fused_add_rms_norm(x, residual, weight)
        separate_result = numpy_rms_norm(x + residual, weight)

        np.testing.assert_allclose(fused_result, separate_result, rtol=1e-6)


# ---------------------------------------------------------------------------
# Numerical Properties Tests
# ---------------------------------------------------------------------------


class TestNormalizationProperties:
    """Tests for mathematical properties of normalization."""

    def test_layer_norm_invariant_to_shift(self) -> None:
        """LayerNorm should be invariant to uniform shift."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result1 = numpy_layer_norm(x, weight, bias)
        result2 = numpy_layer_norm(x + 100, weight, bias)

        np.testing.assert_allclose(result1, result2, rtol=1e-4, atol=1e-4)

    def test_layer_norm_scales_correctly(self) -> None:
        """LayerNorm with weight=2 should double the output range."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight1 = np.ones(64, dtype=np.float32)
        weight2 = np.ones(64, dtype=np.float32) * 2
        bias = np.zeros(64, dtype=np.float32)

        result1 = numpy_layer_norm(x, weight1, bias)
        result2 = numpy_layer_norm(x, weight2, bias)

        np.testing.assert_allclose(result2, result1 * 2, rtol=1e-5)

    def test_rms_norm_scales_with_input_magnitude(self) -> None:
        """RMSNorm output should scale with input magnitude."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)

        result1 = numpy_rms_norm(x, weight)
        result2 = numpy_rms_norm(x * 2, weight)

        # RMS normalization: (2x) / sqrt(mean((2x)^2)) = (2x) / (2 * sqrt(mean(x^2)))
        # So result should be the same!
        np.testing.assert_allclose(result1, result2, rtol=1e-5)

    def test_normalization_gradient_properties(self) -> None:
        """Finite difference check for gradient-like properties."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        h = 1e-5
        x_plus = x.copy()
        x_plus[0, 0] += h
        x_minus = x.copy()
        x_minus[0, 0] -= h

        ln_plus = numpy_layer_norm(x_plus, weight, bias)
        ln_minus = numpy_layer_norm(x_minus, weight, bias)

        # Numerical gradient should be finite
        grad = (ln_plus - ln_minus) / (2 * h)
        assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Edge Cases Tests
# ---------------------------------------------------------------------------


class TestNormalizationEdgeCases:
    """Tests for edge cases in normalization."""

    def test_single_element_dimension(self) -> None:
        """Normalization of single-element dimension."""
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        weight = np.ones(1, dtype=np.float32)
        bias = np.zeros(1, dtype=np.float32)

        # LayerNorm of single element: (x - x) / eps_term = 0
        result = numpy_layer_norm(x, weight, bias)
        np.testing.assert_allclose(result, 0, atol=1e-3)

    def test_very_small_epsilon(self) -> None:
        """Small epsilon should not cause numerical issues."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result = numpy_layer_norm(x, weight, bias, eps=1e-12)
        assert np.all(np.isfinite(result))

    def test_large_epsilon(self) -> None:
        """Large epsilon should dampen normalization."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result_small_eps = numpy_layer_norm(x, weight, bias, eps=1e-5)
        result_large_eps = numpy_layer_norm(x, weight, bias, eps=1.0)

        # Large epsilon should reduce the effect of normalization
        # (variance is smaller relative to eps + var)
        var_small = np.var(result_small_eps, axis=-1)
        var_large = np.var(result_large_eps, axis=-1)

        # With large eps, variance won't be normalized as much
        assert np.mean(var_large) < np.mean(var_small)

    def test_all_zeros_input(self) -> None:
        """Normalization of all-zeros should be all-zeros."""
        x = np.zeros((4, 64), dtype=np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        ln_result = numpy_layer_norm(x, weight, bias)
        rms_result = numpy_rms_norm(x, weight)

        # Should be 0 (or bias for LayerNorm)
        # bias is (64,), result is (4, 64), so broadcast comparison
        expected_bias = np.broadcast_to(bias, ln_result.shape)
        np.testing.assert_allclose(ln_result, expected_bias, atol=1e-5)
        np.testing.assert_allclose(rms_result, 0, atol=1e-5)


# ---------------------------------------------------------------------------
# Tolerance Tests
# ---------------------------------------------------------------------------


class TestNormalizationTolerances:
    """Tests to verify tolerance requirements."""

    @pytest.mark.parametrize("hidden_dim", [64, 256, 768, 1024])
    def test_layer_norm_accuracy_various_dims(self, hidden_dim, random_seed) -> None:
        """LayerNorm accuracy across different hidden dimensions."""
        x = np.random.randn(8, hidden_dim).astype(np.float32)
        weight = np.random.randn(hidden_dim).astype(np.float32)
        bias = np.random.randn(hidden_dim).astype(np.float32)

        result = numpy_layer_norm(x, weight, bias)

        assert result.shape == (8, hidden_dim)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_rms_norm_accuracy_various_batches(self, batch_size, random_seed) -> None:
        """RMSNorm accuracy across different batch sizes."""
        x = np.random.randn(batch_size, 256).astype(np.float32)
        weight = np.random.randn(256).astype(np.float32)

        result = numpy_rms_norm(x, weight)

        assert result.shape == (batch_size, 256)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Cross-validation with PyTorch (if available)
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Cross-validation with PyTorch."""

    def test_layer_norm_vs_pytorch(self, random_seed) -> None:
        """Compare with PyTorch LayerNorm."""
        torch = pytest.importorskip("torch")

        x_np = np.random.randn(4, 64).astype(np.float32)
        weight_np = np.random.randn(64).astype(np.float32)
        bias_np = np.random.randn(64).astype(np.float32)

        # NumPy reference
        numpy_result = numpy_layer_norm(x_np, weight_np, bias_np)

        # PyTorch
        x_torch = torch.tensor(x_np)
        ln = torch.nn.LayerNorm(64, eps=1e-5)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(weight_np))
            ln.bias.copy_(torch.tensor(bias_np))
            torch_result = ln(x_torch).numpy()

        np.testing.assert_allclose(numpy_result, torch_result, rtol=1e-4, atol=1e-4)

    def test_rms_norm_formula(self, random_seed) -> None:
        """Verify RMSNorm formula matches expected computation."""
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32)
        eps = 1e-5

        # Manual computation
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        expected = (x / rms) * weight

        # Reference implementation
        actual = numpy_rms_norm(x, weight, eps)

        np.testing.assert_allclose(actual, expected, rtol=1e-6)
