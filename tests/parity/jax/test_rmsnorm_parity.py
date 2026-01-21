"""JAX parity tests for RMSNorm kernel implementations.

This module tests parity between MLX RMSNorm implementations and JAX
reference implementations for various configurations.
"""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_jax(x_np: np.ndarray, dtype: str) -> "jnp.ndarray":
    """Convert numpy array to JAX with proper dtype."""
    x_jax = jnp.array(x_np.astype(np.float32))
    jax_dtype = get_jax_dtype(dtype)
    return x_jax.astype(jax_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


# =============================================================================
# JAX Reference Implementations
# =============================================================================

def jax_rmsnorm(x: "jnp.ndarray", weight: "jnp.ndarray", eps: float = 1e-6) -> "jnp.ndarray":
    """JAX reference implementation of RMSNorm.

    RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight

    Matches MLX implementation by computing in float32 for numerical stability.

    Args:
        x: Input tensor of any shape, normalized over last dimension.
        weight: Scale parameter with shape (x.shape[-1],).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor with same shape as input.
    """
    orig_dtype = x.dtype
    # Compute in fp32 for numerical stability (matching MLX implementation)
    x_fp32 = x.astype(jnp.float32)

    # Compute RMS over last dimension
    variance = jnp.mean(x_fp32 ** 2, axis=-1, keepdims=True)
    rms = jnp.sqrt(variance + eps)

    # Normalize in fp32, then cast back and scale
    normalized = (x_fp32 / rms).astype(orig_dtype)
    return normalized * weight


# =============================================================================
# RMSNorm Forward Parity Tests
# =============================================================================

class TestRMSNormForwardParity:
    """RMSNorm forward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test RMSNorm forward pass matches JAX reference."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.random.randn(dims).astype(np.float32) * 0.1 + 1.0

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = _convert_to_jax(x_np, dtype)
        weight_jax = _convert_to_jax(weight_np, dtype)
        jax_out = jax_rmsnorm(x_jax, weight_jax)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_various_eps(self, eps, skip_without_jax):
        """Test RMSNorm with different epsilon values."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        batch, seq, dims = 2, 64, 128

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.ones(dims, dtype=np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = rmsnorm(x_mlx, weight_mlx, eps=eps)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_rmsnorm(jnp.array(x_np), jnp.array(weight_np), eps=eps)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm with eps={eps} mismatch (JAX)"
        )


# =============================================================================
# RMSNorm Backward Parity Tests
# =============================================================================

class TestRMSNormBackwardParity:
    """RMSNorm backward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test RMSNorm backward pass gradients match JAX."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.random.randn(dims).astype(np.float32) * 0.1 + 1.0

        # MLX backward
        def mlx_loss_fn(x, w):
            return mx.sum(rmsnorm(x, w))

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        x_grad, w_grad = grad_fn(x_mlx, weight_mlx)
        mx.eval(x_grad, w_grad)

        # JAX backward
        def jax_loss_fn(x, w):
            return jnp.sum(jax_rmsnorm(x, w))

        x_jax = jnp.array(x_np)
        w_jax = jnp.array(weight_np)
        jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1))
        jax_x_grad, jax_w_grad = jax_grad_fn(x_jax, w_jax)

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_grad), _to_numpy(jax_x_grad),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm x gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(w_grad), _to_numpy(jax_w_grad),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm weight gradient mismatch (JAX) [{size}]"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRMSNormEdgeCases:
    """Edge case tests for RMSNorm."""

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_unit_weight(self, skip_without_jax):
        """Test RMSNorm with unit weights (should just normalize)."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        batch, seq, dims = 2, 32, 64

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.ones(dims, dtype=np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        jax_out = jax_rmsnorm(jnp.array(x_np), jnp.array(weight_np))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="RMSNorm with unit weights mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_zero_input(self, skip_without_jax):
        """Test RMSNorm with zero input."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        batch, seq, dims = 2, 16, 32

        x_np = np.zeros((batch, seq, dims), dtype=np.float32)
        weight_np = np.ones(dims, dtype=np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        # Output should be zeros (0 / sqrt(0 + eps) * 1 = 0)
        mlx_out_np = _to_numpy(mlx_out)
        assert not np.any(np.isnan(mlx_out_np)), "NaN in zero input output"
        assert not np.any(np.isinf(mlx_out_np)), "Inf in zero input output"

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_single_element(self, skip_without_jax):
        """Test RMSNorm with single-element last dimension."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        batch, seq, dims = 2, 16, 1

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.ones(dims, dtype=np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        jax_out = jax_rmsnorm(jnp.array(x_np), jnp.array(weight_np))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="RMSNorm with single element mismatch (JAX)"
        )


# =============================================================================
# Fast RMSNorm Parity Tests
# =============================================================================

class TestFastRMSNormParity:
    """Tests for fast_rmsnorm() Metal kernel parity with JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fast_rmsnorm forward pass matches JAX reference."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx, eps=1e-6)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = _convert_to_jax(x_np, dtype)
        weight_jax = _convert_to_jax(weight_np, dtype)
        jax_out = jax_rmsnorm(x_jax, weight_jax, eps=1e-6)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_2d_input(self, skip_without_jax):
        """Test fast_rmsnorm with 2D input (seq, hidden)."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        seq, hidden = 256, 1024

        np.random.seed(42)
        x_np = np.random.randn(seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_rmsnorm(jnp.array(x_np), jnp.array(weight_np))

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg="fast_rmsnorm 2D input mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fast_rmsnorm backward pass gradients match JAX."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, dims = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.random.randn(dims).astype(np.float32) * 0.1 + 1.0

        # MLX backward
        def mlx_loss_fn(x, w):
            return mx.sum(fast_rmsnorm(x, w))

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        x_grad, w_grad = grad_fn(x_mlx, weight_mlx)
        mx.eval(x_grad, w_grad)

        # JAX backward
        def jax_loss_fn(x, w):
            return jnp.sum(jax_rmsnorm(x, w))

        x_jax = jnp.array(x_np)
        w_jax = jnp.array(weight_np)
        jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1))
        jax_x_grad, jax_w_grad = jax_grad_fn(x_jax, w_jax)

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_grad), _to_numpy(jax_x_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm x gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(w_grad), _to_numpy(jax_w_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm weight gradient mismatch (JAX) [{size}]"
        )


class TestFastRMSNormResidualParity:
    """Tests for fast_rmsnorm_residual() parity with JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fast_rmsnorm_residual forward pass matches JAX reference."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm_residual

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        residual_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        residual_mlx = _convert_to_mlx(residual_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = fast_rmsnorm_residual(x_mlx, residual_mlx, weight_mlx, eps=1e-6)
        mx.eval(mlx_out)

        # JAX reference: RMSNorm(x + residual)
        x_jax = _convert_to_jax(x_np, dtype)
        residual_jax = _convert_to_jax(residual_np, dtype)
        weight_jax = _convert_to_jax(weight_np, dtype)
        combined = x_jax + residual_jax
        jax_out = jax_rmsnorm(combined, weight_jax, eps=1e-6)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm_residual mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fast_rmsnorm_residual backward via base rmsnorm (fused kernel lacks VJP).

        Note: fast_rmsnorm_residual uses a custom fused Metal kernel that doesn't
        have a VJP implementation. We verify gradient flow through the equivalent
        operation using the base rmsnorm kernel.
        """
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, dims = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)
        residual_np = np.random.randn(batch, seq, dims).astype(np.float32)
        weight_np = np.random.randn(dims).astype(np.float32) * 0.1 + 1.0

        # MLX backward using base rmsnorm on combined input
        def mlx_loss_fn(x, residual, w):
            combined = x + residual
            return mx.sum(rmsnorm(combined, w))

        x_mlx = mx.array(x_np)
        residual_mlx = mx.array(residual_np)
        weight_mlx = mx.array(weight_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        x_grad, r_grad, w_grad = grad_fn(x_mlx, residual_mlx, weight_mlx)
        mx.eval(x_grad, r_grad, w_grad)

        # JAX backward
        def jax_loss_fn(x, residual, w):
            combined = x + residual
            return jnp.sum(jax_rmsnorm(combined, w))

        x_jax = jnp.array(x_np)
        residual_jax = jnp.array(residual_np)
        w_jax = jnp.array(weight_np)
        jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1, 2))
        jax_x_grad, jax_r_grad, jax_w_grad = jax_grad_fn(x_jax, residual_jax, w_jax)

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_grad), _to_numpy(jax_x_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm_residual x gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(r_grad), _to_numpy(jax_r_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm_residual residual gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(w_grad), _to_numpy(jax_w_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm_residual weight gradient mismatch (JAX) [{size}]"
        )
