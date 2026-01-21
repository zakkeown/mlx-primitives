"""JAX Metal parity tests for fused operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn


# =============================================================================
# Helper Functions
# =============================================================================


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


# =============================================================================
# Fused RMSNorm + Linear Parity Tests
# =============================================================================


class TestFusedRMSNormLinearParity:
    """Fused RMSNorm + Linear parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fused RMSNorm+Linear forward pass parity with JAX."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        out_features = hidden * 4
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        # MLX forward
        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)
        mlx_out = fused_rmsnorm_linear(x_mlx, norm_weight_mlx, linear_weight_mlx, eps=eps)
        mx.eval(mlx_out)

        # JAX reference: RMSNorm then Linear
        jax_dtype = get_jax_dtype(dtype)
        x_jax = jnp.array(x_np).astype(jax_dtype)
        norm_weight_jax = jnp.array(norm_weight_np).astype(jax_dtype)
        linear_weight_jax = jnp.array(linear_weight_np).astype(jax_dtype)

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = jnp.sqrt(jnp.mean(x_jax ** 2, axis=-1, keepdims=True) + eps)
        norm_x = (x_jax / rms) * norm_weight_jax
        # Linear: norm_x @ weight.T
        jax_out = norm_x @ linear_weight_jax.T

        rtol, atol = get_tolerance("fused_ops", "fused_rmsnorm_linear", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(jax_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RMSNorm+Linear forward mismatch (JAX) [{size}, {dtype}]",
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fused RMSNorm+Linear backward pass parity with JAX."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        out_features = hidden * 4
        eps = 1e-5
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        # MLX backward
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_rmsnorm_linear(x, norm_weight_mlx, linear_weight_mlx, eps=eps))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        norm_weight_jax = jnp.array(norm_weight_np)
        linear_weight_jax = jnp.array(linear_weight_np)

        def jax_loss_fn(x):
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
            norm_x = (x / rms) * norm_weight_jax
            out = norm_x @ linear_weight_jax.T
            return jnp.sum(out)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_rmsnorm_linear", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            _to_numpy(jax_grad),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RMSNorm+Linear backward mismatch (JAX) [{size}]",
        )


# =============================================================================
# Fused SwiGLU Parity Tests
# =============================================================================


class TestFusedSwiGLUParity:
    """Fused SwiGLU parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test fused SwiGLU forward pass parity with JAX."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX forward
        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)
        mlx_out = fused_swiglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(mlx_out)

        # JAX reference: silu(x @ W_gate.T) * (x @ W_up.T)
        x_jax = jnp.array(x_np)
        W_gate_jax = jnp.array(W_gate_np)
        W_up_jax = jnp.array(W_up_np)

        gate = jnn.silu(x_jax @ W_gate_jax.T)
        up = x_jax @ W_up_jax.T
        jax_out = gate * up

        rtol, atol = get_tolerance("fused_ops", "fused_swiglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(jax_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused SwiGLU forward mismatch (JAX) [{size}]",
        )


# =============================================================================
# Fused GeGLU Parity Tests
# =============================================================================


class TestFusedGeGLUParity:
    """Fused GeGLU parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test fused GeGLU forward pass parity with JAX."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX forward
        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)
        mlx_out = fused_geglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(mlx_out)

        # JAX reference: gelu(x @ W_gate.T, approximate=True) * (x @ W_up.T)
        # Note: MLX uses tanh approximation, so use approximate=True in JAX
        x_jax = jnp.array(x_np)
        W_gate_jax = jnp.array(W_gate_np)
        W_up_jax = jnp.array(W_up_np)

        gate = jnn.gelu(x_jax @ W_gate_jax.T, approximate=True)
        up = x_jax @ W_up_jax.T
        jax_out = gate * up

        rtol, atol = get_tolerance("fused_ops", "fused_geglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(jax_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused GeGLU forward mismatch (JAX) [{size}]",
        )


# =============================================================================
# Fused RoPE + Attention Parity Tests
# =============================================================================


class TestFusedRoPEAttentionParity:
    """Fused RoPE + Attention parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test fused RoPE+Attention forward pass parity with JAX."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = (
            config["batch"],
            config["seq"],
            config["heads"],
            config["head_dim"],
        )
        scale = 1.0 / (head_dim ** 0.5)
        base = 10000.0

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX forward
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(mlx_out)

        # JAX reference: apply RoPE then attention
        q_jax = jnp.array(q_np)
        k_jax = jnp.array(k_np)
        v_jax = jnp.array(v_np)

        # Compute RoPE frequencies
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        positions = jnp.arange(seq, dtype=jnp.float32)
        angles = positions[:, None] * inv_freq[None, :]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        def apply_rope_jax(x, cos, sin):
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            cos_expanded = cos[None, :, None, :]  # (1, seq, 1, half_dim)
            sin_expanded = sin[None, :, None, :]
            return jnp.concatenate([
                x1 * cos_expanded - x2 * sin_expanded,
                x1 * sin_expanded + x2 * cos_expanded,
            ], axis=-1)

        q_rot = apply_rope_jax(q_jax, cos, sin)
        k_rot = apply_rope_jax(k_jax, cos, sin)

        # Attention: transpose to (batch, heads, seq, dim)
        q_t = jnp.transpose(q_rot, (0, 2, 1, 3))
        k_t = jnp.transpose(k_rot, (0, 2, 1, 3))
        v_t = jnp.transpose(v_jax, (0, 2, 1, 3))

        # Compute attention scores
        scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 1, 3, 2))) * scale

        # Causal mask
        mask = jnp.triu(jnp.full((seq, seq), float("-inf")), k=1)
        scores = scores + mask

        # Softmax and weighted sum
        weights = jax.nn.softmax(scores, axis=-1)
        jax_out = jnp.matmul(weights, v_t)
        jax_out = jnp.transpose(jax_out, (0, 2, 1, 3))  # Back to (batch, seq, heads, dim)

        rtol, atol = get_tolerance("fused_ops", "fused_rope_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(jax_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RoPE+Attention forward mismatch (JAX) [{size}]",
        )
