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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_jax):
        """Test that fused op matches separate RMSNorm and Linear."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        batch, seq, hidden = 4, 128, 512
        out_features = 2048
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.random.randn(hidden).astype(np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)

        # Fused version
        fused_out = fused_rmsnorm_linear(x_mlx, norm_weight_mlx, linear_weight_mlx, eps=eps)
        mx.eval(fused_out)

        # Separate operations (using JAX-style computation)
        rms = mx.sqrt(mx.mean(x_mlx * x_mlx, axis=-1, keepdims=True) + eps)
        norm_x = x_mlx / rms * norm_weight_mlx
        separate_out = norm_x @ linear_weight_mlx.T
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Fused vs separate RMSNorm+Linear mismatch",
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_with_bias(self, skip_without_jax):
        """Test fused op with bias parameter."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        batch, seq, hidden = 2, 64, 256
        out_features = 1024
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02
        linear_bias_np = np.random.randn(out_features).astype(np.float32) * 0.01

        # MLX
        x_mlx = mx.array(x_np)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)
        linear_bias_mlx = mx.array(linear_bias_np)

        mlx_out = fused_rmsnorm_linear(
            x_mlx, norm_weight_mlx, linear_weight_mlx, linear_bias=linear_bias_mlx, eps=eps
        )
        mx.eval(mlx_out)

        # JAX reference
        x_jax = jnp.array(x_np)
        norm_weight_jax = jnp.array(norm_weight_np)
        linear_weight_jax = jnp.array(linear_weight_np)
        linear_bias_jax = jnp.array(linear_bias_np)

        rms = jnp.sqrt(jnp.mean(x_jax ** 2, axis=-1, keepdims=True) + eps)
        norm_x = (x_jax / rms) * norm_weight_jax
        jax_out = norm_x @ linear_weight_jax.T + linear_bias_jax

        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(jax_out),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Fused RMSNorm+Linear with bias mismatch (JAX)",
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

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fused SwiGLU backward pass parity with JAX."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX backward
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_swiglu(x, W_gate_mlx, W_up_mlx))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        W_gate_jax = jnp.array(W_gate_np)
        W_up_jax = jnp.array(W_up_np)

        def jax_loss_fn(x):
            gate = jnn.silu(x @ W_gate_jax.T)
            up = x @ W_up_jax.T
            return jnp.sum(gate * up)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_swiglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            _to_numpy(jax_grad),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused SwiGLU backward mismatch (JAX) [{size}]",
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_jax):
        """Test that fused SwiGLU matches separate ops."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        batch, seq, dim = 4, 128, 512
        hidden_dim = 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        # Fused
        fused_out = fused_swiglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(fused_out)

        # Separate (reference implementation)
        gate = x_mlx @ W_gate_mlx.T
        up = x_mlx @ W_up_mlx.T
        silu_gate = gate * mx.sigmoid(gate)  # silu = x * sigmoid(x)
        separate_out = silu_gate * up
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate SwiGLU mismatch",
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

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fused GeGLU backward pass parity with JAX."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX backward
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_geglu(x, W_gate_mlx, W_up_mlx))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        W_gate_jax = jnp.array(W_gate_np)
        W_up_jax = jnp.array(W_up_np)

        def jax_loss_fn(x):
            gate = jnn.gelu(x @ W_gate_jax.T, approximate=True)
            up = x @ W_up_jax.T
            return jnp.sum(gate * up)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_geglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            _to_numpy(jax_grad),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused GeGLU backward mismatch (JAX) [{size}]",
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_jax):
        """Test that fused GeGLU matches separate ops."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        batch, seq, dim = 4, 128, 512
        hidden_dim = 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        # Fused
        fused_out = fused_geglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(fused_out)

        # Separate (reference implementation using tanh approximation)
        gate = x_mlx @ W_gate_mlx.T
        up = x_mlx @ W_up_mlx.T
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715
        gelu_gate = 0.5 * gate * (1.0 + mx.tanh(sqrt_2_over_pi * (gate + coeff * gate ** 3)))
        separate_out = gelu_gate * up
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate GeGLU mismatch",
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

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fused RoPE+Attention backward pass parity with JAX."""
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

        # MLX backward
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        def mlx_loss_fn(q):
            return mx.sum(fused_rope_attention(q, k_mlx, v_mlx, scale=scale, causal=True))

        q_mlx = mx.array(q_np)
        mlx_grad = mx.grad(mlx_loss_fn)(q_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        k_jax = jnp.array(k_np)
        v_jax = jnp.array(v_np)

        # RoPE frequencies
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        positions = jnp.arange(seq, dtype=jnp.float32)
        angles = positions[:, None] * inv_freq[None, :]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        def apply_rope_jax(x, cos, sin):
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            cos_expanded = cos[None, :, None, :]
            sin_expanded = sin[None, :, None, :]
            return jnp.concatenate([
                x1 * cos_expanded - x2 * sin_expanded,
                x1 * sin_expanded + x2 * cos_expanded,
            ], axis=-1)

        def jax_loss_fn(q):
            k_rot = apply_rope_jax(k_jax, cos, sin)
            q_rot = apply_rope_jax(q, cos, sin)
            q_t = jnp.transpose(q_rot, (0, 2, 1, 3))
            k_t = jnp.transpose(k_rot, (0, 2, 1, 3))
            v_t = jnp.transpose(v_jax, (0, 2, 1, 3))
            scores = jnp.matmul(q_t, jnp.transpose(k_t, (0, 1, 3, 2))) * scale
            mask = jnp.triu(jnp.full((seq, seq), float("-inf")), k=1)
            scores = scores + mask
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.matmul(weights, v_t)
            out = jnp.transpose(out, (0, 2, 1, 3))
            return jnp.sum(out)

        q_jax = jnp.array(q_np)
        jax_grad = jax.grad(jax_loss_fn)(q_jax)

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_rope_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            _to_numpy(jax_grad),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RoPE+Attention backward mismatch (JAX) [{size}]",
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_jax):
        """Test that fused op matches separate RoPE and Attention."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention
        from mlx_primitives.attention.flash import flash_attention

        batch, seq, heads, head_dim = 2, 128, 8, 64
        scale = 1.0 / (head_dim ** 0.5)
        base = 10000.0

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        # Fused
        fused_out = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(fused_out)

        # Separate: apply RoPE manually + flash_attention
        half_dim = head_dim // 2
        positions = mx.arange(seq)
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        angles = positions[:, None] * inv_freq[None, :]
        cos = mx.cos(angles)
        sin = mx.sin(angles)

        def apply_rope_mlx(x, cos, sin):
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            cos_expanded = cos[None, :, None, :]
            sin_expanded = sin[None, :, None, :]
            return mx.concatenate([
                x1 * cos_expanded - x2 * sin_expanded,
                x1 * sin_expanded + x2 * cos_expanded,
            ], axis=-1)

        q_rot = apply_rope_mlx(q_mlx, cos, sin)
        k_rot = apply_rope_mlx(k_mlx, cos, sin)
        separate_out = flash_attention(q_rot, k_rot, v_mlx, scale=scale, causal=True)
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate RoPE+Attention mismatch",
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_causal_masking(self, skip_without_jax):
        """Test fused RoPE+Attention with causal masking."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        batch, seq, heads, head_dim = 2, 64, 4, 32
        scale = 1.0 / (head_dim ** 0.5)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        # Causal
        out_causal = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(out_causal)

        # Non-causal
        out_non_causal = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=False)
        mx.eval(out_non_causal)

        causal_np = _to_numpy(out_causal)
        non_causal_np = _to_numpy(out_non_causal)

        # Both outputs should have valid values (no NaN/Inf)
        assert not np.any(np.isnan(causal_np)), "NaN in causal output"
        assert not np.any(np.isinf(causal_np)), "Inf in causal output"
        assert not np.any(np.isnan(non_causal_np)), "NaN in non-causal output"
        assert not np.any(np.isinf(non_causal_np)), "Inf in non-causal output"

        # Causal and non-causal should produce different results
        assert not np.allclose(
            causal_np, non_causal_np, rtol=1e-3
        ), "Causal and non-causal outputs should differ"

        # Verify outputs have reasonable magnitudes
        assert np.abs(causal_np).mean() > 0.01, "Causal output seems too small"
        assert np.abs(non_causal_np).mean() > 0.01, "Non-causal output seems too small"
