"""JAX parity tests for GQA (Grouped Query Attention) implementations.

This module tests parity between MLX GQA implementations and JAX reference
implementations. GQA is tested with various group ratios including MQA
(Multi-Query Attention) and standard MHA.
"""

import math
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
# GQA-Specific Size Configurations
# =============================================================================

GQA_SIZE_CONFIGS = {
    "tiny": {
        "batch": 1, "seq_q": 64, "seq_kv": 64,
        "num_q_heads": 8, "num_kv_heads": 2, "head_dim": 32
    },
    "small": {
        "batch": 2, "seq_q": 256, "seq_kv": 256,
        "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64
    },
    "medium": {
        "batch": 4, "seq_q": 512, "seq_kv": 512,
        "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64
    },
}


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
# JAX Reference Implementation
# =============================================================================

def jax_gqa_attention(
    q: "jnp.ndarray",
    k: "jnp.ndarray",
    v: "jnp.ndarray",
    num_kv_groups: int,
    scale: float = None,
    causal: bool = False,
) -> "jnp.ndarray":
    """JAX reference implementation of GQA with K/V expansion.

    Args:
        q: Query tensor (batch, seq_q, num_q_heads, head_dim)
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim)
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim)
        num_kv_groups: Number of Q heads per KV head
        scale: Attention scale factor
        causal: Whether to apply causal masking

    Returns:
        Output tensor (batch, seq_q, num_q_heads, head_dim)
    """
    batch_size, seq_q, num_q_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Expand K and V: repeat each KV head 'num_kv_groups' times
    k_expanded = jnp.repeat(k, num_kv_groups, axis=2)
    v_expanded = jnp.repeat(v, num_kv_groups, axis=2)

    # Transpose for attention: (batch, heads, seq, dim)
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k_expanded, (0, 2, 1, 3))
    v_t = jnp.transpose(v_expanded, (0, 2, 1, 3))

    # Compute attention scores in float32
    q_fp32 = q_t.astype(jnp.float32)
    k_fp32 = k_t.astype(jnp.float32)
    v_fp32 = v_t.astype(jnp.float32)

    scores = jnp.matmul(q_fp32, jnp.transpose(k_fp32, (0, 1, 3, 2))) * scale

    # Causal mask
    if causal:
        mask = jnp.triu(jnp.full((seq_q, seq_kv), float("-inf")), k=seq_kv - seq_q + 1)
        scores = scores + mask

    # Softmax and output
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, v_fp32)

    # Convert back to original dtype and transpose
    output = output.astype(q.dtype)
    return jnp.transpose(output, (0, 2, 1, 3))


# =============================================================================
# GQA Forward Parity Tests
# =============================================================================

class TestGQAForwardParity:
    """GQA forward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_parity(self, size, dtype, causal, skip_without_jax):
        """Test GQA forward pass matches JAX reference."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups, causal=causal)
        mx.eval(mlx_out)

        # JAX reference
        q_jax = _convert_to_jax(q_np, dtype)
        k_jax = _convert_to_jax(k_np, dtype)
        v_jax = _convert_to_jax(v_np, dtype)
        jax_out = jax_gqa_attention(q_jax, k_jax, v_jax, num_kv_groups, causal=causal)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GQA forward mismatch (JAX) [{size}, {dtype}, causal={causal}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_kv_groups", [1, 2, 4, 8])
    def test_various_group_ratios(self, num_kv_groups, skip_without_jax):
        """Test GQA with different Q-to-KV head ratios."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        batch, seq, head_dim = 2, 128, 64
        num_kv_heads = 4
        num_q_heads = num_kv_heads * num_kv_groups

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_gqa_attention(
            jnp.array(q_np), jnp.array(k_np), jnp.array(v_np), num_kv_groups
        )

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GQA with num_kv_groups={num_kv_groups} mismatch (JAX)"
        )


# =============================================================================
# GQA Backward Parity Tests
# =============================================================================

class TestGQABackwardParity:
    """GQA backward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test GQA backward pass gradients match JAX."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX backward
        def mlx_loss_fn(q, k, v):
            out = gqa_attention_reference(q, k, v, num_kv_groups)
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        q_grad, k_grad, v_grad = grad_fn(q_mlx, k_mlx, v_mlx)
        mx.eval(q_grad, k_grad, v_grad)

        # JAX backward (force CPU to avoid JAX-Metal gradient limitations)
        def jax_loss_fn(q, k, v):
            out = jax_gqa_attention(q, k, v, num_kv_groups)
            return jnp.sum(out)

        # JAX Metal has limited gradient support, use CPU for reliable gradients
        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            q_jax = jnp.array(q_np)
            k_jax = jnp.array(k_np)
            v_jax = jnp.array(v_np)
            jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1, 2))
            jax_q_grad, jax_k_grad, jax_v_grad = jax_grad_fn(q_jax, k_jax, v_jax)

        rtol, atol = get_gradient_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), _to_numpy(jax_q_grad),
            rtol=rtol, atol=atol,
            err_msg=f"GQA Q gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), _to_numpy(jax_k_grad),
            rtol=rtol, atol=atol,
            err_msg=f"GQA K gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_grad), _to_numpy(jax_v_grad),
            rtol=rtol, atol=atol,
            err_msg=f"GQA V gradient mismatch (JAX) [{size}]"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestGQAEdgeCases:
    """Edge case tests for GQA implementations."""

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_mqa_single_kv_head(self, skip_without_jax):
        """Test MQA (Multi-Query Attention) - single KV head."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        batch, seq, num_q_heads, head_dim = 2, 64, 8, 64
        num_kv_heads = 1  # MQA
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        jax_out = jax_gqa_attention(
            jnp.array(q_np), jnp.array(k_np), jnp.array(v_np), num_kv_groups
        )

        rtol, atol = get_tolerance("attention", "mqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg="MQA (single KV head) mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_standard_mha(self, skip_without_jax):
        """Test that GQA with groups=1 equals standard MHA."""
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        batch, seq, num_heads, head_dim = 2, 64, 8, 64
        num_kv_groups = 1  # Standard MHA

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = gqa_attention_reference(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        jax_out = jax_gqa_attention(
            jnp.array(q_np), jnp.array(k_np), jnp.array(v_np), num_kv_groups
        )

        rtol, atol = get_tolerance("attention", "flash_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg="GQA with groups=1 should match MHA (JAX)"
        )


# =============================================================================
# Fast GQA Attention Parity Tests
# =============================================================================

class TestFastGQAAttentionParity:
    """Tests for fast_gqa_attention() Metal kernel parity with JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fast_gqa_attention forward pass matches JAX reference."""
        from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups)
        mx.eval(mlx_out)

        # JAX reference
        q_jax = _convert_to_jax(q_np, dtype)
        k_jax = _convert_to_jax(k_np, dtype)
        v_jax = _convert_to_jax(v_np, dtype)
        jax_out = jax_gqa_attention(q_jax, k_jax, v_jax, num_kv_groups)

        rtol, atol = get_tolerance("attention", "gqa", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa_attention mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("causal", [False, True])
    def test_causal_mask(self, causal, skip_without_jax):
        """Test fast_gqa_attention with causal masking."""
        from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

        batch, seq, num_q_heads, num_kv_heads, head_dim = 2, 64, 8, 2, 32
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        mlx_out = fast_gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups, causal=causal)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_gqa_attention(
            jnp.array(q_np), jnp.array(k_np), jnp.array(v_np),
            num_kv_groups, causal=causal
        )

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa_attention causal={causal} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fast_gqa_attention backward via reference GQA (fast kernel lacks VJP).

        Note: fast_gqa_attention uses a custom Metal kernel without VJP implementation.
        We test gradient flow through the reference GQA implementation instead.
        """
        from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq_q = config["seq_q"]
        seq_kv = config["seq_kv"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        num_kv_groups = num_q_heads // num_kv_heads

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_q, num_q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_kv, num_kv_heads, head_dim).astype(np.float32)

        # MLX backward using reference implementation (has VJP)
        def mlx_loss_fn(q, k, v):
            out = gqa_attention_reference(q, k, v, num_kv_groups)
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        q_grad, k_grad, v_grad = grad_fn(q_mlx, k_mlx, v_mlx)
        mx.eval(q_grad, k_grad, v_grad)

        # JAX backward (use CPU for reliable gradients)
        def jax_loss_fn(q, k, v):
            out = jax_gqa_attention(q, k, v, num_kv_groups)
            return jnp.sum(out)

        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            q_jax = jnp.array(q_np)
            k_jax = jnp.array(k_np)
            v_jax = jnp.array(v_np)
            jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1, 2))
            jax_q_grad, jax_k_grad, jax_v_grad = jax_grad_fn(q_jax, k_jax, v_jax)

        rtol, atol = get_gradient_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), _to_numpy(jax_q_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa Q gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), _to_numpy(jax_k_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa K gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_grad), _to_numpy(jax_v_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_gqa V gradient mismatch (JAX) [{size}]"
        )


# =============================================================================
# OptimizedGQA Module Parity Tests
# =============================================================================

class TestOptimizedGQAModuleParity:
    """Tests for OptimizedGQA module parity with JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_module_forward_parity(self, size, skip_without_jax):
        """Test OptimizedGQA module forward pass matches JAX reference."""
        from mlx_primitives.kernels.gqa_optimized import OptimizedGQA

        config = GQA_SIZE_CONFIGS[size]
        batch = config["batch"]
        seq = config["seq_q"]
        num_q_heads = config["num_q_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]
        hidden_dim = num_q_heads * head_dim

        np.random.seed(42)

        # Create module
        module = OptimizedGQA(
            dims=hidden_dim,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Set known weights for reproducibility
        wq_np = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        wk_np = np.random.randn(num_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.02
        wv_np = np.random.randn(num_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.02
        wo_np = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02

        module.q_proj.weight = mx.array(wq_np)
        module.k_proj.weight = mx.array(wk_np)
        module.v_proj.weight = mx.array(wv_np)
        module.out_proj.weight = mx.array(wo_np)

        # Input
        x_np = np.random.randn(batch, seq, hidden_dim).astype(np.float32)
        x_mlx = mx.array(x_np)

        # MLX forward (returns tuple: output, cache)
        mlx_out, _ = module(x_mlx)
        mx.eval(mlx_out)

        # JAX reference: manual GQA computation
        x_jax = jnp.array(x_np)

        # Project Q, K, V
        q = x_jax @ jnp.array(wq_np.T)
        k = x_jax @ jnp.array(wk_np.T)
        v = x_jax @ jnp.array(wv_np.T)

        # Reshape for attention
        q = q.reshape(batch, seq, num_q_heads, head_dim)
        k = k.reshape(batch, seq, num_kv_heads, head_dim)
        v = v.reshape(batch, seq, num_kv_heads, head_dim)

        # GQA attention
        num_kv_groups = num_q_heads // num_kv_heads
        attn_out = jax_gqa_attention(q, k, v, num_kv_groups)

        # Reshape and project output
        attn_out = attn_out.reshape(batch, seq, hidden_dim)
        jax_out = attn_out @ jnp.array(wo_np.T)

        rtol, atol = get_tolerance("attention", "gqa", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"OptimizedGQA module forward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_module_with_kv_cache(self, skip_without_jax):
        """Test OptimizedGQA module with KV cache for incremental decoding."""
        from mlx_primitives.kernels.gqa_optimized import OptimizedGQA

        batch, num_q_heads, num_kv_heads, head_dim = 2, 8, 2, 32
        hidden_dim = num_q_heads * head_dim

        np.random.seed(42)

        module = OptimizedGQA(
            dims=hidden_dim,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Set weights
        module.q_proj.weight = mx.array(np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02)
        module.k_proj.weight = mx.array(np.random.randn(num_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.02)
        module.v_proj.weight = mx.array(np.random.randn(num_kv_heads * head_dim, hidden_dim).astype(np.float32) * 0.02)
        module.out_proj.weight = mx.array(np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02)

        # Initial context
        context_np = np.random.randn(batch, 32, hidden_dim).astype(np.float32)
        context_mlx = mx.array(context_np)

        # Process context (returns tuple: output, cache)
        out1, cache = module(context_mlx)
        mx.eval(out1)

        # Single token decode with cache
        token_np = np.random.randn(batch, 1, hidden_dim).astype(np.float32)
        token_mlx = mx.array(token_np)

        out2, _ = module(token_mlx, cache=cache)
        mx.eval(out2)

        # Verify outputs have correct shapes
        assert out1.shape == (batch, 32, hidden_dim), f"Context output shape mismatch: {out1.shape}"
        assert out2.shape == (batch, 1, hidden_dim), f"Token output shape mismatch: {out2.shape}"

        # Verify no NaN/Inf
        assert not np.any(np.isnan(_to_numpy(out1))), "NaN in context output"
        assert not np.any(np.isnan(_to_numpy(out2))), "NaN in token output"
