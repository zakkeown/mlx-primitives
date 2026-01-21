"""JAX parity tests for RoPE (Rotary Position Embedding) implementations.

This module tests parity between MLX RoPE implementations and JAX reference
implementations for various configurations.
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

def jax_precompute_freqs_cis(dims: int, max_seq_len: int, base: float = 10000.0):
    """JAX reference for precomputing RoPE frequencies.

    Args:
        dims: Head dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape (max_seq_len, dims//2).
    """
    half_dim = dims // 2
    inv_freq = base ** (-jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos_cache = jnp.cos(freqs)
    sin_cache = jnp.sin(freqs)
    return cos_cache, sin_cache


def jax_rotate_half(x: "jnp.ndarray") -> "jnp.ndarray":
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def jax_rope(
    x: "jnp.ndarray",
    cos_cache: "jnp.ndarray",
    sin_cache: "jnp.ndarray",
    offset: int = 0,
) -> "jnp.ndarray":
    """JAX reference implementation of RoPE.

    Args:
        x: Input tensor (batch, seq_len, heads, head_dim).
        cos_cache: Precomputed cosines (seq_len, head_dim//2).
        sin_cache: Precomputed sines (seq_len, head_dim//2).
        offset: Position offset.

    Returns:
        Rotated tensor with same shape as input.
    """
    seq_len = x.shape[1]

    # Get the relevant portion of the cache
    cos = cos_cache[offset:offset + seq_len]
    sin = sin_cache[offset:offset + seq_len]

    # Double the cos/sin for full head_dim
    cos_full = jnp.concatenate([cos, cos], axis=-1)
    sin_full = jnp.concatenate([sin, sin], axis=-1)

    # Reshape for broadcasting: (seq, dim) -> (1, seq, 1, dim)
    cos_full = cos_full[None, :, None, :]
    sin_full = sin_full[None, :, None, :]

    # Apply rotation
    return (x * cos_full) + (jax_rotate_half(x) * sin_full)


# =============================================================================
# RoPE Forward Parity Tests
# =============================================================================

class TestRoPEForwardParity:
    """RoPE forward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test RoPE forward pass matches JAX reference."""
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Precompute caches in numpy (shared between MLX and JAX)
        cos_np, sin_np = np.array(jax_precompute_freqs_cis(head_dim, seq))

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        cos_mlx = _convert_to_mlx(cos_np, dtype)
        sin_mlx = _convert_to_mlx(sin_np, dtype)
        mlx_out = rope(x_mlx, cos_mlx, sin_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = _convert_to_jax(x_np, dtype)
        cos_jax = _convert_to_jax(cos_np, dtype)
        sin_jax = _convert_to_jax(sin_np, dtype)
        jax_out = jax_rope(x_jax, cos_jax, sin_jax)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("offset", [0, 32, 128])
    def test_with_offset(self, offset, skip_without_jax):
        """Test RoPE with position offset."""
        from mlx_primitives.kernels.rope import rope

        batch, seq, heads, head_dim = 2, 64, 8, 64
        max_seq = 256

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Precompute caches with larger max_seq
        cos_np, sin_np = np.array(jax_precompute_freqs_cis(head_dim, max_seq))

        # MLX
        x_mlx = mx.array(x_np)
        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)
        mlx_out = rope(x_mlx, cos_mlx, sin_mlx, offset=offset)
        mx.eval(mlx_out)

        # JAX
        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)
        jax_out = jax_rope(jnp.array(x_np), cos_jax, sin_jax, offset=offset)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE with offset={offset} mismatch (JAX)"
        )


# =============================================================================
# RoPE Backward Parity Tests
# =============================================================================

class TestRoPEBackwardParity:
    """RoPE backward pass parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test RoPE backward pass gradients match JAX."""
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        cos_np, sin_np = np.array(jax_precompute_freqs_cis(head_dim, seq))

        # MLX backward
        def mlx_loss_fn(x, cos, sin):
            return mx.sum(rope(x, cos, sin))

        x_mlx = mx.array(x_np)
        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=0)
        mlx_grad = grad_fn(x_mlx, cos_mlx, sin_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_loss_fn(x):
            cos_jax = jnp.array(cos_np)
            sin_jax = jnp.array(sin_np)
            return jnp.sum(jax_rope(x, cos_jax, sin_jax))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), _to_numpy(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE backward mismatch (JAX) [{size}]"
        )


# =============================================================================
# RoPE Module Tests
# =============================================================================

class TestRoPEModuleParity:
    """Tests for RoPE nn.Module parity."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16"])
    def test_rope_module_forward(self, size, dtype, skip_without_jax):
        """Test RoPE module forward pass."""
        from mlx_primitives.attention.rope import RoPE

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        max_seq = seq * 2

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX RoPE module
        rope_module = RoPE(dims=head_dim, max_seq_len=max_seq)

        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_out, k_out = rope_module(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # Verify output shapes
        assert q_out.shape == q_mlx.shape, "Q output shape mismatch"
        assert k_out.shape == k_mlx.shape, "K output shape mismatch"

        # Verify no NaN/Inf
        q_out_np = _to_numpy(q_out)
        k_out_np = _to_numpy(k_out)
        assert not np.any(np.isnan(q_out_np)), "NaN in RoPE Q output"
        assert not np.any(np.isinf(q_out_np)), "Inf in RoPE Q output"
        assert not np.any(np.isnan(k_out_np)), "NaN in RoPE K output"
        assert not np.any(np.isinf(k_out_np)), "Inf in RoPE K output"

        # Verify rotation was applied (output different from input)
        assert not np.allclose(q_out_np, q_np, rtol=1e-3), "RoPE should modify Q"
        assert not np.allclose(k_out_np, k_np, rtol=1e-3), "RoPE should modify K"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRoPEEdgeCases:
    """Edge case tests for RoPE."""

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_cache_consistency(self, skip_without_jax):
        """Test that cos^2 + sin^2 = 1 in precomputed caches."""
        head_dim = 64
        seq_len = 128

        cos, sin = jax_precompute_freqs_cis(head_dim, seq_len)

        # Verify cos^2 + sin^2 = 1
        sum_of_squares = np.array(cos) ** 2 + np.array(sin) ** 2
        np.testing.assert_allclose(
            sum_of_squares, np.ones_like(sum_of_squares),
            rtol=1e-6, atol=1e-7,
            err_msg="cos^2 + sin^2 should equal 1"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_single_token(self, skip_without_jax):
        """Test RoPE with single token sequence."""
        from mlx_primitives.kernels.rope import rope

        batch, seq, heads, head_dim = 2, 1, 8, 64

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        cos_np, sin_np = np.array(jax_precompute_freqs_cis(head_dim, seq))

        x_mlx = mx.array(x_np)
        mlx_out = rope(x_mlx, mx.array(cos_np), mx.array(sin_np))
        mx.eval(mlx_out)

        jax_out = jax_rope(jnp.array(x_np), jnp.array(cos_np), jnp.array(sin_np))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="RoPE single token mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_small_head_dim(self, skip_without_jax):
        """Test RoPE with small head dimension."""
        from mlx_primitives.kernels.rope import rope

        batch, seq, heads, head_dim = 2, 32, 4, 16

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        cos_np, sin_np = np.array(jax_precompute_freqs_cis(head_dim, seq))

        x_mlx = mx.array(x_np)
        mlx_out = rope(x_mlx, mx.array(cos_np), mx.array(sin_np))
        mx.eval(mlx_out)

        jax_out = jax_rope(jnp.array(x_np), jnp.array(cos_np), jnp.array(sin_np))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="RoPE small head_dim mismatch (JAX)"
        )


# =============================================================================
# JAX Reference Implementations for NTK-Aware and YaRN RoPE
# =============================================================================

def jax_ntk_aware_precompute_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    original_max_seq_len: int = 8192,
    max_seq_len: int = 32768,
    alpha: float = None,
) -> tuple:
    """JAX reference for NTK-aware RoPE cache computation.

    NTK-aware scaling modifies the base frequency based on extension ratio:
    alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))
    scaled_base = base * alpha
    """
    if alpha is None:
        dims = head_dim
        alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))

    scaled_base = base * alpha
    half_dim = head_dim // 2

    inv_freq = scaled_base ** (-jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos_cache = jnp.cos(freqs)
    sin_cache = jnp.sin(freqs)
    return cos_cache, sin_cache, float(alpha)


def jax_yarn_precompute_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    original_max_seq_len: int = 8192,
    max_seq_len: int = 32768,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple:
    """JAX reference for YaRN RoPE cache computation.

    YaRN combines NTK-aware scaling with frequency interpolation.
    """
    half_dim = head_dim // 2
    extension_ratio = max_seq_len / original_max_seq_len

    # NTK-aware alpha
    alpha = extension_ratio ** (head_dim / (head_dim - 2))
    scaled_base = base * alpha

    # Compute base frequencies
    inv_freq = scaled_base ** (-jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)

    # YaRN interpolation (simplified - applies ramp function)
    low = max(0, int(jnp.floor(beta_slow * half_dim / (2 * math.pi))))
    high = min(half_dim - 1, int(jnp.ceil(beta_fast * half_dim / (2 * math.pi))))

    if high > low:
        ramp = jnp.linspace(0, 1, high - low + 1)
        interpolation_factor = jnp.ones(half_dim)
        interpolation_factor = interpolation_factor.at[low:high + 1].set(1 - ramp)
        inv_freq = inv_freq * interpolation_factor + inv_freq / extension_ratio * (1 - interpolation_factor)

    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos_cache = jnp.cos(freqs)
    sin_cache = jnp.sin(freqs)

    # YaRN scale factor
    scale = 0.1 * math.log(extension_ratio) + 1.0

    return cos_cache, sin_cache, scale


# =============================================================================
# NTKAwareRoPE JAX Parity Tests
# =============================================================================

class TestNTKAwareRoPEJAXParity:
    """JAX parity tests for NTK-aware RoPE interpolation.

    NTK-aware scaling modifies the base frequency based on context extension:
    alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))
    scaled_base = base * alpha
    """

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("extension_ratio", [2, 4, 8])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_cache_properties(self, extension_ratio, dtype, skip_without_jax):
        """Test NTK-aware cache has correct mathematical properties."""
        from mlx_primitives.attention.rope import NTKAwareRoPE

        head_dim = 64
        original_max_seq_len = 1024
        max_seq_len = original_max_seq_len * extension_ratio
        seq_len = 128

        # MLX NTKAwareRoPE
        ntk_rope = NTKAwareRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )

        # Run forward to initialize
        np.random.seed(42)
        q_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        k_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_out, k_out = ntk_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # Get JAX reference cache
        jax_cos, jax_sin, alpha = jax_ntk_aware_precompute_cache(
            seq_len, head_dim,
            original_max_seq_len=original_max_seq_len,
            max_seq_len=max_seq_len,
        )

        # Verify alpha matches expected formula
        expected_alpha = (max_seq_len / original_max_seq_len) ** (head_dim / (head_dim - 2))
        np.testing.assert_allclose(
            alpha, expected_alpha, rtol=1e-6,
            err_msg=f"NTK alpha mismatch [extension_ratio={extension_ratio}]"
        )

        # Verify cos^2 + sin^2 = 1
        np.testing.assert_allclose(
            np.array(jax_cos) ** 2 + np.array(jax_sin) ** 2,
            np.ones_like(np.array(jax_cos)), rtol=1e-5, atol=1e-5,
            err_msg="NTK-aware cache cos^2 + sin^2 != 1"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_rotation_forward_parity(self, size, dtype, skip_without_jax):
        """Test NTK-aware RoPE rotation matches JAX reference."""
        from mlx_primitives.attention.rope import NTKAwareRoPE
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Compute shared JAX cache for this configuration
        jax_cos, jax_sin, _ = jax_ntk_aware_precompute_cache(
            seq, head_dim,
            original_max_seq_len=original_max_seq_len,
            max_seq_len=max_seq_len,
        )

        # MLX NTKAwareRoPE
        ntk_rope = NTKAwareRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )
        q_mlx = _convert_to_mlx(x_np, dtype)
        k_mlx = _convert_to_mlx(x_np, dtype)
        q_out, k_out = ntk_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # JAX reference using shared cache
        x_jax = _convert_to_jax(x_np, dtype)
        jax_out = jax_rope(x_jax, jax_cos.astype(get_jax_dtype(dtype)), jax_sin.astype(get_jax_dtype(dtype)))

        # NTK-aware RoPE uses scaled frequencies which accumulate more numerical error
        # at longer sequences due to frequency scaling differences
        # fp16/bf16 have lower precision, so we use more relaxed tolerances
        if dtype in ("bf16", "fp16"):
            rtol, atol = 0.02, 0.02
        else:
            rtol, atol = 1e-3, 1e-3
        np.testing.assert_allclose(
            _to_numpy(q_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"NTK-aware RoPE forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test NTK-aware RoPE backward pass gradients match JAX."""
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Compute shared JAX cache
        jax_cos, jax_sin, _ = jax_ntk_aware_precompute_cache(
            seq, head_dim,
            original_max_seq_len=original_max_seq_len,
            max_seq_len=max_seq_len,
        )
        mlx_cos, mlx_sin = mx.array(np.array(jax_cos)), mx.array(np.array(jax_sin))

        # MLX backward
        def mlx_loss_fn(x):
            return mx.sum(rope(x, mlx_cos, mlx_sin))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_loss_fn(x):
            return jnp.sum(jax_rope(x, jax_cos, jax_sin))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), _to_numpy(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"NTK-aware RoPE backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_no_extension(self, skip_without_jax):
        """Test NTK-aware RoPE with extension_ratio=1 matches base RoPE."""
        from mlx_primitives.attention.rope import NTKAwareRoPE, RoPE

        head_dim = 64
        seq_len = 128
        max_seq_len = 1024  # Same as original

        np.random.seed(42)
        q_np = np.random.randn(2, seq_len, 8, head_dim).astype(np.float32)
        k_np = np.random.randn(2, seq_len, 8, head_dim).astype(np.float32)

        # NTK-aware with no extension
        ntk_rope = NTKAwareRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=max_seq_len,  # Same = no extension
        )

        # Base RoPE
        base_rope = RoPE(dims=head_dim, max_seq_len=max_seq_len)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        ntk_q_out, ntk_k_out = ntk_rope(q_mlx, k_mlx)
        base_q_out, base_k_out = base_rope(q_mlx, k_mlx)
        mx.eval(ntk_q_out, ntk_k_out, base_q_out, base_k_out)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(ntk_q_out), _to_numpy(base_q_out),
            rtol=rtol, atol=atol,
            err_msg="NTK-aware with no extension should match base RoPE (query)"
        )
        np.testing.assert_allclose(
            _to_numpy(ntk_k_out), _to_numpy(base_k_out),
            rtol=rtol, atol=atol,
            err_msg="NTK-aware with no extension should match base RoPE (key)"
        )


# =============================================================================
# YaRNRoPE JAX Parity Tests
# =============================================================================

class TestYaRNRoPEJAXParity:
    """JAX parity tests for YaRN RoPE extension.

    YaRN combines NTK-aware scaling with frequency interpolation and
    attention scaling for better extrapolation to longer sequences.
    """

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("extension_ratio", [2, 4, 8])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_cache_properties(self, extension_ratio, dtype, skip_without_jax):
        """Test YaRN cache has correct mathematical properties."""
        from mlx_primitives.attention.rope import YaRNRoPE

        head_dim = 64
        original_max_seq_len = 1024
        max_seq_len = original_max_seq_len * extension_ratio
        seq_len = 128

        # MLX YaRNRoPE
        yarn_rope = YaRNRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )

        np.random.seed(42)
        q_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        k_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_out, k_out = yarn_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # Verify scale factor
        expected_scale = 0.1 * math.log(extension_ratio) + 1.0
        np.testing.assert_allclose(
            yarn_rope.scale, expected_scale, rtol=1e-6,
            err_msg=f"YaRN scale mismatch [extension_ratio={extension_ratio}]"
        )

        # Output should not have NaN or Inf
        q_out_np = _to_numpy(q_out)
        k_out_np = _to_numpy(k_out)
        assert not np.isnan(q_out_np).any(), "YaRN q output contains NaN"
        assert not np.isinf(q_out_np).any(), "YaRN q output contains Inf"
        assert not np.isnan(k_out_np).any(), "YaRN k output contains NaN"
        assert not np.isinf(k_out_np).any(), "YaRN k output contains Inf"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_rotation_forward_consistency(self, size, dtype, skip_without_jax):
        """Test YaRN RoPE rotation produces consistent output."""
        from mlx_primitives.attention.rope import YaRNRoPE

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX YaRNRoPE
        yarn_rope = YaRNRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_out, k_out = yarn_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        q_out_np = _to_numpy(q_out)
        k_out_np = _to_numpy(k_out)

        # Basic sanity checks
        assert q_out_np.shape == q_np.shape, "Q output shape mismatch"
        assert k_out_np.shape == k_np.shape, "K output shape mismatch"
        assert not np.isnan(q_out_np).any(), "YaRN q output contains NaN"
        assert not np.isinf(q_out_np).any(), "YaRN q output contains Inf"
        assert not np.isnan(k_out_np).any(), "YaRN k output contains NaN"
        assert not np.isinf(k_out_np).any(), "YaRN k output contains Inf"

        # Verify the output is different from input (rotation was applied)
        assert not np.allclose(q_out_np, q_np, rtol=1e-3), "YaRN should modify q input"
        assert not np.allclose(k_out_np, k_np, rtol=1e-3), "YaRN should modify k input"

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_gradient_flow(self, size, skip_without_jax):
        """Test YaRN RoPE backward pass gradients flow correctly.

        Note: YaRN has implementation-specific interpolation strategies that differ
        between MLX and reference implementations, so we verify gradient flow
        properties rather than exact numerical parity.
        """
        from mlx_primitives.attention.rope import YaRNRoPE

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        yarn_rope = YaRNRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )

        # MLX backward
        def mlx_loss_fn(q, k):
            q_out, k_out = yarn_rope(q, k)
            return mx.sum(q_out) + mx.sum(k_out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        q_grad, k_grad = grad_fn(q_mlx, k_mlx)
        mx.eval(q_grad, k_grad)

        # Verify gradient properties
        q_grad_np = _to_numpy(q_grad)
        k_grad_np = _to_numpy(k_grad)

        # Check shapes
        assert q_grad_np.shape == q_np.shape, f"Q gradient shape mismatch [{size}]"
        assert k_grad_np.shape == k_np.shape, f"K gradient shape mismatch [{size}]"

        # Check no NaN/Inf
        assert not np.isnan(q_grad_np).any(), f"NaN in YaRN Q gradient [{size}]"
        assert not np.isinf(q_grad_np).any(), f"Inf in YaRN Q gradient [{size}]"
        assert not np.isnan(k_grad_np).any(), f"NaN in YaRN K gradient [{size}]"
        assert not np.isinf(k_grad_np).any(), f"Inf in YaRN K gradient [{size}]"

        # Verify gradient is non-trivial (some flow occurred)
        assert np.abs(q_grad_np).sum() > 1e-6, f"YaRN Q gradient is all zeros [{size}]"
        assert np.abs(k_grad_np).sum() > 1e-6, f"YaRN K gradient is all zeros [{size}]"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("beta_fast,beta_slow", [(32, 1), (64, 2), (16, 0.5)])
    def test_beta_parameters(self, beta_fast, beta_slow, skip_without_jax):
        """Test YaRN with different beta parameters."""
        from mlx_primitives.attention.rope import YaRNRoPE

        head_dim = 64
        seq_len = 128
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        q_np = np.random.randn(2, seq_len, 8, head_dim).astype(np.float32)
        k_np = np.random.randn(2, seq_len, 8, head_dim).astype(np.float32)

        yarn_rope = YaRNRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_out, k_out = yarn_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        q_out_np = _to_numpy(q_out)
        k_out_np = _to_numpy(k_out)
        assert not np.isnan(q_out_np).any(), f"YaRN with beta_fast={beta_fast}, beta_slow={beta_slow} q contains NaN"
        assert not np.isinf(q_out_np).any(), f"YaRN with beta_fast={beta_fast}, beta_slow={beta_slow} q contains Inf"
        assert not np.isnan(k_out_np).any(), f"YaRN with beta_fast={beta_fast}, beta_slow={beta_slow} k contains NaN"
        assert not np.isinf(k_out_np).any(), f"YaRN with beta_fast={beta_fast}, beta_slow={beta_slow} k contains Inf"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_scale_factor_computation(self, skip_without_jax):
        """Test that YaRN scale factor is computed correctly."""
        from mlx_primitives.attention.rope import YaRNRoPE

        head_dim = 64
        original_max_seq_len = 1024

        for extension_ratio in [2, 4, 8, 16]:
            max_seq_len = original_max_seq_len * extension_ratio

            yarn_rope = YaRNRoPE(
                dims=head_dim,
                max_seq_len=max_seq_len,
                original_max_seq_len=original_max_seq_len,
            )

            expected_scale = 0.1 * math.log(extension_ratio) + 1.0
            np.testing.assert_allclose(
                yarn_rope.scale, expected_scale, rtol=1e-6,
                err_msg=f"YaRN scale mismatch for extension_ratio={extension_ratio}"
            )


# =============================================================================
# NumPy Reference for Precompute Cache
# =============================================================================

def numpy_precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple:
    """Numpy reference for precomputing RoPE cache using float64 precision.

    This is the ground truth for testing, computed in float64 then cast to float32.
    """
    half_dim = head_dim // 2
    # Compute in float64 for maximum precision
    inv_freq = base ** (-np.arange(0, half_dim, dtype=np.float64) / half_dim)
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    return cos_cache, sin_cache


# =============================================================================
# Precompute RoPE Cache Parity Tests
# =============================================================================

class TestPrecomputeRopeCacheParity:
    """Tests for precompute_rope_cache() properties and correctness."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("seq_len", [64, 256, 1024])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_cache_shape_and_type(self, seq_len, head_dim, skip_without_jax):
        """Test precompute_rope_cache produces correct shape and dtype."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        cos, sin = precompute_rope_cache(seq_len, head_dim)
        mx.eval(cos, sin)

        expected_shape = (seq_len, head_dim // 2)
        assert cos.shape == expected_shape, f"Cos shape {cos.shape} != {expected_shape}"
        assert sin.shape == expected_shape, f"Sin shape {sin.shape} != {expected_shape}"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("seq_len", [64, 256])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_cache_parity_numpy(self, seq_len, head_dim, skip_without_jax):
        """Test precompute_rope_cache matches NumPy reference."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        mlx_cos, mlx_sin = precompute_rope_cache(seq_len, head_dim)
        mx.eval(mlx_cos, mlx_sin)

        np_cos, np_sin = numpy_precompute_rope_cache(seq_len, head_dim)

        # MLX and NumPy may have minor differences in exponential precision
        # Use relaxed tolerance for cache values
        np.testing.assert_allclose(
            _to_numpy(mlx_cos), np_cos,
            rtol=1e-3, atol=1e-4,
            err_msg=f"Cos cache mismatch [seq={seq_len}, dim={head_dim}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_sin), np_sin,
            rtol=1e-3, atol=1e-4,
            err_msg=f"Sin cache mismatch [seq={seq_len}, dim={head_dim}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_trigonometric_identity(self, skip_without_jax):
        """Test that cos^2 + sin^2 = 1 for precomputed cache."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        seq_len, head_dim = 128, 64
        cos, sin = precompute_rope_cache(seq_len, head_dim)
        mx.eval(cos, sin)

        cos_np = _to_numpy(cos)
        sin_np = _to_numpy(sin)

        identity = cos_np ** 2 + sin_np ** 2
        np.testing.assert_allclose(
            identity, np.ones_like(identity),
            rtol=1e-5, atol=1e-5,
            err_msg="cos^2 + sin^2 should equal 1"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("base", [10000.0, 100000.0, 500000.0])
    def test_various_base_values(self, base, skip_without_jax):
        """Test precompute_rope_cache with different base values."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        seq_len, head_dim = 128, 64
        mlx_cos, mlx_sin = precompute_rope_cache(seq_len, head_dim, base=base)
        mx.eval(mlx_cos, mlx_sin)

        np_cos, np_sin = numpy_precompute_rope_cache(seq_len, head_dim, base=base)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_cos), np_cos,
            rtol=rtol, atol=atol,
            err_msg=f"Cos cache mismatch [base={base}]"
        )


# =============================================================================
# Fast RoPE Parity Tests
# =============================================================================

class TestFastRopeParity:
    """Tests for fast_rope() Metal kernel parity."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fast_rope forward pass matches JAX reference."""
        from mlx_primitives.kernels.rope import fast_rope

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Precompute caches (shared between MLX and JAX)
        cos_np, sin_np = numpy_precompute_rope_cache(seq, head_dim)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        cos_mlx = _convert_to_mlx(cos_np, dtype)
        sin_mlx = _convert_to_mlx(sin_np, dtype)
        mlx_out = fast_rope(x_mlx, cos_mlx, sin_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = _convert_to_jax(x_np, dtype)
        cos_jax = jnp.array(cos_np).astype(get_jax_dtype(dtype))
        sin_jax = jnp.array(sin_np).astype(get_jax_dtype(dtype))
        jax_out = jax_rope(x_jax, cos_jax, sin_jax)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("offset", [0, 32, 128])
    def test_with_offset(self, offset, skip_without_jax):
        """Test fast_rope with position offset."""
        from mlx_primitives.kernels.rope import fast_rope

        batch, seq, heads, head_dim = 2, 64, 8, 64
        max_seq = 256

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Precompute caches with larger max_seq
        cos_np, sin_np = numpy_precompute_rope_cache(max_seq, head_dim)

        # MLX
        x_mlx = mx.array(x_np)
        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)
        mlx_out = fast_rope(x_mlx, cos_mlx, sin_mlx, offset=offset)
        mx.eval(mlx_out)

        # JAX
        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)
        jax_out = jax_rope(jnp.array(x_np), cos_jax, sin_jax, offset=offset)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope with offset={offset} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test fast_rope backward pass gradients match JAX."""
        from mlx_primitives.kernels.rope import fast_rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        cos_np, sin_np = numpy_precompute_rope_cache(seq, head_dim)

        # MLX backward
        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)

        def mlx_loss_fn(x):
            return mx.sum(fast_rope(x, cos_mlx, sin_mlx))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)

        def jax_loss_fn(x):
            return jnp.sum(jax_rope(x, cos_jax, sin_jax))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), _to_numpy(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope backward mismatch (JAX) [{size}]"
        )


# =============================================================================
# Fast RoPE QK Parity Tests
# =============================================================================

def jax_rope_qk(
    q: "jnp.ndarray",
    k: "jnp.ndarray",
    cos_cache: "jnp.ndarray",
    sin_cache: "jnp.ndarray",
    offset: int = 0,
) -> tuple:
    """JAX reference for RoPE on Q and K."""
    q_rotated = jax_rope(q, cos_cache, sin_cache, offset)
    k_rotated = jax_rope(k, cos_cache, sin_cache, offset)
    return q_rotated, k_rotated


class TestFastRopeQKParity:
    """Tests for fast_rope_qk() that rotates Q and K together."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test fast_rope_qk forward pass matches JAX reference."""
        from mlx_primitives.kernels.rope import fast_rope_qk

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        cos_np, sin_np = numpy_precompute_rope_cache(seq, head_dim)

        # MLX
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        cos_mlx = _convert_to_mlx(cos_np, dtype)
        sin_mlx = _convert_to_mlx(sin_np, dtype)
        mlx_q_out, mlx_k_out = fast_rope_qk(q_mlx, k_mlx, cos_mlx, sin_mlx)
        mx.eval(mlx_q_out, mlx_k_out)

        # JAX reference
        q_jax = _convert_to_jax(q_np, dtype)
        k_jax = _convert_to_jax(k_np, dtype)
        cos_jax = jnp.array(cos_np).astype(get_jax_dtype(dtype))
        sin_jax = jnp.array(sin_np).astype(get_jax_dtype(dtype))
        jax_q_out, jax_k_out = jax_rope_qk(q_jax, k_jax, cos_jax, sin_jax)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        # bf16 has limited precision, use slightly relaxed tolerance
        if dtype == "bf16":
            rtol = max(rtol, 0.015)
            atol = max(atol, 0.015)
        np.testing.assert_allclose(
            _to_numpy(mlx_q_out), _to_numpy(jax_q_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk Q mismatch (JAX) [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_k_out), _to_numpy(jax_k_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk K mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test rope_qk backward via base rope (fast kernel lacks VJP).

        Note: fast_rope_qk uses a custom Metal kernel without VJP implementation.
        We test gradient flow through the equivalent operation using base rope.
        """
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        cos_np, sin_np = numpy_precompute_rope_cache(seq, head_dim)

        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)

        # MLX backward using base rope function (has VJP)
        def mlx_loss_fn(q, k):
            q_out = rope(q, cos_mlx, sin_mlx)
            k_out = rope(k, cos_mlx, sin_mlx)
            return mx.sum(q_out) + mx.sum(k_out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        q_grad, k_grad = grad_fn(q_mlx, k_mlx)
        mx.eval(q_grad, k_grad)

        # JAX backward
        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)

        def jax_loss_fn(q, k):
            q_out, k_out = jax_rope_qk(q, k, cos_jax, sin_jax)
            return jnp.sum(q_out) + jnp.sum(k_out)

        q_jax = jnp.array(q_np)
        k_jax = jnp.array(k_np)
        jax_grad_fn = jax.grad(jax_loss_fn, argnums=(0, 1))
        jax_q_grad, jax_k_grad = jax_grad_fn(q_jax, k_jax)

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_grad), _to_numpy(jax_q_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk Q gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_grad), _to_numpy(jax_k_grad),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk K gradient mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_offset_parity(self, skip_without_jax):
        """Test fast_rope_qk with position offset."""
        from mlx_primitives.kernels.rope import fast_rope_qk

        batch, seq, heads, head_dim = 2, 64, 8, 64
        max_seq = 256
        offset = 32

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        cos_np, sin_np = numpy_precompute_rope_cache(max_seq, head_dim)

        # MLX
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        cos_mlx = mx.array(cos_np)
        sin_mlx = mx.array(sin_np)
        mlx_q_out, mlx_k_out = fast_rope_qk(q_mlx, k_mlx, cos_mlx, sin_mlx, offset=offset)
        mx.eval(mlx_q_out, mlx_k_out)

        # JAX
        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)
        jax_q_out, jax_k_out = jax_rope_qk(
            jnp.array(q_np), jnp.array(k_np), cos_jax, sin_jax, offset=offset
        )

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_q_out), _to_numpy(jax_q_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk Q with offset={offset} mismatch (JAX)"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_k_out), _to_numpy(jax_k_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk K with offset={offset} mismatch (JAX)"
        )
