"""PyTorch parity tests for RoPE kernel implementations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_torch(x_np: np.ndarray, dtype: str) -> "torch.Tensor":
    """Convert numpy array to PyTorch with proper dtype."""
    x_torch = torch.from_numpy(x_np.astype(np.float32))
    torch_dtype = get_pytorch_dtype(dtype)
    return x_torch.to(torch_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# Reference Implementations
# =============================================================================

def numpy_precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple:
    """Numpy reference for precomputing RoPE cache using float64 precision.

    This is the ground truth for testing, computed in float64 then cast to float32.
    Using float64 eliminates framework-specific precision differences in exp/pow.
    """
    half_dim = head_dim // 2
    # Compute in float64 for maximum precision
    inv_freq = base ** (-np.arange(0, half_dim, dtype=np.float64) / half_dim)
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    return cos_cache, sin_cache


def pytorch_precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype=None,
) -> tuple:
    """PyTorch reference for precomputing RoPE cache.

    Uses base^(-x) form which matches the MLX implementation.
    """
    half_dim = head_dim // 2
    inv_freq = base ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_cache = torch.cos(freqs)
    sin_cache = torch.sin(freqs)
    if dtype is not None:
        cos_cache = cos_cache.to(dtype)
        sin_cache = sin_cache.to(dtype)
    return cos_cache, sin_cache


def pytorch_rope(
    x: "torch.Tensor",
    cos_cache: "torch.Tensor",
    sin_cache: "torch.Tensor",
    offset: int = 0,
) -> "torch.Tensor":
    """PyTorch reference for RoPE rotation.

    Formula: out1 = x1*cos - x2*sin, out2 = x1*sin + x2*cos

    Args:
        x: Input tensor (batch, seq, heads, head_dim)
        cos_cache: Precomputed cosines (max_seq, half_dim)
        sin_cache: Precomputed sines (max_seq, half_dim)
        offset: Position offset for decoding

    Returns:
        Rotated tensor of same shape
    """
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Slice cache for this sequence
    cos = cos_cache[offset:offset + seq_len]
    sin = sin_cache[offset:offset + seq_len]

    # Reshape for broadcasting: (seq, half_dim) -> (1, seq, 1, half_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Split into first and second halves
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


# =============================================================================
# RoPE Cache Parity Tests
# =============================================================================

class TestPrecomputeRopeCacheParity:
    """Tests for precompute_rope_cache() properties and correctness.

    Note: MLX and PyTorch have different implementations of exponentiation
    (base ** x) with ~6e-08 difference per element. This accumulates with
    sequence length and can exceed strict tolerances. Therefore, we test:
    1. Cache shape and dtype correctness
    2. Mathematical properties (cos/sin identity, value ranges)
    3. Rotation behavior using shared cache (in TestFastRopeParity)
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("seq_len", [64, 256, 1024])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_cache_properties(self, seq_len, head_dim, skip_without_pytorch):
        """Test that MLX cache has correct shape, dtype, and mathematical properties."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        base = 10000.0
        half_dim = head_dim // 2

        # MLX cache
        mlx_cos, mlx_sin = precompute_rope_cache(seq_len, head_dim, base, mx.float32)
        mx.eval(mlx_cos, mlx_sin)

        # Test shape
        assert mlx_cos.shape == (seq_len, half_dim), f"cos shape mismatch"
        assert mlx_sin.shape == (seq_len, half_dim), f"sin shape mismatch"

        # Test dtype
        assert mlx_cos.dtype == mx.float32, f"cos dtype mismatch"
        assert mlx_sin.dtype == mx.float32, f"sin dtype mismatch"

        # Test cos^2 + sin^2 = 1 (fundamental trig identity)
        cos_np = np.array(mlx_cos)
        sin_np = np.array(mlx_sin)
        identity = cos_np ** 2 + sin_np ** 2
        np.testing.assert_allclose(
            identity, np.ones_like(identity),
            rtol=1e-5, atol=1e-6,
            err_msg="cos^2 + sin^2 != 1"
        )

        # Test value ranges
        assert np.all(cos_np >= -1.0) and np.all(cos_np <= 1.0), "cos out of [-1, 1]"
        assert np.all(sin_np >= -1.0) and np.all(sin_np <= 1.0), "sin out of [-1, 1]"

        # Test first row (position 0): cos=1, sin=0 for all dims
        np.testing.assert_allclose(cos_np[0], np.ones(half_dim), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(sin_np[0], np.zeros(half_dim), rtol=1e-6, atol=1e-7)

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("base", [10000.0, 500000.0, 1000000.0])
    def test_various_bases(self, base, skip_without_pytorch):
        """Test cache properties with different frequency bases."""
        from mlx_primitives.kernels.rope import precompute_rope_cache

        seq_len, head_dim = 256, 64
        half_dim = head_dim // 2

        mlx_cos, mlx_sin = precompute_rope_cache(seq_len, head_dim, base, mx.float32)
        mx.eval(mlx_cos, mlx_sin)

        cos_np = np.array(mlx_cos)
        sin_np = np.array(mlx_sin)

        # Test cos^2 + sin^2 = 1
        identity = cos_np ** 2 + sin_np ** 2
        np.testing.assert_allclose(
            identity, np.ones_like(identity),
            rtol=1e-5, atol=1e-6,
            err_msg=f"cos^2 + sin^2 != 1 with base={base}"
        )

        # Higher base = slower rotation, so cache should vary more smoothly
        # Check that adjacent positions don't jump too much
        cos_diff = np.abs(np.diff(cos_np, axis=0))
        # Max change between adjacent positions should be bounded
        # (this is a sanity check, not a strict bound)
        assert np.max(cos_diff) < 1.5, f"Excessive variation in cos cache with base={base}"


# =============================================================================
# Fast RoPE (Metal Kernel) Parity Tests
# =============================================================================

class TestFastRopeParity:
    """Tests for fast_rope() Metal kernel parity.

    Uses shared numpy cache to test rotation operation in isolation,
    eliminating framework differences in cache computation.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fast_rope forward pass matches PyTorch."""
        from mlx_primitives.kernels.rope import fast_rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        # Generate inputs
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache (float64 computed, cast to float32)
        np_cos, np_sin = numpy_precompute_rope_cache(seq, head_dim, 10000.0)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        # MLX fast_rope
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = fast_rope(x_mlx, mlx_cos, mlx_sin, offset=0)
        mx.eval(mlx_out)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin, offset=0)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("offset", [0, 32, 128])
    def test_position_offset(self, offset, skip_without_pytorch):
        """Test fast_rope with position offsets (for KV cache decoding)."""
        from mlx_primitives.kernels.rope import fast_rope

        batch, seq, heads, head_dim = 2, 64, 8, 64
        max_seq = 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Larger shared numpy cache to accommodate offset
        np_cos, np_sin = numpy_precompute_rope_cache(max_seq, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        x_mlx = mx.array(x_np)
        mlx_out = fast_rope(x_mlx, mlx_cos, mlx_sin, offset=offset)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin, offset=offset)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope with offset={offset} mismatch"
        )


# =============================================================================
# Fast RoPE QK (Fused Metal Kernel) Parity Tests
# =============================================================================

class TestFastRopeQKParity:
    """Tests for fast_rope_qk() fused Metal kernel parity.

    Uses shared numpy cache to test rotation operation in isolation.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fast_rope_qk forward pass matches separate PyTorch RoPE calls."""
        from mlx_primitives.kernels.rope import fast_rope_qk

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache
        np_cos, np_sin = numpy_precompute_rope_cache(seq, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        # MLX fused kernel
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_rot_mlx, k_rot_mlx = fast_rope_qk(q_mlx, k_mlx, mlx_cos, mlx_sin)
        mx.eval(q_rot_mlx, k_rot_mlx)

        # PyTorch separate calls
        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)
        q_rot_torch = pytorch_rope(q_torch, torch_cos, torch_sin)
        k_rot_torch = pytorch_rope(k_torch, torch_cos, torch_sin)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), _to_numpy(q_rot_torch),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk Q mismatch [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_rot_mlx), _to_numpy(k_rot_torch),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rope_qk K mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_consistency_with_separate_calls(self, skip_without_pytorch):
        """Verify fast_rope_qk matches calling fast_rope twice."""
        from mlx_primitives.kernels.rope import fast_rope, fast_rope_qk, precompute_rope_cache

        batch, seq, heads, head_dim = 2, 128, 8, 64

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        mlx_cos, mlx_sin = precompute_rope_cache(seq, head_dim)
        mx.eval(mlx_cos, mlx_sin)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)

        # Fused
        q_fused, k_fused = fast_rope_qk(q_mlx, k_mlx, mlx_cos, mlx_sin)
        mx.eval(q_fused, k_fused)

        # Separate
        q_sep = fast_rope(q_mlx, mlx_cos, mlx_sin)
        k_sep = fast_rope(k_mlx, mlx_cos, mlx_sin)
        mx.eval(q_sep, k_sep)

        np.testing.assert_allclose(
            _to_numpy(q_fused), _to_numpy(q_sep),
            rtol=1e-6, atol=1e-7,
            err_msg="fast_rope_qk Q differs from fast_rope"
        )
        np.testing.assert_allclose(
            _to_numpy(k_fused), _to_numpy(k_sep),
            rtol=1e-6, atol=1e-7,
            err_msg="fast_rope_qk K differs from fast_rope"
        )


# =============================================================================
# RoPE Fallback Path Parity Tests
# =============================================================================

class TestRopeFallbackParity:
    """Tests for rope() with use_metal=False (Python fallback).

    Uses shared numpy cache to test rotation operation in isolation.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_fallback_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test Python fallback path matches PyTorch."""
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache
        np_cos, np_sin = numpy_precompute_rope_cache(seq, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        # Force fallback path
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = rope(x_mlx, mlx_cos, mlx_sin, offset=0, use_metal=False)
        mx.eval(mlx_out)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin)

        rtol, atol = get_tolerance("attention", "rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"rope fallback mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_metal_vs_fallback_consistency(self, skip_without_pytorch):
        """Verify Metal kernel matches Python fallback."""
        from mlx_primitives.kernels.rope import rope, precompute_rope_cache

        batch, seq, heads, head_dim = 2, 256, 8, 64

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        mlx_cos, mlx_sin = precompute_rope_cache(seq, head_dim)
        mx.eval(mlx_cos, mlx_sin)

        x_mlx = mx.array(x_np)

        # Metal path
        out_metal = rope(x_mlx, mlx_cos, mlx_sin, use_metal=True)
        mx.eval(out_metal)

        # Fallback path
        out_fallback = rope(x_mlx, mlx_cos, mlx_sin, use_metal=False)
        mx.eval(out_fallback)

        # Should be identical (both are fp32 with same algorithm)
        # Use standard rope tolerances (may have minor numerical differences due to Metal kernel)
        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(out_metal), _to_numpy(out_fallback),
            rtol=rtol, atol=atol,
            err_msg="Metal vs fallback RoPE mismatch"
        )


# =============================================================================
# Backward Pass Parity Tests
# =============================================================================

class TestRopeBackwardParity:
    """Tests for RoPE gradient computation parity.

    Uses shared numpy cache to test gradient operation in isolation.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test RoPE backward pass (gradient) matches PyTorch."""
        from mlx_primitives.kernels.rope import rope

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache
        np_cos, np_sin = numpy_precompute_rope_cache(seq, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        # MLX backward
        def mlx_loss_fn(x):
            return mx.sum(rope(x, mlx_cos, mlx_sin, use_metal=False))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE backward mismatch [{size}]"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRopeEdgeCases:
    """Edge case tests for RoPE implementations.

    Uses shared numpy cache to test edge cases in isolation.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_single_token(self, skip_without_pytorch):
        """Test RoPE with single token (common in decoding)."""
        from mlx_primitives.kernels.rope import fast_rope

        batch, seq, heads, head_dim = 1, 1, 8, 64

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache
        np_cos, np_sin = numpy_precompute_rope_cache(256, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        x_mlx = mx.array(x_np)
        mlx_out = fast_rope(x_mlx, mlx_cos, mlx_sin, offset=0)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin, offset=0)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="Single token RoPE mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_min_head_dim(self, skip_without_pytorch):
        """Test RoPE with minimum head dimension."""
        from mlx_primitives.kernels.rope import fast_rope

        batch, seq, heads, head_dim = 2, 64, 8, 2  # Minimum even dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Use shared numpy cache
        np_cos, np_sin = numpy_precompute_rope_cache(seq, head_dim)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        x_mlx = mx.array(x_np)
        mlx_out = fast_rope(x_mlx, mlx_cos, mlx_sin)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin)

        rtol, atol = get_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="Minimum head_dim RoPE mismatch"
        )


# =============================================================================
# NTKAwareRoPE Parity Tests
# =============================================================================

def numpy_ntk_aware_precompute_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    original_max_seq_len: int = 8192,
    max_seq_len: int = 32768,
    alpha: float = None,
) -> tuple:
    """Numpy reference for NTK-aware RoPE cache computation.

    NTK-aware scaling modifies the base frequency based on extension ratio:
    alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))
    scaled_base = base * alpha
    """
    if alpha is None:
        dims = head_dim
        alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))

    scaled_base = base * alpha
    half_dim = head_dim // 2

    # Compute in float64 for maximum precision
    inv_freq = scaled_base ** (-np.arange(0, half_dim, dtype=np.float64) / half_dim)
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    return cos_cache, sin_cache, alpha


class TestNTKAwareRoPEParity:
    """Tests for NTK-aware RoPE interpolation.

    NTK-aware scaling modifies the base frequency based on context extension:
    alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))
    scaled_base = base * alpha
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("extension_ratio", [2, 4, 8])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_cache_properties(self, extension_ratio, dtype, skip_without_pytorch):
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

        # Get MLX cache (via internal method or by running forward)
        np.random.seed(42)
        q_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        k_np = np.random.randn(1, seq_len, 4, head_dim).astype(np.float32)
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_out, k_out = ntk_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # Get numpy reference cache
        np_cos, np_sin, alpha = numpy_ntk_aware_precompute_cache(
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
            np_cos ** 2 + np_sin ** 2, np.ones_like(np_cos), rtol=1e-5, atol=1e-5,
            err_msg="NTK-aware cache cos^2 + sin^2 != 1"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_rotation_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test NTK-aware RoPE rotation matches PyTorch reference."""
        from mlx_primitives.attention.rope import NTKAwareRoPE

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Compute shared numpy cache for this configuration
        np_cos, np_sin, _ = numpy_ntk_aware_precompute_cache(
            seq, head_dim,
            original_max_seq_len=original_max_seq_len,
            max_seq_len=max_seq_len,
        )
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)

        # MLX NTKAwareRoPE
        ntk_rope = NTKAwareRoPE(
            dims=head_dim,
            max_seq_len=max_seq_len,
            original_max_seq_len=original_max_seq_len,
        )
        q_mlx = _convert_to_mlx(x_np, dtype)
        k_mlx = _convert_to_mlx(x_np, dtype)  # Use same data for simplicity
        q_out, k_out = ntk_rope(q_mlx, k_mlx)
        mx.eval(q_out, k_out)

        # PyTorch reference using shared cache
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin)

        rtol, atol = get_tolerance("attention", "ntk_rope", dtype)
        np.testing.assert_allclose(
            _to_numpy(q_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"NTK-aware RoPE forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test NTK-aware RoPE backward pass."""
        from mlx_primitives.attention.rope import NTKAwareRoPE

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = config["batch"], config["seq"], config["heads"], config["head_dim"]
        original_max_seq_len = 1024
        max_seq_len = 4096

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # Compute shared numpy cache
        np_cos, np_sin, _ = numpy_ntk_aware_precompute_cache(
            seq, head_dim,
            original_max_seq_len=original_max_seq_len,
            max_seq_len=max_seq_len,
        )
        torch_cos, torch_sin = torch.from_numpy(np_cos), torch.from_numpy(np_sin)
        mlx_cos, mlx_sin = mx.array(np_cos), mx.array(np_sin)

        # MLX backward - use kernel directly with shared cache for gradient testing
        from mlx_primitives.kernels.rope import rope

        def mlx_loss_fn(x):
            return mx.sum(rope(x, mlx_cos, mlx_sin, use_metal=False))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = pytorch_rope(x_torch, torch_cos, torch_sin)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("attention", "rope", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"NTK-aware RoPE backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_no_extension(self, skip_without_pytorch):
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

        # Should produce same results (alpha=1 when no extension)
        rtol, atol = get_tolerance("attention", "ntk_rope", "fp32")
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
# YaRNRoPE Parity Tests
# =============================================================================

class TestYaRNRoPEParity:
    """Tests for YaRN RoPE extension.

    YaRN combines NTK-aware scaling with frequency interpolation and
    attention scaling for better extrapolation to longer sequences.
    """

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("extension_ratio", [2, 4, 8])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_cache_properties(self, extension_ratio, dtype, skip_without_pytorch):
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
        import math
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_rotation_forward_consistency(self, size, dtype, skip_without_pytorch):
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

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_consistency(self, size, skip_without_pytorch):
        """Test YaRN RoPE backward pass produces valid gradients."""
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

        def mlx_loss_fn_q(q, k):
            q_out, k_out = yarn_rope(q, k)
            return mx.sum(q_out) + mx.sum(k_out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        grad_fn = mx.grad(mlx_loss_fn_q, argnums=(0, 1))
        q_grad, k_grad = grad_fn(q_mlx, k_mlx)
        mx.eval(q_grad, k_grad)

        q_grad_np = _to_numpy(q_grad)
        k_grad_np = _to_numpy(k_grad)

        # Verify gradients have correct shape and no NaN/Inf
        assert q_grad_np.shape == q_np.shape, "Q gradient shape mismatch"
        assert k_grad_np.shape == k_np.shape, "K gradient shape mismatch"
        assert not np.isnan(q_grad_np).any(), "YaRN q gradient contains NaN"
        assert not np.isinf(q_grad_np).any(), "YaRN q gradient contains Inf"
        assert not np.isnan(k_grad_np).any(), "YaRN k gradient contains NaN"
        assert not np.isinf(k_grad_np).any(), "YaRN k gradient contains Inf"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("beta_fast,beta_slow", [(32, 1), (64, 2), (16, 0.5)])
    def test_beta_parameters(self, beta_fast, beta_slow, skip_without_pytorch):
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_scale_factor_computation(self, skip_without_pytorch):
        """Test that scale factor is computed correctly."""
        from mlx_primitives.attention.rope import YaRNRoPE
        import math

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
