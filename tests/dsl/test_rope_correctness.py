"""Comprehensive correctness tests for DSL RoPE kernels.

RoPE (Rotary Position Embedding) is CRITICAL for LLM inference correctness.
These tests verify:
1. Basic rotation math
2. Position encoding accuracy
3. Cache precomputation
4. Multiple RoPE variants (standard, NeoX, inline)
5. Fused Q/K processing for GQA
6. Incremental decoding with start_pos
"""

import pytest
import numpy as np
from numpy.typing import NDArray


def _mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


# =============================================================================
# NumPy Reference Implementations
# =============================================================================


def numpy_precompute_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[NDArray, NDArray]:
    """Reference RoPE cache computation in NumPy.

    Args:
        max_seq_len: Maximum sequence length
        head_dim: Head dimension (must be even)
        base: RoPE base frequency (default 10000)

    Returns:
        (cos_cache, sin_cache) of shape (max_seq_len, head_dim // 2)
    """
    half_dim = head_dim // 2

    # Compute frequencies: freq[i] = 1 / (base ^ (2i / dim))
    freqs = 1.0 / (base ** (np.arange(0, half_dim) * 2.0 / head_dim))

    # Positions
    positions = np.arange(max_seq_len)

    # Compute angles: positions x freqs -> (max_seq_len, half_dim)
    angles = np.outer(positions, freqs)

    cos_cache = np.cos(angles).astype(np.float32)
    sin_cache = np.sin(angles).astype(np.float32)

    return cos_cache, sin_cache


def numpy_rope_forward(
    x: NDArray,
    cos_cache: NDArray,
    sin_cache: NDArray,
    start_pos: int = 0,
) -> NDArray:
    """Reference RoPE forward pass in NumPy.

    Standard RoPE pairs adjacent elements: (x[2i], x[2i+1])

    Args:
        x: Input tensor (batch, seq, heads, head_dim)
        cos_cache: Precomputed cos values (max_seq, head_dim // 2)
        sin_cache: Precomputed sin values (max_seq, head_dim // 2)
        start_pos: Starting position for incremental decoding

    Returns:
        Output tensor (batch, seq, heads, head_dim)
    """
    batch, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    output = np.zeros_like(x)

    for b in range(batch):
        for s in range(seq_len):
            pos = start_pos + s
            for h in range(num_heads):
                for i in range(half_dim):
                    x0 = x[b, s, h, 2 * i]
                    x1 = x[b, s, h, 2 * i + 1]

                    cos_val = cos_cache[pos, i]
                    sin_val = sin_cache[pos, i]

                    # Apply rotation
                    output[b, s, h, 2 * i] = x0 * cos_val - x1 * sin_val
                    output[b, s, h, 2 * i + 1] = x0 * sin_val + x1 * cos_val

    return output


def numpy_rope_neox(
    x: NDArray,
    cos_cache: NDArray,
    sin_cache: NDArray,
    start_pos: int = 0,
) -> NDArray:
    """Reference NeoX-style RoPE in NumPy.

    NeoX pairs first half with second half: x[i] with x[i + half_dim]

    Args:
        x: Input tensor (batch, seq, heads, head_dim)
        cos_cache: Precomputed cos values (max_seq, head_dim // 2)
        sin_cache: Precomputed sin values (max_seq, head_dim // 2)
        start_pos: Starting position

    Returns:
        Output tensor (batch, seq, heads, head_dim)
    """
    batch, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    output = np.zeros_like(x)

    for b in range(batch):
        for s in range(seq_len):
            pos = start_pos + s
            for h in range(num_heads):
                for i in range(half_dim):
                    x0 = x[b, s, h, i]
                    x1 = x[b, s, h, half_dim + i]

                    cos_val = cos_cache[pos, i]
                    sin_val = sin_cache[pos, i]

                    output[b, s, h, i] = x0 * cos_val - x1 * sin_val
                    output[b, s, h, half_dim + i] = x0 * sin_val + x1 * cos_val

    return output


# =============================================================================
# Test Classes
# =============================================================================


class TestPrecomputeRopeCache:
    """Test cache precomputation correctness."""

    def test_cache_shape(self) -> None:
        """Verify cache dimensions."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        max_seq, head_dim, base = 128, 64, 10000.0
        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, base)

        assert cos_cache.shape == (max_seq, head_dim // 2)
        assert sin_cache.shape == (max_seq, head_dim // 2)

    def test_cache_dtype(self) -> None:
        """Verify cache dtype is float32."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(64, 32, 10000.0)

        assert cos_cache.dtype == np.float32
        assert sin_cache.dtype == np.float32

    def test_cache_values_in_valid_range(self) -> None:
        """Verify cos/sin values are in [-1, 1]."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(1024, 128, 10000.0)

        assert np.all(cos_cache >= -1.0) and np.all(cos_cache <= 1.0)
        assert np.all(sin_cache >= -1.0) and np.all(sin_cache <= 1.0)

    def test_position_zero_identity(self) -> None:
        """At position 0, cos=1.0 and sin=0.0 (identity rotation)."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(10, 64, 10000.0)

        # At position 0, angle is 0 for all frequencies
        # cos(0) = 1.0, sin(0) = 0.0
        np.testing.assert_allclose(cos_cache[0], np.ones(32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sin_cache[0], np.zeros(32), rtol=1e-6, atol=1e-6)

    def test_cache_matches_numpy_reference(self) -> None:
        """Verify cache matches NumPy reference implementation."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        max_seq, head_dim, base = 128, 64, 10000.0

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, base)
        ref_cos, ref_sin = numpy_precompute_rope_cache(max_seq, head_dim, base)

        np.testing.assert_allclose(cos_cache, ref_cos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sin_cache, ref_sin, rtol=1e-6, atol=1e-6)

    def test_frequency_progression(self) -> None:
        """High frequency dims should change faster with position."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(100, 64, 10000.0)

        # Low frequency dim (last dimension) changes slowly
        low_freq_diff = np.abs(cos_cache[1, -1] - cos_cache[0, -1])
        # High frequency dim (first dimension) changes quickly
        high_freq_diff = np.abs(cos_cache[1, 0] - cos_cache[0, 0])

        assert high_freq_diff > low_freq_diff, "High freq should change faster"

    @pytest.mark.parametrize("base", [10000.0, 500000.0, 1000000.0])
    def test_cache_different_bases(self, base: float) -> None:
        """Test different RoPE bases (extended context variants)."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(1024, 128, base)

        # Values should be valid regardless of base
        assert np.all(np.isfinite(cos_cache))
        assert np.all(np.isfinite(sin_cache))
        assert np.all(cos_cache >= -1.0) and np.all(cos_cache <= 1.0)
        assert np.all(sin_cache >= -1.0) and np.all(sin_cache <= 1.0)

    @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
    def test_cache_different_head_dims(self, head_dim: int) -> None:
        """Test common head dimensions from various models."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        cos_cache, sin_cache = precompute_rope_cache(64, head_dim, 10000.0)

        assert cos_cache.shape == (64, head_dim // 2)
        assert sin_cache.shape == (64, head_dim // 2)


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
class TestRopeForwardCorrectness:
    """Test rope_forward kernel correctness."""

    def test_rope_forward_basic(self) -> None:
        """Basic correctness test against NumPy reference."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads, head_dim = 2, 16, 8, 64
        max_seq = 32

        # Precompute cache
        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        # Create input
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        # Run kernel
        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        # NumPy reference
        expected = numpy_rope_forward(x_np, cos_cache, sin_cache, start_pos=0)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("start_pos", [0, 1, 10, 100])
    def test_rope_forward_incremental_decoding(self, start_pos: int) -> None:
        """Test start_pos for incremental decoding (KV cache scenario)."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads, head_dim = 1, 1, 8, 64  # Single token
        max_seq = 1024

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42 + start_pos)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=start_pos,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        expected = numpy_rope_forward(x_np, cos_cache, sin_cache, start_pos=start_pos)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_rope_forward_various_head_dims(self, head_dim: int) -> None:
        """Test common head dimensions from various models."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads = 2, 16, 8
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        expected = numpy_rope_forward(x_np, cos_cache, sin_cache)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
class TestRopeInlineCorrectness:
    """Test rope_inline kernel (computes cos/sin on the fly)."""

    @pytest.mark.xfail(reason="Metal cos/sin type ambiguity in DSL codegen")
    def test_rope_inline_basic(self) -> None:
        """Inline RoPE should match precomputed cache version."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import (
            rope_forward, rope_inline, precompute_rope_cache
        )

        batch, seq, heads, head_dim = 2, 16, 8, 64
        base = 10000.0
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, base)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out1 = mx.zeros_like(x)
        out2 = mx.zeros_like(x)

        # Precomputed version
        precomputed_result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out1,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(precomputed_result, list):
            precomputed_result = precomputed_result[0]
        mx.eval(precomputed_result)

        # Inline version
        inline_result = rope_inline(
            x, out2,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            base=base, start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(inline_result, list):
            inline_result = inline_result[0]
        mx.eval(inline_result)

        np.testing.assert_allclose(
            np.array(inline_result),
            np.array(precomputed_result),
            rtol=1e-4, atol=1e-4
        )

    @pytest.mark.xfail(reason="Metal cos/sin type ambiguity in DSL codegen")
    @pytest.mark.parametrize("base", [10000.0, 500000.0])
    def test_rope_inline_different_bases(self, base: float) -> None:
        """Test inline with different RoPE bases."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_inline, precompute_rope_cache

        batch, seq, heads, head_dim = 1, 8, 4, 32

        cos_cache, sin_cache = precompute_rope_cache(32, head_dim, base)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_inline(
            x, out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            base=base, start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        expected = numpy_rope_forward(x_np, cos_cache, sin_cache)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-3, atol=1e-3
        )


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
class TestRopeQKFusedCorrectness:
    """Test fused Q/K RoPE for GQA models."""

    @pytest.mark.xfail(reason="Kernel output handling for multiple outputs needs adjustment")
    def test_rope_qk_fused_basic(self) -> None:
        """Test fused Q/K processing matches separate processing."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import (
            rope_forward, rope_qk_fused, precompute_rope_cache
        )

        batch, seq, q_heads, kv_heads, head_dim = 2, 16, 32, 8, 64
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, kv_heads, head_dim).astype(np.float32)

        q = mx.array(q_np)
        k = mx.array(k_np)
        q_out = mx.zeros_like(q)
        k_out = mx.zeros_like(k)

        # Fused kernel
        result = rope_qk_fused(
            q, k, mx.array(cos_cache), mx.array(sin_cache), q_out, k_out,
            batch_size=batch, seq_len=seq,
            num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, max(q_heads, kv_heads)),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            q_result = result[0]
            k_result = result[1]
        else:
            q_result = q_out
            k_result = k_out
        mx.eval(q_result, k_result)

        # Separate processing (reference)
        q_expected = numpy_rope_forward(q_np, cos_cache, sin_cache)
        k_expected = numpy_rope_forward(k_np, cos_cache, sin_cache)

        np.testing.assert_allclose(
            np.array(q_result), q_expected, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.array(k_result), k_expected, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.xfail(reason="Kernel output handling for multiple outputs needs adjustment")
    @pytest.mark.parametrize("q_heads,kv_heads", [(32, 8), (32, 4), (16, 2), (8, 1)])
    def test_rope_qk_fused_gqa_ratios(self, q_heads: int, kv_heads: int) -> None:
        """Test different Q/KV head ratios (GQA configurations)."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import (
            rope_qk_fused, precompute_rope_cache
        )

        batch, seq, head_dim = 1, 8, 64
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, q_heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, kv_heads, head_dim).astype(np.float32)

        q = mx.array(q_np)
        k = mx.array(k_np)
        q_out = mx.zeros_like(q)
        k_out = mx.zeros_like(k)

        result = rope_qk_fused(
            q, k, mx.array(cos_cache), mx.array(sin_cache), q_out, k_out,
            batch_size=batch, seq_len=seq,
            num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, max(q_heads, kv_heads)),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            q_result = result[0]
            k_result = result[1]
        else:
            q_result = q_out
            k_result = k_out
        mx.eval(q_result, k_result)

        q_expected = numpy_rope_forward(q_np, cos_cache, sin_cache)
        k_expected = numpy_rope_forward(k_np, cos_cache, sin_cache)

        np.testing.assert_allclose(
            np.array(q_result), q_expected, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.array(k_result), k_expected, rtol=1e-4, atol=1e-4
        )


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
class TestRopeNeoXCorrectness:
    """Test GPT-NeoX style RoPE (different pairing pattern)."""

    def test_rope_neox_basic(self) -> None:
        """Basic NeoX RoPE correctness."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_neox, precompute_rope_cache

        batch, seq, heads, head_dim = 2, 16, 8, 64
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_neox(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        expected = numpy_rope_neox(x_np, cos_cache, sin_cache)

        np.testing.assert_allclose(
            np.array(result), expected, rtol=1e-4, atol=1e-4
        )

    def test_rope_neox_vs_standard_different(self) -> None:
        """Verify NeoX gives different results than standard RoPE."""
        from mlx_primitives.dsl.examples.rope import precompute_rope_cache

        batch, seq, heads, head_dim = 2, 16, 8, 64
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        standard = numpy_rope_forward(x_np, cos_cache, sin_cache)
        neox = numpy_rope_neox(x_np, cos_cache, sin_cache)

        # They should NOT be equal (different pairing patterns)
        assert not np.allclose(standard, neox, rtol=1e-2)


@pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
class TestRoPEEdgeCases:
    """Edge cases and numerical stability for RoPE."""

    def test_rope_position_zero_near_identity(self) -> None:
        """Position 0 should apply near-identity rotation (cos=1, sin=0)."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads, head_dim = 1, 1, 1, 64

        cos_cache, sin_cache = precompute_rope_cache(1, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        # At position 0, rotation is identity (cos=1, sin=0)
        np.testing.assert_allclose(
            np.array(result), x_np, rtol=1e-4, atol=1e-4
        )

    def test_rope_preserves_magnitude(self) -> None:
        """Rotation should preserve vector magnitude for each pair."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads, head_dim = 2, 16, 8, 64
        max_seq = 32

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        result_np = np.array(result)

        # Magnitude of each pair should be preserved
        for i in range(head_dim // 2):
            input_mag = np.sqrt(x_np[..., 2*i]**2 + x_np[..., 2*i+1]**2)
            output_mag = np.sqrt(result_np[..., 2*i]**2 + result_np[..., 2*i+1]**2)
            np.testing.assert_allclose(output_mag, input_mag, rtol=1e-4, atol=1e-4)

    def test_rope_no_nan_or_inf(self) -> None:
        """RoPE should not produce NaN or Inf."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.rope import rope_forward, precompute_rope_cache

        batch, seq, heads, head_dim = 1, 128, 8, 64
        max_seq = 256

        cos_cache, sin_cache = precompute_rope_cache(max_seq, head_dim, 10000.0)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        x = mx.array(x_np)
        out = mx.zeros_like(x)

        result = rope_forward(
            x, mx.array(cos_cache), mx.array(sin_cache), out,
            batch_size=batch, seq_len=seq, num_heads=heads, head_dim=head_dim,
            start_pos=0,
            grid=(batch, seq, heads),
            threadgroup=(min(32, head_dim // 2),),
        )
        if isinstance(result, list):
            result = result[0]
        mx.eval(result)

        result_np = np.array(result)
        assert not np.any(np.isnan(result_np))
        assert not np.any(np.isinf(result_np))


class TestRoPEAnalytical:
    """Analytical tests with known input-output pairs."""

    def test_single_pair_rotation(self) -> None:
        """Test rotation of a single pair with known cos/sin values."""
        # x = [1, 0], cos=cos(pi/4), sin=sin(pi/4) -> rotate by 45 degrees
        x = np.array([1.0, 0.0])
        cos_val = np.cos(np.pi / 4)  # ~0.707
        sin_val = np.sin(np.pi / 4)  # ~0.707

        # Standard RoPE rotation:
        # y0 = x0 * cos - x1 * sin = 1 * 0.707 - 0 * 0.707 = 0.707
        # y1 = x0 * sin + x1 * cos = 1 * 0.707 + 0 * 0.707 = 0.707
        expected = np.array([cos_val, sin_val])

        y0 = x[0] * cos_val - x[1] * sin_val
        y1 = x[0] * sin_val + x[1] * cos_val
        result = np.array([y0, y1])

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_rotation_by_90_degrees(self) -> None:
        """Test 90 degree rotation."""
        # x = [1, 0], rotate by 90 degrees -> [0, 1]
        x = np.array([1.0, 0.0])
        cos_val = np.cos(np.pi / 2)  # 0
        sin_val = np.sin(np.pi / 2)  # 1

        y0 = x[0] * cos_val - x[1] * sin_val
        y1 = x[0] * sin_val + x[1] * cos_val
        result = np.array([y0, y1])

        expected = np.array([0.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_rotation_by_180_degrees(self) -> None:
        """Test 180 degree rotation."""
        # x = [1, 0], rotate by 180 degrees -> [-1, 0]
        x = np.array([1.0, 0.0])
        cos_val = np.cos(np.pi)  # -1
        sin_val = np.sin(np.pi)  # 0

        y0 = x[0] * cos_val - x[1] * sin_val
        y1 = x[0] * sin_val + x[1] * cos_val
        result = np.array([y0, y1])

        expected = np.array([-1.0, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
