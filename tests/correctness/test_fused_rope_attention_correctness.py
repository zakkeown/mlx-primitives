"""Correctness tests for fused RoPE + Flash Attention.

These tests compare the fused implementation against separate RoPE + attention
to verify numerical correctness.
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.kernels import (
    fused_rope_attention,
    fast_fused_rope_attention,
    fast_fused_rope_attention_tiled,
    FusedRoPEFlashAttention,
    precompute_rope_cache,
    rope,
)
from mlx_primitives.attention.flash import flash_attention_forward


# ============================================================================
# Reference Implementations
# ============================================================================


def reference_rope_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    rope_base: float = 10000.0,
    q_offset: int = 0,
    kv_offset: int = 0,
    causal: bool = False,
) -> mx.array:
    """Reference implementation: separate RoPE + attention.

    This is the "slow" path that applies RoPE separately then runs attention.
    The fused kernel should produce identical results.
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Compute max position needed
    max_pos = max(seq_q + q_offset, seq_kv + kv_offset)

    # Precompute RoPE cache
    cos_cache, sin_cache = precompute_rope_cache(max_pos, head_dim, base=rope_base, dtype=q.dtype)

    # Apply RoPE to Q and K separately
    q_rot = rope(q, cos_cache, sin_cache, q_offset)
    k_rot = rope(k, cos_cache, sin_cache, kv_offset)

    # Run flash attention
    return flash_attention_forward(
        q_rot, k_rot, v, scale,
        block_size_q=64,
        block_size_kv=64,
        causal=causal,
    )


# ============================================================================
# Correctness Tests
# ============================================================================


class TestFusedRoPEAttentionCorrectness:
    """Test that fused kernel matches reference implementation."""

    @pytest.mark.parametrize("seq_len,head_dim,num_heads", [
        (16, 32, 4),
        (32, 64, 4),
        (64, 64, 8),
        (128, 64, 12),
    ])
    def test_matches_reference(self, seq_len: int, head_dim: int, num_heads: int):
        """Fused output matches separate RoPE + attention."""
        mx.random.seed(42)
        batch_size = 2

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(head_dim)

        # Reference: separate RoPE + attention
        ref_out = reference_rope_attention(q, k, v, scale, causal=True)
        mx.eval(ref_out)

        # Fused kernel
        fused_out = fused_rope_attention(q, k, v, scale, causal=True)
        mx.eval(fused_out)

        # Compare
        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Fused differs from reference by {max_diff}"

    def test_causal_vs_non_causal(self):
        """Test both causal and non-causal modes."""
        mx.random.seed(42)

        q = mx.random.normal((2, 32, 4, 64))
        k = mx.random.normal((2, 32, 4, 64))
        v = mx.random.normal((2, 32, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        # Non-causal
        ref_non_causal = reference_rope_attention(q, k, v, scale, causal=False)
        fused_non_causal = fused_rope_attention(q, k, v, scale, causal=False)
        mx.eval(ref_non_causal, fused_non_causal)

        max_diff = float(mx.max(mx.abs(ref_non_causal - fused_non_causal)))
        assert max_diff < 1e-3, f"Non-causal differs by {max_diff}"

        # Causal
        ref_causal = reference_rope_attention(q, k, v, scale, causal=True)
        fused_causal = fused_rope_attention(q, k, v, scale, causal=True)
        mx.eval(ref_causal, fused_causal)

        max_diff = float(mx.max(mx.abs(ref_causal - fused_causal)))
        assert max_diff < 1e-3, f"Causal differs by {max_diff}"

        # Causal and non-causal should be different
        diff = float(mx.max(mx.abs(fused_causal - fused_non_causal)))
        assert diff > 0.01, "Causal and non-causal should produce different outputs"


class TestPositionOffsets:
    """Test position offset handling for KV cache scenarios."""

    def test_q_offset(self):
        """Test Q position offset for incremental decoding."""
        mx.random.seed(42)

        # Simulate: new query at position 31 attending to full KV cache
        q = mx.random.normal((1, 1, 4, 64))    # 1 new token
        k = mx.random.normal((1, 32, 4, 64))   # 32 tokens
        v = mx.random.normal((1, 32, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        # Q is at position 31 (0-indexed), KV from position 0
        ref_out = reference_rope_attention(
            q, k, v, scale, q_offset=31, kv_offset=0, causal=True
        )
        fused_out = fused_rope_attention(
            q, k, v, scale, q_offset=31, kv_offset=0, causal=True
        )
        mx.eval(ref_out, fused_out)

        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Q offset test differs by {max_diff}"

    def test_both_offsets(self):
        """Test both Q and KV offsets."""
        mx.random.seed(42)

        q = mx.random.normal((1, 8, 4, 64))
        k = mx.random.normal((1, 8, 4, 64))
        v = mx.random.normal((1, 8, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        # Q starts at 16, KV starts at 16
        ref_out = reference_rope_attention(
            q, k, v, scale, q_offset=16, kv_offset=16, causal=True
        )
        fused_out = fused_rope_attention(
            q, k, v, scale, q_offset=16, kv_offset=16, causal=True
        )
        mx.eval(ref_out, fused_out)

        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Both offsets test differs by {max_diff}"


class TestDifferentSeqLengths:
    """Test with different Q and KV sequence lengths."""

    def test_q_shorter_than_kv(self):
        """Q shorter than KV (incremental decode scenario)."""
        mx.random.seed(42)

        q = mx.random.normal((2, 1, 4, 64))    # 1 query
        k = mx.random.normal((2, 64, 4, 64))   # 64 keys
        v = mx.random.normal((2, 64, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        ref_out = reference_rope_attention(
            q, k, v, scale, q_offset=63, kv_offset=0, causal=True
        )
        fused_out = fused_rope_attention(
            q, k, v, scale, q_offset=63, kv_offset=0, causal=True
        )
        mx.eval(ref_out, fused_out)

        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Q shorter differs by {max_diff}"

    def test_q_longer_than_kv(self):
        """Q longer than KV (cross-attention scenario)."""
        mx.random.seed(42)

        q = mx.random.normal((2, 32, 4, 64))   # 32 queries
        k = mx.random.normal((2, 16, 4, 64))   # 16 keys
        v = mx.random.normal((2, 16, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        # Non-causal for cross-attention
        ref_out = reference_rope_attention(
            q, k, v, scale, causal=False
        )
        fused_out = fused_rope_attention(
            q, k, v, scale, causal=False
        )
        mx.eval(ref_out, fused_out)

        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Q longer differs by {max_diff}"


class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_large_values(self):
        """Test with large input values."""
        mx.random.seed(42)

        q = mx.random.normal((2, 32, 4, 64)) * 10
        k = mx.random.normal((2, 32, 4, 64)) * 10
        v = mx.random.normal((2, 32, 4, 64)) * 10
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        out = fused_rope_attention(q, k, v, scale, causal=True)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "Output contains NaN or Inf"

    def test_small_values(self):
        """Test with small input values."""
        mx.random.seed(42)

        q = mx.random.normal((2, 32, 4, 64)) * 0.01
        k = mx.random.normal((2, 32, 4, 64)) * 0.01
        v = mx.random.normal((2, 32, 4, 64)) * 0.01
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        out = fused_rope_attention(q, k, v, scale, causal=True)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "Output contains NaN or Inf"


class TestRoPEBase:
    """Test different RoPE base values."""

    @pytest.mark.parametrize("rope_base", [10000.0, 1000000.0, 100.0])
    def test_different_bases(self, rope_base: float):
        """Test with different RoPE base values."""
        mx.random.seed(42)

        q = mx.random.normal((2, 32, 4, 64))
        k = mx.random.normal((2, 32, 4, 64))
        v = mx.random.normal((2, 32, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)

        ref_out = reference_rope_attention(
            q, k, v, scale, rope_base=rope_base, causal=True
        )
        fused_out = fused_rope_attention(
            q, k, v, scale, rope_base=rope_base, causal=True
        )
        mx.eval(ref_out, fused_out)

        max_diff = float(mx.max(mx.abs(ref_out - fused_out)))
        assert max_diff < 1e-3, f"Base {rope_base} differs by {max_diff}"


class TestFusedRoPEFlashAttentionModule:
    """Test the FusedRoPEFlashAttention nn.Module."""

    def test_output_shape(self):
        """Module produces correct output shape."""
        mx.random.seed(42)
        dims = 256
        num_heads = 4
        seq_len = 64

        attn = FusedRoPEFlashAttention(
            dims=dims, num_heads=num_heads, causal=True
        )

        x = mx.random.normal((2, seq_len, dims))
        mx.eval(x)

        out, cache = attn(x)
        mx.eval(out)

        assert out.shape == (2, seq_len, dims), f"Wrong shape: {out.shape}"

    def test_cache_shape(self):
        """KV cache has correct shape."""
        mx.random.seed(42)
        dims = 128
        num_heads = 4
        head_dim = dims // num_heads

        attn = FusedRoPEFlashAttention(
            dims=dims, num_heads=num_heads, causal=True
        )

        x = mx.random.normal((2, 32, dims))
        mx.eval(x)

        out, cache = attn(x)
        mx.eval(cache[0], cache[1])

        k_cache, v_cache = cache
        assert k_cache.shape == (2, 32, num_heads, head_dim)
        assert v_cache.shape == (2, 32, num_heads, head_dim)

    def test_incremental_decoding(self):
        """Test incremental decoding with KV cache."""
        mx.random.seed(42)
        dims = 128
        num_heads = 4

        attn = FusedRoPEFlashAttention(
            dims=dims, num_heads=num_heads, causal=True
        )

        # Initial prompt
        prompt = mx.random.normal((1, 32, dims))
        mx.eval(prompt)

        out1, cache = attn(prompt)
        mx.eval(out1, cache[0], cache[1])

        # Incremental decode
        new_token = mx.random.normal((1, 1, dims))
        mx.eval(new_token)

        out2, new_cache = attn(new_token, cache=cache, offset=32)
        mx.eval(out2)

        assert out2.shape == (1, 1, dims)
        assert new_cache[0].shape[1] == 33  # 32 + 1

    def test_self_attention_vs_cross_attention(self):
        """Test self-attention and cross-attention modes."""
        mx.random.seed(42)
        dims = 128
        num_heads = 4

        attn = FusedRoPEFlashAttention(
            dims=dims, num_heads=num_heads, causal=False
        )

        q = mx.random.normal((2, 32, dims))
        kv = mx.random.normal((2, 64, dims))
        mx.eval(q, kv)

        # Cross-attention: Q from one source, K/V from another
        out, _ = attn(q, keys=kv, values=kv)
        mx.eval(out)

        assert out.shape == (2, 32, dims)


class TestTiledKernel:
    """Test the tiled kernel with proper Flash Attention tiling."""

    @pytest.mark.parametrize("seq_len", [64, 256, 512, 1024])
    def test_tiled_matches_reference(self, seq_len: int):
        """Tiled kernel output matches reference."""
        mx.random.seed(42)
        batch_size = 2
        num_heads = 4
        head_dim = 64

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(head_dim)

        # Precompute cache
        cos, sin = precompute_rope_cache(seq_len * 2, head_dim)
        mx.eval(cos, sin)

        # Reference
        ref_out = reference_rope_attention(q, k, v, scale, causal=True)
        mx.eval(ref_out)

        # Tiled kernel (block_size is fixed at 32)
        tiled_out = fast_fused_rope_attention_tiled(
            q, k, v, cos, sin, scale, causal=True, block_size=24
        )
        mx.eval(tiled_out)

        max_diff = float(mx.max(mx.abs(ref_out - tiled_out)))
        assert max_diff < 1e-3, f"Tiled differs by {max_diff}"

    def test_tiled_long_sequence(self):
        """Tiled kernel handles long sequences."""
        mx.random.seed(42)

        q = mx.random.normal((1, 2048, 4, 64))
        k = mx.random.normal((1, 2048, 4, 64))
        v = mx.random.normal((1, 2048, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)
        cos, sin = precompute_rope_cache(4096, 64)
        mx.eval(cos, sin)

        ref_out = reference_rope_attention(q, k, v, scale, causal=True)
        mx.eval(ref_out)

        tiled_out = fast_fused_rope_attention_tiled(
            q, k, v, cos, sin, scale, causal=True, block_size=24
        )
        mx.eval(tiled_out)

        max_diff = float(mx.max(mx.abs(ref_out - tiled_out)))
        assert max_diff < 1e-3, f"Long seq differs by {max_diff}"

    def test_tiled_non_causal(self):
        """Tiled kernel works in non-causal mode."""
        mx.random.seed(42)

        q = mx.random.normal((2, 512, 4, 64))
        k = mx.random.normal((2, 512, 4, 64))
        v = mx.random.normal((2, 512, 4, 64))
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(64)
        cos, sin = precompute_rope_cache(1024, 64)
        mx.eval(cos, sin)

        ref_out = reference_rope_attention(q, k, v, scale, causal=False)
        mx.eval(ref_out)

        tiled_out = fast_fused_rope_attention_tiled(
            q, k, v, cos, sin, scale, causal=False, block_size=24
        )
        mx.eval(tiled_out)

        max_diff = float(mx.max(mx.abs(ref_out - tiled_out)))
        assert max_diff < 1e-3, f"Non-causal differs by {max_diff}"


