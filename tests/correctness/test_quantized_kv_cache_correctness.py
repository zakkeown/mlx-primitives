"""Correctness tests for QuantizedKVCache.

Tests verify:
1. Quantization/dequantization roundtrip accuracy
2. Cache storage and retrieval correctness
3. Attention output quality vs unquantized baseline
4. Memory compression verification
5. Numerical stability with edge cases
"""

import pytest
import mlx.core as mx

from mlx_primitives.attention.quantized_kv_cache import (
    QuantizedKVCache,
    QuantizedKVCacheAttention,
    quantize_kv_for_cache,
    dequantize_kv_from_cache,
)
from mlx_primitives.attention import flash_attention_forward


# =============================================================================
# Reference Implementations
# =============================================================================


def naive_int8_symmetric_quantize(x: mx.array):
    """Reference implementation: per-head, per-token symmetric INT8 quantization.

    Args:
        x: Input tensor (batch, seq, heads, dim) or (seq, heads, dim).

    Returns:
        Tuple of (quantized, scales).
    """
    # Compute per-head, per-token abs max
    abs_max = mx.max(mx.abs(x), axis=-1, keepdims=True)
    scale = abs_max / 127.0
    # Avoid division by zero
    scale = mx.maximum(scale, 1e-10)
    # Quantize
    x_quant = mx.round(x / scale)
    x_quant = mx.clip(x_quant, -127, 127)
    return x_quant.astype(mx.int8), scale.astype(mx.float32)


def naive_int8_symmetric_dequantize(x_quant: mx.array, scale: mx.array):
    """Reference implementation: symmetric INT8 dequantization."""
    return x_quant.astype(mx.float32) * scale


def naive_kv_cache_attention(q, k_list, v_list, scale, causal=True):
    """Reference attention using list-based KV cache (no quantization)."""
    # Concatenate all K/V
    k = mx.concatenate(k_list, axis=1) if len(k_list) > 1 else k_list[0]
    v = mx.concatenate(v_list, axis=1) if len(v_list) > 1 else v_list[0]

    # Standard attention
    return flash_attention_forward(q, k, v, scale=scale, causal=causal)


# =============================================================================
# Test Classes
# =============================================================================


class TestQuantizeDequantizeRoundtrip:
    """Test quantization/dequantization accuracy."""

    def test_symmetric_roundtrip_small_values(self):
        """Symmetric quantization roundtrip with normal-range values."""
        mx.random.seed(42)
        k = mx.random.normal((2, 64, 8, 128))
        v = mx.random.normal((2, 64, 8, 128))
        mx.eval(k, v)

        # Quantize and dequantize
        k_quant, v_quant, k_scale, v_scale, k_zero, v_zero = quantize_kv_for_cache(
            k, v, symmetric=True
        )
        mx.eval(k_quant, v_quant, k_scale, v_scale)

        k_dequant, v_dequant = dequantize_kv_from_cache(
            k_quant, v_quant, k_scale, v_scale, k_zero, v_zero
        )
        mx.eval(k_dequant, v_dequant)

        # Compute relative error for K
        max_abs_k = float(mx.max(mx.abs(k)))
        max_error_k = float(mx.max(mx.abs(k - k_dequant)))
        relative_error_k = max_error_k / max_abs_k

        # INT8 symmetric should have < 2% relative error for normal values
        assert relative_error_k < 0.02, f"K relative error {relative_error_k:.4f} exceeds 2%"

        # Compute relative error for V
        max_abs_v = float(mx.max(mx.abs(v)))
        max_error_v = float(mx.max(mx.abs(v - v_dequant)))
        relative_error_v = max_error_v / max_abs_v

        assert relative_error_v < 0.02, f"V relative error {relative_error_v:.4f} exceeds 2%"

    def test_symmetric_roundtrip_large_values(self):
        """Symmetric quantization roundtrip with large values."""
        mx.random.seed(42)
        k = mx.random.normal((2, 64, 8, 128)) * 100.0
        v = mx.random.normal((2, 64, 8, 128)) * 100.0
        mx.eval(k, v)

        k_quant, v_quant, k_scale, v_scale, k_zero, v_zero = quantize_kv_for_cache(
            k, v, symmetric=True
        )
        mx.eval(k_quant, v_quant, k_scale, v_scale)

        k_dequant, v_dequant = dequantize_kv_from_cache(
            k_quant, v_quant, k_scale, v_scale, k_zero, v_zero
        )
        mx.eval(k_dequant, v_dequant)

        max_abs_k = float(mx.max(mx.abs(k)))
        max_error_k = float(mx.max(mx.abs(k - k_dequant)))
        relative_error_k = max_error_k / max_abs_k

        assert relative_error_k < 0.02, f"K relative error {relative_error_k:.4f} exceeds 2%"

    def test_vs_naive_implementation(self):
        """Compare our quantization to naive reference."""
        mx.random.seed(42)
        k = mx.random.normal((2, 32, 8, 64))
        v = mx.random.normal((2, 32, 8, 64))
        mx.eval(k, v)

        # Our implementation
        k_quant_ours, _, k_scale_ours, _, _, _ = quantize_kv_for_cache(k, v, symmetric=True)
        mx.eval(k_quant_ours, k_scale_ours)

        # Naive reference
        k_quant_ref, k_scale_ref = naive_int8_symmetric_quantize(k)
        mx.eval(k_quant_ref, k_scale_ref)

        # Dequantize both
        k_dequant_ours, _ = dequantize_kv_from_cache(
            k_quant_ours, k_quant_ours, k_scale_ours, k_scale_ours
        )
        k_dequant_ref = naive_int8_symmetric_dequantize(k_quant_ref, k_scale_ref)
        mx.eval(k_dequant_ours, k_dequant_ref)

        # Both should produce similar results
        max_diff = float(mx.max(mx.abs(k_dequant_ours - k_dequant_ref)))
        assert max_diff < 1e-5, f"Implementations differ by {max_diff}"


class TestQuantizedKVCacheStorage:
    """Test cache storage and retrieval correctness."""

    def test_single_token_append_retrieval(self):
        """Single token append and retrieve matches input."""
        mx.random.seed(42)
        cache = QuantizedKVCache(num_heads=8, head_dim=64)

        # Add single token
        k = mx.random.normal((1, 1, 8, 64))
        v = mx.random.normal((1, 1, 8, 64))
        mx.eval(k, v)

        cache.update(k, v)

        # Retrieve
        k_out, v_out = cache.get_dequantized()
        mx.eval(k_out, v_out)

        # Check shapes
        assert k_out.shape == (1, 1, 8, 64)
        assert v_out.shape == (1, 1, 8, 64)

        # Check accuracy (allow for quantization error)
        k_error = float(mx.max(mx.abs(k - k_out)))
        v_error = float(mx.max(mx.abs(v - v_out)))

        # Relative to input magnitude
        k_rel_error = k_error / float(mx.max(mx.abs(k)))
        v_rel_error = v_error / float(mx.max(mx.abs(v)))

        assert k_rel_error < 0.02, f"K relative error {k_rel_error:.4f} > 2%"
        assert v_rel_error < 0.02, f"V relative error {v_rel_error:.4f} > 2%"

    def test_multi_token_append_retrieval(self):
        """Multiple token append and retrieve matches concatenated input."""
        mx.random.seed(42)
        cache = QuantizedKVCache(num_heads=8, head_dim=64)

        # Add tokens in batches
        k1 = mx.random.normal((1, 10, 8, 64))
        v1 = mx.random.normal((1, 10, 8, 64))
        k2 = mx.random.normal((1, 5, 8, 64))
        v2 = mx.random.normal((1, 5, 8, 64))
        mx.eval(k1, v1, k2, v2)

        cache.update(k1, v1)
        cache.update(k2, v2)

        # Expected concatenation
        k_expected = mx.concatenate([k1, k2], axis=1)
        v_expected = mx.concatenate([v1, v2], axis=1)

        # Retrieve
        k_out, v_out = cache.get_dequantized()
        mx.eval(k_out, v_out)

        # Check shapes
        assert k_out.shape == (1, 15, 8, 64)
        assert v_out.shape == (1, 15, 8, 64)

        # Check accuracy
        k_error = float(mx.max(mx.abs(k_expected - k_out)))
        v_error = float(mx.max(mx.abs(v_expected - v_out)))

        k_rel_error = k_error / float(mx.max(mx.abs(k_expected)))
        v_rel_error = v_error / float(mx.max(mx.abs(v_expected)))

        assert k_rel_error < 0.02, f"K relative error {k_rel_error:.4f} > 2%"
        assert v_rel_error < 0.02, f"V relative error {v_rel_error:.4f} > 2%"

    def test_sequence_length_tracking(self):
        """Cache correctly tracks sequence length."""
        cache = QuantizedKVCache(num_heads=4, head_dim=32)

        assert cache.seq_len == 0

        k1 = mx.random.normal((1, 10, 4, 32))
        v1 = mx.random.normal((1, 10, 4, 32))
        cache.update(k1, v1)
        assert cache.seq_len == 10

        k2 = mx.random.normal((1, 5, 4, 32))
        v2 = mx.random.normal((1, 5, 4, 32))
        cache.update(k2, v2)
        assert cache.seq_len == 15

        cache.reset()
        assert cache.seq_len == 0


class TestAttentionOutputQuality:
    """Test attention output quality with quantized cache."""

    def test_attention_vs_unquantized(self):
        """Quantized cache attention output close to unquantized baseline."""
        mx.random.seed(42)

        batch, seq_len, num_heads, head_dim = 1, 64, 8, 64
        dims = num_heads * head_dim

        # Create attention modules
        quant_attn = QuantizedKVCacheAttention(
            dims=dims,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            causal=True,
        )

        # Input sequence
        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        # Quantized attention output
        quant_output = quant_attn(x)
        mx.eval(quant_output)

        # Unquantized reference: compute Q, K, V manually
        q = quant_attn.q_proj(x).reshape(batch, seq_len, num_heads, head_dim)
        k = quant_attn.k_proj(x).reshape(batch, seq_len, num_heads, head_dim)
        v = quant_attn.v_proj(x).reshape(batch, seq_len, num_heads, head_dim)
        mx.eval(q, k, v)

        scale = 1.0 / (head_dim ** 0.5)
        ref_attn_out = flash_attention_forward(q, k, v, scale=scale, causal=True)
        ref_output = quant_attn.out_proj(ref_attn_out.reshape(batch, seq_len, dims))
        mx.eval(ref_output)

        # Compare outputs
        max_diff = float(mx.max(mx.abs(quant_output - ref_output)))
        relative_diff = max_diff / float(mx.max(mx.abs(ref_output)))

        # Allow 5% relative difference due to quantization
        assert relative_diff < 0.05, (
            f"Quantized attention differs by {relative_diff:.2%} from unquantized"
        )

    def test_incremental_generation(self):
        """Test incremental token generation with quantized cache."""
        mx.random.seed(42)

        batch, num_heads, head_dim = 1, 8, 64
        dims = num_heads * head_dim

        attn = QuantizedKVCacheAttention(
            dims=dims,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            causal=True,
        )

        # Prefill
        prefill = mx.random.normal((batch, 32, dims))
        mx.eval(prefill)
        out_prefill = attn(prefill)
        mx.eval(out_prefill)

        assert attn._cache.seq_len == 32

        # Generate tokens one at a time
        for i in range(5):
            token = mx.random.normal((batch, 1, dims))
            mx.eval(token)
            out_token = attn(token)
            mx.eval(out_token)

        assert attn._cache.seq_len == 37

        # Output should be valid
        assert mx.all(mx.isfinite(out_token))


class TestMemoryCompression:
    """Test memory compression properties."""

    def test_compression_ratio(self):
        """Verify ~4x compression ratio."""
        cache = QuantizedKVCache(num_heads=32, head_dim=128)

        # Add significant amount of data
        k = mx.random.normal((1, 512, 32, 128))
        v = mx.random.normal((1, 512, 32, 128))
        mx.eval(k, v)

        cache.update(k, v)

        stats = cache.get_memory_stats()

        # Float32 equivalent should be much larger than quantized
        compression = stats["compression_ratio"]

        # Should be close to 4x (allowing some overhead for scales)
        assert compression > 3.5, f"Compression ratio {compression:.2f} < 3.5x"
        assert compression < 4.5, f"Compression ratio {compression:.2f} > 4.5x"


class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_large_input_values(self):
        """Quantization handles large input values."""
        mx.random.seed(42)
        cache = QuantizedKVCache(num_heads=8, head_dim=64)

        # Large values (x1000)
        k = mx.random.normal((1, 32, 8, 64)) * 1000.0
        v = mx.random.normal((1, 32, 8, 64)) * 1000.0
        mx.eval(k, v)

        cache.update(k, v)
        k_out, v_out = cache.get_dequantized()
        mx.eval(k_out, v_out)

        # Should be finite
        assert mx.all(mx.isfinite(k_out)), "K output contains NaN/Inf"
        assert mx.all(mx.isfinite(v_out)), "V output contains NaN/Inf"

        # Should maintain reasonable accuracy
        k_rel_error = float(mx.max(mx.abs(k - k_out))) / float(mx.max(mx.abs(k)))
        assert k_rel_error < 0.02, f"K relative error {k_rel_error:.4f} > 2%"

    def test_small_input_values(self):
        """Quantization handles small input values."""
        mx.random.seed(42)
        cache = QuantizedKVCache(num_heads=8, head_dim=64)

        # Small values (x0.001)
        k = mx.random.normal((1, 32, 8, 64)) * 0.001
        v = mx.random.normal((1, 32, 8, 64)) * 0.001
        mx.eval(k, v)

        cache.update(k, v)
        k_out, v_out = cache.get_dequantized()
        mx.eval(k_out, v_out)

        # Should be finite
        assert mx.all(mx.isfinite(k_out)), "K output contains NaN/Inf"
        assert mx.all(mx.isfinite(v_out)), "V output contains NaN/Inf"

        # Small values will have larger relative error due to quantization granularity
        # but absolute error should still be small
        k_abs_error = float(mx.max(mx.abs(k - k_out)))
        assert k_abs_error < 0.001, f"K absolute error {k_abs_error} > 0.001"

    def test_mixed_magnitude_values(self):
        """Quantization handles mixed magnitude values per head."""
        mx.random.seed(42)
        cache = QuantizedKVCache(num_heads=8, head_dim=64)

        k = mx.random.normal((1, 32, 8, 64))
        v = mx.random.normal((1, 32, 8, 64))

        # Scale different heads differently
        scales = mx.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 0.001, 50.0])
        scales = scales.reshape(1, 1, 8, 1)
        k = k * scales
        v = v * scales
        mx.eval(k, v)

        cache.update(k, v)
        k_out, v_out = cache.get_dequantized()
        mx.eval(k_out, v_out)

        # Per-head quantization should handle this well
        # Check each head's relative error
        for head_idx in range(8):
            k_head = k[:, :, head_idx, :]
            k_head_out = k_out[:, :, head_idx, :]

            head_max = float(mx.max(mx.abs(k_head)))
            if head_max > 1e-6:  # Skip near-zero heads
                head_error = float(mx.max(mx.abs(k_head - k_head_out)))
                head_rel_error = head_error / head_max
                assert head_rel_error < 0.02, (
                    f"Head {head_idx} relative error {head_rel_error:.4f} > 2%"
                )
