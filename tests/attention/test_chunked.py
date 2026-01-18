"""Tests for chunked cross-attention implementation."""

import math

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.attention.chunked import (
    chunked_cross_attention,
    ChunkedCrossAttention,
    estimate_memory_savings,
    _reference_cross_attention,
)


def numpy_cross_attention(q, k, v, scale, causal):
    """NumPy reference implementation for testing."""
    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)

    batch, seq_q, heads, dim = q_np.shape
    _, seq_kv, _, _ = k_np.shape

    # Transpose to (batch, heads, seq, dim)
    q_t = np.transpose(q_np, (0, 2, 1, 3))
    k_t = np.transpose(k_np, (0, 2, 1, 3))
    v_t = np.transpose(v_np, (0, 2, 1, 3))

    # Attention scores: (batch, heads, seq_q, seq_kv)
    scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) * scale

    if causal:
        # Causal mask: kv_pos <= q_pos
        q_pos = np.arange(seq_q)[:, None]
        kv_pos = np.arange(seq_kv)[None, :]
        mask = kv_pos <= q_pos
        scores = np.where(mask, scores, -1e9)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Output
    output = np.matmul(weights, v_t)
    return np.transpose(output, (0, 2, 1, 3))


class TestChunkedAttentionCorrectness:
    """Verify chunked attention matches standard attention."""

    def test_vs_numpy_small_bidirectional(self) -> None:
        """Small sequences, no causal masking."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 2, 16, 32, 4, 32

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        chunked_out = chunked_cross_attention(
            q, k, v, chunk_size=8, scale=scale, causal=False, use_metal=False
        )
        numpy_out = numpy_cross_attention(q, k, v, scale, causal=False)

        np.testing.assert_allclose(
            np.array(chunked_out), numpy_out, rtol=1e-4, atol=1e-5
        )

    def test_vs_numpy_small_causal(self) -> None:
        """Small sequences with causal masking."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 32, 4, 32

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        chunked_out = chunked_cross_attention(
            q, k, v, chunk_size=8, scale=scale, causal=True, use_metal=False
        )
        numpy_out = numpy_cross_attention(q, k, v, scale, causal=True)

        np.testing.assert_allclose(
            np.array(chunked_out), numpy_out, rtol=1e-4, atol=1e-5
        )

    def test_vs_reference_medium_sequence(self) -> None:
        """Medium KV sequence length."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 2, 32, 256, 8, 64

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))

        chunked_out = chunked_cross_attention(
            q, k, v, chunk_size=64, causal=False, use_metal=False
        )
        ref_out = _reference_cross_attention(q, k, v, 1.0 / math.sqrt(dim), False)

        np.testing.assert_allclose(
            np.array(chunked_out), np.array(ref_out), rtol=1e-3, atol=1e-4
        )

    def test_different_chunk_sizes(self) -> None:
        """Different chunk sizes should produce same result."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 2, 16, 128, 4, 32

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        ref_out = numpy_cross_attention(q, k, v, scale, causal=False)

        for chunk_size in [8, 16, 32, 64, 128]:
            chunked_out = chunked_cross_attention(
                q, k, v, chunk_size=chunk_size, scale=scale, causal=False, use_metal=False
            )
            np.testing.assert_allclose(
                np.array(chunked_out), ref_out, rtol=1e-3, atol=1e-4,
                err_msg=f"Failed for chunk_size={chunk_size}"
            )

    def test_chunk_size_larger_than_kv(self) -> None:
        """When chunk_size >= seq_kv, should match standard attention."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 2, 16, 64, 4, 32

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))

        # Chunk size larger than KV length
        chunked_out = chunked_cross_attention(
            q, k, v, chunk_size=128, causal=False, use_metal=False
        )
        ref_out = _reference_cross_attention(q, k, v, 1.0 / math.sqrt(dim), False)

        np.testing.assert_allclose(
            np.array(chunked_out), np.array(ref_out), rtol=1e-5, atol=1e-6
        )


class TestChunkedAttentionNumericalStability:
    """Test numerical stability with various inputs."""

    def test_large_values(self) -> None:
        """Large input values should not cause overflow."""
        mx.random.seed(42)
        q = mx.random.normal((1, 16, 4, 64)) * 10
        k = mx.random.normal((1, 64, 4, 64)) * 10
        v = mx.random.normal((1, 64, 4, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=16, use_metal=False)

        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()

    def test_many_chunks(self) -> None:
        """Many small chunks should accumulate correctly."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 1, 8, 256, 2, 32

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        # Very small chunks = many accumulation steps
        chunked_out = chunked_cross_attention(
            q, k, v, chunk_size=8, scale=scale, causal=False, use_metal=False
        )
        ref_out = numpy_cross_attention(q, k, v, scale, causal=False)

        np.testing.assert_allclose(
            np.array(chunked_out), ref_out, rtol=1e-3, atol=1e-4
        )

    def test_chunk_boundary_stability(self) -> None:
        """Verify numerical stability at chunk boundaries."""
        mx.random.seed(42)
        batch, seq_q, seq_kv, heads, dim = 1, 16, 64, 4, 32

        # Create adversarial case: high scores in one chunk, low in another
        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, seq_kv, heads, dim))
        v = mx.random.normal((batch, seq_kv, heads, dim))

        # Boost scores for first chunk only using concatenation
        k_first_boosted = k[:, :16, :, :] * 5
        k_boosted = mx.concatenate([k_first_boosted, k[:, 16:, :, :]], axis=1)

        out = chunked_cross_attention(
            q, k_boosted, v, chunk_size=16, causal=False, use_metal=False
        )

        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()


class TestChunkedAttentionShapes:
    """Test shape handling and edge cases."""

    def test_different_q_kv_lengths(self) -> None:
        """Q and KV can have different sequence lengths."""
        mx.random.seed(42)

        # Short queries, long context
        q = mx.random.normal((2, 32, 4, 64))
        k = mx.random.normal((2, 512, 4, 64))
        v = mx.random.normal((2, 512, 4, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=64, use_metal=False)

        assert out.shape == q.shape

    def test_single_query_long_context(self) -> None:
        """Single query attending to long context."""
        mx.random.seed(42)

        q = mx.random.normal((1, 1, 4, 64))
        k = mx.random.normal((1, 256, 4, 64))
        v = mx.random.normal((1, 256, 4, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=32, use_metal=False)
        ref = _reference_cross_attention(q, k, v, 1.0 / math.sqrt(64), False)

        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-4, atol=1e-5
        )

    def test_batch_size_one(self) -> None:
        """Batch size of 1."""
        mx.random.seed(42)

        q = mx.random.normal((1, 32, 8, 64))
        k = mx.random.normal((1, 128, 8, 64))
        v = mx.random.normal((1, 128, 8, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=32, use_metal=False)

        assert out.shape == q.shape


class TestChunkedCrossAttentionModule:
    """Test the ChunkedCrossAttention module class."""

    def test_module_basic(self) -> None:
        """Basic module usage."""
        mx.random.seed(42)
        attn = ChunkedCrossAttention(num_heads=8, head_dim=64, chunk_size=64)

        q = mx.random.normal((2, 32, 8, 64))
        k = mx.random.normal((2, 256, 8, 64))
        v = mx.random.normal((2, 256, 8, 64))

        out = attn(q, k, v)

        assert out.shape == q.shape

    def test_module_causal(self) -> None:
        """Module with causal masking."""
        mx.random.seed(42)
        attn = ChunkedCrossAttention(num_heads=4, head_dim=32, chunk_size=32, causal=True)

        q = mx.random.normal((2, 64, 4, 32))
        k = mx.random.normal((2, 64, 4, 32))
        v = mx.random.normal((2, 64, 4, 32))

        out = attn(q, k, v)
        ref = numpy_cross_attention(q, k, v, 1.0 / math.sqrt(32), causal=True)

        np.testing.assert_allclose(
            np.array(out), ref, rtol=1e-3, atol=1e-4
        )


class TestMemorySavingsEstimate:
    """Test the memory savings estimation function."""

    def test_basic_estimate(self) -> None:
        """Basic memory estimate calculation."""
        standard, chunked, ratio = estimate_memory_savings(
            seq_q=512,
            seq_kv=100000,
            num_heads=8,
            head_dim=64,
            chunk_size=4096,
        )

        # Standard should be much larger
        assert standard > chunked
        # Ratio should reflect the savings
        assert ratio > 10  # At least 10x savings on attention matrix

    def test_small_sequence_ratio(self) -> None:
        """Small sequences have modest savings."""
        standard, chunked, ratio = estimate_memory_savings(
            seq_q=64,
            seq_kv=256,
            num_heads=8,
            head_dim=64,
            chunk_size=64,
        )

        # Still should have some savings
        assert ratio >= 1


class TestChunkedAttentionGradients:
    """Test gradient computation through chunked attention."""

    def test_gradient_flows(self) -> None:
        """Verify gradients flow through chunked attention."""
        mx.random.seed(42)
        q = mx.random.normal((2, 16, 4, 32))
        k = mx.random.normal((2, 64, 4, 32))
        v = mx.random.normal((2, 64, 4, 32))

        def loss_fn(q, k, v):
            out = chunked_cross_attention(q, k, v, chunk_size=16, use_metal=False)
            return mx.sum(out ** 2)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        # Verify gradients exist and have correct shapes
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

        # Verify gradients are not all zeros
        assert mx.any(grads[0] != 0).item()
        assert mx.any(grads[1] != 0).item()
        assert mx.any(grads[2] != 0).item()


@pytest.mark.benchmark
class TestChunkedAttentionBenchmarks:
    """Benchmark tests."""

    def test_medium_kv_sequence(self) -> None:
        """Medium KV sequence benchmark."""
        mx.random.seed(42)
        q = mx.random.normal((2, 64, 8, 64))
        k = mx.random.normal((2, 2048, 8, 64))
        v = mx.random.normal((2, 2048, 8, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=256)
        mx.eval(out)

        assert out.shape == q.shape

    @pytest.mark.slow
    def test_long_kv_sequence(self) -> None:
        """Long KV sequence benchmark."""
        mx.random.seed(42)
        q = mx.random.normal((1, 128, 8, 64))
        k = mx.random.normal((1, 16384, 8, 64))
        v = mx.random.normal((1, 16384, 8, 64))

        out = chunked_cross_attention(q, k, v, chunk_size=1024)
        mx.eval(out)

        assert out.shape == q.shape
