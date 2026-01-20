"""Tests for Flash Attention implementation."""

import math

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.attention.flash import (
    flash_attention,
    FlashAttention,
    _reference_flash_attention,
    get_optimal_flash_config,
)
from mlx_primitives.attention._online_softmax import (
    online_softmax_merge,
    compute_chunk_attention,
)
from mlx_primitives.constants import ATTENTION_MASK_VALUE


def numpy_attention(q, k, v, scale, causal):
    """NumPy reference implementation for testing.

    Uses ATTENTION_MASK_VALUE from constants for consistency with
    production code paths.
    """
    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)

    batch, seq, heads, dim = q_np.shape

    # Transpose to (batch, heads, seq, dim)
    q_t = np.transpose(q_np, (0, 2, 1, 3))
    k_t = np.transpose(k_np, (0, 2, 1, 3))
    v_t = np.transpose(v_np, (0, 2, 1, 3))

    # Attention scores
    scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) * scale

    if causal:
        mask = np.tril(np.ones((seq, seq)))
        # Use the same mask value as production code for consistency
        scores = np.where(mask == 1, scores, ATTENTION_MASK_VALUE)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Output
    output = np.matmul(weights, v_t)
    return np.transpose(output, (0, 2, 1, 3))


class TestFlashAttentionCorrectness:
    """Verify Flash Attention matches standard attention."""

    def test_vs_numpy_small_causal(self) -> None:
        """Small sequence with causal masking."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 16, 4, 32

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        flash_out = flash_attention(q, k, v, scale=scale, causal=True, use_metal=False)
        numpy_out = numpy_attention(q, k, v, scale, causal=True)

        np.testing.assert_allclose(
            np.array(flash_out), numpy_out, rtol=1e-4, atol=1e-5
        )

    def test_vs_numpy_small_bidirectional(self) -> None:
        """Small sequence without causal masking."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 16, 4, 32

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))
        scale = 1.0 / math.sqrt(dim)

        flash_out = flash_attention(q, k, v, scale=scale, causal=False, use_metal=False)
        numpy_out = numpy_attention(q, k, v, scale, causal=False)

        np.testing.assert_allclose(
            np.array(flash_out), numpy_out, rtol=1e-4, atol=1e-5
        )

    def test_vs_reference_medium_sequence(self) -> None:
        """Medium sequence length."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 128, 8, 64

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        flash_out = flash_attention(q, k, v, causal=True, use_metal=False)
        ref_out = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(flash_out), np.array(ref_out), rtol=1e-3, atol=1e-4
        )

    def test_different_block_sizes(self) -> None:
        """Test various block sizes produce same result."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 64, 4, 32

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        ref_out = _reference_flash_attention(q, k, v, None, True)

        for block_q, block_kv in [(8, 8), (16, 16), (32, 32), (64, 64)]:
            flash_out = flash_attention(
                q, k, v, causal=True, block_q=block_q, block_kv=block_kv, use_metal=False
            )
            np.testing.assert_allclose(
                np.array(flash_out), np.array(ref_out), rtol=1e-3, atol=1e-4,
                err_msg=f"Failed for block_q={block_q}, block_kv={block_kv}"
            )

    def test_head_dim_64(self) -> None:
        """Test with head_dim=64 (common in many models)."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 64, 8, 64

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        flash_out = flash_attention(q, k, v, causal=True, use_metal=False)
        ref_out = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(flash_out), np.array(ref_out), rtol=1e-3, atol=1e-4
        )

    def test_head_dim_128(self) -> None:
        """Test with head_dim=128 (used in larger models)."""
        mx.random.seed(42)
        batch, seq, heads, dim = 2, 32, 4, 128

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        flash_out = flash_attention(q, k, v, causal=True, block_q=16, block_kv=16, use_metal=False)
        ref_out = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(flash_out), np.array(ref_out), rtol=1e-3, atol=1e-4
        )


class TestFlashAttentionNumericalStability:
    """Test numerical stability edge cases."""

    def test_large_values(self) -> None:
        """Large input values should not cause overflow."""
        mx.random.seed(42)
        q = mx.random.normal((1, 32, 4, 64)) * 10
        k = mx.random.normal((1, 32, 4, 64)) * 10
        v = mx.random.normal((1, 32, 4, 64))

        out = flash_attention(q, k, v, causal=True, use_metal=False)

        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()

    def test_small_values(self) -> None:
        """Small input values should not cause underflow."""
        mx.random.seed(42)
        q = mx.random.normal((1, 32, 4, 64)) * 0.001
        k = mx.random.normal((1, 32, 4, 64)) * 0.001
        v = mx.random.normal((1, 32, 4, 64))

        out = flash_attention(q, k, v, causal=True, use_metal=False)

        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()

    def test_extreme_score_differences(self) -> None:
        """Handle cases where some scores are much larger than others."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 16, 2, 32

        # Create Q and K such that one position has much higher score
        q = mx.zeros((batch, seq, heads, dim))
        k = mx.zeros((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        # Make first query strongly attend to first key
        # Use concatenation since MLX doesn't support .at[].set()
        q_first = mx.ones((batch, 1, heads, dim)) * 10
        q = mx.concatenate([q_first, q[:, 1:, :, :]], axis=1)

        k_first = mx.ones((batch, 1, heads, dim)) * 10
        k = mx.concatenate([k_first, k[:, 1:, :, :]], axis=1)

        out = flash_attention(q, k, v, causal=True, use_metal=False)

        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()


class TestFlashAttentionShapes:
    """Test shape handling and edge cases."""

    def test_single_token(self) -> None:
        """Single token sequence."""
        mx.random.seed(42)
        q = mx.random.normal((2, 1, 4, 64))
        k = mx.random.normal((2, 1, 4, 64))
        v = mx.random.normal((2, 1, 4, 64))

        out = flash_attention(q, k, v, causal=True, use_metal=False)

        assert out.shape == q.shape
        # Single token should just return normalized v
        np.testing.assert_allclose(
            np.array(out), np.array(v), rtol=1e-5, atol=1e-6
        )

    def test_batch_size_one(self) -> None:
        """Batch size of 1."""
        mx.random.seed(42)
        q = mx.random.normal((1, 64, 8, 64))
        k = mx.random.normal((1, 64, 8, 64))
        v = mx.random.normal((1, 64, 8, 64))

        out = flash_attention(q, k, v, causal=True, use_metal=False)
        ref = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-3, atol=1e-4
        )

    def test_many_heads(self) -> None:
        """Many attention heads."""
        mx.random.seed(42)
        q = mx.random.normal((2, 32, 32, 32))  # 32 heads
        k = mx.random.normal((2, 32, 32, 32))
        v = mx.random.normal((2, 32, 32, 32))

        out = flash_attention(q, k, v, causal=True, use_metal=False)
        ref = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-3, atol=1e-4
        )

    def test_sequence_not_multiple_of_block(self) -> None:
        """Sequence length not divisible by block size."""
        mx.random.seed(42)
        # 47 is prime, won't divide evenly by any block size
        q = mx.random.normal((2, 47, 4, 64))
        k = mx.random.normal((2, 47, 4, 64))
        v = mx.random.normal((2, 47, 4, 64))

        out = flash_attention(q, k, v, causal=True, block_q=16, block_kv=16, use_metal=False)
        ref = _reference_flash_attention(q, k, v, None, True)

        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-3, atol=1e-4
        )


class TestFlashAttentionModule:
    """Test the FlashAttention module class."""

    def test_module_basic(self) -> None:
        """Basic module usage."""
        mx.random.seed(42)
        attn = FlashAttention(num_heads=8, head_dim=64, causal=True)

        q = mx.random.normal((2, 64, 8, 64))
        k = mx.random.normal((2, 64, 8, 64))
        v = mx.random.normal((2, 64, 8, 64))

        out = attn(q, k, v)

        assert out.shape == q.shape

    def test_module_bidirectional(self) -> None:
        """Module with causal=False."""
        mx.random.seed(42)
        attn = FlashAttention(num_heads=4, head_dim=32, causal=False)

        q = mx.random.normal((2, 32, 4, 32))
        k = mx.random.normal((2, 32, 4, 32))
        v = mx.random.normal((2, 32, 4, 32))

        out = attn(q, k, v)
        ref = _reference_flash_attention(q, k, v, 1.0 / math.sqrt(32), False)

        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-3, atol=1e-4
        )

    def test_module_custom_blocks(self) -> None:
        """Module with custom block sizes."""
        attn = FlashAttention(num_heads=8, head_dim=128, causal=True, block_q=8, block_kv=8)

        assert attn.block_q == 8
        assert attn.block_kv == 8


class TestOptimalConfig:
    """Test the optimal configuration function."""

    def test_small_head_dim(self) -> None:
        """Small head dim should allow larger blocks."""
        block_q, block_kv = get_optimal_flash_config(seq_len=1024, head_dim=32, num_heads=8)
        assert block_q >= 16
        assert block_kv >= 16

    def test_large_head_dim(self) -> None:
        """Large head dim should use smaller blocks."""
        block_q, block_kv = get_optimal_flash_config(seq_len=1024, head_dim=128, num_heads=8)
        assert block_q <= 32
        assert block_kv <= 32

    def test_short_sequence(self) -> None:
        """Short sequence shouldn't exceed sequence length."""
        block_q, block_kv = get_optimal_flash_config(seq_len=16, head_dim=64, num_heads=8)
        assert block_q <= 16
        assert block_kv <= 16


class TestFlashAttentionGradients:
    """Test gradient computation through Flash Attention."""

    def test_gradient_flows(self) -> None:
        """Verify gradients flow through Flash Attention."""
        mx.random.seed(42)
        q = mx.random.normal((2, 32, 4, 64))
        k = mx.random.normal((2, 32, 4, 64))
        v = mx.random.normal((2, 32, 4, 64))

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, causal=True, use_metal=False)
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
class TestFlashAttentionBenchmarks:
    """Benchmark tests."""

    def test_medium_sequence(self) -> None:
        """Medium sequence benchmark."""
        mx.random.seed(42)
        q = mx.random.normal((4, 512, 8, 64))
        k = mx.random.normal((4, 512, 8, 64))
        v = mx.random.normal((4, 512, 8, 64))

        out = flash_attention(q, k, v, causal=True)
        mx.eval(out)

        assert out.shape == q.shape

    @pytest.mark.slow
    def test_long_sequence(self) -> None:
        """Long sequence benchmark."""
        mx.random.seed(42)
        q = mx.random.normal((2, 2048, 8, 64))
        k = mx.random.normal((2, 2048, 8, 64))
        v = mx.random.normal((2, 2048, 8, 64))

        out = flash_attention(q, k, v, causal=True)
        mx.eval(out)

        assert out.shape == q.shape


class TestOnlineSoftmaxMerge:
    """Test the online softmax merge operation directly.

    These tests validate that chunked attention with merging produces
    the same results as computing attention over the full sequence.
    """

    def test_two_chunk_merge_correctness(self) -> None:
        """Verify merging two chunks produces correct output."""
        mx.random.seed(42)
        batch, seq_q, heads, dim = 2, 8, 4, 32
        chunk_size = 16
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, chunk_size * 2, heads, dim))
        v = mx.random.normal((batch, chunk_size * 2, heads, dim))

        # Compute reference: full attention over all KV
        ref_out = numpy_attention(
            q,
            k,
            v,
            scale,
            causal=False
        )

        # Compute in two chunks and merge
        k_chunk1 = k[:, :chunk_size, :, :]
        v_chunk1 = v[:, :chunk_size, :, :]
        k_chunk2 = k[:, chunk_size:, :, :]
        v_chunk2 = v[:, chunk_size:, :, :]

        # Process first chunk
        out1, max1, sum1 = compute_chunk_attention(
            q, k_chunk1, v_chunk1, scale, causal=False
        )

        # Process second chunk
        out2, max2, sum2 = compute_chunk_attention(
            q, k_chunk2, v_chunk2, scale, causal=False, kv_offset=chunk_size
        )

        # Merge the two chunks
        merged_out, _, merged_sum = online_softmax_merge(
            out1, max1, sum1,
            out2, max2, sum2
        )

        # Normalize at the end (as per the online softmax algorithm)
        final_out = merged_out / (merged_sum[..., None] + 1e-6)

        np.testing.assert_allclose(
            np.array(final_out), ref_out, rtol=1e-4, atol=1e-5,
            err_msg="Two-chunk merge doesn't match full attention"
        )

    def test_many_chunk_merge_accumulation(self) -> None:
        """Verify correctness with many small chunks (tests error accumulation)."""
        mx.random.seed(42)
        batch, seq_q, heads, dim = 2, 4, 2, 32
        num_chunks = 8
        chunk_size = 4
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq_q, heads, dim))
        k = mx.random.normal((batch, num_chunks * chunk_size, heads, dim))
        v = mx.random.normal((batch, num_chunks * chunk_size, heads, dim))

        # Compute reference
        ref_out = numpy_attention(q, k, v, scale, causal=False)

        # Process chunk by chunk with merging
        acc_out, acc_max, acc_sum = compute_chunk_attention(
            q, k[:, :chunk_size, :, :], v[:, :chunk_size, :, :], scale, causal=False
        )

        for i in range(1, num_chunks):
            kv_offset = i * chunk_size
            new_out, new_max, new_sum = compute_chunk_attention(
                q,
                k[:, kv_offset:kv_offset + chunk_size, :, :],
                v[:, kv_offset:kv_offset + chunk_size, :, :],
                scale,
                causal=False,
                kv_offset=kv_offset
            )
            acc_out, acc_max, acc_sum = online_softmax_merge(
                acc_out, acc_max, acc_sum,
                new_out, new_max, new_sum
            )

        # Normalize at the end
        final_out = acc_out / (acc_sum[..., None] + 1e-6)

        np.testing.assert_allclose(
            np.array(final_out), ref_out, rtol=1e-3, atol=1e-4,
            err_msg="Multi-chunk merge doesn't match full attention"
        )

    def test_merge_with_identical_max_values(self) -> None:
        """Edge case: both chunks have same max score."""
        mx.random.seed(42)
        batch, seq_q, heads, dim = 1, 4, 2, 16

        # Use same k for both chunks to get similar max scores
        q = mx.random.normal((batch, seq_q, heads, dim))
        k_base = mx.random.normal((batch, 8, heads, dim))
        v = mx.random.normal((batch, 16, heads, dim))

        # Duplicate K to ensure similar scores
        k = mx.concatenate([k_base, k_base], axis=1)
        scale = 1.0 / math.sqrt(dim)

        # Reference
        ref_out = numpy_attention(q, k, v, scale, causal=False)

        # Chunked
        out1, max1, sum1 = compute_chunk_attention(
            q, k[:, :8, :, :], v[:, :8, :, :], scale, causal=False
        )
        out2, max2, sum2 = compute_chunk_attention(
            q, k[:, 8:, :, :], v[:, 8:, :, :], scale, causal=False, kv_offset=8
        )
        merged_out, _, merged_sum = online_softmax_merge(out1, max1, sum1, out2, max2, sum2)

        # Normalize at the end
        final_out = merged_out / (merged_sum[..., None] + 1e-6)

        np.testing.assert_allclose(
            np.array(final_out), ref_out, rtol=1e-4, atol=1e-5,
            err_msg="Merge with identical max values failed"
        )

    def test_merge_with_large_score_difference(self) -> None:
        """Edge case: one chunk has much larger scores than the other."""
        mx.random.seed(42)
        batch, seq_q, heads, dim = 1, 4, 2, 16
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq_q, heads, dim))

        # First chunk with large K values (high scores)
        k_large = mx.random.normal((batch, 8, heads, dim)) * 5
        v1 = mx.random.normal((batch, 8, heads, dim))

        # Second chunk with small K values (low scores)
        k_small = mx.random.normal((batch, 8, heads, dim)) * 0.1
        v2 = mx.random.normal((batch, 8, heads, dim))

        k = mx.concatenate([k_large, k_small], axis=1)
        v = mx.concatenate([v1, v2], axis=1)

        # Reference
        ref_out = numpy_attention(q, k, v, scale, causal=False)

        # Chunked
        out1, max1, sum1 = compute_chunk_attention(
            q, k[:, :8, :, :], v[:, :8, :, :], scale, causal=False
        )
        out2, max2, sum2 = compute_chunk_attention(
            q, k[:, 8:, :, :], v[:, 8:, :, :], scale, causal=False, kv_offset=8
        )
        merged_out, _, merged_sum = online_softmax_merge(out1, max1, sum1, out2, max2, sum2)

        # Normalize at the end
        final_out = merged_out / (merged_sum[..., None] + 1e-6)

        np.testing.assert_allclose(
            np.array(final_out), ref_out, rtol=1e-4, atol=1e-5,
            err_msg="Merge with large score difference failed"
        )
