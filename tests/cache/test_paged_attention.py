"""Tests for paged attention with NumPy reference validation."""

import math
import pytest
import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from mlx_primitives.cache import (
    paged_attention,
    create_block_table_from_lengths,
    KVCache,
    KVCacheConfig,
)


def numpy_standard_attention(
    q: NDArray,
    k: NDArray,
    v: NDArray,
    scale: float,
    causal: bool = True,
) -> NDArray:
    """NumPy reference implementation of standard attention.

    Args:
        q: Query (batch, seq_q, heads, dim)
        k: Key (batch, seq_k, heads, dim)
        v: Value (batch, seq_k, heads, dim)
        scale: Attention scale
        causal: Apply causal masking

    Returns:
        Output (batch, seq_q, heads, dim)
    """
    batch, seq_q, heads, dim = q.shape
    seq_k = k.shape[1]

    # Transpose to (batch, heads, seq, dim)
    q_t = np.transpose(q, (0, 2, 1, 3))
    k_t = np.transpose(k, (0, 2, 1, 3))
    v_t = np.transpose(v, (0, 2, 1, 3))

    # Attention scores: (batch, heads, seq_q, seq_k)
    scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) * scale

    if causal:
        # Causal mask: q can attend to k where k_pos <= q_pos
        # For decode (seq_q=1), this means attend to all previous positions
        if seq_q == 1:
            pass  # All positions valid
        else:
            mask = np.tril(np.ones((seq_q, seq_k)))
            scores = np.where(mask == 1, scores, -1e9)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-10)

    # Output
    output = np.matmul(weights, v_t)
    return np.transpose(output, (0, 2, 1, 3))


def numpy_paged_attention_reference(
    q: NDArray,
    k_pool: NDArray,
    v_pool: NDArray,
    block_tables: NDArray,
    context_lens: NDArray,
    scale: float,
    block_size: int,
) -> NDArray:
    """NumPy reference implementation of paged attention.

    This reconstructs the full K, V from blocks and uses standard attention.

    Args:
        q: Query (batch, seq_q, heads, dim)
        k_pool: Key block pool (num_blocks, block_size, heads, dim)
        v_pool: Value block pool (num_blocks, block_size, heads, dim)
        block_tables: Block indices (batch, max_blocks), -1 for invalid
        context_lens: Context length per sequence (batch,)
        scale: Attention scale
        block_size: Tokens per block

    Returns:
        Output (batch, seq_q, heads, dim)
    """
    batch, seq_q, heads, dim = q.shape
    max_context = int(np.max(context_lens))

    # Reconstruct K, V from blocks
    k_full = np.zeros((batch, max_context, heads, dim), dtype=q.dtype)
    v_full = np.zeros((batch, max_context, heads, dim), dtype=q.dtype)

    for b in range(batch):
        ctx_len = int(context_lens[b])
        token_idx = 0

        for block_idx in range(block_tables.shape[1]):
            block_id = int(block_tables[b, block_idx])
            if block_id < 0:
                break

            tokens_remaining = ctx_len - token_idx
            tokens_in_block = min(block_size, tokens_remaining)

            if tokens_in_block <= 0:
                break

            k_full[b, token_idx:token_idx + tokens_in_block] = k_pool[block_id, :tokens_in_block]
            v_full[b, token_idx:token_idx + tokens_in_block] = v_pool[block_id, :tokens_in_block]
            token_idx += tokens_in_block

    # Create attention mask for variable context lengths
    # For decode mode (seq_q=1), each query attends to its context
    outputs = []
    for b in range(batch):
        ctx_len = int(context_lens[b])
        q_b = q[b:b+1, :, :, :]  # (1, seq_q, heads, dim)
        k_b = k_full[b:b+1, :ctx_len, :, :]  # (1, ctx_len, heads, dim)
        v_b = v_full[b:b+1, :ctx_len, :, :]

        out_b = numpy_standard_attention(q_b, k_b, v_b, scale, causal=False)
        outputs.append(out_b)

    return np.concatenate(outputs, axis=0)


class TestPagedAttentionCorrectness:
    """Verify paged attention matches reference implementation."""

    def test_decode_single_sequence(self) -> None:
        """Test decode mode with single sequence."""
        mx.random.seed(42)
        batch, heads, dim = 1, 8, 64
        block_size = 16
        context_len = 30  # Uses 2 blocks

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((10, block_size, heads, dim))
        v_pool = mx.random.normal((10, block_size, heads, dim))
        block_tables = mx.array([[0, 1]], dtype=mx.int32)
        context_lens = mx.array([context_len], dtype=mx.int32)

        scale = 1.0 / math.sqrt(dim)
        mlx_out = paged_attention(
            q, k_pool, v_pool, block_tables, context_lens,
            scale=scale, block_size=block_size
        )

        numpy_out = numpy_paged_attention_reference(
            np.array(q), np.array(k_pool), np.array(v_pool),
            np.array(block_tables), np.array(context_lens),
            scale, block_size
        )

        np.testing.assert_allclose(
            np.array(mlx_out), numpy_out, rtol=1e-3, atol=1e-4
        )

    def test_decode_batch(self) -> None:
        """Test decode mode with batched sequences."""
        mx.random.seed(42)
        batch, heads, dim = 4, 8, 64
        block_size = 16

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((20, block_size, heads, dim))
        v_pool = mx.random.normal((20, block_size, heads, dim))

        # Different context lengths
        context_lens = mx.array([16, 32, 48, 25], dtype=mx.int32)
        block_tables = mx.array([
            [0, -1, -1],
            [1, 2, -1],
            [3, 4, 5],
            [6, 7, -1],
        ], dtype=mx.int32)

        scale = 1.0 / math.sqrt(dim)
        mlx_out = paged_attention(
            q, k_pool, v_pool, block_tables, context_lens,
            scale=scale, block_size=block_size
        )

        numpy_out = numpy_paged_attention_reference(
            np.array(q), np.array(k_pool), np.array(v_pool),
            np.array(block_tables), np.array(context_lens),
            scale, block_size
        )

        np.testing.assert_allclose(
            np.array(mlx_out), numpy_out, rtol=1e-3, atol=1e-4
        )

    def test_full_blocks_only(self) -> None:
        """Test with exactly filled blocks."""
        mx.random.seed(42)
        batch, heads, dim = 2, 8, 64
        block_size = 16

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((10, block_size, heads, dim))
        v_pool = mx.random.normal((10, block_size, heads, dim))

        # Exactly 32 and 48 tokens (2 and 3 full blocks)
        context_lens = mx.array([32, 48], dtype=mx.int32)
        block_tables = mx.array([
            [0, 1, -1],
            [2, 3, 4],
        ], dtype=mx.int32)

        scale = 1.0 / math.sqrt(dim)
        mlx_out = paged_attention(
            q, k_pool, v_pool, block_tables, context_lens,
            scale=scale, block_size=block_size
        )

        numpy_out = numpy_paged_attention_reference(
            np.array(q), np.array(k_pool), np.array(v_pool),
            np.array(block_tables), np.array(context_lens),
            scale, block_size
        )

        np.testing.assert_allclose(
            np.array(mlx_out), numpy_out, rtol=1e-3, atol=1e-4
        )

    def test_single_token_context(self) -> None:
        """Test with minimal context (1 token)."""
        mx.random.seed(42)
        batch, heads, dim = 1, 8, 64
        block_size = 16

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((5, block_size, heads, dim))
        v_pool = mx.random.normal((5, block_size, heads, dim))

        context_lens = mx.array([1], dtype=mx.int32)
        block_tables = mx.array([[0]], dtype=mx.int32)

        scale = 1.0 / math.sqrt(dim)
        mlx_out = paged_attention(
            q, k_pool, v_pool, block_tables, context_lens,
            scale=scale, block_size=block_size
        )

        numpy_out = numpy_paged_attention_reference(
            np.array(q), np.array(k_pool), np.array(v_pool),
            np.array(block_tables), np.array(context_lens),
            scale, block_size
        )

        np.testing.assert_allclose(
            np.array(mlx_out), numpy_out, rtol=1e-3, atol=1e-4
        )


class TestPagedAttentionProperties:
    """Test properties of paged attention."""

    def test_output_shape(self) -> None:
        """Test output shape matches query shape."""
        mx.random.seed(42)
        batch, seq_q, heads, dim = 2, 1, 8, 64
        block_size = 16

        q = mx.random.normal((batch, seq_q, heads, dim))
        k_pool = mx.random.normal((10, block_size, heads, dim))
        v_pool = mx.random.normal((10, block_size, heads, dim))
        block_tables = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
        context_lens = mx.array([20, 25], dtype=mx.int32)

        output = paged_attention(q, k_pool, v_pool, block_tables, context_lens)
        assert output.shape == q.shape

    def test_deterministic(self) -> None:
        """Test output is deterministic."""
        mx.random.seed(42)
        batch, heads, dim = 2, 8, 64
        block_size = 16

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((10, block_size, heads, dim))
        v_pool = mx.random.normal((10, block_size, heads, dim))
        block_tables = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
        context_lens = mx.array([20, 25], dtype=mx.int32)

        out1 = paged_attention(q, k_pool, v_pool, block_tables, context_lens)
        out2 = paged_attention(q, k_pool, v_pool, block_tables, context_lens)

        np.testing.assert_array_equal(np.array(out1), np.array(out2))

    def test_different_scales(self) -> None:
        """Test with different scale values."""
        mx.random.seed(42)
        batch, heads, dim = 1, 8, 64
        block_size = 16

        q = mx.random.normal((batch, 1, heads, dim))
        k_pool = mx.random.normal((5, block_size, heads, dim))
        v_pool = mx.random.normal((5, block_size, heads, dim))
        block_tables = mx.array([[0, 1]], dtype=mx.int32)
        context_lens = mx.array([20], dtype=mx.int32)

        out1 = paged_attention(q, k_pool, v_pool, block_tables, context_lens, scale=0.1)
        out2 = paged_attention(q, k_pool, v_pool, block_tables, context_lens, scale=0.5)

        # Outputs should differ with different scales
        assert not np.allclose(np.array(out1), np.array(out2))


class TestBlockTableCreation:
    """Tests for block table creation utility."""

    def test_create_block_table_basic(self) -> None:
        """Test basic block table creation."""
        seq_lengths = mx.array([16, 32, 48], dtype=mx.int32)
        block_tables = create_block_table_from_lengths(seq_lengths, block_size=16)

        assert block_tables.shape == (3, 3)  # 3 seqs, max 3 blocks
        # First sequence: 1 block
        assert int(block_tables[0, 0]) == 0
        assert int(block_tables[0, 1]) == -1

    def test_create_block_table_partial_blocks(self) -> None:
        """Test with partial blocks."""
        seq_lengths = mx.array([10, 25], dtype=mx.int32)
        block_tables = create_block_table_from_lengths(seq_lengths, block_size=16)

        assert block_tables.shape == (2, 2)
        # 10 tokens = 1 block, 25 tokens = 2 blocks
        assert int(block_tables[0, 1]) == -1  # Padding


class TestKVCacheIntegration:
    """Integration tests with KVCache."""

    def test_cache_then_paged_attention(self) -> None:
        """Test using paged attention with KVCache."""
        mx.random.seed(42)
        block_size = 16  # Use explicit block size for consistency
        config = KVCacheConfig(
            num_heads=8, head_dim=64, num_layers=1, max_memory_gb=0.1,
            block_size=block_size
        )
        cache = KVCache(config)

        # Create sequence and add tokens
        seq_id = cache.create_sequence()
        k = mx.random.normal((32, 8, 64))
        v = mx.random.normal((32, 8, 64))
        cache.update(seq_id, k, v, layer_idx=0)

        # Get paged data
        k_pool, v_pool, block_tables, context_lens = cache.get_kv_paged([seq_id], layer_idx=0)

        # Run paged attention with matching block_size
        q = mx.random.normal((1, 1, 8, 64))
        output = paged_attention(q, k_pool, v_pool, block_tables, context_lens,
                                  block_size=block_size)

        assert output.shape == (1, 1, 8, 64)

    def test_batched_cache_attention(self) -> None:
        """Test paged attention with multiple cached sequences."""
        mx.random.seed(42)
        block_size = 16  # Use explicit block size for consistency
        config = KVCacheConfig(
            num_heads=8, head_dim=64, num_layers=1, max_memory_gb=0.5,
            block_size=block_size
        )
        cache = KVCache(config)

        # Create multiple sequences
        seq_ids = []
        for length in [20, 35, 50]:
            seq_id = cache.create_sequence()
            k = mx.random.normal((length, 8, 64))
            v = mx.random.normal((length, 8, 64))
            cache.update(seq_id, k, v, layer_idx=0)
            seq_ids.append(seq_id)

        # Get paged data for all sequences
        k_pool, v_pool, block_tables, context_lens = cache.get_kv_paged(seq_ids, layer_idx=0)

        # Run paged attention with matching block_size
        q = mx.random.normal((3, 1, 8, 64))
        output = paged_attention(q, k_pool, v_pool, block_tables, context_lens,
                                  block_size=block_size)

        assert output.shape == (3, 1, 8, 64)
