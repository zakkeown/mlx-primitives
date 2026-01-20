"""Correctness tests for PagedKVCache (vLLM-style paged attention).

Tests verify:
1. Basic storage and retrieval correctness
2. Multi-layer consistency
3. Block boundary handling
4. Sequence isolation
5. Copy-on-write (COW) fork semantics
6. Comparison against naive KV cache
7. Integration with attention
"""

import pytest
import mlx.core as mx

from mlx_primitives.advanced.paged_attention import (
    PagedKVCache as PagedKVCacheV2,
    BlockManager,
    BlockConfig,
)
from mlx_primitives.attention import flash_attention_forward


# =============================================================================
# Reference Implementations
# =============================================================================


class NaiveKVCache:
    """Simple dict-based KV cache for comparison."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # k_cache[layer_idx] = list of (seq, heads, dim) tensors
        self.k_cache = {i: [] for i in range(num_layers)}
        self.v_cache = {i: [] for i in range(num_layers)}
        self.num_tokens = 0

    def append(self, layer_idx: int, k: mx.array, v: mx.array):
        """Append K/V for a layer. k, v shape: (num_tokens, heads, dim)."""
        self.k_cache[layer_idx].append(k)
        self.v_cache[layer_idx].append(v)
        if layer_idx == 0:
            self.num_tokens += k.shape[0]

    def get(self, layer_idx: int):
        """Get concatenated K/V for a layer."""
        if not self.k_cache[layer_idx]:
            return None, None
        k = mx.concatenate(self.k_cache[layer_idx], axis=0)
        v = mx.concatenate(self.v_cache[layer_idx], axis=0)
        return k, v


# =============================================================================
# Test Classes
# =============================================================================


class TestBasicStorageRetrieval:
    """Test basic storage and retrieval correctness."""

    def test_single_layer_exact_match(self):
        """Store and retrieve K/V for single layer, exact float32 match."""
        cache = PagedKVCacheV2(
            num_kv_heads=8,
            head_dim=64,
            num_layers=1,
            block_size=16,
            max_blocks=64,
            dtype=mx.float32,  # Use float32 for exact match
        )

        seq_id = cache.create_sequence()

        # Add tokens
        k_in = mx.random.normal((20, 8, 64))
        v_in = mx.random.normal((20, 8, 64))
        mx.eval(k_in, v_in)

        cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k_in, v_batch=v_in)

        # Retrieve
        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_out, v_out)

        # Should be exact match with float32
        k_diff = float(mx.max(mx.abs(k_in - k_out)))
        v_diff = float(mx.max(mx.abs(v_in - v_out)))

        assert k_diff == 0.0, f"K differs by {k_diff}"
        assert v_diff == 0.0, f"V differs by {v_diff}"

    def test_incremental_append(self):
        """Incremental token appends accumulate correctly."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=8,
            max_blocks=32,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        # Append in batches
        k_parts = []
        v_parts = []
        for _ in range(5):
            k = mx.random.normal((10, 4, 32))
            v = mx.random.normal((10, 4, 32))
            mx.eval(k, v)
            k_parts.append(k)
            v_parts.append(v)
            cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k, v_batch=v)

        # Expected concatenation
        k_expected = mx.concatenate(k_parts, axis=0)
        v_expected = mx.concatenate(v_parts, axis=0)
        mx.eval(k_expected, v_expected)

        # Retrieve
        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_out, v_out)

        assert k_out.shape == (50, 4, 32)
        assert v_out.shape == (50, 4, 32)

        k_diff = float(mx.max(mx.abs(k_expected - k_out)))
        v_diff = float(mx.max(mx.abs(v_expected - v_out)))

        assert k_diff == 0.0, f"K differs by {k_diff}"
        assert v_diff == 0.0, f"V differs by {v_diff}"


class TestMultiLayerConsistency:
    """Test multi-layer storage consistency."""

    def test_same_data_all_layers(self):
        """Same K/V stored to all layers retrieves identically."""
        num_layers = 4
        cache = PagedKVCacheV2(
            num_kv_heads=8,
            head_dim=64,
            num_layers=num_layers,
            block_size=16,
            max_blocks=64,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        k_in = mx.random.normal((32, 8, 64))
        v_in = mx.random.normal((32, 8, 64))
        mx.eval(k_in, v_in)

        # Store same data to all layers
        for layer_idx in range(num_layers):
            cache.append_kv_batch(seq_id, layer_idx, k_in, v_in)

        # Retrieve from each layer
        for layer_idx in range(num_layers):
            k_out, v_out = cache.get_kv(seq_id, layer_idx)
            mx.eval(k_out, v_out)

            k_diff = float(mx.max(mx.abs(k_in - k_out)))
            v_diff = float(mx.max(mx.abs(v_in - v_out)))

            assert k_diff == 0.0, f"Layer {layer_idx} K differs by {k_diff}"
            assert v_diff == 0.0, f"Layer {layer_idx} V differs by {v_diff}"

    def test_different_data_per_layer(self):
        """Different K/V per layer stored and retrieved correctly."""
        num_layers = 4
        cache = PagedKVCacheV2(
            num_kv_heads=8,
            head_dim=64,
            num_layers=num_layers,
            block_size=16,
            max_blocks=64,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        # Store different data per layer
        layer_data = {}
        for layer_idx in range(num_layers):
            mx.random.seed(layer_idx * 100)
            k = mx.random.normal((32, 8, 64))
            v = mx.random.normal((32, 8, 64))
            mx.eval(k, v)
            layer_data[layer_idx] = (k, v)
            cache.append_kv_batch(seq_id, layer_idx, k, v)

        # Verify each layer
        for layer_idx in range(num_layers):
            k_expected, v_expected = layer_data[layer_idx]
            k_out, v_out = cache.get_kv(seq_id, layer_idx)
            mx.eval(k_out, v_out)

            k_diff = float(mx.max(mx.abs(k_expected - k_out)))
            v_diff = float(mx.max(mx.abs(v_expected - v_out)))

            assert k_diff == 0.0, f"Layer {layer_idx} K differs"
            assert v_diff == 0.0, f"Layer {layer_idx} V differs"


class TestBlockBoundaryHandling:
    """Test correctness at block boundaries."""

    @pytest.mark.parametrize("block_size", [8, 16, 32])
    def test_exact_block_fill(self, block_size):
        """Filling exactly one block works correctly."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=block_size,
            max_blocks=16,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        k = mx.random.normal((block_size, 4, 32))
        v = mx.random.normal((block_size, 4, 32))
        mx.eval(k, v)

        cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k, v_batch=v)

        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_out, v_out)

        assert float(mx.max(mx.abs(k - k_out))) == 0.0
        assert float(mx.max(mx.abs(v - v_out))) == 0.0

    def test_cross_block_boundary(self):
        """Tokens spanning multiple blocks stored correctly."""
        block_size = 16
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=block_size,
            max_blocks=16,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        # Add tokens that span 3 blocks (16 + 16 + 8 = 40 tokens)
        k = mx.random.normal((40, 4, 32))
        v = mx.random.normal((40, 4, 32))
        mx.eval(k, v)

        cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k, v_batch=v)

        # Should use 3 blocks (ceil(40/16))
        stats = cache.get_memory_stats()
        assert stats["allocated_blocks"] == 3

        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_out, v_out)

        assert float(mx.max(mx.abs(k - k_out))) == 0.0

    def test_boundary_position_accuracy(self):
        """Positions at block boundaries (15, 16, 17) stored correctly."""
        block_size = 16
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=block_size,
            max_blocks=16,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        # Add 20 tokens (boundary at 15/16)
        k = mx.arange(20 * 4 * 32).reshape(20, 4, 32).astype(mx.float32)
        v = mx.arange(20 * 4 * 32).reshape(20, 4, 32).astype(mx.float32) + 1000
        mx.eval(k, v)

        cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k, v_batch=v)

        k_out, v_out = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_out, v_out)

        # Check specific boundary positions
        for pos in [14, 15, 16, 17]:
            k_expected = k[pos]
            k_actual = k_out[pos]
            assert mx.allclose(k_expected, k_actual), f"K mismatch at position {pos}"


class TestSequenceIsolation:
    """Test that multiple sequences don't interfere."""

    def test_two_sequences_independent(self):
        """Two sequences store/retrieve independently."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=16,
            max_blocks=32,
            dtype=mx.float32,
        )

        seq1 = cache.create_sequence()
        seq2 = cache.create_sequence()

        # Different data for each sequence
        mx.random.seed(1)
        k1 = mx.random.normal((20, 4, 32))
        v1 = mx.random.normal((20, 4, 32))
        mx.eval(k1, v1)

        mx.random.seed(2)
        k2 = mx.random.normal((30, 4, 32))
        v2 = mx.random.normal((30, 4, 32))
        mx.eval(k2, v2)

        cache.append_kv_batch(seq1, layer_idx=0, k_batch=k1, v_batch=v1)
        cache.append_kv_batch(seq2, layer_idx=0, k_batch=k2, v_batch=v2)

        # Retrieve and verify
        k1_out, v1_out = cache.get_kv(seq1, layer_idx=0)
        k2_out, v2_out = cache.get_kv(seq2, layer_idx=0)
        mx.eval(k1_out, v1_out, k2_out, v2_out)

        assert k1_out.shape == (20, 4, 32)
        assert k2_out.shape == (30, 4, 32)

        assert float(mx.max(mx.abs(k1 - k1_out))) == 0.0
        assert float(mx.max(mx.abs(k2 - k2_out))) == 0.0

    def test_delete_sequence_frees_blocks(self):
        """Deleting a sequence frees its blocks."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=16,
            max_blocks=32,
            dtype=mx.float32,
        )

        initial_free = cache._block_manager.num_free_blocks

        seq = cache.create_sequence()
        k = mx.random.normal((32, 4, 32))
        v = mx.random.normal((32, 4, 32))
        mx.eval(k, v)

        cache.append_kv_batch(seq, layer_idx=0, k_batch=k, v_batch=v)

        # Blocks allocated
        after_alloc = cache._block_manager.num_free_blocks
        assert after_alloc < initial_free

        # Delete sequence
        cache.delete_sequence(seq)

        # Blocks freed
        after_delete = cache._block_manager.num_free_blocks
        assert after_delete == initial_free


class TestCopyOnWriteFork:
    """Test copy-on-write fork semantics for beam search."""

    def test_fork_shares_data_initially(self):
        """Forked sequence initially returns same data as parent."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=16,
            max_blocks=32,
            dtype=mx.float32,
        )

        parent = cache.create_sequence()

        k = mx.random.normal((20, 4, 32))
        v = mx.random.normal((20, 4, 32))
        mx.eval(k, v)

        cache.append_kv_batch(parent, layer_idx=0, k_batch=k, v_batch=v)

        # Fork
        child = cache.fork_sequence(parent)

        # Both should have same data
        k_parent, _ = cache.get_kv(parent, layer_idx=0)
        k_child, _ = cache.get_kv(child, layer_idx=0)
        mx.eval(k_parent, k_child)

        assert float(mx.max(mx.abs(k_parent - k_child))) == 0.0
        assert cache.get_sequence_length(parent) == cache.get_sequence_length(child)

    def test_fork_modification_isolates(self):
        """Modifying forked sequence doesn't affect parent."""
        cache = PagedKVCacheV2(
            num_kv_heads=4,
            head_dim=32,
            num_layers=1,
            block_size=16,
            max_blocks=32,
            dtype=mx.float32,
        )

        parent = cache.create_sequence()

        k_orig = mx.random.normal((20, 4, 32))
        v_orig = mx.random.normal((20, 4, 32))
        mx.eval(k_orig, v_orig)

        cache.append_kv_batch(parent, layer_idx=0, k_batch=k_orig, v_batch=v_orig)

        # Fork and modify child
        child = cache.fork_sequence(parent)

        k_new = mx.random.normal((5, 4, 32))
        v_new = mx.random.normal((5, 4, 32))
        mx.eval(k_new, v_new)

        cache.append_kv_batch(child, layer_idx=0, k_batch=k_new, v_batch=v_new)

        # Parent should be unchanged
        k_parent, _ = cache.get_kv(parent, layer_idx=0)
        mx.eval(k_parent)

        assert k_parent.shape == (20, 4, 32)
        assert float(mx.max(mx.abs(k_orig - k_parent))) == 0.0

        # Child should have new data
        k_child, _ = cache.get_kv(child, layer_idx=0)
        mx.eval(k_child)

        assert k_child.shape == (25, 4, 32)


class TestVsNaiveKVCache:
    """Compare against naive implementation."""

    def test_prefill_matches_naive(self):
        """Prefill phase matches naive cache."""
        num_layers = 2
        cache = PagedKVCacheV2(
            num_kv_heads=8,
            head_dim=64,
            num_layers=num_layers,
            block_size=16,
            max_blocks=64,
            dtype=mx.float32,
        )
        naive = NaiveKVCache(num_layers)

        seq_id = cache.create_sequence()

        # Prefill
        k = mx.random.normal((50, 8, 64))
        v = mx.random.normal((50, 8, 64))
        mx.eval(k, v)

        for layer_idx in range(num_layers):
            cache.append_kv_batch(seq_id, layer_idx, k, v)
            naive.append(layer_idx, k, v)

        # Compare each layer
        for layer_idx in range(num_layers):
            k_paged, v_paged = cache.get_kv(seq_id, layer_idx)
            k_naive, v_naive = naive.get(layer_idx)
            mx.eval(k_paged, v_paged, k_naive, v_naive)

            assert float(mx.max(mx.abs(k_paged - k_naive))) == 0.0
            assert float(mx.max(mx.abs(v_paged - v_naive))) == 0.0

    def test_generation_matches_naive(self):
        """Token-by-token generation matches naive cache."""
        num_layers = 2
        cache = PagedKVCacheV2(
            num_kv_heads=8,
            head_dim=64,
            num_layers=num_layers,
            block_size=16,
            max_blocks=64,
            dtype=mx.float32,
        )
        naive = NaiveKVCache(num_layers)

        seq_id = cache.create_sequence()

        # Prefill
        k_pre = mx.random.normal((32, 8, 64))
        v_pre = mx.random.normal((32, 8, 64))
        mx.eval(k_pre, v_pre)

        for layer_idx in range(num_layers):
            cache.append_kv_batch(seq_id, layer_idx, k_pre, v_pre)
            naive.append(layer_idx, k_pre, v_pre)

        # Generate 10 tokens one at a time
        # Note: For single-token generation, we process all layers for each token
        for tok_idx in range(10):
            k_tok = mx.random.normal((8, 64))
            v_tok = mx.random.normal((8, 64))
            mx.eval(k_tok, v_tok)

            for layer_idx in range(num_layers):
                cache.append_kv(seq_id, layer_idx, k_tok, v_tok)
                # Naive cache expects (seq, heads, dim), add seq dim
                naive.append(layer_idx, k_tok[None, :, :], v_tok[None, :, :])

        # Compare
        for layer_idx in range(num_layers):
            k_paged, v_paged = cache.get_kv(seq_id, layer_idx)
            k_naive, v_naive = naive.get(layer_idx)
            mx.eval(k_paged, v_paged, k_naive, v_naive)

            assert k_paged.shape == k_naive.shape, f"Shape mismatch: {k_paged.shape} vs {k_naive.shape}"
            assert float(mx.max(mx.abs(k_paged - k_naive))) == 0.0, f"Layer {layer_idx} K values differ"


class TestAttentionIntegration:
    """Test integration with attention computation."""

    def test_attention_output_matches(self):
        """Attention with paged cache matches attention with direct K/V."""
        mx.random.seed(42)

        batch, seq_len, num_heads, head_dim = 1, 64, 8, 64

        # Create paged cache
        cache = PagedKVCacheV2(
            num_kv_heads=num_heads,
            head_dim=head_dim,
            num_layers=1,
            block_size=16,
            max_blocks=32,
            dtype=mx.float32,
        )

        seq_id = cache.create_sequence()

        # Create Q, K, V
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch, seq_len, num_heads, head_dim))
        mx.eval(q, k, v)

        # Store K/V in paged cache (remove batch dim for storage)
        cache.append_kv_batch(seq_id, layer_idx=0, k_batch=k[0], v_batch=v[0])

        # Retrieve from cache
        k_cached, v_cached = cache.get_kv(seq_id, layer_idx=0)
        mx.eval(k_cached, v_cached)

        # Add batch dim back
        k_cached = k_cached[None, :, :, :]
        v_cached = v_cached[None, :, :, :]

        # Compute attention both ways
        scale = 1.0 / (head_dim ** 0.5)
        out_direct = flash_attention_forward(q, k, v, scale=scale, causal=True)
        out_cached = flash_attention_forward(q, k_cached, v_cached, scale=scale, causal=True)
        mx.eval(out_direct, out_cached)

        # Should match exactly (float32 storage)
        max_diff = float(mx.max(mx.abs(out_direct - out_cached)))
        assert max_diff == 0.0, f"Attention outputs differ by {max_diff}"
