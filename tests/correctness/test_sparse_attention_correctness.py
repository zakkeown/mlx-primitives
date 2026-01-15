"""Correctness tests for Sparse Attention implementations.

Tests verify:
1. BlockSparseAttention: mask structure, output vs dense, gradient flow, causal
2. LongformerAttention: sliding window pattern, global tokens, vs dense
3. BigBirdAttention: random pattern, window/global components, determinism
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.attention.sparse import (
    create_block_sparse_mask,
    create_sliding_window_mask,
    create_bigbird_mask,
    BlockSparseAttention,
    LongformerAttention,
    BigBirdAttention,
)


# =============================================================================
# Reference Implementations
# =============================================================================


def naive_dense_attention(q, k, v, mask=None, scale=None):
    """Reference dense attention implementation.

    Args:
        q: Query (batch, heads, seq, head_dim)
        k: Key (batch, heads, seq, head_dim)
        v: Value (batch, heads, seq, head_dim)
        mask: Boolean mask (seq, seq) or None
        scale: Scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output (batch, heads, seq, head_dim)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        scores = mx.where(mask[None, None, :, :], scores, float("-inf"))

    attn_weights = mx.softmax(scores, axis=-1)
    # Handle NaN from all-masked rows
    attn_weights = mx.where(mx.isnan(attn_weights), 0.0, attn_weights)

    return attn_weights @ v


def naive_sliding_window_mask(seq_len, window_size):
    """Reference sliding window mask implementation."""
    mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= window_size:
                mask = mx.where(
                    (mx.arange(seq_len) == i)[:, None]
                    & (mx.arange(seq_len) == j)[None, :],
                    True,
                    mask,
                )
    # Simpler vectorized version
    positions = mx.arange(seq_len)
    mask = mx.abs(positions[:, None] - positions[None, :]) <= window_size
    return mask


def naive_block_diagonal_mask(seq_len, block_size):
    """Reference block diagonal mask implementation."""
    block_indices = mx.arange(seq_len) // block_size
    return block_indices[:, None] == block_indices[None, :]


# =============================================================================
# BlockSparseAttention Tests
# =============================================================================


class TestBlockSparseMaskStructure:
    """Test block-sparse mask correctness."""

    def test_diagonal_blocks_present(self):
        """Each position should attend to its own block."""
        seq_len = 64
        block_size = 16
        mask = create_block_sparse_mask(seq_len, block_size)
        mx.eval(mask)

        # Check diagonal blocks
        for block_idx in range(seq_len // block_size):
            start = block_idx * block_size
            end = start + block_size
            # Every position in block should attend to every other position in same block
            block_mask = mask[start:end, start:end]
            assert mx.all(block_mask), f"Block {block_idx} diagonal not fully connected"

    def test_off_diagonal_blocks_sparse(self):
        """Off-diagonal blocks should be sparse (zero without global/random)."""
        seq_len = 64
        block_size = 16
        mask = create_block_sparse_mask(
            seq_len, block_size, num_random_blocks=0, num_global_blocks=0
        )
        mx.eval(mask)

        # Check that off-diagonal blocks are zero
        for i in range(seq_len // block_size):
            for j in range(seq_len // block_size):
                if i != j:
                    i_start, i_end = i * block_size, (i + 1) * block_size
                    j_start, j_end = j * block_size, (j + 1) * block_size
                    block = mask[i_start:i_end, j_start:j_end]
                    assert not mx.any(block), f"Off-diagonal block ({i},{j}) should be zero"

    def test_global_blocks_connect_all(self):
        """Global blocks should connect to all positions."""
        seq_len = 64
        block_size = 16
        num_global_blocks = 1
        mask = create_block_sparse_mask(
            seq_len, block_size, num_random_blocks=0, num_global_blocks=num_global_blocks
        )
        mx.eval(mask)

        global_end = num_global_blocks * block_size

        # Global tokens (first block) should attend to all
        global_rows = mask[:global_end, :]
        assert mx.all(global_rows), "Global tokens should attend to all positions"

        # All tokens should attend to global tokens
        global_cols = mask[:, :global_end]
        assert mx.all(global_cols), "All positions should attend to global tokens"

    def test_mask_vs_naive_block_diagonal(self):
        """Block-sparse mask (no extras) should match naive block diagonal."""
        seq_len = 48
        block_size = 16
        mask = create_block_sparse_mask(
            seq_len, block_size, num_random_blocks=0, num_global_blocks=0
        )
        naive_mask = naive_block_diagonal_mask(seq_len, block_size)
        mx.eval(mask, naive_mask)

        assert mx.array_equal(mask, naive_mask), "Block sparse mask differs from naive"


class TestBlockSparseAttentionOutput:
    """Test BlockSparseAttention output correctness."""

    def test_output_vs_dense_with_same_mask(self):
        """BlockSparse output should match dense attention with same mask."""
        mx.random.seed(42)
        batch, seq_len, dims, num_heads = 2, 64, 128, 4
        block_size = 16
        head_dim = dims // num_heads

        # Create module and get mask
        attn = BlockSparseAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
            num_random_blocks=0,
            num_global_blocks=0,
        )

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        # Get the sparse mask
        mask = create_block_sparse_mask(seq_len, block_size, 0, 0)
        mx.eval(mask)

        # Compute Q, K, V manually
        q = attn.q_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        mx.eval(q, k, v)

        # Dense attention with mask
        dense_out = naive_dense_attention(q, k, v, mask=mask, scale=head_dim**-0.5)
        dense_out = dense_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, dims)
        dense_out = attn.out_proj(dense_out)
        mx.eval(dense_out)

        # Module output
        module_out = attn(x)
        mx.eval(module_out)

        max_diff = float(mx.max(mx.abs(dense_out - module_out)))
        assert max_diff < 1e-5, f"Output differs by {max_diff}"

    def test_causal_block_sparse(self):
        """Causal mask should combine with block sparse correctly."""
        mx.random.seed(42)
        seq_len = 64
        block_size = 16

        # Create causal block sparse mask manually
        block_mask = create_block_sparse_mask(seq_len, block_size, 0, 0)
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        combined_mask = block_mask & causal_mask
        mx.eval(combined_mask)

        # Verify causal property: no position attends to future
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert not bool(combined_mask[i, j]), f"Position {i} attends to future {j}"

        # Verify block structure preserved within causal constraint
        for block_idx in range(seq_len // block_size):
            start = block_idx * block_size
            end = start + block_size
            block = combined_mask[start:end, start:end]
            # Lower triangle of block should be all True
            expected = mx.tril(mx.ones((block_size, block_size), dtype=mx.bool_))
            assert mx.array_equal(block, expected), f"Block {block_idx} causal structure wrong"


class TestBlockSparseGradients:
    """Test gradient flow through block sparse attention."""

    def test_gradient_flow_through_mask(self):
        """Gradients should flow only through non-masked positions."""
        mx.random.seed(42)
        batch, seq_len, dims, num_heads = 1, 32, 64, 2
        block_size = 16
        head_dim = dims // num_heads

        q = mx.random.normal((batch, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch, num_heads, seq_len, head_dim))
        mx.eval(q, k, v)

        mask = create_block_sparse_mask(seq_len, block_size, 0, 0)
        mx.eval(mask)

        def loss_fn(q, k, v):
            out = naive_dense_attention(q, k, v, mask=mask)
            return mx.sum(out)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)

        # Gradients should be non-zero
        q_grad, k_grad, v_grad = grads
        assert float(mx.sum(mx.abs(q_grad))) > 0, "Q gradient is zero"
        assert float(mx.sum(mx.abs(k_grad))) > 0, "K gradient is zero"
        assert float(mx.sum(mx.abs(v_grad))) > 0, "V gradient is zero"


# =============================================================================
# LongformerAttention Tests
# =============================================================================


class TestSlidingWindowMask:
    """Test sliding window mask correctness."""

    def test_window_pattern_correct(self):
        """Each position should attend to exactly window_size neighbors on each side."""
        seq_len = 32
        window_size = 4
        mask = create_sliding_window_mask(seq_len, window_size, global_indices=None)
        mx.eval(mask)

        for i in range(seq_len):
            for j in range(seq_len):
                expected = abs(i - j) <= window_size
                actual = bool(mask[i, j])
                assert actual == expected, f"Mask[{i},{j}] = {actual}, expected {expected}"

    def test_window_vs_naive(self):
        """Sliding window should match naive implementation."""
        seq_len = 48
        window_size = 8
        mask = create_sliding_window_mask(seq_len, window_size, global_indices=None)
        naive = naive_sliding_window_mask(seq_len, window_size)
        mx.eval(mask, naive)

        assert mx.array_equal(mask, naive), "Sliding window differs from naive"

    def test_global_tokens_in_mask(self):
        """Global tokens should attend to/from all positions."""
        seq_len = 32
        window_size = 4
        global_indices = [0, 5, 10]
        mask = create_sliding_window_mask(seq_len, window_size, global_indices)
        mx.eval(mask)

        for g_idx in global_indices:
            # Global token attends to all
            assert mx.all(mask[g_idx, :]), f"Global token {g_idx} doesn't attend to all"
            # All attend to global token
            assert mx.all(mask[:, g_idx]), f"Not all attend to global token {g_idx}"


class TestLongformerAttentionOutput:
    """Test LongformerAttention output correctness."""

    def test_output_shape(self):
        """Output should have correct shape."""
        mx.random.seed(42)
        batch, seq_len, dims = 2, 64, 128
        attn = LongformerAttention(dims=dims, num_heads=4, window_size=16)

        x = mx.random.normal((batch, seq_len, dims))
        out = attn(x)
        mx.eval(out)

        assert out.shape == (batch, seq_len, dims)

    def test_local_attention_matches_dense_windowed(self):
        """Local (non-global) attention should match dense with window mask."""
        mx.random.seed(42)
        batch, seq_len, dims, num_heads = 1, 32, 64, 2
        window_size = 8
        head_dim = dims // num_heads

        # Create Longformer with no global tokens
        attn = LongformerAttention(
            dims=dims, num_heads=num_heads, window_size=window_size, num_global_tokens=0
        )

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        # Create expected mask
        mask = create_sliding_window_mask(seq_len, window_size, global_indices=[])
        mx.eval(mask)

        # Get Q, K, V from local projections
        q = attn.q_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        mx.eval(q, k, v)

        # Dense attention with window mask
        expected = naive_dense_attention(q, k, v, mask=mask, scale=head_dim**-0.5)
        expected = expected.transpose(0, 2, 1, 3).reshape(batch, seq_len, dims)
        expected = attn.out_proj(expected)
        mx.eval(expected)

        # Module output (no global tokens)
        actual = attn(x, global_indices=[])
        mx.eval(actual)

        max_diff = float(mx.max(mx.abs(expected - actual)))
        assert max_diff < 1e-4, f"Output differs by {max_diff}"


class TestLongformerGradients:
    """Test gradient flow through Longformer attention."""

    def test_gradients_nonzero(self):
        """Gradients should flow through the module."""
        mx.random.seed(42)
        batch, seq_len, dims = 1, 32, 64
        attn = LongformerAttention(dims=dims, num_heads=2, window_size=8, num_global_tokens=0)

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(attn(x, global_indices=[]))

        grad_fn = mx.grad(loss_fn)
        x_grad = grad_fn(x)
        mx.eval(x_grad)

        # Check that gradients are non-zero
        grad_norm = float(mx.sum(mx.abs(x_grad)))
        assert grad_norm > 0, "Input gradients are zero"


# =============================================================================
# BigBirdAttention Tests
# =============================================================================


class TestBigBirdMask:
    """Test BigBird mask correctness."""

    def test_window_component_present(self):
        """BigBird mask should include sliding window."""
        seq_len = 64
        block_size = 16
        window_size = 16  # Will become window_size * block_size in module
        mask = create_bigbird_mask(
            seq_len,
            block_size,
            window_size,
            num_global_tokens=0,
            num_random_blocks=0,
            seed=42,
        )
        mx.eval(mask)

        # Check window pattern for central positions
        for i in range(window_size, seq_len - window_size):
            for j in range(seq_len):
                if abs(i - j) <= window_size:
                    assert bool(mask[i, j]), f"Window position ({i},{j}) should be True"

    def test_global_tokens_present(self):
        """Global tokens should attend to/from all."""
        seq_len = 64
        num_global = 4
        mask = create_bigbird_mask(
            seq_len,
            block_size=16,
            window_size=8,
            num_global_tokens=num_global,
            num_random_blocks=0,
            seed=42,
        )
        mx.eval(mask)

        # First num_global tokens are global
        for g in range(num_global):
            assert mx.all(mask[g, :]), f"Global token {g} doesn't attend to all"
            assert mx.all(mask[:, g]), f"Not all attend to global token {g}"

    def test_determinism_with_seed(self):
        """Same seed should produce same random pattern."""
        seq_len = 64
        seed = 123

        mask1 = create_bigbird_mask(
            seq_len,
            block_size=16,
            window_size=8,
            num_global_tokens=2,
            num_random_blocks=2,
            seed=seed,
        )
        mx.eval(mask1)

        mask2 = create_bigbird_mask(
            seq_len,
            block_size=16,
            window_size=8,
            num_global_tokens=2,
            num_random_blocks=2,
            seed=seed,
        )
        mx.eval(mask2)

        assert mx.array_equal(mask1, mask2), "Same seed should produce same mask"

    def test_different_seeds_differ(self):
        """Different seeds should produce different patterns (with random blocks)."""
        seq_len = 64

        mask1 = create_bigbird_mask(
            seq_len,
            block_size=16,
            window_size=8,
            num_global_tokens=2,
            num_random_blocks=2,
            seed=1,
        )
        mx.eval(mask1)

        mask2 = create_bigbird_mask(
            seq_len,
            block_size=16,
            window_size=8,
            num_global_tokens=2,
            num_random_blocks=2,
            seed=999,
        )
        mx.eval(mask2)

        # They should differ somewhere (random blocks)
        diff_count = int(mx.sum(mask1 != mask2))
        assert diff_count > 0, "Different seeds should produce different masks"


class TestBigBirdAttentionOutput:
    """Test BigBirdAttention output correctness."""

    def test_output_shape(self):
        """Output should have correct shape."""
        mx.random.seed(42)
        batch, seq_len, dims = 2, 64, 128
        attn = BigBirdAttention(dims=dims, num_heads=4, block_size=16)

        x = mx.random.normal((batch, seq_len, dims))
        out = attn(x)
        mx.eval(out)

        assert out.shape == (batch, seq_len, dims)

    def test_output_vs_dense_with_mask(self):
        """BigBird output should match dense attention with same mask."""
        mx.random.seed(42)
        batch, seq_len, dims, num_heads = 1, 64, 64, 2
        block_size = 16
        head_dim = dims // num_heads

        attn = BigBirdAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
            window_size=1,  # Will be 1 * block_size = 16
            num_global_tokens=2,
            num_random_blocks=0,  # No random for reproducibility
        )

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        # Get the mask
        mask = attn._get_mask(seq_len)
        mx.eval(mask)

        # Compute Q, K, V
        q = attn.q_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        mx.eval(q, k, v)

        # Dense attention with BigBird mask
        expected = naive_dense_attention(q, k, v, mask=mask, scale=head_dim**-0.5)
        expected = expected.transpose(0, 2, 1, 3).reshape(batch, seq_len, dims)
        expected = attn.out_proj(expected)
        mx.eval(expected)

        # Module output
        actual = attn(x)
        mx.eval(actual)

        max_diff = float(mx.max(mx.abs(expected - actual)))
        assert max_diff < 1e-4, f"Output differs by {max_diff}"

    def test_no_nan_in_output(self):
        """Output should not contain NaN values."""
        mx.random.seed(42)
        batch, seq_len, dims = 2, 128, 64
        attn = BigBirdAttention(dims=dims, num_heads=2, block_size=32)

        x = mx.random.normal((batch, seq_len, dims))
        out = attn(x)
        mx.eval(out)

        has_nan = bool(mx.any(mx.isnan(out)))
        assert not has_nan, "Output contains NaN"


class TestBigBirdGradients:
    """Test gradient flow through BigBird attention."""

    def test_gradients_nonzero(self):
        """Gradients should flow through the module."""
        mx.random.seed(42)
        # Use smaller dimensions to avoid memory issues
        batch, seq_len, dims = 1, 32, 32
        attn = BigBirdAttention(
            dims=dims,
            num_heads=2,
            block_size=8,
            num_random_blocks=0,  # Simpler pattern
        )

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(attn(x))

        grad_fn = mx.grad(loss_fn)
        x_grad = grad_fn(x)
        mx.eval(x_grad)

        # Check that gradients are non-zero
        grad_norm = float(mx.sum(mx.abs(x_grad)))
        assert grad_norm > 0, "Input gradients are zero"


# =============================================================================
# Cross-Implementation Tests
# =============================================================================


class TestSparseAttentionConsistency:
    """Test consistency across sparse attention implementations."""

    def test_all_sparse_handle_same_input(self):
        """All sparse attention modules should handle the same input without error."""
        mx.random.seed(42)
        batch, seq_len, dims = 2, 64, 128

        x = mx.random.normal((batch, seq_len, dims))
        mx.eval(x)

        # BlockSparse
        block_sparse = BlockSparseAttention(dims=dims, num_heads=4, block_size=16)
        out1 = block_sparse(x)
        mx.eval(out1)
        assert out1.shape == (batch, seq_len, dims)

        # Longformer
        longformer = LongformerAttention(dims=dims, num_heads=4, window_size=16)
        out2 = longformer(x)
        mx.eval(out2)
        assert out2.shape == (batch, seq_len, dims)

        # BigBird
        bigbird = BigBirdAttention(dims=dims, num_heads=4, block_size=16)
        out3 = bigbird(x)
        mx.eval(out3)
        assert out3.shape == (batch, seq_len, dims)

    def test_sparse_more_efficient_pattern(self):
        """Sparse masks should have fewer True values than dense."""
        seq_len = 128
        dense_count = seq_len * seq_len

        # Block sparse
        block_mask = create_block_sparse_mask(seq_len, block_size=32, num_random_blocks=0, num_global_blocks=0)
        block_count = int(mx.sum(block_mask))
        assert block_count < dense_count, "Block sparse should be sparser than dense"

        # Sliding window
        window_mask = create_sliding_window_mask(seq_len, window_size=16, global_indices=None)
        window_count = int(mx.sum(window_mask))
        assert window_count < dense_count, "Window should be sparser than dense"

        # BigBird
        bigbird_mask = create_bigbird_mask(seq_len, block_size=32, window_size=16, num_global_tokens=2, num_random_blocks=1)
        bigbird_count = int(mx.sum(bigbird_mask))
        assert bigbird_count < dense_count, "BigBird should be sparser than dense"
