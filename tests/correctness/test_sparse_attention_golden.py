"""Golden file tests for sparse attention mechanisms.

These tests compare MLX sparse attention implementations against
reference outputs stored in golden files.

Coverage:
- BlockSparseAttention: Block-sparse attention patterns
- LongformerAttention: Sliding window + global attention
- BigBirdAttention: Random + window + global attention

To generate golden files:
    python scripts/validation/generate_all.py --category attention

To run tests:
    pytest tests/correctness/test_sparse_attention_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx

from golden_utils import load_golden, assert_close_golden, golden_exists

from mlx_primitives.attention.sparse import (
    BlockSparseAttention,
    LongformerAttention,
    BigBirdAttention,
)


# =============================================================================
# Block Sparse Attention
# =============================================================================


class TestBlockSparseGolden:
    """Test BlockSparseAttention against golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "block_sparse_b16_small",
            "block_sparse_b32_medium",
            "block_sparse_b64_large",
        ],
    )
    def test_block_sparse_sizes(self, config):
        """BlockSparse output matches reference for various block sizes."""
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        params = metadata["params"]
        block_size = params["block_size"]
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = BlockSparseAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
        )

        # Combine Q, K, V into format expected by module
        batch, seq_len = q.shape[0], q.shape[1]
        x = mx.zeros((batch, seq_len, dims))  # Dummy input for shape

        # Call forward with pre-computed Q, K, V
        out = attn._compute_attention(q, k, v, seq_len)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    def test_block_sparse_causal(self):
        """BlockSparse with causal masking matches reference."""
        config = "block_sparse_b32_causal"
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        params = metadata["params"]
        block_size = params["block_size"]
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = BlockSparseAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
            causal=True,
        )

        batch, seq_len = q.shape[0], q.shape[1]
        out = attn._compute_attention(q, k, v, seq_len)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Longformer Attention
# =============================================================================


class TestLongformerGolden:
    """Test LongformerAttention against golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "longformer_w64_g4_small",
            "longformer_w128_g8_medium",
            "longformer_w256_g16_large",
        ],
    )
    def test_longformer_sizes(self, config):
        """Longformer output matches reference for various window sizes."""
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        params = metadata["params"]
        window_size = params["window_size"]
        num_global_tokens = params.get("num_global_tokens", 4)
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = LongformerAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
        )

        batch, seq_len = q.shape[0], q.shape[1]
        out = attn._compute_attention(q, k, v, seq_len)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    def test_longformer_global_tokens(self):
        """Longformer correctly handles global token attention."""
        config = "longformer_w64_g8_global"
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        global_mask = mx.array(golden.get("global_mask", None))

        metadata = golden["__metadata__"]
        params = metadata["params"]
        window_size = params["window_size"]
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = LongformerAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
        )

        batch, seq_len = q.shape[0], q.shape[1]
        out = attn._compute_attention(q, k, v, seq_len, global_mask=global_mask)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# BigBird Attention
# =============================================================================


class TestBigBirdGolden:
    """Test BigBirdAttention against golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "bigbird_w32_g4_r8_small",
            "bigbird_w64_g8_r16_medium",
            "bigbird_w128_g16_r32_large",
        ],
    )
    def test_bigbird_sizes(self, config):
        """BigBird output matches reference for various configurations."""
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        params = metadata["params"]
        window_size = params["window_size"]
        num_global_tokens = params.get("num_global_tokens", 4)
        num_random_tokens = params.get("num_random_tokens", 8)
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = BigBirdAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            num_random_tokens=num_random_tokens,
        )

        batch, seq_len = q.shape[0], q.shape[1]

        # Set seed for reproducible random attention
        np.random.seed(golden.get("__metadata__", {}).get("seed", 42))

        out = attn._compute_attention(q, k, v, seq_len)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    def test_bigbird_block_sparse_mode(self):
        """BigBird in block-sparse mode matches reference."""
        config = "bigbird_block_sparse"
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        params = metadata["params"]
        block_size = params.get("block_size", 64)
        num_heads = params["num_heads"]
        head_dim = q.shape[-1]
        dims = num_heads * head_dim

        attn = BigBirdAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
            use_block_sparse=True,
        )

        batch, seq_len = q.shape[0], q.shape[1]
        out = attn._compute_attention(q, k, v, seq_len)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestSparseAttentionGradients:
    """Test gradient flow through sparse attention mechanisms."""

    def test_block_sparse_gradient_flow(self):
        """Verify gradients flow through BlockSparseAttention."""
        dims, num_heads, block_size = 256, 8, 32
        batch, seq_len = 2, 128

        attn = BlockSparseAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
        )

        x = mx.random.normal((batch, seq_len, dims))

        def loss_fn(x):
            return mx.sum(attn(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)), "Gradients contain NaN"
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"

    def test_longformer_gradient_flow(self):
        """Verify gradients flow through LongformerAttention."""
        dims, num_heads, window_size = 256, 8, 64
        batch, seq_len = 2, 256

        attn = LongformerAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
        )

        x = mx.random.normal((batch, seq_len, dims))

        def loss_fn(x):
            return mx.sum(attn(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)), "Gradients contain NaN"
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"

    def test_bigbird_gradient_flow(self):
        """Verify gradients flow through BigBirdAttention."""
        dims, num_heads = 256, 8
        batch, seq_len = 2, 256

        attn = BigBirdAttention(
            dims=dims,
            num_heads=num_heads,
        )

        x = mx.random.normal((batch, seq_len, dims))

        def loss_fn(x):
            return mx.sum(attn(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)), "Gradients contain NaN"
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"
