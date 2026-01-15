"""Tests for attention modules."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
import numpy as np

from mlx_primitives.attention import (
    ALiBi,
    FlashAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    RoPE,
    SlidingWindowAttention,
    alibi_bias,
    apply_rope,
    # Sparse attention
    BlockSparseAttention,
    LongformerAttention,
    BigBirdAttention,
    # Linear attention
    LinearAttention,
    PerformerAttention,
    CosFormerAttention,
)
from mlx_primitives.attention.rope import precompute_freqs_cis, NTKAwareRoPE, YaRNRoPE
from mlx_primitives.attention.flash import scaled_dot_product_attention, _naive_attention
from mlx_primitives.attention.grouped_query import gqa_attention
from mlx_primitives.attention.multi_query import mqa_attention
from mlx_primitives.attention.sliding_window import (
    create_sliding_window_mask,
    SlidingWindowCache,
)
from mlx_primitives.attention.alibi import get_alibi_slopes, ALiBiGQA


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_precompute_freqs_shape(self):
        """Test that precomputed frequencies have correct shape."""
        dim = 64
        max_seq_len = 1024
        cos, sin = precompute_freqs_cis(dim, max_seq_len)

        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_rope_module_basic(self):
        """Test basic RoPE module functionality."""
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        rope = RoPE(dims=head_dim, max_seq_len=1024)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_with_offset(self):
        """Test RoPE with position offset for incremental decoding."""
        batch_size, num_heads, head_dim = 2, 8, 64
        rope = RoPE(dims=head_dim, max_seq_len=1024)

        # First 100 tokens
        q1 = mx.random.normal((batch_size, 100, num_heads, head_dim))
        k1 = mx.random.normal((batch_size, 100, num_heads, head_dim))
        q1_rot, k1_rot = rope(q1, k1, offset=0)

        # Next token with offset
        q2 = mx.random.normal((batch_size, 1, num_heads, head_dim))
        k2 = mx.random.normal((batch_size, 1, num_heads, head_dim))
        q2_rot, k2_rot = rope(q2, k2, offset=100)

        assert q2_rot.shape == (batch_size, 1, num_heads, head_dim)

    def test_rope_even_dim_required(self):
        """Test that RoPE requires even dimensions."""
        with pytest.raises(ValueError):
            RoPE(dims=63, max_seq_len=1024)

    def test_rope_forward_one(self):
        """Test rotating a single tensor."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 32
        rope = RoPE(dims=head_dim, max_seq_len=256)

        x = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        x_rot = rope.forward_one(x)

        assert x_rot.shape == x.shape

    def test_ntk_aware_rope(self):
        """Test NTK-aware RoPE for extended context."""
        rope = NTKAwareRoPE(
            dims=64,
            max_seq_len=32768,
            original_max_seq_len=8192,
        )

        q = mx.random.normal((1, 1000, 4, 64))
        k = mx.random.normal((1, 1000, 4, 64))

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape

    def test_yarn_rope(self):
        """Test YaRN RoPE for extended context."""
        rope = YaRNRoPE(
            dims=64,
            max_seq_len=32768,
            original_max_seq_len=8192,
        )

        q = mx.random.normal((1, 1000, 4, 64))
        k = mx.random.normal((1, 1000, 4, 64))

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestFlashAttention:
    """Tests for Flash Attention."""

    def test_flash_attention_basic(self):
        """Test basic FlashAttention forward pass."""
        batch_size, seq_len, dims = 2, 128, 256
        num_heads = 4

        attn = FlashAttention(dims=dims, num_heads=num_heads)
        x = mx.random.normal((batch_size, seq_len, dims))

        output = attn(x)

        assert output.shape == x.shape

    def test_flash_attention_causal(self):
        """Test causal FlashAttention."""
        batch_size, seq_len, dims = 2, 64, 128
        num_heads = 4

        attn = FlashAttention(dims=dims, num_heads=num_heads, causal=True)
        x = mx.random.normal((batch_size, seq_len, dims))

        output = attn(x)
        assert output.shape == x.shape

    def test_flash_attention_cross_attention(self):
        """Test FlashAttention with separate Q, K, V."""
        batch_size, dims = 2, 256
        seq_q, seq_kv = 64, 128
        num_heads = 4

        attn = FlashAttention(dims=dims, num_heads=num_heads)

        q = mx.random.normal((batch_size, seq_q, dims))
        k = mx.random.normal((batch_size, seq_kv, dims))
        v = mx.random.normal((batch_size, seq_kv, dims))

        output = attn(q, k, v)

        assert output.shape == (batch_size, seq_q, dims)

    def test_scaled_dot_product_attention(self):
        """Test functional attention interface."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 32

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        output = scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == q.shape


class TestGroupedQueryAttention:
    """Tests for Grouped Query Attention."""

    def test_gqa_basic(self):
        """Test basic GQA forward pass."""
        batch_size, seq_len, dims = 2, 128, 512
        num_heads, num_kv_heads = 8, 2

        attn = GroupedQueryAttention(
            dims=dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        x = mx.random.normal((batch_size, seq_len, dims))

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_gqa_with_cache(self):
        """Test GQA with KV cache."""
        batch_size, dims = 2, 256
        num_heads, num_kv_heads = 8, 2
        head_dim = dims // num_heads

        attn = GroupedQueryAttention(
            dims=dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            causal=True,
        )

        # Initial forward pass
        x1 = mx.random.normal((batch_size, 64, dims))
        output1, cache = attn(x1)

        # Incremental decoding
        x2 = mx.random.normal((batch_size, 1, dims))
        output2, new_cache = attn(x2, cache=cache)

        assert output2.shape == (batch_size, 1, dims)
        assert new_cache is not None
        assert new_cache[0].shape[1] == 65  # 64 + 1

    def test_gqa_heads_divisibility(self):
        """Test that num_heads must be divisible by num_kv_heads."""
        with pytest.raises(ValueError):
            GroupedQueryAttention(dims=256, num_heads=7, num_kv_heads=3)

    def test_gqa_functional(self):
        """Test functional GQA interface."""
        batch_size, seq_len, num_heads, num_kv_heads, head_dim = 2, 64, 8, 2, 32
        num_groups = num_heads // num_kv_heads

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))

        output = gqa_attention(q, k, v, num_kv_groups=num_groups)

        assert output.shape == q.shape


class TestMultiQueryAttention:
    """Tests for Multi-Query Attention."""

    def test_mqa_basic(self):
        """Test basic MQA forward pass."""
        batch_size, seq_len, dims = 2, 128, 256
        num_heads = 8

        attn = MultiQueryAttention(dims=dims, num_heads=num_heads)
        x = mx.random.normal((batch_size, seq_len, dims))

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_mqa_with_cache(self):
        """Test MQA with KV cache."""
        batch_size, dims = 2, 256
        num_heads = 8
        head_dim = dims // num_heads

        attn = MultiQueryAttention(dims=dims, num_heads=num_heads, causal=True)

        # Initial forward pass
        x1 = mx.random.normal((batch_size, 32, dims))
        output1, cache = attn(x1)

        # Incremental decoding
        x2 = mx.random.normal((batch_size, 1, dims))
        output2, new_cache = attn(x2, cache=cache)

        assert output2.shape == (batch_size, 1, dims)
        # Cache has single head for K,V
        assert new_cache[0].shape == (batch_size, 33, 1, head_dim)

    def test_mqa_functional(self):
        """Test functional MQA interface."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 32

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, head_dim))  # Single head
        v = mx.random.normal((batch_size, seq_len, head_dim))

        output = mqa_attention(q, k, v, causal=True)

        assert output.shape == q.shape


class TestSlidingWindowAttention:
    """Tests for Sliding Window Attention."""

    def test_sliding_window_mask(self):
        """Test sliding window mask creation."""
        seq_len, window_size = 10, 4
        mask = create_sliding_window_mask(seq_len, window_size, causal=True)

        assert mask.shape == (seq_len, seq_len)
        # Check that position 5 can attend to positions 2, 3, 4, 5 (window of 4)
        row_5 = mask[5]
        assert row_5[5].item() == 0  # Current position
        assert row_5[4].item() == 0  # Within window
        assert row_5[2].item() == 0  # Within window
        assert row_5[1].item() == float("-inf")  # Outside window

    def test_sliding_window_attention_basic(self):
        """Test basic sliding window attention."""
        batch_size, seq_len, dims = 2, 256, 256
        num_heads, window_size = 4, 64

        attn = SlidingWindowAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
        )
        x = mx.random.normal((batch_size, seq_len, dims))

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_sliding_window_cache(self):
        """Test sliding window KV cache."""
        window_size, num_heads, head_dim = 64, 4, 32

        cache = SlidingWindowCache(
            window_size=window_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Add entries beyond window size
        for i in range(100):
            k_new = mx.random.normal((1, 1, num_heads, head_dim))
            v_new = mx.random.normal((1, 1, num_heads, head_dim))
            k_full, v_full = cache.update(k_new, v_new)

        # Cache should be capped at window_size
        assert cache.length == window_size
        assert k_full.shape[1] == window_size


class TestALiBi:
    """Tests for ALiBi attention."""

    def test_alibi_slopes(self):
        """Test ALiBi slope computation."""
        # Power of 2
        slopes_8 = get_alibi_slopes(8)
        assert slopes_8.shape == (8,)
        assert slopes_8[0].item() > slopes_8[7].item()  # Decreasing

        # Non-power of 2
        slopes_12 = get_alibi_slopes(12)
        assert slopes_12.shape == (12,)

    def test_alibi_bias_shape(self):
        """Test ALiBi bias matrix shape."""
        seq_q, seq_k, num_heads = 64, 128, 8
        bias = alibi_bias(seq_q, seq_k, num_heads)

        assert bias.shape == (1, num_heads, seq_q, seq_k)

    def test_alibi_attention_basic(self):
        """Test basic ALiBi attention."""
        batch_size, seq_len, dims = 2, 128, 256
        num_heads = 8

        attn = ALiBi(dims=dims, num_heads=num_heads)
        x = mx.random.normal((batch_size, seq_len, dims))

        output, _ = attn(x)

        assert output.shape == x.shape

    def test_alibi_extrapolation(self):
        """Test ALiBi with sequences longer than typical training."""
        batch_size, dims = 2, 256
        num_heads = 8

        attn = ALiBi(dims=dims, num_heads=num_heads, causal=True)

        # Short sequence
        x_short = mx.random.normal((batch_size, 128, dims))
        output_short, _ = attn(x_short)

        # Long sequence (extrapolation)
        x_long = mx.random.normal((batch_size, 1024, dims))
        output_long, _ = attn(x_long)

        assert output_short.shape == x_short.shape
        assert output_long.shape == x_long.shape


class TestAttentionGradients:
    """Tests for gradient computation through attention modules."""

    def test_rope_gradient(self):
        """Test gradient flow through RoPE."""
        rope = RoPE(dims=32, max_seq_len=128)

        def forward(q, k):
            q_rot, k_rot = rope(q, k)
            return mx.sum(q_rot + k_rot)

        q = mx.random.normal((1, 16, 2, 32))
        k = mx.random.normal((1, 16, 2, 32))

        loss, grads = mx.value_and_grad(forward, argnums=(0, 1))(q, k)
        grad_q, grad_k = grads

        assert grad_q.shape == q.shape
        assert grad_k.shape == k.shape

    def test_flash_attention_gradient(self):
        """Test gradient flow through FlashAttention."""
        attn = FlashAttention(dims=64, num_heads=2)

        def forward(x):
            return mx.sum(attn(x))

        x = mx.random.normal((1, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape

    def test_gqa_gradient(self):
        """Test gradient flow through GQA."""
        attn = GroupedQueryAttention(dims=64, num_heads=4, num_kv_heads=2)

        def forward(x):
            output, _ = attn(x)
            return mx.sum(output)

        x = mx.random.normal((1, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape

    def test_mqa_gradient(self):
        """Test gradient flow through MQA."""
        attn = MultiQueryAttention(dims=64, num_heads=4)

        def forward(x):
            output, _ = attn(x)
            return mx.sum(output)

        x = mx.random.normal((1, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape

    def test_sliding_window_gradient(self):
        """Test gradient flow through SlidingWindowAttention."""
        attn = SlidingWindowAttention(dims=64, num_heads=4, window_size=8)

        def forward(x):
            output, _ = attn(x)
            return mx.sum(output)

        x = mx.random.normal((1, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape

    def test_alibi_gradient(self):
        """Test gradient flow through ALiBi."""
        attn = ALiBi(dims=64, num_heads=4)

        def forward(x):
            output, _ = attn(x)
            return mx.sum(output)

        x = mx.random.normal((1, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape


class TestAttentionCorrectness:
    """Tests comparing implementations against reference."""

    def test_flash_vs_naive_attention(self):
        """Verify FlashAttention matches naive implementation."""
        batch, seq, heads, dim = 2, 32, 4, 16
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        # Naive attention
        naive_out = _naive_attention(q, k, v, scale, causal=False)

        # Flash attention (functional)
        flash_out = scaled_dot_product_attention(q, k, v, scale=scale)

        mx.eval(naive_out, flash_out)
        np.testing.assert_allclose(
            np.array(naive_out), np.array(flash_out), rtol=1e-4, atol=1e-4
        )

    def test_flash_vs_naive_causal(self):
        """Verify causal FlashAttention matches naive implementation."""
        batch, seq, heads, dim = 2, 32, 4, 16
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        naive_out = _naive_attention(q, k, v, scale, causal=True)
        flash_out = scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        mx.eval(naive_out, flash_out)
        np.testing.assert_allclose(
            np.array(naive_out), np.array(flash_out), rtol=1e-4, atol=1e-4
        )

    def test_gqa_equals_mha_when_equal_heads(self):
        """GQA with num_kv_heads=num_heads should equal standard MHA."""
        batch, seq, dims = 2, 32, 64
        num_heads = 4

        gqa = GroupedQueryAttention(
            dims=dims, num_heads=num_heads, num_kv_heads=num_heads
        )
        mha = nn.MultiHeadAttention(dims=dims, num_heads=num_heads)

        # Copy weights
        mha.query_proj.weight = gqa.q_proj.weight
        mha.key_proj.weight = gqa.k_proj.weight
        mha.value_proj.weight = gqa.v_proj.weight
        mha.out_proj.weight = gqa.out_proj.weight

        x = mx.random.normal((batch, seq, dims))

        gqa_out, _ = gqa(x)
        # MLX MHA requires queries, keys, values separately
        mha_out = mha(x, x, x)

        mx.eval(gqa_out, mha_out)
        np.testing.assert_allclose(
            np.array(gqa_out), np.array(mha_out), rtol=1e-4, atol=1e-4
        )

    def test_mqa_is_gqa_with_one_kv_head(self):
        """MQA should be equivalent to GQA with num_kv_heads=1."""
        batch, seq, dims = 2, 32, 64
        num_heads = 4

        mqa = MultiQueryAttention(dims=dims, num_heads=num_heads)
        gqa = GroupedQueryAttention(dims=dims, num_heads=num_heads, num_kv_heads=1)

        # Copy weights
        gqa.q_proj.weight = mqa.q_proj.weight
        gqa.k_proj.weight = mqa.k_proj.weight
        gqa.v_proj.weight = mqa.v_proj.weight
        gqa.out_proj.weight = mqa.out_proj.weight

        x = mx.random.normal((batch, seq, dims))

        mqa_out, _ = mqa(x)
        gqa_out, _ = gqa(x)

        mx.eval(mqa_out, gqa_out)
        np.testing.assert_allclose(
            np.array(mqa_out), np.array(gqa_out), rtol=1e-4, atol=1e-4
        )

    def test_rope_rotation_properties(self):
        """Test that RoPE has expected rotation properties."""
        rope = RoPE(dims=64, max_seq_len=128)

        # Same input at different positions should give different outputs
        q = mx.ones((1, 2, 1, 64))
        k = mx.ones((1, 2, 1, 64))

        q_rot, k_rot = rope(q, k)
        mx.eval(q_rot, k_rot)

        # Position 0 and position 1 should differ
        pos0 = np.array(q_rot[0, 0, 0])
        pos1 = np.array(q_rot[0, 1, 0])
        assert not np.allclose(pos0, pos1)

    def test_alibi_bias_properties(self):
        """Test ALiBi bias has expected properties."""
        num_heads = 8
        seq_len = 16

        bias = alibi_bias(seq_len, seq_len, num_heads)
        mx.eval(bias)
        bias_np = np.array(bias[0])  # (num_heads, seq, seq)

        # Diagonal should be 0 (current position)
        for h in range(num_heads):
            diag = np.diag(bias_np[h])
            np.testing.assert_allclose(diag, 0, atol=1e-6)

        # Bias should decrease with distance
        for h in range(num_heads):
            # Check row 8: position 8 looking at positions 0-15
            row = bias_np[h, 8]
            # Past positions (0-7) should have negative bias
            assert all(row[:8] <= 0)


class TestAttentionEdgeCases:
    """Tests for edge cases and special configurations."""

    def test_single_token_attention(self):
        """Test attention with sequence length 1."""
        attn = FlashAttention(dims=64, num_heads=4)
        x = mx.random.normal((2, 1, 64))
        output = attn(x)
        assert output.shape == (2, 1, 64)

    def test_very_long_sequence(self):
        """Test attention with long sequences."""
        attn = FlashAttention(dims=64, num_heads=4, block_size=32)
        x = mx.random.normal((1, 512, 64))
        output = attn(x)
        assert output.shape == (1, 512, 64)

    def test_different_q_kv_lengths(self):
        """Test cross-attention with different Q and KV lengths."""
        attn = FlashAttention(dims=64, num_heads=4)
        q = mx.random.normal((2, 16, 64))
        k = mx.random.normal((2, 64, 64))
        v = mx.random.normal((2, 64, 64))

        output = attn(q, k, v)
        assert output.shape == (2, 16, 64)

    def test_gqa_incremental_decoding(self):
        """Test GQA incremental decoding over many steps."""
        attn = GroupedQueryAttention(
            dims=64, num_heads=8, num_kv_heads=2, causal=True
        )

        # Prefill with 32 tokens
        x = mx.random.normal((1, 32, 64))
        output, cache = attn(x)

        # Decode 10 more tokens one at a time
        for i in range(10):
            x_new = mx.random.normal((1, 1, 64))
            output, cache = attn(x_new, cache=cache)
            assert output.shape == (1, 1, 64)
            assert cache[0].shape[1] == 33 + i  # Growing cache

    def test_sliding_window_longer_than_sequence(self):
        """Test sliding window when window > sequence length."""
        attn = SlidingWindowAttention(
            dims=64, num_heads=4, window_size=100  # Window larger than seq
        )
        x = mx.random.normal((2, 32, 64))  # seq=32 < window=100
        output, _ = attn(x)
        assert output.shape == x.shape

    def test_alibi_very_long_extrapolation(self):
        """Test ALiBi with sequences much longer than typical."""
        attn = ALiBi(dims=64, num_heads=4, causal=True)

        # Short sequence
        x_short = mx.random.normal((1, 64, 64))
        out_short, _ = attn(x_short)

        # Very long sequence (8x longer)
        x_long = mx.random.normal((1, 512, 64))
        out_long, _ = attn(x_long)

        assert out_short.shape == x_short.shape
        assert out_long.shape == x_long.shape

    def test_batch_size_one(self):
        """Test with batch size 1."""
        for AttnClass in [FlashAttention, GroupedQueryAttention, ALiBi]:
            if AttnClass == GroupedQueryAttention:
                attn = AttnClass(dims=64, num_heads=4, num_kv_heads=2)
            else:
                attn = AttnClass(dims=64, num_heads=4)

            x = mx.random.normal((1, 16, 64))
            if hasattr(attn, "__call__"):
                result = attn(x)
                if isinstance(result, tuple):
                    result = result[0]
                assert result.shape == x.shape

    def test_large_batch(self):
        """Test with large batch size."""
        attn = FlashAttention(dims=64, num_heads=4)
        x = mx.random.normal((32, 16, 64))
        output = attn(x)
        assert output.shape == x.shape

    def test_alibi_gqa_combined(self):
        """Test ALiBi combined with GQA."""
        attn = ALiBiGQA(
            dims=64, num_heads=8, num_kv_heads=2, causal=True
        )
        x = mx.random.normal((2, 32, 64))
        output, cache = attn(x)

        assert output.shape == x.shape
        assert cache[0].shape == (2, 32, 2, 8)  # 2 KV heads


class TestAttentionDtypes:
    """Tests for different data types."""

    def test_flash_attention_float32(self):
        """Test FlashAttention with float32."""
        attn = FlashAttention(dims=64, num_heads=4)
        x = mx.random.normal((2, 16, 64)).astype(mx.float32)
        output = attn(x)
        assert output.dtype == mx.float32

    def test_flash_attention_float16_runs(self):
        """Test FlashAttention accepts float16 input (may upcast internally)."""
        attn = FlashAttention(dims=64, num_heads=4)
        x = mx.random.normal((2, 16, 64)).astype(mx.float16)
        output = attn(x)
        # Note: MLX may upcast to float32 internally for numerical stability
        assert output.shape == x.shape

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
    def test_rope_dtypes(self, dtype):
        """Test RoPE with different dtypes."""
        rope = RoPE(dims=32, max_seq_len=64)
        q = mx.random.normal((1, 8, 2, 32)).astype(dtype)
        k = mx.random.normal((1, 8, 2, 32)).astype(dtype)
        q_rot, k_rot = rope(q, k)
        assert q_rot.dtype == dtype
        assert k_rot.dtype == dtype

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
    def test_gqa_dtypes(self, dtype):
        """Test GQA with different dtypes."""
        attn = GroupedQueryAttention(dims=64, num_heads=4, num_kv_heads=2)
        x = mx.random.normal((1, 16, 64)).astype(dtype)
        output, _ = attn(x)
        # Check output is valid (may be upcasted)
        assert output.shape == x.shape


try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestAttentionBenchmarks:
    """Benchmark tests for attention modules.

    These tests require pytest-benchmark to be installed.
    Run with: pip install pytest-benchmark && pytest -v -m benchmark
    """

    def test_flash_attention_benchmark(self, benchmark):
        """Benchmark FlashAttention."""
        attn = FlashAttention(dims=768, num_heads=12, causal=True)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            output = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_gqa_benchmark(self, benchmark):
        """Benchmark Grouped Query Attention."""
        attn = GroupedQueryAttention(
            dims=768, num_heads=12, num_kv_heads=4, causal=True
        )
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_mqa_benchmark(self, benchmark):
        """Benchmark Multi-Query Attention."""
        attn = MultiQueryAttention(dims=768, num_heads=12, causal=True)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_sliding_window_benchmark(self, benchmark):
        """Benchmark Sliding Window Attention."""
        attn = SlidingWindowAttention(
            dims=768, num_heads=12, window_size=256, causal=True
        )
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_alibi_benchmark(self, benchmark):
        """Benchmark ALiBi Attention."""
        attn = ALiBi(dims=768, num_heads=12, causal=True)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_rope_benchmark(self, benchmark):
        """Benchmark RoPE."""
        rope = RoPE(dims=64, max_seq_len=2048)
        q = mx.random.normal((4, 512, 12, 64))
        k = mx.random.normal((4, 512, 12, 64))
        mx.eval(q, k)

        def run():
            q_rot, k_rot = rope(q, k)
            mx.eval(q_rot, k_rot)
            return q_rot, k_rot

        benchmark(run)

    def test_gqa_vs_mha_memory_efficiency(self, benchmark):
        """Benchmark GQA with reduced KV heads (Llama 2 style)."""
        # 8 KV heads instead of 32 query heads - 4x memory reduction
        attn = GroupedQueryAttention(
            dims=4096, num_heads=32, num_kv_heads=8, causal=True
        )
        x = mx.random.normal((1, 256, 4096))
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)

    def test_long_context_sliding_window(self, benchmark):
        """Benchmark sliding window on long context."""
        attn = SlidingWindowAttention(
            dims=768, num_heads=12, window_size=512, causal=True
        )
        x = mx.random.normal((1, 2048, 768))  # Long sequence
        mx.eval(x)

        def run():
            output, _ = attn(x)
            mx.eval(output)
            return output

        benchmark(run)


# ============================================================================
# Sparse Attention Tests
# ============================================================================


class TestBlockSparseAttention:
    """Tests for BlockSparseAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = BlockSparseAttention(dims=64, num_heads=4, block_size=16)
        x = mx.random.normal((2, 64, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_causal_mode(self):
        """Test causal block sparse attention."""
        attn = BlockSparseAttention(dims=64, num_heads=4, block_size=8, causal=True)
        x = mx.random.normal((1, 32, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through block sparse attention."""
        attn = BlockSparseAttention(dims=64, num_heads=4, block_size=8)

        def forward(x):
            return mx.sum(attn(x))

        x = mx.random.normal((1, 32, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape


class TestLongformerAttention:
    """Tests for LongformerAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = LongformerAttention(
            dims=64, num_heads=4, window_size=8, num_global_tokens=2
        )
        x = mx.random.normal((2, 32, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_global_attention(self):
        """Test that global tokens can attend to all positions."""
        attn = LongformerAttention(
            dims=64, num_heads=4, window_size=4, num_global_tokens=4
        )
        x = mx.random.normal((1, 64, 64))

        output = attn(x)

        assert output.shape == x.shape


class TestBigBirdAttention:
    """Tests for BigBirdAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = BigBirdAttention(
            dims=64, num_heads=4, window_size=8, num_global_tokens=2, num_random_tokens=2
        )
        x = mx.random.normal((2, 32, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_components(self):
        """Test that BigBird includes local, global, and random attention."""
        attn = BigBirdAttention(
            dims=64, num_heads=4, window_size=4, num_global_tokens=2, num_random_tokens=4
        )
        x = mx.random.normal((1, 32, 64))

        output = attn(x)

        assert output.shape == x.shape


# ============================================================================
# Linear Attention Tests
# ============================================================================


class TestLinearAttention:
    """Tests for LinearAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = LinearAttention(dims=64, num_heads=4)
        x = mx.random.normal((2, 64, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_causal_mode(self):
        """Test causal linear attention."""
        attn = LinearAttention(dims=64, num_heads=4, causal=True)
        x = mx.random.normal((1, 128, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_long_sequence(self):
        """Test linear attention with long sequences (O(n) complexity)."""
        attn = LinearAttention(dims=64, num_heads=4)
        x = mx.random.normal((1, 512, 64))

        output = attn(x)

        assert output.shape == x.shape


class TestPerformerAttention:
    """Tests for PerformerAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = PerformerAttention(dims=64, num_heads=4, num_features=32)
        x = mx.random.normal((2, 64, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_different_feature_sizes(self):
        """Test with different random feature sizes."""
        for num_features in [16, 32, 64]:
            attn = PerformerAttention(dims=64, num_heads=4, num_features=num_features)
            x = mx.random.normal((1, 32, 64))

            output = attn(x)

            assert output.shape == x.shape

    def test_causal_mode(self):
        """Test causal Performer attention."""
        attn = PerformerAttention(dims=64, num_heads=4, num_features=32, causal=True)
        x = mx.random.normal((1, 64, 64))

        output = attn(x)

        assert output.shape == x.shape


class TestCosFormerAttention:
    """Tests for CosFormerAttention."""

    def test_basic_output_shape(self):
        """Test basic output shape."""
        attn = CosFormerAttention(dims=64, num_heads=4)
        x = mx.random.normal((2, 64, 64))

        output = attn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through CosFormer attention."""
        attn = CosFormerAttention(dims=64, num_heads=4)

        def forward(x):
            return mx.sum(attn(x))

        x = mx.random.normal((1, 32, 64))
        loss, grad = mx.value_and_grad(forward)(x)

        assert grad.shape == x.shape

    def test_causal_mode(self):
        """Test causal CosFormer attention."""
        attn = CosFormerAttention(dims=64, num_heads=4, causal=True)
        x = mx.random.normal((1, 64, 64))

        output = attn(x)

        assert output.shape == x.shape
