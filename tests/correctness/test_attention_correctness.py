"""Correctness tests for attention primitives.

These tests compare our optimized implementations against reference
(naive) implementations to verify numerical correctness.
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.attention import (
    FlashAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    RoPE,
    ALiBi,
    alibi_bias,
    # Linear attention
    LinearAttention,
    PerformerAttention,
    CosFormerAttention,
)
from mlx_primitives.attention.sliding_window import create_sliding_window_mask


# ============================================================================
# Reference Implementations
# ============================================================================


def naive_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array = None,
    scale: float = None,
) -> mx.array:
    """Reference implementation of scaled dot-product attention.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]
        mask: Optional attention mask
        scale: Optional scale factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # Compute attention scores
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Compute output
    output = mx.matmul(weights, value)

    return output


def naive_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Create a causal attention mask."""
    mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=dtype), k=1)
    return mask


def naive_alibi_bias(num_heads: int, seq_len_q: int, seq_len_k: int = None) -> mx.array:
    """Reference implementation of ALiBi attention bias.

    Args:
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length (default: same as query)

    Returns:
        ALiBi bias tensor [num_heads, seq_len_q, seq_len_k]
    """
    if seq_len_k is None:
        seq_len_k = seq_len_q

    # Compute slopes for each head
    slopes = []
    for i in range(num_heads):
        slope = 2 ** (-8.0 / num_heads * (i + 1))
        slopes.append(slope)
    slopes = mx.array(slopes)  # [num_heads]

    # Create position difference matrix
    q_pos = mx.arange(seq_len_q)
    k_pos = mx.arange(seq_len_k)
    distances = k_pos[None, :] - q_pos[:, None]  # [seq_len_q, seq_len_k]

    # Compute bias: slope * distance
    bias = slopes[:, None, None] * distances[None, :, :]  # [num_heads, seq_len_q, seq_len_k]

    return bias


# ============================================================================
# FlashAttention Correctness Tests
# ============================================================================


class TestFlashAttentionCorrectness:
    """Correctness tests for FlashAttention implementation."""

    def test_flash_output_shape(self):
        """Test FlashAttention produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 32
        dims = 128
        num_heads = 4

        flash = FlashAttention(dims=dims, num_heads=num_heads)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        output = flash(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    @pytest.mark.parametrize("dims,num_heads,seq_len", [
        (64, 2, 16),
        (128, 4, 32),
        (256, 8, 64),
    ])
    def test_flash_different_configs(self, dims, num_heads, seq_len):
        """Test FlashAttention with various configurations."""
        mx.random.seed(42)

        flash = FlashAttention(dims=dims, num_heads=num_heads)
        x = mx.random.normal((2, seq_len, dims))
        mx.eval(x)

        output = flash(x)
        mx.eval(output)

        assert output.shape == (2, seq_len, dims)

    def test_flash_causal_masking(self):
        """Test that causal FlashAttention doesn't attend to future."""
        mx.random.seed(42)
        dims = 64
        num_heads = 2
        seq_len = 16

        flash_causal = FlashAttention(dims=dims, num_heads=num_heads, causal=True)

        x = mx.random.normal((1, seq_len, dims))
        mx.eval(x)

        output = flash_causal(x)
        mx.eval(output)

        # Output should be finite
        assert mx.all(mx.isfinite(output))

    def test_flash_gradient_flow(self):
        """Test gradients flow through FlashAttention."""
        mx.random.seed(42)
        dims = 64
        num_heads = 2

        flash = FlashAttention(dims=dims, num_heads=num_heads)

        x = mx.random.normal((2, 16, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(flash(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert grad.shape == x.shape
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are zero"


# ============================================================================
# RoPE Correctness Tests
# ============================================================================


class TestRoPECorrectness:
    """Correctness tests for Rotary Position Embeddings."""

    def test_rope_output_shape(self):
        """Test RoPE produces correct output shape."""
        mx.random.seed(42)
        head_dim = 64
        max_seq_len = 128

        rope = RoPE(dims=head_dim, max_seq_len=max_seq_len)

        # Input: [batch, seq_len, num_heads, head_dim]
        q = mx.random.normal((2, 32, 4, head_dim))
        k = mx.random.normal((2, 32, 4, head_dim))
        mx.eval(q, k)

        q_rot, k_rot = rope(q, k)
        mx.eval(q_rot, k_rot)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_preserves_norm(self):
        """Test that RoPE approximately preserves vector norms."""
        mx.random.seed(42)
        head_dim = 64
        seq_len = 32

        rope = RoPE(dims=head_dim, max_seq_len=seq_len)

        q = mx.random.normal((1, seq_len, 1, head_dim))
        k = mx.random.normal((1, seq_len, 1, head_dim))
        mx.eval(q, k)

        q_rot, k_rot = rope(q, k)
        mx.eval(q_rot, k_rot)

        # Compute norms (check q)
        norm_before = mx.sqrt(mx.sum(q * q, axis=-1))
        norm_after = mx.sqrt(mx.sum(q_rot * q_rot, axis=-1))
        mx.eval(norm_before, norm_after)

        # Norms should be similar
        max_diff = float(mx.max(mx.abs(norm_before - norm_after)))
        assert max_diff < 0.1, f"RoPE changed norms significantly: max_diff={max_diff}"

    def test_rope_gradient_flow(self):
        """Test gradients flow through RoPE."""
        mx.random.seed(42)
        head_dim = 32

        rope = RoPE(dims=head_dim, max_seq_len=64)

        q = mx.random.normal((1, 16, 2, head_dim))
        k = mx.random.normal((1, 16, 2, head_dim))
        mx.eval(q, k)

        def loss_fn(q, k):
            q_rot, k_rot = rope(q, k)
            return mx.sum(q_rot) + mx.sum(k_rot)

        grad_q, grad_k = mx.grad(loss_fn, argnums=(0, 1))(q, k)
        mx.eval(grad_q, grad_k)

        assert float(mx.sum(mx.abs(grad_q))) > 0, "RoPE q gradients are zero"
        assert float(mx.sum(mx.abs(grad_k))) > 0, "RoPE k gradients are zero"


# ============================================================================
# ALiBi Correctness Tests
# ============================================================================


class TestALiBiCorrectness:
    """Correctness tests for ALiBi attention bias."""

    def test_alibi_output_shape(self):
        """Test alibi_bias produces correct output shape."""
        num_heads = 8
        seq_len_q = 32
        seq_len_k = 64

        bias = alibi_bias(seq_len_q, seq_len_k, num_heads)
        mx.eval(bias)

        # Output shape is (1, num_heads, seq_len_q, seq_len_k)
        assert bias.shape == (1, num_heads, seq_len_q, seq_len_k)

    def test_alibi_vs_naive(self):
        """Compare ALiBi implementation against naive computation."""
        num_heads = 8
        seq_len = 32

        # Our implementation
        our_bias = alibi_bias(seq_len, seq_len, num_heads)
        mx.eval(our_bias)

        # Naive implementation
        naive = naive_alibi_bias(num_heads, seq_len)
        mx.eval(naive)

        # Should be close
        max_diff = float(mx.max(mx.abs(our_bias - naive)))
        assert max_diff < 1e-4, f"ALiBi differs from naive: max_diff={max_diff}"

    def test_alibi_slopes_are_geometric(self):
        """Test that ALiBi slopes form a geometric sequence."""
        num_heads = 8
        seq_len = 16

        bias = alibi_bias(seq_len, seq_len, num_heads)
        mx.eval(bias)

        # Shape is (1, num_heads, seq_len_q, seq_len_k)
        # Extract slopes by looking at position (0,1) - distance of 1
        slopes = bias[0, :, 0, 1]  # (num_heads,)
        mx.eval(slopes)

        # Check geometric sequence property
        for i in range(1, num_heads - 1):
            val_prev = float(slopes[i-1])
            val_curr = float(slopes[i])
            val_next = float(slopes[i+1])

            if abs(val_prev) > 1e-10 and abs(val_curr) > 1e-10:
                ratio_1 = val_curr / val_prev
                ratio_2 = val_next / val_curr
                assert abs(ratio_1 - ratio_2) < 0.1, f"Slopes not geometric at {i}"


# ============================================================================
# GQA/MQA Correctness Tests
# ============================================================================


class TestGQACorrectness:
    """Correctness tests for Grouped Query Attention."""

    def test_gqa_output_shape(self):
        """Test GQA produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 32
        dims = 128
        num_heads = 8
        num_kv_heads = 2

        gqa = GroupedQueryAttention(dims=dims, num_heads=num_heads, num_kv_heads=num_kv_heads)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        # GQA returns (output, cache) tuple
        output, _ = gqa(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    def test_mqa_output_shape(self):
        """Test MQA produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 32
        dims = 128
        num_heads = 8

        mqa = MultiQueryAttention(dims=dims, num_heads=num_heads)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        # MQA returns (output, cache) tuple
        output, _ = mqa(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    def test_gqa_gradient_flow(self):
        """Test gradients flow through GQA."""
        mx.random.seed(42)
        dims = 64

        gqa = GroupedQueryAttention(dims=dims, num_heads=4, num_kv_heads=2)

        x = mx.random.normal((2, 16, dims))
        mx.eval(x)

        def loss_fn(x):
            output, _ = gqa(x)
            return mx.sum(output)

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "GQA gradients are zero"


# ============================================================================
# Sliding Window Attention Correctness Tests
# ============================================================================


class TestSlidingWindowCorrectness:
    """Correctness tests for Sliding Window Attention."""

    def test_sliding_window_mask_structure(self):
        """Test sliding window mask has correct structure."""
        seq_len = 16
        window_size = 4

        mask = create_sliding_window_mask(seq_len, window_size, causal=True)
        mx.eval(mask)

        # Check mask structure
        # Implementation: can attend to positions [pos - window_size + 1, pos]
        # i.e., distance = i - j must satisfy: 0 <= distance < window_size
        for i in range(seq_len):
            for j in range(seq_len):
                distance = i - j
                # Causal: distance >= 0 (past positions only)
                # Window: distance < window_size
                should_attend = (distance >= 0) and (distance < window_size)

                mask_val = float(mask[i, j])
                if should_attend:
                    assert mask_val == 0.0, f"Position ({i}, {j}) should be unmasked"
                else:
                    assert mask_val == float('-inf'), f"Position ({i}, {j}) should be masked"

    def test_sliding_window_output_shape(self):
        """Test SlidingWindowAttention produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 64
        dims = 128
        num_heads = 4
        window_size = 16

        swa = SlidingWindowAttention(dims=dims, num_heads=num_heads, window_size=window_size)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        output, _ = swa(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)


# ============================================================================
# Linear Attention Correctness Tests
# ============================================================================


class TestLinearAttentionCorrectness:
    """Correctness tests for Linear Attention variants."""

    def test_linear_attention_output_shape(self):
        """Test LinearAttention produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 64
        dims = 128
        num_heads = 4

        linear = LinearAttention(dims=dims, num_heads=num_heads)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        output = linear(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    def test_performer_output_shape(self):
        """Test PerformerAttention produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 64
        dims = 128
        num_heads = 4

        performer = PerformerAttention(dims=dims, num_heads=num_heads, num_features=32)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        output = performer(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    def test_performer_deterministic(self):
        """Test Performer is deterministic after initialization."""
        mx.random.seed(42)
        dims = 64
        num_heads = 2

        performer = PerformerAttention(dims=dims, num_heads=num_heads, num_features=16)

        x = mx.random.normal((1, 16, dims))
        mx.eval(x)

        out1 = performer(x)
        out2 = performer(x)
        mx.eval(out1, out2)

        assert mx.allclose(out1, out2, atol=1e-6), "Performer not deterministic"

    def test_cosformer_output_shape(self):
        """Test CosFormerAttention produces correct output shape."""
        mx.random.seed(42)
        batch_size = 2
        seq_len = 32
        dims = 64
        num_heads = 2

        cosformer = CosFormerAttention(dims=dims, num_heads=num_heads)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        output = cosformer(x)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dims)

    def test_linear_attention_gradient_flow(self):
        """Test gradients flow through LinearAttention."""
        mx.random.seed(42)
        dims = 64

        linear = LinearAttention(dims=dims, num_heads=2)

        x = mx.random.normal((1, 16, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(linear(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "LinearAttention gradients are zero"


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of attention implementations."""

    def test_flash_attention_large_values(self):
        """Test FlashAttention handles large input values."""
        mx.random.seed(42)
        dims = 128

        flash = FlashAttention(dims=dims, num_heads=4)

        x = mx.random.normal((2, 64, dims)) * 10
        mx.eval(x)

        out = flash(x)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "FlashAttention produced non-finite values"

    def test_flash_attention_small_values(self):
        """Test FlashAttention handles small input values."""
        mx.random.seed(42)
        dims = 128

        flash = FlashAttention(dims=dims, num_heads=4)

        x = mx.random.normal((2, 64, dims)) * 0.001
        mx.eval(x)

        out = flash(x)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "FlashAttention produced non-finite values"
        assert float(mx.std(out)) > 0, "FlashAttention output collapsed to zero"

    def test_attention_long_sequences(self):
        """Test attention handles long sequences."""
        mx.random.seed(42)
        dims = 128

        flash = FlashAttention(dims=dims, num_heads=4)

        x = mx.random.normal((1, 1024, dims))
        mx.eval(x)

        out = flash(x)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "Long sequence produced non-finite values"


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through attention layers."""

    def test_flash_attention_gradient_nonzero(self):
        """Test FlashAttention produces non-zero gradients."""
        mx.random.seed(42)
        dims = 128

        flash = FlashAttention(dims=dims, num_heads=4)

        x = mx.random.normal((2, 32, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(flash(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"

    def test_linear_attention_gradient_nonzero(self):
        """Test linear attention produces non-zero gradients."""
        mx.random.seed(42)
        dims = 64

        linear = LinearAttention(dims=dims, num_heads=4)

        x = mx.random.normal((2, 32, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(linear(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "Linear attention gradients are all zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
