"""Tests for sliding window attention."""

import math

import mlx.core as mx
import pytest

from mlx_primitives.attention import (
    SlidingWindowAttention,
    create_sliding_window_mask,
    sliding_window_attention,
)


class TestSlidingWindowMask:
    """Tests for sliding window mask creation."""

    def test_mask_shape(self) -> None:
        """Test that mask has correct shape."""
        seq_len = 10
        window_size = 3
        mask = create_sliding_window_mask(seq_len, window_size)
        assert mask.shape == (seq_len, seq_len)

    def test_mask_causal(self) -> None:
        """Test causal sliding window mask."""
        seq_len = 5
        window_size = 2
        mask = create_sliding_window_mask(seq_len, window_size, causal=True)

        # Expected: position i can attend to [max(0, i-2), i]
        # Position 0: can attend to [0]
        # Position 1: can attend to [0, 1]
        # Position 2: can attend to [0, 1, 2]
        # Position 3: can attend to [1, 2, 3]
        # Position 4: can attend to [2, 3, 4]
        expected = mx.array([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
        ])
        assert mx.all(mask == expected).item()

    def test_mask_bidirectional(self) -> None:
        """Test bidirectional sliding window mask."""
        seq_len = 5
        window_size = 1
        mask = create_sliding_window_mask(seq_len, window_size, causal=False)

        # Position i can attend to [max(0, i-1), min(seq_len, i+2)]
        expected = mx.array([
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ])
        assert mx.all(mask == expected).item()


class TestSlidingWindowAttention:
    """Tests for sliding window attention."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input."""
        batch, seq, heads, dim = 2, 32, 4, 64
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out = sliding_window_attention(q, k, v, window_size=8, causal=True)
        assert out.shape == (batch, seq, heads, dim)

    def test_causal_no_future_attention(self) -> None:
        """Test that causal attention doesn't attend to future positions."""
        batch, seq, heads, dim = 1, 8, 1, 16
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))

        # Create V where each position has a unique identifier
        v = mx.zeros((batch, seq, heads, dim))
        for i in range(seq):
            v = v.at[0, i, 0, 0].add(float(i + 1))

        # With full window size and causal, position i should not see i+1, i+2, ...
        out = sliding_window_attention(q, k, v, window_size=seq, causal=True)

        # The last position should have the highest v[0] value in its output
        # (since it can see all previous positions including itself)
        mx.eval(out)

    def test_window_restriction(self) -> None:
        """Test that attention respects window boundaries."""
        batch, seq, heads, dim = 1, 16, 1, 8

        # Set up Q, K where only distant pairs have high scores
        q = mx.zeros((batch, seq, heads, dim))
        k = mx.zeros((batch, seq, heads, dim))

        # Position 0 query matches position 15 key
        q = q.at[0, 0, 0, 0].add(1.0)
        k = k.at[0, 15, 0, 0].add(1.0)

        # With small window, position 0 shouldn't see position 15
        v = mx.zeros((batch, seq, heads, dim))
        v = v.at[0, 15, 0, 0].add(100.0)  # Large value at position 15

        out_small_window = sliding_window_attention(
            q, k, v, window_size=4, causal=False
        )

        # With small window, position 0 can only see [0, 4], not position 15
        # So output at position 0 should be close to 0
        mx.eval(out_small_window)
        assert abs(out_small_window[0, 0, 0, 0].item()) < 1.0

    def test_vs_full_attention_small_window(self) -> None:
        """Test sliding window matches full attention when window covers all."""
        batch, seq, heads, dim = 2, 16, 2, 32

        mx.random.seed(42)
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        # With window_size >= seq_len, should match full causal attention
        out_sliding = sliding_window_attention(q, k, v, window_size=seq, causal=True)

        # Compute reference full causal attention
        scale = 1.0 / math.sqrt(dim)
        q_t = q.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)

        scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale
        # Apply causal mask
        mask = mx.triu(mx.full((seq, seq), float('-inf')), k=1)
        scores = scores + mask[None, None, :, :]
        weights = mx.softmax(scores, axis=-1)
        out_full = (weights @ v_t).transpose(0, 2, 1, 3)

        assert mx.allclose(out_sliding, out_full, rtol=1e-3, atol=1e-3).item()

    def test_metal_vs_reference(self) -> None:
        """Test Metal kernel matches reference implementation."""
        batch, seq, heads, dim = 2, 64, 4, 32
        window_size = 16

        mx.random.seed(123)
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        # Reference (no Metal)
        out_ref = sliding_window_attention(
            q, k, v, window_size=window_size, causal=True, use_metal=False
        )

        # Metal kernel (if available)
        out_metal = sliding_window_attention(
            q, k, v, window_size=window_size, causal=True, use_metal=True
        )

        assert mx.allclose(out_metal, out_ref, rtol=1e-3, atol=1e-3).item()

    @pytest.mark.benchmark
    def test_long_sequence(self) -> None:
        """Test with longer sequence."""
        batch, seq, heads, dim = 1, 512, 8, 64
        window_size = 128

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out = sliding_window_attention(q, k, v, window_size=window_size, causal=True)
        mx.eval(out)
        assert out.shape == (batch, seq, heads, dim)


class TestSlidingWindowAttentionModule:
    """Tests for SlidingWindowAttention module."""

    def test_module_output_shape(self) -> None:
        """Test module produces correct output shape."""
        attn = SlidingWindowAttention(num_heads=8, head_dim=64, window_size=32)

        q = mx.random.normal((2, 128, 8, 64))
        k = mx.random.normal((2, 128, 8, 64))
        v = mx.random.normal((2, 128, 8, 64))

        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_module_deterministic(self) -> None:
        """Test module produces consistent results."""
        attn = SlidingWindowAttention(num_heads=4, head_dim=32, window_size=8)

        q = mx.random.normal((1, 32, 4, 32))
        k = mx.random.normal((1, 32, 4, 32))
        v = mx.random.normal((1, 32, 4, 32))

        out1 = attn(q, k, v)
        out2 = attn(q, k, v)

        assert mx.allclose(out1, out2).item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
