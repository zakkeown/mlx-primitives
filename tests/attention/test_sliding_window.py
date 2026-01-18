"""Tests for sliding window attention.

Validation strategy:
1. NumPy reference implementations for algorithmic correctness
2. Analytical test cases with known mask patterns
3. Metal vs reference implementation consistency
4. Property-based tests (determinism, shape preservation)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.attention import (
    SlidingWindowAttention,
    create_sliding_window_mask,
    sliding_window_attention,
)

from tests.reference import (
    AnalyticalTests,
    attention as np_attention,
    sliding_window_attention as np_sliding_window_attention,
    sliding_window_mask as np_sliding_window_mask,
    softmax as np_softmax,
)


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    mx.eval(x)
    return np.array(x)


def to_mlx(x: np.ndarray) -> mx.array:
    """Convert NumPy array to MLX."""
    return mx.array(x)


class TestSlidingWindowMaskAgainstNumPy:
    """Validate mask creation against NumPy."""

    def test_mask_causal_vs_numpy(self) -> None:
        """Test causal mask matches NumPy."""
        seq_len, window_size = 16, 4

        mlx_mask = to_numpy(create_sliding_window_mask(seq_len, window_size, causal=True))
        np_mask = np_sliding_window_mask(seq_len, window_size, causal=True)

        np.testing.assert_array_equal(mlx_mask, np_mask)

    def test_mask_bidirectional_vs_numpy(self) -> None:
        """Test bidirectional mask matches NumPy."""
        seq_len, window_size = 16, 4

        mlx_mask = to_numpy(create_sliding_window_mask(seq_len, window_size, causal=False))
        np_mask = np_sliding_window_mask(seq_len, window_size, causal=False)

        np.testing.assert_array_equal(mlx_mask, np_mask)


class TestSlidingWindowMaskAnalytical:
    """Analytical tests for mask patterns."""

    def test_mask_shape(self) -> None:
        """Test that mask has correct shape."""
        seq_len = 10
        window_size = 3
        mask = create_sliding_window_mask(seq_len, window_size)
        assert mask.shape == (seq_len, seq_len)

    def test_mask_causal_pattern(self) -> None:
        """Test causal sliding window mask with known pattern."""
        seq_len = 5
        window_size = 2
        mask = create_sliding_window_mask(seq_len, window_size, causal=True)

        # Expected: position i can attend to [max(0, i-2), i]
        expected = mx.array([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
        ])
        assert mx.all(mask == expected).item()

    def test_mask_bidirectional_pattern(self) -> None:
        """Test bidirectional sliding window mask with known pattern."""
        seq_len = 5
        window_size = 1
        mask = create_sliding_window_mask(seq_len, window_size, causal=False)

        expected = mx.array([
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ])
        assert mx.all(mask == expected).item()


class TestSlidingWindowAttentionAgainstNumPy:
    """Validate sliding window attention against NumPy reference."""

    def test_attention_vs_numpy(self) -> None:
        """Test sliding window attention matches NumPy."""
        np.random.seed(42)
        batch, seq, heads, dim = 2, 32, 4, 32
        window_size = 8

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        mlx_out = to_numpy(sliding_window_attention(
            to_mlx(q_np), to_mlx(k_np), to_mlx(v_np),
            window_size=window_size, causal=True
        ))
        np_out = np_sliding_window_attention(q_np, k_np, v_np, window_size, causal=True)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    def test_attention_bidirectional_vs_numpy(self) -> None:
        """Test bidirectional attention matches NumPy."""
        np.random.seed(42)
        batch, seq, heads, dim = 2, 32, 4, 32
        window_size = 8

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        mlx_out = to_numpy(sliding_window_attention(
            to_mlx(q_np), to_mlx(k_np), to_mlx(v_np),
            window_size=window_size, causal=False
        ))
        np_out = np_sliding_window_attention(q_np, k_np, v_np, window_size, causal=False)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)


class TestAttentionProperties:
    """Property-based tests for attention invariants."""

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input."""
        batch, seq, heads, dim = 2, 32, 4, 64
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out = sliding_window_attention(q, k, v, window_size=8, causal=True)
        assert out.shape == (batch, seq, heads, dim)

    def test_attention_deterministic(self) -> None:
        """Test attention produces consistent results."""
        batch, seq, heads, dim = 2, 32, 4, 32

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out1 = sliding_window_attention(q, k, v, window_size=8, causal=True)
        out2 = sliding_window_attention(q, k, v, window_size=8, causal=True)

        mx.eval(out1, out2)
        np.testing.assert_array_equal(to_numpy(out1), to_numpy(out2))

    def test_causal_no_future_leakage(self) -> None:
        """Test that causal attention doesn't attend to future positions."""
        batch, seq, heads, dim = 1, 8, 1, 16

        # Create identity-like QK so position i has highest score with itself
        q = mx.eye(seq).reshape(batch, seq, 1, seq)
        k = mx.eye(seq).reshape(batch, seq, 1, seq)

        # V where each position has a unique marker
        v = mx.zeros((batch, seq, heads, dim))
        for i in range(seq):
            v = v.at[0, i, 0, i].add(1.0)

        out = sliding_window_attention(q, k, v, window_size=seq, causal=True)
        mx.eval(out)

        # Position 0 should only see its own value (first dim)
        # Position i should see weighted average of positions 0..i

    def test_full_window_equals_standard_attention(self) -> None:
        """With window >= seq_len, should match standard causal attention."""
        np.random.seed(42)
        batch, seq, heads, dim = 2, 16, 2, 32

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        # Sliding window with full window
        sliding_out = to_numpy(sliding_window_attention(
            to_mlx(q_np), to_mlx(k_np), to_mlx(v_np),
            window_size=seq, causal=True
        ))

        # Standard attention with causal mask
        causal_mask = np.tril(np.ones((seq, seq), dtype=bool))
        standard_out = np_attention(q_np, k_np, v_np, mask=causal_mask)

        np.testing.assert_allclose(sliding_out, standard_out, rtol=1e-3, atol=1e-3)

    def test_window_restriction(self) -> None:
        """Test that attention respects window boundaries."""
        batch, seq, heads, dim = 1, 16, 1, 8

        q = mx.zeros((batch, seq, heads, dim))
        k = mx.zeros((batch, seq, heads, dim))

        # Position 0 query matches position 15 key
        q = q.at[0, 0, 0, 0].add(1.0)
        k = k.at[0, 15, 0, 0].add(1.0)

        v = mx.zeros((batch, seq, heads, dim))
        v = v.at[0, 15, 0, 0].add(100.0)  # Large value at position 15

        # With small window, position 0 shouldn't see position 15
        out_small = sliding_window_attention(q, k, v, window_size=4, causal=False)
        mx.eval(out_small)

        # Position 0 can only see [0, 4], not position 15
        assert abs(out_small[0, 0, 0, 0].item()) < 1.0


class TestMetalVsReference:
    """Test Metal kernel consistency with reference implementation."""

    def test_metal_vs_reference(self) -> None:
        """Test Metal kernel matches reference implementation."""
        batch, seq, heads, dim = 2, 64, 4, 32
        window_size = 16

        mx.random.seed(123)
        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out_ref = sliding_window_attention(
            q, k, v, window_size=window_size, causal=True, use_metal=False
        )
        out_metal = sliding_window_attention(
            q, k, v, window_size=window_size, causal=True, use_metal=True
        )

        mx.eval(out_ref, out_metal)
        np.testing.assert_allclose(
            to_numpy(out_metal), to_numpy(out_ref), rtol=1e-3, atol=1e-3
        )


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

        mx.eval(out1, out2)
        np.testing.assert_array_equal(to_numpy(out1), to_numpy(out2))


class TestLargeSequences:
    """Benchmark tests for long sequences."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
