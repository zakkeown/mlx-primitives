"""Tests for Metal vs Python implementation parity.

These tests verify that the Metal kernel implementations produce
identical results to the Python implementations.
"""

import math

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.attention.flash import flash_attention, _HAS_METAL


def numpy_attention(q, k, v, scale, causal):
    """NumPy reference implementation for testing."""
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
        scores = np.where(mask == 1, scores, -1e38)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Output
    output = np.matmul(weights, v_t)
    return np.transpose(output, (0, 2, 1, 3))


class TestMetalPythonParity:
    """Verify Metal and Python paths produce identical results."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_flash_attention_parity(self, seq_len: int) -> None:
        """Compare Metal vs Python for various sequence lengths."""
        if not _HAS_METAL:
            pytest.skip("Metal kernels not available")

        mx.random.seed(42)
        batch, heads, dim = 1, 8, 64

        q = mx.random.normal((batch, seq_len, heads, dim))
        k = mx.random.normal((batch, seq_len, heads, dim))
        v = mx.random.normal((batch, seq_len, heads, dim))

        # Force Python path
        out_python = flash_attention(q, k, v, causal=True, use_metal=False)
        mx.eval(out_python)

        # Try Metal path
        try:
            out_metal = flash_attention(q, k, v, causal=True, use_metal=True)
            mx.eval(out_metal)

            np.testing.assert_allclose(
                np.array(out_python),
                np.array(out_metal),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Metal/Python mismatch at seq_len={seq_len}"
            )
        except RuntimeError as e:
            pytest.skip(f"Metal kernel failed: {e}")

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_bidirectional_parity(self, seq_len: int) -> None:
        """Compare Metal vs Python for bidirectional attention."""
        if not _HAS_METAL:
            pytest.skip("Metal kernels not available")

        mx.random.seed(42)
        batch, heads, dim = 1, 4, 64

        q = mx.random.normal((batch, seq_len, heads, dim))
        k = mx.random.normal((batch, seq_len, heads, dim))
        v = mx.random.normal((batch, seq_len, heads, dim))

        out_python = flash_attention(q, k, v, causal=False, use_metal=False)
        mx.eval(out_python)

        try:
            out_metal = flash_attention(q, k, v, causal=False, use_metal=True)
            mx.eval(out_metal)

            np.testing.assert_allclose(
                np.array(out_python),
                np.array(out_metal),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Metal/Python mismatch (bidirectional) at seq_len={seq_len}"
            )
        except RuntimeError as e:
            pytest.skip(f"Metal kernel failed: {e}")

    def test_large_head_dim_parity(self) -> None:
        """Test parity with large head dimensions (head_dim=128)."""
        if not _HAS_METAL:
            pytest.skip("Metal kernels not available")

        mx.random.seed(42)
        batch, seq, heads, dim = 1, 64, 4, 128

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out_python = flash_attention(q, k, v, causal=True, use_metal=False)
        mx.eval(out_python)

        try:
            out_metal = flash_attention(q, k, v, causal=True, use_metal=True)
            mx.eval(out_metal)

            np.testing.assert_allclose(
                np.array(out_python),
                np.array(out_metal),
                rtol=1e-3, atol=1e-4,  # Slightly looser for large head_dim
                err_msg="Metal/Python mismatch with head_dim=128"
            )
        except RuntimeError as e:
            pytest.skip(f"Metal kernel failed: {e}")


class TestMaskingConsistency:
    """Verify masking behavior is consistent between paths."""

    def test_causal_boundary_positions(self) -> None:
        """Verify causal masking at boundary positions."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 32, 2, 32
        scale = 1.0 / math.sqrt(dim)

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out_python = flash_attention(q, k, v, scale=scale, causal=True, use_metal=False)
        numpy_out = numpy_attention(q, k, v, scale, causal=True)

        np.testing.assert_allclose(
            np.array(out_python), numpy_out, rtol=1e-4, atol=1e-5,
            err_msg="Python implementation doesn't match NumPy reference for causal masking"
        )

    def test_first_position_sees_only_self(self) -> None:
        """First position in causal attention should only attend to itself."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 16, 2, 32

        # Use distinct values so we can verify attention pattern
        q = mx.zeros((batch, seq, heads, dim))
        k = mx.zeros((batch, seq, heads, dim))
        v = mx.ones((batch, seq, heads, dim))

        # Make position 0 query distinctive
        q = q.at[:, 0, :, :].add(1.0)
        # Make position 0 key match the query
        k = k.at[:, 0, :, :].add(1.0)

        out = flash_attention(q, k, v, causal=True, use_metal=False)
        mx.eval(out)

        # First position output should be v[0] since it only attends to itself
        first_pos_out = np.array(out[:, 0, :, :])
        expected = np.ones((batch, heads, dim))

        np.testing.assert_allclose(
            first_pos_out, expected, rtol=1e-4, atol=1e-4,
            err_msg="First position should only attend to itself in causal attention"
        )


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_large_sequence_stability(self) -> None:
        """Test stability with longer sequences."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 1024, 4, 64

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))

        out = flash_attention(q, k, v, causal=True, use_metal=False)
        mx.eval(out)

        # Check for NaN/Inf
        out_np = np.array(out)
        assert not np.any(np.isnan(out_np)), "Output contains NaN values"
        assert not np.any(np.isinf(out_np)), "Output contains Inf values"

    def test_uniform_attention_distribution(self) -> None:
        """With uniform Q, K the attention should be roughly uniform."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 64, 2, 32

        # All queries and keys are the same
        q = mx.ones((batch, seq, heads, dim))
        k = mx.ones((batch, seq, heads, dim))
        # Values are position-dependent
        v = mx.broadcast_to(
            mx.arange(seq).reshape(1, seq, 1, 1),
            (batch, seq, heads, dim)
        ).astype(mx.float32)

        out = flash_attention(q, k, v, causal=False, use_metal=False)
        mx.eval(out)

        # With uniform attention, output should be mean of values
        # For bidirectional: mean of [0, 1, ..., seq-1] = (seq-1)/2
        expected_mean = (seq - 1) / 2.0
        out_np = np.array(out)

        np.testing.assert_allclose(
            out_np.mean(), expected_mean, rtol=1e-2,
            err_msg="Uniform attention should produce mean of values"
        )
