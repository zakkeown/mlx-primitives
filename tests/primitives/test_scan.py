"""Tests for associative scan primitive."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.primitives import associative_scan
from mlx_primitives.primitives.scan import selective_scan


class TestAssociativeScanAdd:
    """Tests for additive associative scan (cumsum)."""

    def test_simple_1d(self) -> None:
        """Test simple 1D cumsum."""
        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = associative_scan(x, operator="add")
        expected = mx.cumsum(x)
        assert mx.allclose(result, expected).item()

    def test_2d_last_axis(self) -> None:
        """Test 2D scan along last axis."""
        x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = associative_scan(x, operator="add", axis=-1)
        expected = mx.cumsum(x, axis=-1)
        assert mx.allclose(result, expected).item()

    def test_batch_scan(self) -> None:
        """Test batched scan."""
        batch_size = 4
        seq_len = 128
        x = mx.random.normal((batch_size, seq_len))
        result = associative_scan(x, operator="add", axis=-1)
        expected = mx.cumsum(x, axis=-1)
        assert mx.allclose(result, expected, rtol=1e-4, atol=1e-4).item()

    def test_3d_tensor(self) -> None:
        """Test 3D tensor scan."""
        x = mx.random.normal((2, 64, 32))
        result = associative_scan(x, operator="add", axis=1)
        expected = mx.cumsum(x, axis=1)
        assert mx.allclose(result, expected, rtol=1e-4, atol=1e-4).item()

    def test_reverse_scan(self) -> None:
        """Test reverse scan."""
        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = associative_scan(x, operator="add", reverse=True)
        # Reverse, cumsum, reverse back
        expected = mx.cumsum(x[::-1])[::-1]
        assert mx.allclose(result, expected).item()

    @pytest.mark.benchmark
    def test_large_sequence(self) -> None:
        """Benchmark large sequence scan."""
        x = mx.random.normal((1, 512))
        result = associative_scan(x, operator="add", axis=-1)
        expected = mx.cumsum(x, axis=-1)
        mx.eval(result)
        mx.eval(expected)
        assert mx.allclose(result, expected, rtol=1e-4, atol=1e-4).item()


class TestAssociativeScanMul:
    """Tests for multiplicative associative scan (cumprod)."""

    def test_simple_1d(self) -> None:
        """Test simple 1D cumprod."""
        x = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = associative_scan(x, operator="mul")
        expected = mx.cumprod(x)
        assert mx.allclose(result, expected).item()


class TestSSMScan:
    """Tests for SSM scan: h[t] = A[t] * h[t-1] + x[t]."""

    def test_simple_recurrence(self) -> None:
        """Test simple SSM recurrence."""
        # A = 0.9 (decay), x = 1.0 (constant input)
        # h[0] = 0.9 * 0 + 1 = 1
        # h[1] = 0.9 * 1 + 1 = 1.9
        # h[2] = 0.9 * 1.9 + 1 = 2.71
        A = mx.full((1, 3, 1), 0.9)
        x = mx.full((1, 3, 1), 1.0)
        result = associative_scan(x, operator="ssm", A=A, axis=1)

        expected = mx.array([[[1.0], [1.9], [2.71]]])
        assert mx.allclose(result, expected, rtol=1e-4, atol=1e-4).item()

    def test_batch_ssm(self) -> None:
        """Test batched SSM scan."""
        batch_size = 4
        seq_len = 64
        d_inner = 32

        A = mx.random.uniform(0.8, 0.99, (batch_size, seq_len, d_inner))
        x = mx.random.normal((batch_size, seq_len, d_inner))

        result = associative_scan(x, operator="ssm", A=A, axis=1)

        # Verify against sequential implementation
        h_seq = _sequential_ssm(A, x)
        assert mx.allclose(result, h_seq, rtol=1e-3, atol=1e-3).item()

    def test_zero_decay(self) -> None:
        """Test with A=0 (no memory)."""
        A = mx.zeros((1, 5, 1))
        x = mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        result = associative_scan(x, operator="ssm", A=A, axis=1)
        # With A=0, h[t] = x[t]
        assert mx.allclose(result, x).item()

    def test_identity_decay(self) -> None:
        """Test with A=1 (perfect memory, cumsum)."""
        A = mx.ones((1, 5, 1))
        x = mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        result = associative_scan(x, operator="ssm", A=A, axis=1)
        # With A=1, h[t] = sum(x[0:t+1])
        expected = mx.cumsum(x, axis=1)
        assert mx.allclose(result, expected).item()


class TestSelectiveScan:
    """Tests for Mamba-style selective scan."""

    def test_selective_scan_shape(self) -> None:
        """Test selective scan output shape."""
        batch_size = 2
        seq_len = 16
        d_inner = 8
        d_state = 4

        x = mx.random.normal((batch_size, seq_len, d_inner))
        delta = mx.random.uniform(0.001, 0.1, (batch_size, seq_len, d_inner))
        A = -mx.random.uniform(0.1, 1.0, (d_inner, d_state))
        B = mx.random.normal((batch_size, seq_len, d_state))
        C = mx.random.normal((batch_size, seq_len, d_state))
        D = mx.random.normal((d_inner,))

        y = selective_scan(x, delta, A, B, C, D, use_metal=False)

        assert y.shape == (batch_size, seq_len, d_inner)

    def test_selective_scan_correctness(self) -> None:
        """Test selective scan against reference implementation."""
        batch_size = 1
        seq_len = 8
        d_inner = 4
        d_state = 2

        mx.random.seed(42)
        x = mx.random.normal((batch_size, seq_len, d_inner))
        delta = mx.random.uniform(0.01, 0.1, (batch_size, seq_len, d_inner))
        A = -mx.random.uniform(0.1, 1.0, (d_inner, d_state))
        B = mx.random.normal((batch_size, seq_len, d_state))
        C = mx.random.normal((batch_size, seq_len, d_state))

        y = selective_scan(x, delta, A, B, C, use_metal=False)

        # Reference sequential implementation
        y_ref = _reference_selective_scan(x, delta, A, B, C)

        assert mx.allclose(y, y_ref, rtol=1e-3, atol=1e-3).item()


def _sequential_ssm(A: mx.array, x: mx.array) -> mx.array:
    """Sequential reference for SSM scan."""
    batch_size, seq_len, d_inner = x.shape
    h_prev = mx.zeros((batch_size, d_inner))
    outputs = []
    for t in range(seq_len):
        h_t = A[:, t, :] * h_prev + x[:, t, :]
        outputs.append(h_t)
        h_prev = h_t
    return mx.stack(outputs, axis=1)


def _reference_selective_scan(
    x: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
) -> mx.array:
    """Reference sequential selective scan."""
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Initialize hidden state
    h = mx.zeros((batch_size, d_inner, d_state))

    outputs = []
    for t in range(seq_len):
        # Discretization
        delta_t = delta[:, t, :]  # (batch, d_inner)
        A_bar = mx.exp(delta_t[:, :, None] * A[None, :, :])  # (batch, d_inner, d_state)

        # SSM step
        B_t = B[:, t, :]  # (batch, d_state)
        x_t = x[:, t, :]  # (batch, d_inner)
        B_x = delta_t[:, :, None] * B_t[:, None, :] * x_t[:, :, None]  # (batch, d_inner, d_state)

        h = A_bar * h + B_x  # (batch, d_inner, d_state)

        # Output
        C_t = C[:, t, :]  # (batch, d_state)
        y_t = mx.sum(C_t[:, None, :] * h, axis=-1)  # (batch, d_inner)
        outputs.append(y_t)

    return mx.stack(outputs, axis=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
