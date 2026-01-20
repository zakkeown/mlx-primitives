"""Stress tests for very long sequences.

Tests sequences longer than typical use cases (16K+) to validate:
1. Multi-block kernel correctness at scale
2. Numerical stability over long sequences
3. Memory efficiency for large tensors
"""

import numpy as np
import pytest

import mlx.core as mx

from mlx_primitives import associative_scan, flash_attention
from tests.reference import ssm_scan_sequential as np_ssm_scan


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    mx.eval(x)
    return np.array(x)


class TestLongSequenceScan:
    """Stress tests for scan operations with very long sequences."""

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", [16384, 32768])
    def test_cumsum_very_long(self, seq_len: int) -> None:
        """Test cumsum on very long sequences (16K-32K)."""
        np.random.seed(42)
        x_np = np.random.randn(1, seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(mx.array(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        # Relaxed tolerance for very long sequences
        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_cumsum_65536(self) -> None:
        """Test cumsum at 65K sequence length (64 blocks)."""
        np.random.seed(42)
        seq_len = 65536
        x_np = np.random.randn(seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(mx.array(x_np), operator="add"))
        np_out = np.cumsum(x_np)

        # Check first/last elements and overall shape
        assert mlx_out.shape == np_out.shape
        np.testing.assert_allclose(mlx_out[:100], np_out[:100], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(mlx_out[-100:], np_out[-100:], rtol=1e-2, atol=1e-2)

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", [16384, 32768])
    def test_ssm_scan_very_long(self, seq_len: int) -> None:
        """Test SSM scan on very long sequences (16K-32K)."""
        np.random.seed(42)
        batch, state = 1, 4

        # Use high decay values to limit numerical drift
        A_np = np.random.uniform(0.95, 0.99, (batch, seq_len, state)).astype(np.float32)
        x_np = np.random.randn(batch, seq_len, state).astype(np.float32) * 0.1

        mlx_out = to_numpy(associative_scan(
            mx.array(x_np), operator="ssm", A=mx.array(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, x_np)

        # Very relaxed tolerance for extremely long sequences
        np.testing.assert_allclose(mlx_out, np_out, rtol=0.15, atol=0.15)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_ssm_scan_with_varying_decay(self) -> None:
        """Test SSM scan stability with varying decay rates over long sequences."""
        np.random.seed(42)
        seq_len = 8192
        batch, state = 2, 8

        # Create varying decay pattern (some positions have high decay, some low)
        A_np = np.zeros((batch, seq_len, state), dtype=np.float32)
        for i in range(seq_len):
            # Alternate between high and low decay
            decay = 0.99 if (i // 64) % 2 == 0 else 0.5
            A_np[:, i, :] = decay

        x_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            mx.array(x_np), operator="ssm", A=mx.array(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, x_np)

        # Check that results are reasonable (not NaN/Inf)
        assert np.all(np.isfinite(mlx_out)), "MLX output contains NaN or Inf"
        np.testing.assert_allclose(mlx_out, np_out, rtol=0.1, atol=0.1)


class TestLongSequenceAttention:
    """Stress tests for attention with very long sequences."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_flash_attention_4096(self) -> None:
        """Test flash attention with 4096 sequence length."""
        np.random.seed(42)
        batch, seq, heads, dim = 1, 4096, 4, 32

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        result = flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=True, use_metal=False
        )
        mx.eval(result)

        assert result.shape == (batch, seq, heads, dim)
        assert np.all(np.isfinite(to_numpy(result))), "Result contains NaN or Inf"

    @pytest.mark.stress
    @pytest.mark.slow
    def test_flash_attention_8192(self) -> None:
        """Test flash attention with 8192 sequence length."""
        np.random.seed(42)
        batch, seq, heads, dim = 1, 8192, 2, 32

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        result = flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=True, use_metal=False
        )
        mx.eval(result)

        assert result.shape == (batch, seq, heads, dim)
        assert np.all(np.isfinite(to_numpy(result))), "Result contains NaN or Inf"

    @pytest.mark.stress
    @pytest.mark.slow
    def test_flash_attention_numerical_stability(self) -> None:
        """Test flash attention numerical stability with extreme values."""
        np.random.seed(42)
        batch, seq, heads, dim = 1, 2048, 4, 64

        # Create inputs with some extreme values (testing online softmax stability)
        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        # Add some large values that would cause overflow with naive softmax
        q_np[:, 0, :, :] = 10.0
        k_np[:, 0, :, :] = 10.0

        result = flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=True, use_metal=False
        )
        mx.eval(result)

        out_np = to_numpy(result)
        assert np.all(np.isfinite(out_np)), "Result contains NaN or Inf with extreme inputs"


class TestBatchedLongSequences:
    """Stress tests for batched operations with long sequences."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_batched_cumsum_long_sequence(self) -> None:
        """Test batched cumsum with long sequences."""
        np.random.seed(42)
        batch_size = 16
        seq_len = 4096

        x_np = np.random.randn(batch_size, seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(mx.array(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_batched_ssm_scan_long_sequence(self) -> None:
        """Test batched SSM scan with long sequences."""
        np.random.seed(42)
        batch_size = 4
        seq_len = 4096
        state_dim = 16

        A_np = np.random.uniform(0.9, 0.99, (batch_size, seq_len, state_dim)).astype(np.float32)
        x_np = np.random.randn(batch_size, seq_len, state_dim).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            mx.array(x_np), operator="ssm", A=mx.array(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=5e-2, atol=5e-2)


class TestRepeatedOperations:
    """Stress tests for repeated operations (stability over time)."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_repeated_scan_stability(self) -> None:
        """Test that repeated scan operations produce consistent results."""
        np.random.seed(42)
        x_np = np.random.randn(4, 2048).astype(np.float32)
        x = mx.array(x_np)

        results = []
        for _ in range(10):
            result = associative_scan(x, operator="add", axis=-1)
            mx.eval(result)
            results.append(to_numpy(result))

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    @pytest.mark.stress
    @pytest.mark.slow
    def test_repeated_attention_stability(self) -> None:
        """Test that repeated attention operations produce consistent results."""
        np.random.seed(42)
        batch, seq, heads, dim = 2, 1024, 8, 64

        q = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))
        k = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))
        v = mx.array(np.random.randn(batch, seq, heads, dim).astype(np.float32))

        results = []
        for _ in range(10):
            result = flash_attention(q, k, v, causal=True, use_metal=False)
            mx.eval(result)
            results.append(to_numpy(result))

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
