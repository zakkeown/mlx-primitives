"""Tests for associative scan primitive.

Validation strategy:
1. NumPy reference implementations (cumsum, cumprod, sequential SSM)
2. Analytical test cases with known outputs
3. Property-based tests (associativity, identity elements)
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.primitives import associative_scan
from mlx_primitives.primitives.scan import selective_scan

from tests.reference import (
    AnalyticalTests,
    cumsum as np_cumsum,
    cumprod as np_cumprod,
    ssm_scan_sequential as np_ssm_scan,
)


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    mx.eval(x)
    return np.array(x)


def to_mlx(x: np.ndarray) -> mx.array:
    """Convert NumPy array to MLX."""
    return mx.array(x)


class TestCumsumAgainstNumPy:
    """Validate cumulative sum against NumPy."""

    def test_cumsum_1d_vs_numpy(self) -> None:
        """Test 1D cumsum matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)

    def test_cumsum_2d_vs_numpy(self) -> None:
        """Test 2D cumsum along last axis matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(8, 64).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np_cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)

    def test_cumsum_3d_vs_numpy(self) -> None:
        """Test 3D cumsum along middle axis matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(2, 64, 32).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=1))
        np_out = np_cumsum(x_np, axis=1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_cumsum_analytical_values(self) -> None:
        """Test cumsum with known inputs."""
        x_np, expected = AnalyticalTests.cumsum_known_values()

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))

        np.testing.assert_allclose(mlx_out, expected, rtol=1e-6, atol=1e-6)


class TestCumprodAgainstNumPy:
    """Validate cumulative product against NumPy."""

    def test_cumprod_1d_vs_numpy(self) -> None:
        """Test 1D cumprod matches NumPy."""
        # Use small positive values to avoid overflow
        np.random.seed(42)
        x_np = np.random.uniform(0.5, 1.5, 20).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="mul"))
        np_out = np_cumprod(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)


class TestSSMScanAgainstNumPy:
    """Validate SSM scan against NumPy sequential implementation."""

    def test_ssm_scan_vs_numpy(self) -> None:
        """Test SSM scan matches NumPy sequential implementation."""
        np.random.seed(42)
        batch, seq, state = 4, 64, 32

        A_np = np.random.uniform(0.8, 0.99, (batch, seq, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    def test_ssm_analytical_values(self) -> None:
        """Test SSM scan with known inputs."""
        A_np, h_np, expected = AnalyticalTests.ssm_scan_known_values()

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))

        np.testing.assert_allclose(mlx_out, expected, rtol=1e-4, atol=1e-4)


class TestScanProperties:
    """Property-based tests for scan invariants."""

    def test_cumsum_associativity(self) -> None:
        """Cumsum is associative: scan(a ++ b) relates to scan(a) and scan(b)."""
        np.random.seed(42)
        a_np = np.random.randn(10).astype(np.float32)
        b_np = np.random.randn(10).astype(np.float32)
        ab_np = np.concatenate([a_np, b_np])

        # Scan of concatenation
        scan_ab = to_numpy(associative_scan(to_mlx(ab_np), operator="add"))

        # Scan of parts
        scan_a = to_numpy(associative_scan(to_mlx(a_np), operator="add"))
        scan_b = to_numpy(associative_scan(to_mlx(b_np), operator="add"))

        # scan(a ++ b) = scan(a) ++ (last(scan(a)) + scan(b))
        expected = np.concatenate([scan_a, scan_a[-1] + scan_b])

        np.testing.assert_allclose(scan_ab, expected, rtol=1e-5, atol=1e-6)

    def test_cumsum_first_element(self) -> None:
        """First element of cumsum equals first input element."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        result = to_numpy(associative_scan(to_mlx(x_np), operator="add"))

        assert abs(result[0] - x_np[0]) < 1e-6

    def test_cumprod_identity(self) -> None:
        """Cumprod of ones is ones."""
        x = mx.ones(100)
        result = associative_scan(x, operator="mul")
        mx.eval(result)

        expected = np.ones(100)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_ssm_zero_decay_equals_input(self) -> None:
        """With A=0 (no memory), SSM output equals input."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8).astype(np.float32)
        A_np = np.zeros((2, 16, 8), dtype=np.float32)

        result = to_numpy(associative_scan(
            to_mlx(x_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))

        np.testing.assert_allclose(result, x_np, rtol=1e-6, atol=1e-6)

    def test_ssm_full_decay_equals_cumsum(self) -> None:
        """With A=1 (perfect memory), SSM output equals cumsum."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8).astype(np.float32)
        A_np = np.ones((2, 16, 8), dtype=np.float32)

        result = to_numpy(associative_scan(
            to_mlx(x_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        expected = np.cumsum(x_np, axis=1)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestReverseScan:
    """Tests for reverse scan functionality."""

    def test_reverse_cumsum_vs_numpy(self) -> None:
        """Test reverse cumsum matches reversed NumPy cumsum."""
        np.random.seed(42)
        x_np = np.random.randn(50).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", reverse=True))

        # Reverse, cumsum, reverse back
        np_out = np.cumsum(x_np[::-1])[::-1]

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)


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

    def test_selective_scan_vs_reference(self) -> None:
        """Test selective scan against sequential reference."""
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
        y_ref = _reference_selective_scan(x, delta, A, B, C)

        mx.eval(y, y_ref)
        np.testing.assert_allclose(
            to_numpy(y), to_numpy(y_ref), rtol=1e-3, atol=1e-3
        )


class TestLargeSequences:
    """Benchmark tests for large sequences."""

    @pytest.mark.benchmark
    def test_large_sequence_cumsum(self) -> None:
        """Test cumsum on large sequence."""
        np.random.seed(42)
        x_np = np.random.randn(1, 512).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.benchmark
    def test_large_batch_ssm(self) -> None:
        """Test SSM scan on large batch."""
        np.random.seed(42)
        batch, seq, state = 16, 256, 64

        A_np = np.random.uniform(0.9, 0.99, (batch, seq, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq, state).astype(np.float32)

        result = associative_scan(to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1)
        mx.eval(result)

        assert result.shape == (batch, seq, state)


class TestMultiBlockScan:
    """Tests for multi-block scan (sequences > 1024)."""

    @pytest.mark.parametrize("seq_len", [1025, 2048, 4096])
    def test_multiblock_cumsum_vs_numpy(self, seq_len: int) -> None:
        """Test multi-block cumsum matches NumPy for long sequences."""
        np.random.seed(42)
        x_np = np.random.randn(seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("seq_len", [1025, 2048])
    def test_multiblock_batched_cumsum(self, seq_len: int) -> None:
        """Test multi-block cumsum with batched input."""
        np.random.seed(42)
        batch_size = 4
        x_np = np.random.randn(batch_size, seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_multiblock_3d_tensor(self) -> None:
        """Test multi-block scan on 3D tensor."""
        np.random.seed(42)
        x_np = np.random.randn(2, 2048, 16).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=1))
        np_out = np.cumsum(x_np, axis=1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)


class TestBoundaryConditions:
    """Tests for boundary conditions around block size limits."""

    @pytest.mark.parametrize("seq_len", [1, 32, 33, 1023, 1024, 1025])
    def test_boundary_cumsum(self, seq_len: int) -> None:
        """Test cumsum at boundary sequence lengths."""
        np.random.seed(42)
        x_np = np.random.randn(seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024])
    def test_power_of_2_cumsum(self, seq_len: int) -> None:
        """Test cumsum at power-of-2 sequence lengths."""
        np.random.seed(42)
        x_np = np.random.randn(seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-5)

    def test_single_element(self) -> None:
        """Test scan of single element."""
        x = mx.array([3.14])
        result = associative_scan(x, operator="add")
        mx.eval(result)

        np.testing.assert_allclose(to_numpy(result), [3.14], rtol=1e-6)

    def test_two_elements(self) -> None:
        """Test scan of two elements."""
        x = mx.array([1.0, 2.0])
        result = associative_scan(x, operator="add")
        mx.eval(result)

        np.testing.assert_allclose(to_numpy(result), [1.0, 3.0], rtol=1e-6)


class TestSIMDOptimization:
    """Tests specifically for SIMD-optimized kernels."""

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024])
    def test_simd_cumsum_correctness(self, seq_len: int) -> None:
        """Test SIMD-optimized cumsum correctness at various sizes."""
        np.random.seed(42)
        x_np = np.random.randn(8, seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_simd_warp_boundary(self) -> None:
        """Test at SIMD warp boundary (32 elements)."""
        np.random.seed(42)
        # Exactly 32 elements (one warp)
        x_np = np.random.randn(32).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-5)

    def test_simd_multiple_warps(self) -> None:
        """Test with multiple complete warps."""
        np.random.seed(42)
        # 128 elements (4 warps)
        x_np = np.random.randn(128).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add"))
        np_out = np_cumsum(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-5)


class TestSSMScanOptimization:
    """Tests for SIMD-optimized SSM scan."""

    @pytest.mark.parametrize("seq_len", [32, 64, 256, 512, 1024])
    def test_ssm_simd_correctness(self, seq_len: int) -> None:
        """Test SIMD-optimized SSM scan correctness."""
        np.random.seed(42)
        batch, state = 4, 32

        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    def test_ssm_small_state(self) -> None:
        """Test SSM scan with small state dimension."""
        np.random.seed(42)
        batch, seq, state = 2, 256, 4

        A_np = np.random.uniform(0.9, 0.99, (batch, seq, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)

    def test_ssm_large_state(self) -> None:
        """Test SSM scan with large state dimension."""
        np.random.seed(42)
        batch, seq, state = 2, 128, 256

        A_np = np.random.uniform(0.9, 0.99, (batch, seq, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-3, atol=1e-3)


class TestMultiBlockSSMScan:
    """Tests for multi-block SSM scan (sequences > 1024)."""

    @pytest.mark.parametrize("seq_len", [1025, 2048, 4096])
    def test_multiblock_ssm_vs_numpy(self, seq_len: int) -> None:
        """Test multi-block SSM scan matches NumPy for long sequences."""
        np.random.seed(42)
        batch, state = 2, 16

        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        # Slightly relaxed tolerance for long sequences due to accumulating numerical error
        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-2, atol=1e-2)

    def test_multiblock_ssm_vs_numpy_8192(self) -> None:
        """Test multi-block SSM scan at 8192 sequence length."""
        np.random.seed(42)
        batch, state = 1, 8

        seq_len = 8192
        A_np = np.random.uniform(0.9, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        # Tolerance scaled for sequence length: expect ~sqrt(seq_len) numerical accumulation
        # For 8192 elements: ~90 ULPs accumulated, rtol=1e-2 gives ~2000 ULP margin
        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-2, atol=1e-2)

    @pytest.mark.slow
    def test_multiblock_ssm_vs_numpy_16384(self) -> None:
        """Test multi-block SSM scan at 16384 sequence length (stress test)."""
        np.random.seed(42)
        batch, state = 1, 4

        seq_len = 16384
        A_np = np.random.uniform(0.95, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        # Tolerance scaled for sequence length: 2e-2 gives reasonable margin
        # while still catching algorithmic bugs (was 1e-1 which is too loose)
        np.testing.assert_allclose(mlx_out, np_out, rtol=2e-2, atol=2e-2)

    def test_multiblock_ssm_boundary_1025(self) -> None:
        """Test SSM scan at exact boundary (1025 = 1024 + 1)."""
        np.random.seed(42)
        batch, state = 2, 32

        seq_len = 1025
        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-2, atol=1e-2)

    def test_multiblock_ssm_single_batch(self) -> None:
        """Test multi-block SSM scan with single batch."""
        np.random.seed(42)
        batch, seq_len, state = 1, 2048, 64

        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-2, atol=1e-2)

    def test_multiblock_ssm_large_batch(self) -> None:
        """Test multi-block SSM scan with larger batch size."""
        np.random.seed(42)
        batch, seq_len, state = 8, 2048, 16

        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        mlx_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        np_out = np_ssm_scan(A_np, h_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-2, atol=1e-2)

    def test_multiblock_ssm_properties(self) -> None:
        """Test SSM properties hold for multi-block scan."""
        np.random.seed(42)
        seq_len = 2048

        # Test: A=1 should equal cumsum
        x_np = np.random.randn(1, seq_len, 8).astype(np.float32)
        A_np = np.ones((1, seq_len, 8), dtype=np.float32)

        result = to_numpy(associative_scan(
            to_mlx(x_np), operator="ssm", A=to_mlx(A_np), axis=1
        ))
        expected = np.cumsum(x_np, axis=1)

        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


class TestVectorizedOptimization:
    """Tests for vectorized (float4) memory access optimization."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    def test_vectorized_cumsum_correctness(self, seq_len: int) -> None:
        """Test vectorized cumsum correctness at various sizes."""
        np.random.seed(42)
        x_np = np.random.randn(8, seq_len).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_vectorized_non_divisible_by_4(self) -> None:
        """Test vectorized kernel with seq_len not divisible by 4."""
        np.random.seed(42)
        # 65 elements - not divisible by 4
        x_np = np.random.randn(4, 65).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_vectorized_large_batch(self) -> None:
        """Test vectorized kernel with large batch size."""
        np.random.seed(42)
        x_np = np.random.randn(64, 256).astype(np.float32)

        mlx_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", axis=-1))
        np_out = np.cumsum(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)


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

    h = mx.zeros((batch_size, d_inner, d_state))

    outputs = []
    for t in range(seq_len):
        delta_t = delta[:, t, :]
        A_bar = mx.exp(delta_t[:, :, None] * A[None, :, :])

        B_t = B[:, t, :]
        x_t = x[:, t, :]
        B_x = delta_t[:, :, None] * B_t[:, None, :] * x_t[:, :, None]

        h = A_bar * h + B_x

        C_t = C[:, t, :]
        y_t = mx.sum(C_t[:, None, :] * h, axis=-1)
        outputs.append(y_t)

    return mx.stack(outputs, axis=1)


class TestMetalFallbackConsistency:
    """Tests that Metal and fallback implementations produce identical results."""

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512])
    def test_cumsum_metal_vs_fallback(self, seq_len: int) -> None:
        """Test Metal and fallback cumsum produce same results."""
        np.random.seed(42)
        x_np = np.random.randn(4, seq_len).astype(np.float32)

        metal_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", use_metal=True))
        fallback_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", use_metal=False))

        np.testing.assert_allclose(metal_out, fallback_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512])
    def test_ssm_metal_vs_fallback(self, seq_len: int) -> None:
        """Test Metal and fallback SSM scan produce same results."""
        np.random.seed(42)
        batch, state = 2, 16

        A_np = np.random.uniform(0.8, 0.99, (batch, seq_len, state)).astype(np.float32)
        h_np = np.random.randn(batch, seq_len, state).astype(np.float32)

        metal_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1, use_metal=True
        ))
        fallback_out = to_numpy(associative_scan(
            to_mlx(h_np), operator="ssm", A=to_mlx(A_np), axis=1, use_metal=False
        ))

        np.testing.assert_allclose(metal_out, fallback_out, rtol=1e-4, atol=1e-4)

    def test_multiblock_cumsum_metal_vs_fallback(self) -> None:
        """Test Metal and fallback for multi-block cumsum."""
        np.random.seed(42)
        x_np = np.random.randn(2, 2048).astype(np.float32)

        metal_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", use_metal=True))
        fallback_out = to_numpy(associative_scan(to_mlx(x_np), operator="add", use_metal=False))

        np.testing.assert_allclose(metal_out, fallback_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
