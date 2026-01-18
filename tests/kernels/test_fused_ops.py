"""Tests for fused operation kernels.

Validation strategy:
1. NumPy reference implementations for algorithmic correctness
2. Analytical test cases with known outputs
3. Metal vs reference implementation consistency
4. Property-based tests for mathematical invariants
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.kernels import (
    FusedRMSNormLinear,
    GeGLU,
    SwiGLU,
    fused_geglu,
    fused_rmsnorm_linear,
    fused_swiglu,
    gelu,
    rmsnorm,
    silu,
)

from tests.reference import (
    AnalyticalTests,
    fused_rmsnorm_linear as np_fused_rmsnorm_linear,
    gelu_approximate as np_gelu_approximate,
    gelu_exact as np_gelu_exact,
    geglu as np_geglu,
    rmsnorm as np_rmsnorm,
    silu as np_silu,
    swiglu as np_swiglu,
)


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    mx.eval(x)
    return np.array(x)


def to_mlx(x: np.ndarray) -> mx.array:
    """Convert NumPy array to MLX."""
    return mx.array(x)


class TestSiLUAgainstNumPy:
    """Validate SiLU against NumPy reference."""

    def test_silu_vs_numpy(self) -> None:
        """Test SiLU matches NumPy implementation."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        mlx_out = to_numpy(silu(to_mlx(x_np)))
        np_out = np_silu(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)

    def test_silu_analytical_values(self) -> None:
        """Test SiLU at known points."""
        x_np, expected = AnalyticalTests.silu_known_values()

        mlx_out = to_numpy(silu(to_mlx(x_np)))

        np.testing.assert_allclose(mlx_out, expected, rtol=1e-5, atol=1e-6)

    def test_silu_zero(self) -> None:
        """SiLU(0) = 0 exactly."""
        x = mx.array([0.0])
        out = silu(x)
        mx.eval(out)

        assert float(out[0]) == 0.0

    def test_silu_properties(self) -> None:
        """Test mathematical properties of SiLU."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)
        x = to_mlx(x_np)

        out = to_numpy(silu(x))

        # Property: silu(x) = x * sigmoid(x)
        # Property: sign(silu(x)) = sign(x) for x != 0
        for i, xi in enumerate(x_np):
            if xi > 0:
                assert out[i] > 0
            elif xi < 0:
                assert out[i] < 0


class TestGELUAgainstNumPy:
    """Validate GELU against NumPy reference."""

    def test_gelu_approximate_vs_numpy(self) -> None:
        """Test approximate GELU matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        mlx_out = to_numpy(gelu(to_mlx(x_np), approximate=True))
        np_out = np_gelu_approximate(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)

    def test_gelu_exact_vs_numpy(self) -> None:
        """Test exact GELU matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float32)

        mlx_out = to_numpy(gelu(to_mlx(x_np), approximate=False))
        np_out = np_gelu_exact(x_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-5)

    def test_gelu_analytical_values(self) -> None:
        """Test GELU at known points."""
        x_np, expected = AnalyticalTests.gelu_known_values()

        # Exact GELU should match analytical values
        mlx_out = to_numpy(gelu(to_mlx(x_np), approximate=False))

        np.testing.assert_allclose(mlx_out, expected, rtol=1e-4, atol=1e-4)

    def test_gelu_zero(self) -> None:
        """GELU(0) = 0 exactly."""
        x = mx.array([0.0])
        out = gelu(x, approximate=True)
        mx.eval(out)

        assert abs(float(out[0])) < 1e-7


class TestRMSNormAgainstNumPy:
    """Validate RMSNorm against NumPy reference."""

    def test_rmsnorm_vs_numpy(self) -> None:
        """Test RMSNorm matches NumPy."""
        np.random.seed(42)
        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        weight_np = np.random.randn(8).astype(np.float32)

        mlx_out = to_numpy(rmsnorm(to_mlx(x_np), to_mlx(weight_np)))
        np_out = np_rmsnorm(x_np, weight_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-5, atol=1e-6)

    def test_rmsnorm_analytical_values(self) -> None:
        """Test RMSNorm with known inputs."""
        x_np, weight_np, expected = AnalyticalTests.rmsnorm_known_values()

        mlx_out = to_numpy(rmsnorm(to_mlx(x_np), to_mlx(weight_np)))

        np.testing.assert_allclose(mlx_out, expected, rtol=1e-5, atol=1e-6)

    def test_rmsnorm_unit_weight(self) -> None:
        """With weight=1, RMSNorm normalizes to unit RMS."""
        np.random.seed(42)
        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        weight_np = np.ones(8, dtype=np.float32)

        mlx_out = to_numpy(rmsnorm(to_mlx(x_np), to_mlx(weight_np)))

        # RMS of output should be approximately 1
        rms = np.sqrt(np.mean(mlx_out * mlx_out, axis=-1))
        np.testing.assert_allclose(rms, 1.0, rtol=1e-4, atol=1e-4)


class TestFusedRMSNormLinearAgainstNumPy:
    """Validate fused RMSNorm+Linear against NumPy reference."""

    def test_fused_vs_numpy(self) -> None:
        """Test fused operation matches NumPy."""
        np.random.seed(42)
        batch, seq, hidden = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_w_np = np.random.randn(hidden).astype(np.float32)
        linear_w_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.1

        mlx_out = to_numpy(fused_rmsnorm_linear(
            to_mlx(x_np), to_mlx(norm_w_np), to_mlx(linear_w_np)
        ))
        np_out = np_fused_rmsnorm_linear(x_np, norm_w_np, linear_w_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_fused_with_bias_vs_numpy(self) -> None:
        """Test fused operation with bias matches NumPy."""
        np.random.seed(42)
        batch, seq, hidden = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_w_np = np.random.randn(hidden).astype(np.float32)
        linear_w_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.1
        bias_np = np.random.randn(out_features).astype(np.float32) * 0.01

        mlx_out = to_numpy(fused_rmsnorm_linear(
            to_mlx(x_np), to_mlx(norm_w_np), to_mlx(linear_w_np), to_mlx(bias_np)
        ))
        np_out = np_fused_rmsnorm_linear(x_np, norm_w_np, linear_w_np, bias_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_metal_vs_reference(self) -> None:
        """Test Metal kernel matches reference implementation."""
        np.random.seed(42)
        batch, seq, hidden = 2, 32, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_w_np = np.random.randn(hidden).astype(np.float32)
        linear_w_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.1

        # Metal (use_metal=True, seq>=8)
        metal_out = to_numpy(fused_rmsnorm_linear(
            to_mlx(x_np), to_mlx(norm_w_np), to_mlx(linear_w_np), use_metal=True
        ))
        # Reference (use_metal=False)
        ref_out = to_numpy(fused_rmsnorm_linear(
            to_mlx(x_np), to_mlx(norm_w_np), to_mlx(linear_w_np), use_metal=False
        ))

        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-4, atol=1e-4)


class TestFusedSwiGLUAgainstNumPy:
    """Validate fused SwiGLU against NumPy reference."""

    def test_swiglu_vs_numpy(self) -> None:
        """Test fused SwiGLU matches NumPy."""
        np.random.seed(42)
        batch, seq, in_features = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, in_features).astype(np.float32)
        W_gate_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        W_up_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1

        mlx_out = to_numpy(fused_swiglu(
            to_mlx(x_np), to_mlx(W_gate_np), to_mlx(W_up_np)
        ))
        np_out = np_swiglu(x_np, W_gate_np, W_up_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_metal_vs_reference(self) -> None:
        """Test Metal kernel matches reference."""
        np.random.seed(42)
        batch, seq, in_features = 2, 32, 64
        out_features = 128

        x = mx.random.normal((batch, seq, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        metal_out = to_numpy(fused_swiglu(x, W_gate, W_up, use_metal=True))
        ref_out = to_numpy(fused_swiglu(x, W_gate, W_up, use_metal=False))

        np.testing.assert_allclose(metal_out, ref_out, rtol=1e-4, atol=1e-4)


class TestFusedGeGLUAgainstNumPy:
    """Validate fused GeGLU against NumPy reference."""

    def test_geglu_vs_numpy(self) -> None:
        """Test fused GeGLU matches NumPy."""
        np.random.seed(42)
        batch, seq, in_features = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, in_features).astype(np.float32)
        W_gate_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        W_up_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1

        mlx_out = to_numpy(fused_geglu(
            to_mlx(x_np), to_mlx(W_gate_np), to_mlx(W_up_np)
        ))
        np_out = np_geglu(x_np, W_gate_np, W_up_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)


class TestLayerClasses:
    """Test layer wrapper classes."""

    def test_fused_rmsnorm_linear_class(self) -> None:
        """Test FusedRMSNormLinear class."""
        layer = FusedRMSNormLinear(hidden_dim=64, out_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 128)

    def test_swiglu_class(self) -> None:
        """Test SwiGLU class produces correct output shape."""
        layer = SwiGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        # SwiGLU has down projection back to in_features
        assert out.shape == (2, 16, 64)

    def test_geglu_class(self) -> None:
        """Test GeGLU class produces correct output shape."""
        layer = GeGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 64)


class TestInputValidation:
    """Tests for input validation."""

    def test_fused_rmsnorm_linear_wrong_dim(self) -> None:
        """Test that 2D input raises error."""
        x = mx.random.normal((2, 64))
        norm_weight = mx.ones((64,))
        linear_weight = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="Expected 3D"):
            fused_rmsnorm_linear(x, norm_weight, linear_weight)

    def test_fused_rmsnorm_linear_dim_mismatch(self) -> None:
        """Test dimension mismatch raises error."""
        x = mx.random.normal((2, 16, 64))
        norm_weight = mx.ones((32,))  # Wrong dim
        linear_weight = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="norm_weight dim"):
            fused_rmsnorm_linear(x, norm_weight, linear_weight)

    def test_fused_swiglu_wrong_dim(self) -> None:
        """Test that 2D input raises error for SwiGLU."""
        x = mx.random.normal((2, 64))
        W_gate = mx.random.normal((128, 64))
        W_up = mx.random.normal((128, 64))

        with pytest.raises(ValueError, match="Expected 3D"):
            fused_swiglu(x, W_gate, W_up)

    def test_fused_swiglu_weight_mismatch(self) -> None:
        """Test weight shape mismatch raises error."""
        x = mx.random.normal((2, 16, 64))
        W_gate = mx.random.normal((128, 64))
        W_up = mx.random.normal((64, 64))  # Different out_features

        with pytest.raises(ValueError, match="must have same shape"):
            fused_swiglu(x, W_gate, W_up)


class TestPropertyBased:
    """Property-based tests for mathematical invariants."""

    def test_silu_bounds(self) -> None:
        """SiLU(x) is bounded: silu(x) >= -0.278 for all x."""
        np.random.seed(42)
        x = mx.array(np.random.randn(1000).astype(np.float32))

        out = silu(x)
        mx.eval(out)

        # SiLU has a minimum around x ≈ -1.28, value ≈ -0.278
        assert float(mx.min(out)) >= -0.3

    def test_gelu_monotonic_positive(self) -> None:
        """GELU is monotonically increasing for x > 0."""
        x_sorted = mx.array(np.linspace(0.01, 5.0, 100).astype(np.float32))

        out = gelu(x_sorted)
        mx.eval(out)

        out_np = to_numpy(out)
        diffs = np.diff(out_np)

        # All differences should be positive (monotonic increase)
        assert np.all(diffs > 0)

    def test_rmsnorm_scale_invariance(self) -> None:
        """RMSNorm(c*x) / c = RMSNorm(x) for c > 0."""
        np.random.seed(42)
        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        weight_np = np.ones(8, dtype=np.float32)
        c = 5.0

        out_x = to_numpy(rmsnorm(to_mlx(x_np), to_mlx(weight_np)))
        out_cx = to_numpy(rmsnorm(to_mlx(c * x_np), to_mlx(weight_np)))

        # RMSNorm(c*x) = c * RMSNorm(x) due to scale-equivariance
        # Actually RMSNorm normalizes by RMS, so scaling x by c:
        # RMSNorm(c*x) = (c*x) / (c*RMS(x)) = x / RMS(x) = RMSNorm(x)
        np.testing.assert_allclose(out_cx, out_x, rtol=1e-5, atol=1e-6)

    def test_fused_equals_separate(self) -> None:
        """Fused operations equal sequential separate operations."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 64).astype(np.float32)
        norm_w_np = np.random.randn(64).astype(np.float32)
        linear_w_np = np.random.randn(128, 64).astype(np.float32) * 0.1

        x = to_mlx(x_np)
        norm_w = to_mlx(norm_w_np)
        linear_w = to_mlx(linear_w_np)

        # Fused
        fused_out = fused_rmsnorm_linear(x, norm_w, linear_w)

        # Separate
        norm_out = rmsnorm(x, norm_w)
        separate_out = norm_out @ linear_w.T

        mx.eval(fused_out, separate_out)

        np.testing.assert_allclose(
            to_numpy(fused_out), to_numpy(separate_out), rtol=1e-4, atol=1e-4
        )
