"""Tests for fused operation kernels."""

import math

import mlx.core as mx
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


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_rmsnorm_basic(self) -> None:
        """Test basic RMSNorm computation."""
        x = mx.random.normal((2, 4, 8))
        weight = mx.ones((8,))

        out = rmsnorm(x, weight)
        mx.eval(out)

        assert out.shape == x.shape

    def test_rmsnorm_correctness(self) -> None:
        """Test RMSNorm produces correct values."""
        x = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
        weight = mx.ones((4,))

        out = rmsnorm(x, weight)
        mx.eval(out)

        # Manual calculation: RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        # Normalized: [0.365, 0.730, 1.095, 1.461]
        rms = math.sqrt(sum([1, 4, 9, 16]) / 4)
        expected = mx.array([[[1 / rms, 2 / rms, 3 / rms, 4 / rms]]])

        assert mx.allclose(out, expected, atol=1e-5)

    def test_rmsnorm_with_scaling(self) -> None:
        """Test RMSNorm with non-unit weight."""
        x = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
        weight = mx.array([2.0, 2.0, 2.0, 2.0])

        out = rmsnorm(x, weight)
        mx.eval(out)

        rms = math.sqrt(sum([1, 4, 9, 16]) / 4)
        expected = mx.array([[[2 / rms, 4 / rms, 6 / rms, 8 / rms]]])

        assert mx.allclose(out, expected, atol=1e-5)


class TestFusedRMSNormLinear:
    """Tests for fused RMSNorm + Linear."""

    def test_fused_rmsnorm_linear_basic(self) -> None:
        """Test basic fused RMSNorm + Linear."""
        batch, seq_len, hidden_dim = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, hidden_dim))
        norm_weight = mx.ones((hidden_dim,))
        linear_weight = mx.random.normal((out_features, hidden_dim)) * 0.1

        out = fused_rmsnorm_linear(x, norm_weight, linear_weight)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_fused_vs_separate(self) -> None:
        """Test that fused result matches separate operations."""
        batch, seq_len, hidden_dim = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, hidden_dim))
        norm_weight = mx.random.normal((hidden_dim,))
        linear_weight = mx.random.normal((out_features, hidden_dim)) * 0.1

        # Fused (with Metal)
        fused_out = fused_rmsnorm_linear(x, norm_weight, linear_weight, use_metal=True)
        mx.eval(fused_out)

        # Reference (separate ops)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        norm_x = x / rms * norm_weight
        ref_out = norm_x @ linear_weight.T
        mx.eval(ref_out)

        assert mx.allclose(fused_out, ref_out, atol=1e-4)

    def test_fused_with_bias(self) -> None:
        """Test fused RMSNorm + Linear with bias."""
        batch, seq_len, hidden_dim = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, hidden_dim))
        norm_weight = mx.ones((hidden_dim,))
        linear_weight = mx.random.normal((out_features, hidden_dim)) * 0.1
        linear_bias = mx.random.normal((out_features,)) * 0.01

        out = fused_rmsnorm_linear(x, norm_weight, linear_weight, linear_bias)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

        # Verify bias is applied
        out_no_bias = fused_rmsnorm_linear(x, norm_weight, linear_weight)
        mx.eval(out_no_bias)

        # They should be different
        assert not mx.allclose(out, out_no_bias, atol=1e-6)

    def test_fused_reference_fallback(self) -> None:
        """Test reference implementation fallback for small sequences."""
        batch, seq_len, hidden_dim = 2, 4, 64  # Small seq_len triggers fallback
        out_features = 128

        x = mx.random.normal((batch, seq_len, hidden_dim))
        norm_weight = mx.ones((hidden_dim,))
        linear_weight = mx.random.normal((out_features, hidden_dim)) * 0.1

        out = fused_rmsnorm_linear(x, norm_weight, linear_weight)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_fused_rmsnorm_linear_class(self) -> None:
        """Test FusedRMSNormLinear class."""
        layer = FusedRMSNormLinear(hidden_dim=64, out_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 128)


class TestSiLU:
    """Tests for SiLU activation."""

    def test_silu_basic(self) -> None:
        """Test basic SiLU computation."""
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = silu(x)
        mx.eval(out)

        expected = x * mx.sigmoid(x)
        assert mx.allclose(out, expected, atol=1e-6)

    def test_silu_zero(self) -> None:
        """Test SiLU at zero."""
        x = mx.array([0.0])
        out = silu(x)
        mx.eval(out)

        assert mx.allclose(out, mx.array([0.0]), atol=1e-6)


class TestGELU:
    """Tests for GELU activation."""

    def test_gelu_approximate(self) -> None:
        """Test approximate GELU."""
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = gelu(x, approximate=True)
        mx.eval(out)

        # Known approximate values
        # GELU(0) = 0
        assert abs(float(out[2]) - 0.0) < 1e-5

    def test_gelu_exact(self) -> None:
        """Test exact GELU using erf."""
        x = mx.array([0.0, 1.0])
        out = gelu(x, approximate=False)
        mx.eval(out)

        # GELU(0) = 0, GELU(1) ≈ 0.8413
        assert abs(float(out[0]) - 0.0) < 1e-5
        assert abs(float(out[1]) - 0.8413) < 0.01


class TestFusedSwiGLU:
    """Tests for fused SwiGLU."""

    def test_fused_swiglu_basic(self) -> None:
        """Test basic fused SwiGLU."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        out = fused_swiglu(x, W_gate, W_up)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_fused_swiglu_vs_reference(self) -> None:
        """Test that fused SwiGLU matches reference."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        # Fused
        fused_out = fused_swiglu(x, W_gate, W_up, use_metal=True)
        mx.eval(fused_out)

        # Reference: silu(x @ W_gate.T) * (x @ W_up.T)
        gate = x @ W_gate.T
        up = x @ W_up.T
        ref_out = mx.sigmoid(gate) * gate * up  # silu(gate) * up
        mx.eval(ref_out)

        assert mx.allclose(fused_out, ref_out, atol=1e-4)

    def test_fused_swiglu_reference_fallback(self) -> None:
        """Test reference fallback for small sequences."""
        batch, seq_len, in_features = 2, 4, 64  # Small seq triggers fallback
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        out = fused_swiglu(x, W_gate, W_up)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_swiglu_class(self) -> None:
        """Test SwiGLU class."""
        layer = SwiGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        # Output should be same as input (down projection back to in_features)
        assert out.shape == (2, 16, 64)


class TestFusedGeGLU:
    """Tests for fused GeGLU."""

    def test_fused_geglu_basic(self) -> None:
        """Test basic fused GeGLU."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        out = fused_geglu(x, W_gate, W_up)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_fused_geglu_vs_reference(self) -> None:
        """Test that fused GeGLU matches reference."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        W_gate = mx.random.normal((out_features, in_features)) * 0.1
        W_up = mx.random.normal((out_features, in_features)) * 0.1

        # Fused
        fused_out = fused_geglu(x, W_gate, W_up, use_metal=True)
        mx.eval(fused_out)

        # Reference: gelu(x @ W_gate.T) * (x @ W_up.T)
        gate = x @ W_gate.T
        up = x @ W_up.T
        # GELU tanh approximation
        gelu_gate = 0.5 * gate * (1 + mx.tanh(0.7978845608 * (gate + 0.044715 * gate**3)))
        ref_out = gelu_gate * up
        mx.eval(ref_out)

        assert mx.allclose(fused_out, ref_out, atol=1e-4)

    def test_geglu_class(self) -> None:
        """Test GeGLU class."""
        layer = GeGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))

        out = layer(x)
        mx.eval(out)

        # Output should be same as input (down projection back to in_features)
        assert out.shape == (2, 16, 64)


class TestInputValidation:
    """Tests for input validation."""

    def test_fused_rmsnorm_linear_input_dim(self) -> None:
        """Test that 2D input raises error."""
        x = mx.random.normal((2, 64))  # 2D instead of 3D
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

    def test_fused_swiglu_input_dim(self) -> None:
        """Test that 2D input raises error for SwiGLU."""
        x = mx.random.normal((2, 64))  # 2D instead of 3D
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
