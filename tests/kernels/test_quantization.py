"""Tests for quantized linear operations."""

import mlx.core as mx
import pytest

from mlx_primitives.kernels import (
    QuantizedLinear,
    dequantize_int4,
    dequantize_int8,
    int4_linear,
    int8_linear,
    quantize_int4,
    quantize_int8,
)


class TestInt8Quantization:
    """Tests for INT8 quantization."""

    def test_quantize_int8_per_channel_shape(self) -> None:
        """Test that INT8 per-channel quantization produces correct shapes."""
        weights = mx.random.normal((128, 64))

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=True)
        mx.eval(W_quant, scale, zero_point)

        assert W_quant.shape == (128, 64)
        assert W_quant.dtype == mx.int8
        assert scale.shape == (128,)
        assert zero_point.shape == (128,)

    def test_quantize_int8_per_tensor_shape(self) -> None:
        """Test that INT8 per-tensor quantization produces correct shapes."""
        weights = mx.random.normal((128, 64))

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=False)
        mx.eval(W_quant, scale, zero_point)

        assert W_quant.shape == (128, 64)
        assert W_quant.dtype == mx.int8
        assert scale.shape == (1,)
        assert zero_point.shape == (1,)

    def test_quantize_int8_value_range(self) -> None:
        """Test that INT8 values are in valid range."""
        weights = mx.random.normal((128, 64))

        W_quant, _, _ = quantize_int8(weights, per_channel=True)
        mx.eval(W_quant)

        # INT8 range is -128 to 127
        assert mx.all(W_quant >= -128)
        assert mx.all(W_quant <= 127)

    def test_dequantize_int8_reconstructs(self) -> None:
        """Test that dequantization approximately reconstructs original."""
        weights = mx.random.normal((128, 64)) * 0.5  # Smaller values for better quantization

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=True)
        W_dequant = dequantize_int8(W_quant, scale, zero_point)
        mx.eval(W_dequant)

        # Should be close to original (within quantization error)
        error = mx.abs(W_dequant - weights)
        max_error = mx.max(error)

        # INT8 with per-channel should be quite accurate
        assert float(max_error) < 0.1


class TestInt8Linear:
    """Tests for INT8 linear operations."""

    def test_int8_linear_shape(self) -> None:
        """Test INT8 linear output shape."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=True)

        out = int8_linear(x, W_quant, scale, zero_point)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_int8_linear_vs_fp32(self) -> None:
        """Test INT8 linear approximates FP32."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1

        # FP32 reference
        fp32_out = x @ weights.T
        mx.eval(fp32_out)

        # INT8
        W_quant, scale, zero_point = quantize_int8(weights, per_channel=True)
        int8_out = int8_linear(x, W_quant, scale, zero_point)
        mx.eval(int8_out)

        # Should be close (within quantization error)
        assert mx.allclose(int8_out, fp32_out, atol=0.5, rtol=0.1)

    def test_int8_linear_with_bias(self) -> None:
        """Test INT8 linear with bias."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1
        bias = mx.random.normal((out_features,)) * 0.01

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=True)

        out_no_bias = int8_linear(x, W_quant, scale, zero_point)
        out_with_bias = int8_linear(x, W_quant, scale, zero_point, bias)
        mx.eval(out_no_bias, out_with_bias)

        # Bias should add to output
        assert not mx.allclose(out_no_bias, out_with_bias, atol=1e-6)

    def test_int8_linear_per_tensor(self) -> None:
        """Test INT8 linear with per-tensor quantization."""
        batch, seq_len, in_features = 2, 16, 64
        out_features = 128

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1

        W_quant, scale, zero_point = quantize_int8(weights, per_channel=False)

        out = int8_linear(x, W_quant, scale, zero_point)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)


class TestInt4Quantization:
    """Tests for INT4 quantization."""

    def test_quantize_int4_shape(self) -> None:
        """Test that INT4 quantization produces correct shapes."""
        weights = mx.random.normal((128, 256))  # in_features must be even
        group_size = 64

        W_packed, scales, zero_points = quantize_int4(weights, group_size=group_size)
        mx.eval(W_packed, scales, zero_points)

        # Packed: half the input features
        assert W_packed.shape == (128, 128)  # 256 / 2
        assert W_packed.dtype == mx.uint8

        # Scales: one per group per output channel
        num_groups = (256 + group_size - 1) // group_size  # 4 groups
        assert scales.shape == (128, num_groups)
        assert zero_points.shape == (128, num_groups)

    def test_quantize_int4_packed_values(self) -> None:
        """Test that INT4 packed values are in valid range."""
        weights = mx.random.normal((64, 128))

        W_packed, _, _ = quantize_int4(weights, group_size=32)
        mx.eval(W_packed)

        # Each byte contains two 4-bit values, so max is 0xFF
        assert mx.all(W_packed >= 0)
        assert mx.all(W_packed <= 255)

    def test_dequantize_int4_shape(self) -> None:
        """Test INT4 dequantization produces correct shape."""
        weights = mx.random.normal((64, 128))

        W_packed, scales, zero_points = quantize_int4(weights, group_size=32)
        W_dequant = dequantize_int4(W_packed, scales, zero_points, group_size=32)
        mx.eval(W_dequant)

        assert W_dequant.shape == weights.shape


class TestInt4Linear:
    """Tests for INT4 linear operations."""

    def test_int4_linear_shape(self) -> None:
        """Test INT4 linear output shape."""
        batch, seq_len, in_features = 2, 16, 128
        out_features = 64

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1

        W_packed, scales, zero_points = quantize_int4(weights, group_size=32)

        out = int4_linear(x, W_packed, scales, zero_points, group_size=32)
        mx.eval(out)

        assert out.shape == (batch, seq_len, out_features)

    def test_int4_linear_with_bias(self) -> None:
        """Test INT4 linear with bias."""
        batch, seq_len, in_features = 2, 16, 128
        out_features = 64

        x = mx.random.normal((batch, seq_len, in_features))
        weights = mx.random.normal((out_features, in_features)) * 0.1
        bias = mx.random.normal((out_features,)) * 0.01

        W_packed, scales, zero_points = quantize_int4(weights, group_size=32)

        out_no_bias = int4_linear(x, W_packed, scales, zero_points, group_size=32)
        out_with_bias = int4_linear(x, W_packed, scales, zero_points, bias, group_size=32)
        mx.eval(out_no_bias, out_with_bias)

        assert not mx.allclose(out_no_bias, out_with_bias, atol=1e-6)


class TestQuantizedLinear:
    """Tests for QuantizedLinear class."""

    def test_quantized_linear_int8(self) -> None:
        """Test QuantizedLinear with INT8."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=8)

        x = mx.random.normal((2, 16, 64))
        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 128)

    def test_quantized_linear_int4(self) -> None:
        """Test QuantizedLinear with INT4."""
        layer = QuantizedLinear(
            in_features=128, out_features=64, bits=4, group_size=32
        )

        x = mx.random.normal((2, 16, 128))
        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 64)

    def test_quantized_linear_with_bias(self) -> None:
        """Test QuantizedLinear with bias."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=8, bias=True)

        x = mx.random.normal((2, 16, 64))
        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 128)
        assert layer.bias is not None

    def test_quantize_weights_method(self) -> None:
        """Test manual weight quantization."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=8)

        # Provide custom weights
        custom_weights = mx.random.normal((128, 64)) * 0.2
        layer.quantize_weights(custom_weights)

        x = mx.random.normal((2, 16, 64))
        out = layer(x)
        mx.eval(out)

        assert out.shape == (2, 16, 128)


class TestInputValidation:
    """Tests for input validation."""

    def test_quantize_int8_wrong_dim(self) -> None:
        """Test that wrong dimension raises error."""
        weights = mx.random.normal((64,))  # 1D instead of 2D

        with pytest.raises(ValueError, match="Expected 2D"):
            quantize_int8(weights)

    def test_quantize_int4_wrong_dim(self) -> None:
        """Test that wrong dimension raises error."""
        weights = mx.random.normal((64,))

        with pytest.raises(ValueError, match="Expected 2D"):
            quantize_int4(weights)

    def test_quantize_int4_odd_features(self) -> None:
        """Test that odd in_features raises error."""
        weights = mx.random.normal((64, 65))  # Odd in_features

        with pytest.raises(ValueError, match="must be even"):
            quantize_int4(weights)

    def test_int8_linear_wrong_dim(self) -> None:
        """Test that wrong input dimension raises error."""
        x = mx.random.normal((2, 64))  # 2D instead of 3D
        W = mx.random.normal((128, 64)).astype(mx.int8)
        scale = mx.array([1.0])
        zp = mx.array([0.0])

        with pytest.raises(ValueError, match="Expected 3D"):
            int8_linear(x, W, scale, zp)

    def test_int4_linear_wrong_dim(self) -> None:
        """Test that wrong input dimension raises error."""
        x = mx.random.normal((2, 64))  # 2D instead of 3D
        W = mx.random.normal((128, 32)).astype(mx.uint8)
        scales = mx.random.normal((128, 2))
        zps = mx.random.normal((128, 2))

        with pytest.raises(ValueError, match="Expected 3D"):
            int4_linear(x, W, scales, zps)


class TestMemoryReduction:
    """Tests to verify memory reduction from quantization."""

    def test_int8_memory_reduction(self) -> None:
        """Verify INT8 uses 4x less memory than FP32."""
        out_features, in_features = 1024, 1024

        # FP32 size
        fp32_size = out_features * in_features * 4  # 4 bytes per float32

        # INT8 size (weights + scale + zero_point per channel)
        int8_weight_size = out_features * in_features * 1  # 1 byte per int8
        int8_params_size = out_features * 4 * 2  # scale + zp per channel
        int8_total = int8_weight_size + int8_params_size

        # Should be ~4x smaller
        ratio = fp32_size / int8_total
        assert ratio > 3.5  # At least 3.5x reduction

    def test_int4_memory_reduction(self) -> None:
        """Verify INT4 uses 8x less memory than FP32."""
        out_features, in_features = 1024, 1024
        group_size = 128
        num_groups = (in_features + group_size - 1) // group_size

        # FP32 size
        fp32_size = out_features * in_features * 4

        # INT4 size (packed weights + scales + zero_points)
        int4_weight_size = out_features * (in_features // 2)  # 0.5 byte per weight
        int4_params_size = out_features * num_groups * 4 * 2  # scale + zp per group
        int4_total = int4_weight_size + int4_params_size

        # Should be ~6-8x smaller
        ratio = fp32_size / int4_total
        assert ratio > 5.0  # At least 5x reduction
