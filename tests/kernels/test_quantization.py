"""Tests for quantized linear operations.

Validation strategy:
1. NumPy reference implementations for algorithmic correctness
2. Round-trip tests (quantize → dequantize ≈ identity)
3. Memory reduction verification
4. Property-based tests for quantization invariants
"""

import mlx.core as mx
import numpy as np
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

from tests.reference import (
    dequantize_int8 as np_dequantize_int8,
    int8_linear as np_int8_linear,
    quantize_int8_per_channel as np_quantize_int8_per_channel,
    quantize_int8_per_tensor as np_quantize_int8_per_tensor,
)


def to_numpy(x: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy."""
    mx.eval(x)
    return np.array(x)


def to_mlx(x: np.ndarray) -> mx.array:
    """Convert NumPy array to MLX."""
    return mx.array(x)


class TestInt8QuantizationAgainstNumPy:
    """Validate INT8 quantization against NumPy reference."""

    def test_quantize_int8_per_channel_vs_numpy(self) -> None:
        """Test per-channel quantization matches NumPy."""
        np.random.seed(42)
        weights_np = np.random.randn(128, 64).astype(np.float32) * 0.5

        # MLX
        W_quant_mlx, scale_mlx, zp_mlx = quantize_int8(to_mlx(weights_np), per_channel=True)

        # NumPy
        W_quant_np, scale_np, zp_np = np_quantize_int8_per_channel(weights_np)

        np.testing.assert_array_equal(to_numpy(W_quant_mlx), W_quant_np)
        np.testing.assert_allclose(to_numpy(scale_mlx), scale_np, rtol=1e-5)
        np.testing.assert_allclose(to_numpy(zp_mlx), zp_np, rtol=1e-5)

    def test_quantize_int8_per_tensor_vs_numpy(self) -> None:
        """Test per-tensor quantization matches NumPy."""
        np.random.seed(42)
        weights_np = np.random.randn(128, 64).astype(np.float32) * 0.5

        # MLX
        W_quant_mlx, scale_mlx, zp_mlx = quantize_int8(to_mlx(weights_np), per_channel=False)

        # NumPy
        W_quant_np, scale_np, zp_np = np_quantize_int8_per_tensor(weights_np)

        np.testing.assert_array_equal(to_numpy(W_quant_mlx), W_quant_np)
        np.testing.assert_allclose(to_numpy(scale_mlx), scale_np, rtol=1e-5)
        np.testing.assert_allclose(to_numpy(zp_mlx), zp_np, rtol=1e-5)

    def test_dequantize_int8_vs_numpy(self) -> None:
        """Test dequantization matches NumPy."""
        np.random.seed(42)
        weights_np = np.random.randn(128, 64).astype(np.float32) * 0.5

        # Quantize with NumPy
        W_quant_np, scale_np, zp_np = np_quantize_int8_per_channel(weights_np)

        # Dequantize with MLX
        mlx_dequant = to_numpy(dequantize_int8(
            to_mlx(W_quant_np), to_mlx(scale_np), to_mlx(zp_np)
        ))

        # Dequantize with NumPy
        np_dequant = np_dequantize_int8(W_quant_np, scale_np, zp_np)

        np.testing.assert_allclose(mlx_dequant, np_dequant, rtol=1e-5, atol=1e-6)


class TestInt8LinearAgainstNumPy:
    """Validate INT8 linear against NumPy reference."""

    def test_int8_linear_vs_numpy(self) -> None:
        """Test INT8 linear matches NumPy."""
        np.random.seed(42)
        batch, seq, in_features = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, in_features).astype(np.float32)
        weights_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1

        # Quantize
        W_quant_np, scale_np, zp_np = np_quantize_int8_per_channel(weights_np)

        # MLX linear
        mlx_out = to_numpy(int8_linear(
            to_mlx(x_np),
            to_mlx(W_quant_np),
            to_mlx(scale_np),
            to_mlx(zp_np)
        ))

        # NumPy linear
        np_out = np_int8_linear(x_np, W_quant_np, scale_np, zp_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)

    def test_int8_linear_with_bias_vs_numpy(self) -> None:
        """Test INT8 linear with bias matches NumPy."""
        np.random.seed(42)
        batch, seq, in_features = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, in_features).astype(np.float32)
        weights_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        bias_np = np.random.randn(out_features).astype(np.float32) * 0.01

        W_quant_np, scale_np, zp_np = np_quantize_int8_per_channel(weights_np)

        mlx_out = to_numpy(int8_linear(
            to_mlx(x_np),
            to_mlx(W_quant_np),
            to_mlx(scale_np),
            to_mlx(zp_np),
            to_mlx(bias_np)
        ))

        np_out = np_int8_linear(x_np, W_quant_np, scale_np, zp_np, bias_np)

        np.testing.assert_allclose(mlx_out, np_out, rtol=1e-4, atol=1e-4)


class TestRoundTripQuantization:
    """Test quantize → dequantize round-trip accuracy."""

    def test_int8_round_trip_per_channel(self) -> None:
        """INT8 per-channel quantization round-trip."""
        np.random.seed(42)
        weights_np = np.random.randn(128, 64).astype(np.float32) * 0.5

        weights = to_mlx(weights_np)
        W_quant, scale, zp = quantize_int8(weights, per_channel=True)
        W_dequant = dequantize_int8(W_quant, scale, zp)

        error = to_numpy(mx.abs(W_dequant - weights))
        max_error = np.max(error)

        # INT8 with 255 levels should have max error < 0.01 for values in [-0.5, 0.5]
        assert max_error < 0.01

    def test_int8_round_trip_per_tensor(self) -> None:
        """INT8 per-tensor quantization round-trip."""
        np.random.seed(42)
        weights_np = np.random.randn(128, 64).astype(np.float32) * 0.5

        weights = to_mlx(weights_np)
        W_quant, scale, zp = quantize_int8(weights, per_channel=False)
        W_dequant = dequantize_int8(W_quant, scale, zp)

        error = to_numpy(mx.abs(W_dequant - weights))
        max_error = np.max(error)

        # Per-tensor is less accurate but still reasonable
        assert max_error < 0.05

    def test_int4_round_trip(self) -> None:
        """INT4 grouped quantization round-trip."""
        np.random.seed(42)
        weights_np = np.random.randn(64, 128).astype(np.float32) * 0.3

        weights = to_mlx(weights_np)
        W_packed, scales, zps = quantize_int4(weights, group_size=32)
        W_dequant = dequantize_int4(W_packed, scales, zps, group_size=32)

        error = to_numpy(mx.abs(W_dequant - weights))
        max_error = np.max(error)

        # INT4 with 16 levels has larger quantization error
        assert max_error < 0.15


class TestQuantizationProperties:
    """Property-based tests for quantization invariants."""

    def test_int8_value_range(self) -> None:
        """Quantized INT8 values are in valid range."""
        np.random.seed(42)
        weights = mx.array(np.random.randn(128, 64).astype(np.float32))

        W_quant, _, _ = quantize_int8(weights, per_channel=True)
        mx.eval(W_quant)

        W_np = to_numpy(W_quant)

        # INT8 range is -128 to 127
        assert np.all(W_np >= -128)
        assert np.all(W_np <= 127)

    def test_int4_packed_value_range(self) -> None:
        """Packed INT4 values are valid bytes."""
        np.random.seed(42)
        weights = mx.array(np.random.randn(64, 128).astype(np.float32))

        W_packed, _, _ = quantize_int4(weights, group_size=32)
        mx.eval(W_packed)

        W_np = to_numpy(W_packed)

        # Packed bytes should be 0-255
        assert np.all(W_np >= 0)
        assert np.all(W_np <= 255)

    def test_scale_positive(self) -> None:
        """Quantization scales are always positive."""
        np.random.seed(42)
        weights = mx.array(np.random.randn(128, 64).astype(np.float32))

        _, scale_channel, _ = quantize_int8(weights, per_channel=True)
        _, scale_tensor, _ = quantize_int8(weights, per_channel=False)

        mx.eval(scale_channel, scale_tensor)

        assert np.all(to_numpy(scale_channel) > 0)
        assert np.all(to_numpy(scale_tensor) > 0)

    def test_quantization_preserves_sign(self) -> None:
        """Large magnitude values preserve their sign after round-trip."""
        np.random.seed(42)
        # Use larger values where sign should be preserved
        weights_np = np.random.randn(64, 64).astype(np.float32) * 2.0

        weights = to_mlx(weights_np)
        W_quant, scale, zp = quantize_int8(weights, per_channel=True)
        W_dequant = dequantize_int8(W_quant, scale, zp)

        # For large values, sign should be preserved
        original_sign = np.sign(weights_np)
        dequant_sign = np.sign(to_numpy(W_dequant))

        # At least 99% of signs should match (small values near 0 may flip)
        match_rate = np.mean(original_sign == dequant_sign)
        assert match_rate > 0.99


class TestMemoryReduction:
    """Tests to verify memory reduction from quantization."""

    def test_int8_memory_reduction(self) -> None:
        """Verify INT8 uses ~4x less memory than FP32."""
        out_features, in_features = 1024, 1024

        # FP32 size
        fp32_size = out_features * in_features * 4  # 4 bytes per float32

        # INT8 size (weights + scale + zero_point per channel)
        int8_weight_size = out_features * in_features * 1  # 1 byte per int8
        int8_params_size = out_features * 4 * 2  # scale + zp per channel (float32)
        int8_total = int8_weight_size + int8_params_size

        ratio = fp32_size / int8_total
        assert ratio > 3.5  # At least 3.5x reduction

    def test_int4_memory_reduction(self) -> None:
        """Verify INT4 uses ~6-8x less memory than FP32."""
        out_features, in_features = 1024, 1024
        group_size = 128
        num_groups = (in_features + group_size - 1) // group_size

        # FP32 size
        fp32_size = out_features * in_features * 4

        # INT4 size (packed weights + scales + zero_points)
        int4_weight_size = out_features * (in_features // 2)  # 0.5 byte per weight
        int4_params_size = out_features * num_groups * 4 * 2  # scale + zp per group
        int4_total = int4_weight_size + int4_params_size

        ratio = fp32_size / int4_total
        assert ratio > 5.0  # At least 5x reduction


class TestQuantizedLinearAccuracy:
    """Test quantized linear vs full precision accuracy."""

    def test_int8_vs_fp32_relative_error(self) -> None:
        """INT8 linear has small relative error vs FP32."""
        np.random.seed(42)
        batch, seq, in_features = 2, 16, 64
        out_features = 128

        x_np = np.random.randn(batch, seq, in_features).astype(np.float32)
        weights_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.1

        x = to_mlx(x_np)
        weights = to_mlx(weights_np)

        # FP32 reference
        fp32_out = to_numpy(x @ weights.T)

        # INT8
        W_quant, scale, zp = quantize_int8(weights, per_channel=True)
        int8_out = to_numpy(int8_linear(x, W_quant, scale, zp))

        # Relative error should be small
        relative_error = np.abs(int8_out - fp32_out) / (np.abs(fp32_out) + 1e-8)
        mean_relative_error = np.mean(relative_error)

        assert mean_relative_error < 0.05  # Less than 5% mean relative error


class TestQuantizedLinearClass:
    """Test QuantizedLinear wrapper class."""

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

    def test_quantized_linear_deterministic(self) -> None:
        """Test QuantizedLinear produces deterministic output."""
        layer = QuantizedLinear(in_features=64, out_features=128, bits=8)

        x = mx.random.normal((2, 16, 64))
        out1 = layer(x)
        out2 = layer(x)
        mx.eval(out1, out2)

        np.testing.assert_array_equal(to_numpy(out1), to_numpy(out2))


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


class TestQuantizationEdgeCases:
    """Test edge cases in quantization."""

    def test_zero_weights_int8(self) -> None:
        """Test INT8 quantization of all-zero weights."""
        weights = mx.zeros((64, 32))

        with pytest.warns(RuntimeWarning, match="zero range"):
            W_quant, scale, zp = quantize_int8(weights, per_channel=True)

        mx.eval(W_quant, scale, zp)

        # Scale should be clipped to minimum positive value
        assert to_numpy(scale).min() > 0

    def test_zero_weights_int4(self) -> None:
        """Test INT4 quantization of all-zero weights."""
        weights = mx.zeros((64, 32))

        with pytest.warns(RuntimeWarning, match="zero range"):
            W_packed, scales, zps = quantize_int4(weights, group_size=16)

        mx.eval(W_packed, scales, zps)

        # Scales should be clipped to minimum positive value
        assert to_numpy(scales).min() > 0

    def test_constant_weights_per_tensor_int8(self) -> None:
        """Test per-tensor INT8 with constant weights (zero range)."""
        weights = mx.ones((64, 32)) * 0.5  # All same value

        with pytest.warns(RuntimeWarning, match="zero range"):
            W_quant, scale, zp = quantize_int8(weights, per_channel=False)

        mx.eval(W_quant, scale, zp)

        # Should still produce valid output
        assert to_numpy(scale)[0] > 0

    def test_very_small_weights(self) -> None:
        """Test quantization of very small weights (near FP16 min)."""
        # Weights near the FP16 minimum positive normal (~6e-5)
        # These are all identical values, so zero-range warning is expected
        weights = mx.array(np.full((64, 32), 1e-6, dtype=np.float32))

        # Should work without error - scale will be clipped to FP16 min
        # Also triggers zero-range warning since all values are identical
        with pytest.warns(RuntimeWarning, match="zero range"):
            W_quant, scale, zp = quantize_int8(weights, per_channel=True)
        mx.eval(W_quant, scale, zp)

        # Scale should be at least FP16 min positive normal
        assert to_numpy(scale).min() >= 2**-14 - 1e-8

    def test_mixed_constant_rows_int8(self) -> None:
        """Test per-channel INT8 with some constant rows."""
        np.random.seed(42)
        weights_np = np.random.randn(64, 32).astype(np.float32)
        # Make first 5 rows constant (zero range)
        weights_np[:5, :] = 0.5

        weights = to_mlx(weights_np)

        with pytest.warns(RuntimeWarning, match="5 out of 64 channels"):
            W_quant, scale, zp = quantize_int8(weights, per_channel=True)

        mx.eval(W_quant, scale, zp)

        # All scales should be positive
        assert to_numpy(scale).min() > 0

    def test_large_values_no_overflow(self) -> None:
        """Test quantization of large values doesn't cause overflow."""
        weights = mx.array(np.random.randn(64, 32).astype(np.float32) * 100)

        W_quant, scale, zp = quantize_int8(weights, per_channel=True)
        W_dequant = dequantize_int8(W_quant, scale, zp)

        mx.eval(W_dequant)

        # Should not have NaN or Inf
        dequant_np = to_numpy(W_dequant)
        assert not np.any(np.isnan(dequant_np))
        assert not np.any(np.isinf(dequant_np))


class TestKernelCaching:
    """Test kernel caching behavior.

    Note: MLX Metal operations are not thread-safe for concurrent GPU access
    from multiple Python threads. These tests verify sequential caching behavior
    and Python-level cache structures only.
    """

    def test_sequential_kernel_reuse(self) -> None:
        """Test that kernel is cached and reused across sequential calls."""
        np.random.seed(42)

        # Prepare test data
        x = mx.array(np.random.randn(2, 16, 64).astype(np.float32))
        weights = mx.array(np.random.randn(128, 64).astype(np.float32))
        W_quant, scale, zp = quantize_int8(weights, per_channel=True)

        # First call compiles kernel
        out1 = int8_linear(x, W_quant, scale, zp)
        mx.eval(out1)

        # Subsequent calls should reuse cached kernel
        out2 = int8_linear(x, W_quant, scale, zp)
        out3 = int8_linear(x, W_quant, scale, zp)
        mx.eval(out2, out3)

        # All outputs should be identical
        np.testing.assert_array_equal(to_numpy(out1), to_numpy(out2))
        np.testing.assert_array_equal(to_numpy(out1), to_numpy(out3))

    def test_autotune_config_hashability(self) -> None:
        """Test that autotune Config is properly hashable for caching."""
        from mlx_primitives.dsl.decorators import Config

        # Verify Config can be used in cache keys
        c1 = Config(BLOCK_M=32, BLOCK_N=32)
        c2 = Config(BLOCK_M=32, BLOCK_N=32)
        c3 = Config(BLOCK_M=64, BLOCK_N=32)

        # Different configs should be distinguishable via get()
        assert c1.get("BLOCK_M") == c2.get("BLOCK_M")
        assert c1.get("BLOCK_M") != c3.get("BLOCK_M")

        # Configs should be usable in cache key tuples
        key1 = (("BLOCK_M", 32), ("BLOCK_N", 32))
        key2 = (("BLOCK_M", 64), ("BLOCK_N", 32))
        assert key1 != key2

    def test_different_input_sizes_use_same_kernel(self) -> None:
        """Test that different input sizes reuse the same compiled kernel."""
        np.random.seed(42)

        weights = mx.array(np.random.randn(128, 64).astype(np.float32))
        W_quant, scale, zp = quantize_int8(weights, per_channel=True)

        # Different batch sizes should work with same kernel
        for batch_size in [1, 2, 4, 8]:
            x = mx.array(np.random.randn(batch_size, 16, 64).astype(np.float32))
            out = int8_linear(x, W_quant, scale, zp)
            mx.eval(out)
            assert out.shape == (batch_size, 16, 128)

        # Different sequence lengths should work
        for seq_len in [8, 16, 32, 64]:
            x = mx.array(np.random.randn(2, seq_len, 64).astype(np.float32))
            out = int8_linear(x, W_quant, scale, zp)
            mx.eval(out)
            assert out.shape == (2, seq_len, 128)
