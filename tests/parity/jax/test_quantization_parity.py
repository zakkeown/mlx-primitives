"""JAX Metal parity tests for quantization operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from tests.reference_jax_extended import (
        jax_quantize_int8,
        jax_dequantize_int8,
        jax_quantize_int4,
        jax_dequantize_int4,
        jax_int8_linear,
        jax_int4_linear,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestINT8QuantizationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test INT8 quantization forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import quantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        n, k = config["n"], config["k"]

        np.random.seed(42)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # MLX INT8 quantization
        weight_mlx = mx.array(weight_np)
        mlx_q, mlx_scale, mlx_zp = quantize_tensor(
            weight_mlx, num_bits=8, per_channel=False, symmetric=True
        )
        mx.eval(mlx_q, mlx_scale)

        # JAX reference
        jax_q, jax_scale, jax_zp = jax_quantize_int8(
            weight_np, per_channel=False, symmetric=True
        )

        # Compare quantized weights (exact match for integers)
        np.testing.assert_array_equal(
            _to_numpy(mlx_q), jax_q,
            err_msg=f"INT8 quantization mismatch (JAX) [{size}]"
        )

        # Compare scales (should be very close)
        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_scale), jax_scale,
            rtol=rtol, atol=atol,
            err_msg=f"INT8 scale mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_scale_computation(self, skip_without_jax):
        """Test INT8 scale computation matches JAX reference."""
        from mlx_primitives.advanced.quantization import quantize_tensor

        np.random.seed(42)
        x_np = np.random.randn(64, 64).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        _, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        mx.eval(scale_mlx)

        # JAX reference: scale = absmax / 127
        x_absmax = np.max(np.abs(x_np))
        scale_jax = x_absmax / 127.0

        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(scale_mlx),
            scale_jax,
            rtol=rtol, atol=atol,
            err_msg="INT8 scale computation mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_zero_point_computation(self, skip_without_jax):
        """Test INT8 zero point computation for asymmetric quantization."""
        from mlx_primitives.advanced.quantization import quantize_tensor

        np.random.seed(42)
        # Use asymmetric data (all positive) to test zero point
        x_np = np.random.uniform(0, 10, (64, 64)).astype(np.float32)

        # MLX asymmetric quantization
        x_mlx = mx.array(x_np)
        _, scale_mlx, zp_mlx = quantize_tensor(x_mlx, num_bits=8, symmetric=False)
        mx.eval(scale_mlx, zp_mlx)

        # JAX reference asymmetric quantization
        x_min = np.min(x_np)
        x_max = np.max(x_np)
        qmin, qmax = -128, 127
        scale_jax = (x_max - x_min) / (qmax - qmin)
        scale_jax = max(scale_jax, 1e-8)
        zp_jax = qmin - np.round(x_min / scale_jax)
        zp_jax = np.clip(zp_jax, qmin, qmax)

        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        # Allow 1 unit difference due to rounding
        np.testing.assert_allclose(
            _to_numpy(zp_mlx),
            zp_jax,
            rtol=rtol, atol=max(atol, 1.0),
            err_msg="INT8 zero point computation mismatch (JAX)"
        )


class TestINT8DequantizationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test INT8 dequantization forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import quantize_tensor, dequantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        n, k = config["n"], config["k"]

        np.random.seed(42)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # First quantize using MLX
        weight_mlx = mx.array(weight_np)
        mlx_q, mlx_scale, mlx_zp = quantize_tensor(
            weight_mlx, num_bits=8, per_channel=False, symmetric=True
        )
        mx.eval(mlx_q, mlx_scale)

        # MLX dequantization
        mlx_deq = dequantize_tensor(mlx_q, mlx_scale, mlx_zp)
        mx.eval(mlx_deq)

        # JAX reference dequantization (using same quantized values)
        jax_deq = jax_dequantize_int8(
            _to_numpy(mlx_q), _to_numpy(mlx_scale), None  # symmetric has no zero point
        )

        rtol, atol = get_tolerance("quantization", "int8_dequantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_deq), jax_deq,
            rtol=rtol, atol=atol,
            err_msg=f"INT8 dequantization mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_jax):
        """Test quantize -> dequantize roundtrip error matches JAX."""
        from mlx_primitives.advanced.quantization import quantize_tensor, dequantize_tensor

        np.random.seed(42)
        x_np = np.random.randn(128, 128).astype(np.float32)

        # MLX roundtrip
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        x_deq_mlx = dequantize_tensor(x_q_mlx, scale_mlx, None)
        mx.eval(x_deq_mlx)
        mlx_error = np.abs(_to_numpy(x_deq_mlx) - x_np).mean()

        # JAX reference roundtrip
        jax_q, jax_scale, _ = jax_quantize_int8(x_np, per_channel=False, symmetric=True)
        jax_deq = jax_dequantize_int8(jax_q, jax_scale, None)
        jax_error = np.abs(jax_deq - x_np).mean()

        # Roundtrip errors should be comparable
        rtol, atol = get_tolerance("quantization", "int8_dequantize", "fp32")
        np.testing.assert_allclose(
            mlx_error, jax_error,
            rtol=rtol, atol=atol,
            err_msg="INT8 roundtrip error mismatch (JAX)"
        )


class TestINT4QuantizationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_forward_parity(self, size, group_size, skip_without_jax):
        """Test INT4 quantization forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        n, k = config["n"], config["k"]

        # Ensure dimensions are compatible with group_size
        k = max(k, group_size)

        np.random.seed(42)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # Create MLX linear layer and convert to Int4
        linear_mlx = nn.Linear(k, n, bias=False)
        linear_mlx.weight = mx.array(weight_np)
        mx.eval(linear_mlx.parameters())

        int4_mlx = Int4Linear.from_linear(linear_mlx, group_size=group_size)
        mx.eval(int4_mlx.scales)

        # JAX reference quantization
        jax_q, jax_scales, jax_num_groups = jax_quantize_int4(weight_np, group_size=group_size)

        # Compare scales (main quantization parameter)
        rtol, atol = get_tolerance("quantization", "int4_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(int4_mlx.scales), jax_scales,
            rtol=rtol, atol=atol,
            err_msg=f"INT4 scales mismatch (JAX) [{size}, group_size={group_size}]"
        )


class TestINT4DequantizationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_forward_parity(self, size, group_size, skip_without_jax):
        """Test INT4 dequantization forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        n, k = config["n"], config["k"]

        # Ensure dimensions are compatible with group_size
        k = max(k, group_size)

        np.random.seed(42)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # Create MLX Int4Linear
        linear_mlx = nn.Linear(k, n, bias=False)
        linear_mlx.weight = mx.array(weight_np)
        mx.eval(linear_mlx.parameters())

        int4_mlx = Int4Linear.from_linear(linear_mlx, group_size=group_size)
        mx.eval(int4_mlx.scales, int4_mlx.weight_packed)

        # Get unpacked (dequantized) weights from MLX
        mlx_deq = int4_mlx.unpack_weights()
        mx.eval(mlx_deq)

        # JAX reference: quantize then dequantize
        jax_q, jax_scales, _ = jax_quantize_int4(weight_np, group_size=group_size)
        jax_deq = jax_dequantize_int4(jax_q, jax_scales, group_size)

        rtol, atol = get_tolerance("quantization", "int4_dequantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_deq), jax_deq,
            rtol=rtol, atol=atol,
            err_msg=f"INT4 dequantization mismatch (JAX) [{size}, group_size={group_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_jax):
        """Test INT4 quantize -> dequantize roundtrip error."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        np.random.seed(42)
        n, k = 128, 128
        group_size = 32
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # MLX roundtrip via Int4Linear
        linear_mlx = nn.Linear(k, n, bias=False)
        linear_mlx.weight = mx.array(weight_np)
        mx.eval(linear_mlx.parameters())

        int4_mlx = Int4Linear.from_linear(linear_mlx, group_size=group_size)
        mx.eval(int4_mlx.scales, int4_mlx.weight_packed)

        mlx_deq = int4_mlx.unpack_weights()
        mx.eval(mlx_deq)
        mlx_error = np.abs(_to_numpy(mlx_deq) - weight_np).mean()

        # JAX reference roundtrip
        jax_q, jax_scales, _ = jax_quantize_int4(weight_np, group_size=group_size)
        jax_deq = jax_dequantize_int4(jax_q, jax_scales, group_size)
        jax_error = np.abs(jax_deq - weight_np).mean()

        # Roundtrip errors should be comparable (INT4 has more error than INT8)
        rtol, atol = get_tolerance("quantization", "int4_dequantize", "fp32")
        np.testing.assert_allclose(
            mlx_error, jax_error,
            rtol=rtol, atol=atol,
            err_msg="INT4 roundtrip error mismatch (JAX)"
        )


class TestINT8LinearParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test INT8 linear layer forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import QuantizedLinear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, k).astype(np.float32)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02
        bias_np = np.random.randn(n).astype(np.float32) * 0.01

        # Create MLX QuantizedLinear
        linear_mlx = nn.Linear(k, n, bias=True)
        linear_mlx.weight = mx.array(weight_np)
        linear_mlx.bias = mx.array(bias_np)
        mx.eval(linear_mlx.parameters())

        qlinear_mlx = QuantizedLinear.from_linear(linear_mlx, num_bits=8, per_channel=True)
        mx.eval(qlinear_mlx.weight_q, qlinear_mlx.weight_scale)

        # MLX forward
        x_mlx = mx.array(x_np)
        mlx_out = qlinear_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_int8_linear(
            x_np,
            _to_numpy(qlinear_mlx.weight_q),
            _to_numpy(qlinear_mlx.weight_scale),
            None,  # symmetric
            bias_np,
        )

        rtol, atol = get_tolerance("quantization", "int8_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"INT8 linear mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_jax):
        """Test INT8 linear vs FP32 linear error bounds."""
        from mlx_primitives.advanced.quantization import QuantizedLinear
        import mlx.nn as nn

        np.random.seed(42)
        batch_seq, in_features, out_features = 64, 256, 256
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02

        # MLX FP32 baseline
        fp32_layer = nn.Linear(in_features, out_features, bias=False)
        fp32_layer.weight = mx.array(weight_np)
        mx.eval(fp32_layer.parameters())
        x_mlx = mx.array(x_np)
        fp32_out = fp32_layer(x_mlx)
        mx.eval(fp32_out)

        # MLX INT8 quantized
        qlinear_mlx = QuantizedLinear.from_linear(fp32_layer, num_bits=8, per_channel=True)
        mx.eval(qlinear_mlx.weight_q, qlinear_mlx.weight_scale)
        int8_out = qlinear_mlx(x_mlx)
        mx.eval(int8_out)

        mlx_error = np.abs(_to_numpy(int8_out) - _to_numpy(fp32_out)).mean()

        # JAX FP32 baseline
        fp32_out_jax = x_np @ weight_np.T

        # JAX INT8 quantized (per-channel)
        w_absmax = np.max(np.abs(weight_np), axis=1, keepdims=True)
        w_scale = w_absmax / 127.0
        w_scale = np.clip(w_scale, 1e-8, None)
        w_q = np.round(weight_np / w_scale)
        w_q = np.clip(w_q, -128, 127)
        w_deq = w_q * w_scale
        int8_out_jax = x_np @ w_deq.T

        jax_error = np.abs(int8_out_jax - fp32_out_jax).mean()

        # Quantization error should be similar between frameworks
        rtol, atol = get_tolerance("quantization", "int8_linear", "fp32")
        np.testing.assert_allclose(
            mlx_error, jax_error,
            rtol=rtol, atol=atol,
            err_msg="INT8 vs FP32 error bounds mismatch (JAX)"
        )


class TestINT4LinearParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_forward_parity(self, size, group_size, skip_without_jax):
        """Test INT4 linear layer forward pass parity with JAX."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        # Ensure dimensions are compatible with group_size
        k = max(k, group_size)

        np.random.seed(42)
        x_np = np.random.randn(m, k).astype(np.float32)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02
        bias_np = np.random.randn(n).astype(np.float32) * 0.01

        # Create MLX Int4Linear
        linear_mlx = nn.Linear(k, n, bias=True)
        linear_mlx.weight = mx.array(weight_np)
        linear_mlx.bias = mx.array(bias_np)
        mx.eval(linear_mlx.parameters())

        int4_mlx = Int4Linear.from_linear(linear_mlx, group_size=group_size)
        int4_mlx.bias = mx.array(bias_np)
        mx.eval(int4_mlx.scales, int4_mlx.weight_packed, int4_mlx.bias)

        # MLX forward
        x_mlx = mx.array(x_np)
        mlx_out = int4_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference: first get quantized weights from JAX
        jax_q, jax_scales, _ = jax_quantize_int4(weight_np, group_size=group_size)
        jax_out = jax_int4_linear(x_np, jax_q, jax_scales, group_size, bias_np)

        rtol, atol = get_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"INT4 linear mismatch (JAX) [{size}, group_size={group_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_jax):
        """Test INT4 linear vs FP32 linear error bounds."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        np.random.seed(42)
        batch_seq, in_features, out_features = 64, 256, 256
        group_size = 32
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02

        # MLX FP32 baseline
        fp32_layer = nn.Linear(in_features, out_features, bias=False)
        fp32_layer.weight = mx.array(weight_np)
        mx.eval(fp32_layer.parameters())
        x_mlx = mx.array(x_np)
        fp32_out = fp32_layer(x_mlx)
        mx.eval(fp32_out)

        # MLX INT4 quantized
        int4_layer = Int4Linear.from_linear(fp32_layer, group_size=group_size)
        mx.eval(int4_layer.scales, int4_layer.weight_packed)
        int4_out = int4_layer(x_mlx)
        mx.eval(int4_out)

        mlx_error = np.abs(_to_numpy(int4_out) - _to_numpy(fp32_out)).mean()

        # JAX FP32 baseline
        fp32_out_jax = x_np @ weight_np.T

        # JAX INT4 reference
        jax_q, jax_scales, _ = jax_quantize_int4(weight_np, group_size=group_size)
        int4_out_jax = jax_int4_linear(x_np, jax_q, jax_scales, group_size, None)

        jax_error = np.abs(int4_out_jax - fp32_out_jax).mean()

        # Quantization error should be similar between frameworks
        # INT4 typically has ~5-10% relative error (higher than INT8)
        rtol, atol = get_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            mlx_error, jax_error,
            rtol=rtol, atol=atol,
            err_msg="INT4 vs FP32 error bounds mismatch (JAX)"
        )


# =============================================================================
# Backward Parity Tests
# =============================================================================

class TestINT8LinearBackwardParity:
    """INT8 linear backward parity tests.

    Quantized linear layers support backpropagation by dequantizing weights
    during the backward pass. We verify gradients flow correctly.
    """

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test INT8 linear backward pass gradients flow correctly."""
        from mlx_primitives.advanced.quantization import QuantizedLinear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, k).astype(np.float32)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # Create MLX QuantizedLinear
        linear_mlx = nn.Linear(k, n, bias=False)
        linear_mlx.weight = mx.array(weight_np)
        mx.eval(linear_mlx.parameters())

        qlinear_mlx = QuantizedLinear.from_linear(linear_mlx, num_bits=8, per_channel=True)
        mx.eval(qlinear_mlx.weight_q, qlinear_mlx.weight_scale)

        # MLX backward
        def mlx_loss_fn(x):
            return mx.sum(qlinear_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # Verify gradient has correct shape and no NaN/Inf
        assert mlx_grad.shape == x_mlx.shape, "Gradient shape mismatch"

        grad_np = _to_numpy(mlx_grad)
        assert not np.isnan(grad_np).any(), f"NaN in INT8 linear gradient [{size}]"
        assert not np.isinf(grad_np).any(), f"Inf in INT8 linear gradient [{size}]"

        # Verify gradient is non-trivial (some flow occurred)
        assert np.abs(grad_np).sum() > 1e-6, f"INT8 linear gradient is all zeros [{size}]"


class TestINT4LinearBackwardParity:
    """INT4 linear backward parity tests.

    INT4 quantized linear layers support backpropagation by dequantizing weights
    during the backward pass. We verify gradients flow correctly.
    """

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("group_size", [32, 64])
    def test_backward_parity(self, size, group_size, skip_without_jax):
        """Test INT4 linear backward pass gradients flow correctly."""
        from mlx_primitives.advanced.quantization import Int4Linear
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]
        k = max(k, group_size)

        np.random.seed(42)
        x_np = np.random.randn(m, k).astype(np.float32)
        weight_np = np.random.randn(n, k).astype(np.float32) * 0.02

        # Create MLX Int4Linear
        linear_mlx = nn.Linear(k, n, bias=False)
        linear_mlx.weight = mx.array(weight_np)
        mx.eval(linear_mlx.parameters())

        int4_mlx = Int4Linear.from_linear(linear_mlx, group_size=group_size)
        mx.eval(int4_mlx.scales, int4_mlx.weight_packed)

        # MLX backward
        def mlx_loss_fn(x):
            return mx.sum(int4_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # Verify gradient has correct shape and no NaN/Inf
        assert mlx_grad.shape == x_mlx.shape, "Gradient shape mismatch"

        grad_np = _to_numpy(mlx_grad)
        assert not np.isnan(grad_np).any(), f"NaN in INT4 linear gradient [{size}, group_size={group_size}]"
        assert not np.isinf(grad_np).any(), f"Inf in INT4 linear gradient [{size}, group_size={group_size}]"

        # Verify gradient is non-trivial (some flow occurred)
        assert np.abs(grad_np).sum() > 1e-6, f"INT4 linear gradient is all zeros [{size}, group_size={group_size}]"
