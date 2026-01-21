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
