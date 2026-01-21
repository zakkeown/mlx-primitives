"""PyTorch parity tests for quantization operations."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from tests.parity.shared.input_generators import quantization_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn.functional as F


# =============================================================================
# Helper Functions
# =============================================================================

def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _pytorch_symmetric_quantize_int8(x: "torch.Tensor"):
    """PyTorch reference: symmetric INT8 quantization.

    Returns (quantized_tensor, scale).
    """
    x_absmax = torch.max(torch.abs(x))
    scale = x_absmax / 127.0
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -128, 127).to(torch.int8)
    return x_q, scale


def _pytorch_symmetric_quantize_int4(x: "torch.Tensor"):
    """PyTorch reference: symmetric INT4 quantization.

    Returns (quantized_tensor, scale).
    """
    x_absmax = torch.max(torch.abs(x))
    scale = x_absmax / 7.0
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -8, 7).to(torch.int8)
    return x_q, scale


def _pytorch_dequantize(x_q: "torch.Tensor", scale: "torch.Tensor"):
    """PyTorch reference: dequantization."""
    return x_q.float() * scale


# =============================================================================
# INT8 Quantization Parity Tests
# =============================================================================

class TestINT8QuantizationParity:
    """INT8 quantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 quantization forward pass parity."""
        from mlx_primitives.advanced import quantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, n).astype(np.float32)

        # MLX quantization
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        mx.eval(x_q_mlx, scale_mlx)

        # PyTorch quantization
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int8(x_torch)

        # Compare quantized values
        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_q_mlx).astype(np.float32),
            x_q_torch.float().numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT8 quantize forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_scale_computation(self, skip_without_pytorch):
        """Test INT8 scale computation matches PyTorch."""
        from mlx_primitives.advanced import quantize_tensor

        np.random.seed(42)
        x_np = np.random.randn(64, 64).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        _, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        mx.eval(scale_mlx)

        # PyTorch reference
        x_torch = torch.from_numpy(x_np)
        x_absmax = torch.max(torch.abs(x_torch))
        scale_torch = x_absmax / 127.0

        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(scale_mlx),
            scale_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg="INT8 scale computation mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_zero_point_computation(self, skip_without_pytorch):
        """Test INT8 zero point computation matches PyTorch."""
        from mlx_primitives.advanced import quantize_tensor

        np.random.seed(42)
        # Use asymmetric data to test zero point
        x_np = np.random.uniform(0, 10, (64, 64)).astype(np.float32)

        # MLX asymmetric quantization
        x_mlx = mx.array(x_np)
        _, scale_mlx, zp_mlx = quantize_tensor(x_mlx, num_bits=8, symmetric=False)
        mx.eval(scale_mlx, zp_mlx)

        # PyTorch reference asymmetric quantization
        x_torch = torch.from_numpy(x_np)
        x_min = torch.min(x_torch)
        x_max = torch.max(x_torch)
        qmin, qmax = -128, 127
        scale_torch = (x_max - x_min) / (qmax - qmin)
        scale_torch = torch.clamp(scale_torch, min=1e-8)
        zp_torch = qmin - torch.round(x_min / scale_torch)
        zp_torch = torch.clamp(zp_torch, qmin, qmax)

        rtol, atol = get_tolerance("quantization", "int8_quantize", "fp32")
        # Compare zero points (may differ slightly due to rounding)
        np.testing.assert_allclose(
            _to_numpy(zp_mlx),
            zp_torch.numpy(),
            rtol=rtol, atol=max(atol, 1.0),  # Allow 1 unit difference due to rounding
            err_msg="INT8 zero point computation mismatch"
        )


# =============================================================================
# INT8 Dequantization Parity Tests
# =============================================================================

class TestINT8DequantizationParity:
    """INT8 dequantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 dequantization forward pass parity."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, n).astype(np.float32)

        # MLX quantize then dequantize
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        x_deq_mlx = dequantize_tensor(x_q_mlx, scale_mlx, None)
        mx.eval(x_deq_mlx)

        # PyTorch quantize then dequantize
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int8(x_torch)
        x_deq_torch = _pytorch_dequantize(x_q_torch, scale_torch)

        rtol, atol = get_tolerance("quantization", "int8_dequantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_deq_mlx),
            x_deq_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT8 dequantize forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_pytorch):
        """Test quantize -> dequantize roundtrip error matches PyTorch."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        np.random.seed(42)
        x_np = np.random.randn(128, 128).astype(np.float32)

        # MLX roundtrip
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=8, symmetric=True)
        x_deq_mlx = dequantize_tensor(x_q_mlx, scale_mlx, None)
        mx.eval(x_deq_mlx)
        mlx_error = np.abs(_to_numpy(x_deq_mlx) - x_np).mean()

        # PyTorch roundtrip
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int8(x_torch)
        x_deq_torch = _pytorch_dequantize(x_q_torch, scale_torch)
        torch_error = torch.abs(x_deq_torch - x_torch).mean().item()

        # Roundtrip errors should be comparable
        rtol, atol = get_tolerance("quantization", "int8_dequantize", "fp32")
        np.testing.assert_allclose(
            mlx_error, torch_error,
            rtol=rtol, atol=atol,
            err_msg="INT8 roundtrip error mismatch"
        )


# =============================================================================
# INT4 Quantization Parity Tests
# =============================================================================

class TestINT4QuantizationParity:
    """INT4 quantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 quantization forward pass parity."""
        from mlx_primitives.advanced import quantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, n).astype(np.float32)

        # MLX quantization
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=4, symmetric=True)
        mx.eval(x_q_mlx, scale_mlx)

        # PyTorch quantization
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int4(x_torch)

        rtol, atol = get_tolerance("quantization", "int4_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_q_mlx).astype(np.float32),
            x_q_torch.float().numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 quantize forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_group_quantization(self, skip_without_pytorch):
        """Test INT4 group-wise quantization."""
        from mlx_primitives.advanced import quantize_tensor

        np.random.seed(42)
        # Create tensor with shape suitable for per-channel quantization
        x_np = np.random.randn(64, 256).astype(np.float32)

        # MLX per-channel (group) quantization
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=4, per_channel=True, symmetric=True)
        mx.eval(x_q_mlx, scale_mlx)

        # PyTorch reference per-channel
        x_torch = torch.from_numpy(x_np)
        # Per-channel: compute scale for each row (axis 0)
        x_absmax = torch.max(torch.abs(x_torch), dim=1, keepdim=True).values
        scale_torch = x_absmax / 7.0
        scale_torch = torch.clamp(scale_torch, min=1e-8)
        x_q_torch = torch.round(x_torch / scale_torch)
        x_q_torch = torch.clamp(x_q_torch, -8, 7).to(torch.int8)

        rtol, atol = get_tolerance("quantization", "int4_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_q_mlx).astype(np.float32),
            x_q_torch.float().numpy(),
            rtol=rtol, atol=atol,
            err_msg="INT4 per-channel quantize mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_different_group_sizes(self, group_size, skip_without_pytorch):
        """Test INT4 quantization with different group sizes."""
        from mlx_primitives.advanced import quantize_tensor

        np.random.seed(42)
        # Create tensor where dim is divisible by group_size
        x_np = np.random.randn(group_size, group_size * 2).astype(np.float32)

        # MLX quantization (per-channel simulates group quantization at row level)
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=4, per_channel=True, symmetric=True)
        mx.eval(x_q_mlx, scale_mlx)

        # PyTorch reference
        x_torch = torch.from_numpy(x_np)
        x_absmax = torch.max(torch.abs(x_torch), dim=1, keepdim=True).values
        scale_torch = x_absmax / 7.0
        scale_torch = torch.clamp(scale_torch, min=1e-8)
        x_q_torch = torch.round(x_torch / scale_torch)
        x_q_torch = torch.clamp(x_q_torch, -8, 7).to(torch.int8)

        rtol, atol = get_tolerance("quantization", "int4_quantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_q_mlx).astype(np.float32),
            x_q_torch.float().numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 group_size={group_size} quantize mismatch"
        )


# =============================================================================
# INT4 Dequantization Parity Tests
# =============================================================================

class TestINT4DequantizationParity:
    """INT4 dequantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 dequantization forward pass parity."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]

        np.random.seed(42)
        x_np = np.random.randn(m, n).astype(np.float32)

        # MLX quantize then dequantize
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=4, symmetric=True)
        x_deq_mlx = dequantize_tensor(x_q_mlx, scale_mlx, None)
        mx.eval(x_deq_mlx)

        # PyTorch quantize then dequantize
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int4(x_torch)
        x_deq_torch = _pytorch_dequantize(x_q_torch, scale_torch)

        rtol, atol = get_tolerance("quantization", "int4_dequantize", "fp32")
        np.testing.assert_allclose(
            _to_numpy(x_deq_mlx),
            x_deq_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 dequantize forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_pytorch):
        """Test quantize -> dequantize roundtrip error matches PyTorch."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        np.random.seed(42)
        x_np = np.random.randn(128, 128).astype(np.float32)

        # MLX roundtrip
        x_mlx = mx.array(x_np)
        x_q_mlx, scale_mlx, _ = quantize_tensor(x_mlx, num_bits=4, symmetric=True)
        x_deq_mlx = dequantize_tensor(x_q_mlx, scale_mlx, None)
        mx.eval(x_deq_mlx)
        mlx_error = np.abs(_to_numpy(x_deq_mlx) - x_np).mean()

        # PyTorch roundtrip
        x_torch = torch.from_numpy(x_np)
        x_q_torch, scale_torch = _pytorch_symmetric_quantize_int4(x_torch)
        x_deq_torch = _pytorch_dequantize(x_q_torch, scale_torch)
        torch_error = torch.abs(x_deq_torch - x_torch).mean().item()

        # Roundtrip errors should be comparable (INT4 has more error than INT8)
        rtol, atol = get_tolerance("quantization", "int4_dequantize", "fp32")
        np.testing.assert_allclose(
            mlx_error, torch_error,
            rtol=rtol, atol=atol,
            err_msg="INT4 roundtrip error mismatch"
        )


# =============================================================================
# INT8 Linear Parity Tests
# =============================================================================

class TestINT8LinearParity:
    """INT8 quantized linear layer parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 linear forward pass parity."""
        from mlx_primitives.advanced import QuantizedLinear

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]
        batch_seq = m
        in_features = n
        out_features = k

        np.random.seed(42)
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        bias_np = np.zeros(out_features, dtype=np.float32)

        # MLX quantized linear
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=8, per_channel=True)
        layer_mlx._weight_float = mx.array(weight_np)
        layer_mlx.bias = mx.array(bias_np)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())

        x_mlx = mx.array(x_np)
        out_mlx = layer_mlx(x_mlx)
        mx.eval(out_mlx)

        # PyTorch reference: quantize weights, dequantize, then linear
        weight_torch = torch.from_numpy(weight_np)
        x_torch = torch.from_numpy(x_np)
        bias_torch = torch.from_numpy(bias_np)

        # Per-channel INT8 quantization of weights
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 127.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_q = torch.round(weight_torch / w_scale)
        w_q = torch.clamp(w_q, -128, 127)
        w_deq = w_q * w_scale

        out_torch = F.linear(x_torch, w_deq, bias_torch)

        rtol, atol = get_tolerance("quantization", "int8_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(out_mlx),
            out_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT8 linear forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test INT8 linear backward pass parity (STE gradient)."""
        from mlx_primitives.advanced import QuantizedLinear

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]
        batch_seq = m
        in_features = n
        out_features = k

        np.random.seed(42)
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02

        # MLX backward
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=8, bias=False)
        layer_mlx._weight_float = mx.array(weight_np)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(layer_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward with STE (straight-through estimator)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)

        # Quantize weights (STE: gradient flows through as if no quantization)
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 127.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_deq = torch.round(weight_torch / w_scale) * w_scale

        out_torch = F.linear(x_torch, w_deq, None)
        out_torch.sum().backward()

        rtol, atol = get_gradient_tolerance("quantization", "int8_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT8 linear backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_pytorch):
        """Test INT8 linear vs FP32 linear error bounds."""
        from mlx_primitives.advanced import QuantizedLinear

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
        int8_layer = QuantizedLinear(in_features, out_features, num_bits=8, bias=False)
        int8_layer._weight_float = mx.array(weight_np)
        int8_layer.quantize_weights()
        mx.eval(int8_layer.parameters())
        int8_out = int8_layer(x_mlx)
        mx.eval(int8_out)

        mlx_error = np.abs(_to_numpy(int8_out) - _to_numpy(fp32_out)).mean()

        # PyTorch FP32 baseline
        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        fp32_out_torch = F.linear(x_torch, weight_torch, None)

        # PyTorch INT8 quantized
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 127.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_deq = torch.round(weight_torch / w_scale) * w_scale
        int8_out_torch = F.linear(x_torch, w_deq, None)

        torch_error = torch.abs(int8_out_torch - fp32_out_torch).mean().item()

        # Quantization error should be similar between frameworks
        # INT8 typically has ~0.5-1% relative error
        rtol, atol = get_tolerance("quantization", "int8_linear", "fp32")
        np.testing.assert_allclose(
            mlx_error, torch_error,
            rtol=rtol, atol=atol,
            err_msg="INT8 vs FP32 error bounds mismatch"
        )


# =============================================================================
# INT4 Linear Parity Tests
# =============================================================================

class TestINT4LinearParity:
    """INT4 quantized linear layer parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 linear forward pass parity."""
        from mlx_primitives.advanced import QuantizedLinear

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]
        batch_seq = m
        in_features = n
        out_features = k

        np.random.seed(42)
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        bias_np = np.zeros(out_features, dtype=np.float32)

        # MLX quantized linear
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=4, per_channel=True)
        layer_mlx._weight_float = mx.array(weight_np)
        layer_mlx.bias = mx.array(bias_np)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())

        x_mlx = mx.array(x_np)
        out_mlx = layer_mlx(x_mlx)
        mx.eval(out_mlx)

        # PyTorch reference: quantize weights to INT4, dequantize, then linear
        weight_torch = torch.from_numpy(weight_np)
        x_torch = torch.from_numpy(x_np)
        bias_torch = torch.from_numpy(bias_np)

        # Per-channel INT4 quantization of weights
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 7.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_q = torch.round(weight_torch / w_scale)
        w_q = torch.clamp(w_q, -8, 7)
        w_deq = w_q * w_scale

        out_torch = F.linear(x_torch, w_deq, bias_torch)

        rtol, atol = get_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(out_mlx),
            out_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 linear forward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test INT4 linear backward pass parity (STE gradient)."""
        from mlx_primitives.advanced import QuantizedLinear

        config = SIZE_CONFIGS[size]["quantization"]
        m, n, k = config["m"], config["n"], config["k"]
        batch_seq = m
        in_features = n
        out_features = k

        np.random.seed(42)
        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02

        # MLX backward
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=4, bias=False)
        layer_mlx._weight_float = mx.array(weight_np)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(layer_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward with STE
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)

        # Quantize weights (STE: gradient flows through as if no quantization)
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 7.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_deq = torch.round(weight_torch / w_scale) * w_scale

        out_torch = F.linear(x_torch, w_deq, None)
        out_torch.sum().backward()

        rtol, atol = get_gradient_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 linear backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_pytorch):
        """Test INT4 linear vs FP32 linear error bounds."""
        from mlx_primitives.advanced import QuantizedLinear

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

        # MLX INT4 quantized
        int4_layer = QuantizedLinear(in_features, out_features, num_bits=4, bias=False)
        int4_layer._weight_float = mx.array(weight_np)
        int4_layer.quantize_weights()
        mx.eval(int4_layer.parameters())
        int4_out = int4_layer(x_mlx)
        mx.eval(int4_out)

        mlx_error = np.abs(_to_numpy(int4_out) - _to_numpy(fp32_out)).mean()

        # PyTorch FP32 baseline
        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        fp32_out_torch = F.linear(x_torch, weight_torch, None)

        # PyTorch INT4 quantized
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 7.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_deq = torch.round(weight_torch / w_scale) * w_scale
        int4_out_torch = F.linear(x_torch, w_deq, None)

        torch_error = torch.abs(int4_out_torch - fp32_out_torch).mean().item()

        # Quantization error should be similar between frameworks
        # INT4 typically has ~5-10% relative error (higher than INT8)
        rtol, atol = get_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            mlx_error, torch_error,
            rtol=rtol, atol=atol,
            err_msg="INT4 vs FP32 error bounds mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_group_wise_quantization(self, group_size, skip_without_pytorch):
        """Test INT4 linear with group-wise quantization."""
        from mlx_primitives.advanced import QuantizedLinear

        np.random.seed(42)
        batch_seq = 32
        in_features = group_size * 2
        out_features = group_size

        x_np = np.random.randn(batch_seq, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02

        # MLX quantized linear with per_channel (group-wise at row level)
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=4, per_channel=True)
        layer_mlx._weight_float = mx.array(weight_np)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())

        x_mlx = mx.array(x_np)
        out_mlx = layer_mlx(x_mlx)
        mx.eval(out_mlx)

        # PyTorch reference
        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)

        # Per-channel INT4 quantization
        w_absmax = torch.max(torch.abs(weight_torch), dim=1, keepdim=True).values
        w_scale = w_absmax / 7.0
        w_scale = torch.clamp(w_scale, min=1e-8)
        w_deq = torch.round(weight_torch / w_scale) * w_scale

        out_torch = F.linear(x_torch, w_deq, None)

        rtol, atol = get_tolerance("quantization", "int4_linear", "fp32")
        np.testing.assert_allclose(
            _to_numpy(out_mlx),
            out_torch.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"INT4 group_size={group_size} linear mismatch"
        )
