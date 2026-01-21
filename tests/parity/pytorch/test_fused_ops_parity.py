"""PyTorch parity tests for fused operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn.functional as F


# =============================================================================
# Helper Functions
# =============================================================================


def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_torch(x_np: np.ndarray, dtype: str) -> "torch.Tensor":
    """Convert numpy array to PyTorch with proper dtype."""
    x_torch = torch.from_numpy(x_np.astype(np.float32))
    torch_dtype = get_pytorch_dtype(dtype)
    return x_torch.to(torch_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# Fused RMSNorm + Linear Parity Tests
# =============================================================================


class TestFusedRMSNormLinearParity:
    """Fused RMSNorm + Linear parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused RMSNorm+Linear forward pass parity."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        out_features = hidden * 4
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)
        mlx_out = fused_rmsnorm_linear(x_mlx, norm_weight_mlx, linear_weight_mlx, eps=eps)
        mx.eval(mlx_out)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        norm_weight_torch = torch.from_numpy(norm_weight_np).to(x_torch.dtype)
        linear_weight_torch = torch.from_numpy(linear_weight_np).to(x_torch.dtype)

        # RMSNorm + Linear
        rms = torch.sqrt(torch.mean(x_torch**2, dim=-1, keepdim=True) + eps)
        norm_x = (x_torch / rms) * norm_weight_torch
        torch_out = F.linear(norm_x, linear_weight_torch)

        rtol, atol = get_tolerance("fused_ops", "fused_rmsnorm_linear", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(torch_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RMSNorm+Linear forward mismatch [{size}, {dtype}]",
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused RMSNorm+Linear backward pass parity."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        out_features = hidden * 4
        eps = 1e-5
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        # MLX backward
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_rmsnorm_linear(x, norm_weight_mlx, linear_weight_mlx, eps=eps))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        norm_weight_torch = torch.from_numpy(norm_weight_np)
        linear_weight_torch = torch.from_numpy(linear_weight_np)

        rms = torch.sqrt(torch.mean(x_torch**2, dim=-1, keepdim=True) + eps)
        norm_x = (x_torch / rms) * norm_weight_torch
        torch_out = F.linear(norm_x, linear_weight_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_rmsnorm_linear", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            x_torch.grad.numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RMSNorm+Linear backward mismatch [{size}]",
        )

    @pytest.mark.forward_parity
    def test_vs_separate_ops(self):
        """Test that fused op matches separate RMSNorm and Linear."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        batch, seq, hidden = 4, 128, 512
        out_features = 2048
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.random.randn(hidden).astype(np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)

        # Fused version
        fused_out = fused_rmsnorm_linear(x_mlx, norm_weight_mlx, linear_weight_mlx, eps=eps)
        mx.eval(fused_out)

        # Separate operations
        rms = mx.sqrt(mx.mean(x_mlx * x_mlx, axis=-1, keepdims=True) + eps)
        norm_x = x_mlx / rms * norm_weight_mlx
        separate_out = norm_x @ linear_weight_mlx.T
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Fused vs separate RMSNorm+Linear mismatch",
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_bias(self, skip_without_pytorch):
        """Test fused op with bias parameter - comparing MLX to PyTorch."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        batch, seq, hidden = 2, 64, 256
        out_features = 1024
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        norm_weight_np = np.ones(hidden, dtype=np.float32)
        linear_weight_np = np.random.randn(out_features, hidden).astype(np.float32) * 0.02
        linear_bias_np = np.random.randn(out_features).astype(np.float32) * 0.01

        # MLX
        x_mlx = mx.array(x_np)
        norm_weight_mlx = mx.array(norm_weight_np)
        linear_weight_mlx = mx.array(linear_weight_np)
        linear_bias_mlx = mx.array(linear_bias_np)

        mlx_out = fused_rmsnorm_linear(
            x_mlx, norm_weight_mlx, linear_weight_mlx, linear_bias=linear_bias_mlx, eps=eps
        )
        mx.eval(mlx_out)

        # PyTorch
        x_torch = torch.from_numpy(x_np)
        norm_weight_torch = torch.from_numpy(norm_weight_np)
        linear_weight_torch = torch.from_numpy(linear_weight_np)
        linear_bias_torch = torch.from_numpy(linear_bias_np)

        rms = torch.sqrt(torch.mean(x_torch**2, dim=-1, keepdim=True) + eps)
        norm_x = (x_torch / rms) * norm_weight_torch
        torch_out = F.linear(norm_x, linear_weight_torch, linear_bias_torch)

        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(torch_out),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Fused RMSNorm+Linear with bias mismatch",
        )


# =============================================================================
# Fused SwiGLU Parity Tests
# =============================================================================


class TestFusedSwiGLUParity:
    """Fused SwiGLU parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused SwiGLU forward pass parity."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)
        mlx_out = fused_swiglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(mlx_out)

        # PyTorch: silu(x @ W_gate.T) * (x @ W_up.T)
        x_torch = _convert_to_torch(x_np, dtype)
        W_gate_torch = torch.from_numpy(W_gate_np).to(x_torch.dtype)
        W_up_torch = torch.from_numpy(W_up_np).to(x_torch.dtype)

        gate = F.silu(x_torch @ W_gate_torch.T)
        up = x_torch @ W_up_torch.T
        torch_out = gate * up

        rtol, atol = get_tolerance("fused_ops", "fused_swiglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(torch_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused SwiGLU forward mismatch [{size}, {dtype}]",
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused SwiGLU backward pass parity."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX backward
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_swiglu(x, W_gate_mlx, W_up_mlx))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        W_gate_torch = torch.from_numpy(W_gate_np)
        W_up_torch = torch.from_numpy(W_up_np)

        gate = F.silu(x_torch @ W_gate_torch.T)
        up = x_torch @ W_up_torch.T
        torch_out = gate * up
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_swiglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            x_torch.grad.numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused SwiGLU backward mismatch [{size}]",
        )

    @pytest.mark.forward_parity
    def test_vs_separate_ops(self):
        """Test that fused SwiGLU matches separate ops."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        batch, seq, dim = 4, 128, 512
        hidden_dim = 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        # Fused
        fused_out = fused_swiglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(fused_out)

        # Separate (reference implementation)
        gate = x_mlx @ W_gate_mlx.T
        up = x_mlx @ W_up_mlx.T
        silu_gate = gate * mx.sigmoid(gate)  # silu = x * sigmoid(x)
        separate_out = silu_gate * up
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate SwiGLU mismatch",
        )


# =============================================================================
# Fused GeGLU Parity Tests
# =============================================================================


class TestFusedGeGLUParity:
    """Fused GeGLU parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused GeGLU forward pass parity."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)
        mlx_out = fused_geglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(mlx_out)

        # PyTorch: gelu(x @ W_gate.T) * (x @ W_up.T)
        x_torch = _convert_to_torch(x_np, dtype)
        W_gate_torch = torch.from_numpy(W_gate_np).to(x_torch.dtype)
        W_up_torch = torch.from_numpy(W_up_np).to(x_torch.dtype)

        gate = F.gelu(x_torch @ W_gate_torch.T, approximate="tanh")
        up = x_torch @ W_up_torch.T
        torch_out = gate * up

        rtol, atol = get_tolerance("fused_ops", "fused_geglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(torch_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused GeGLU forward mismatch [{size}, {dtype}]",
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused GeGLU backward pass parity."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        # MLX backward
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        def mlx_loss_fn(x):
            return mx.sum(fused_geglu(x, W_gate_mlx, W_up_mlx))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        W_gate_torch = torch.from_numpy(W_gate_np)
        W_up_torch = torch.from_numpy(W_up_np)

        gate = F.gelu(x_torch @ W_gate_torch.T, approximate="tanh")
        up = x_torch @ W_up_torch.T
        torch_out = gate * up
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_geglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            x_torch.grad.numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused GeGLU backward mismatch [{size}]",
        )

    @pytest.mark.forward_parity
    def test_vs_separate_ops(self):
        """Test that fused GeGLU matches separate ops."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        batch, seq, dim = 4, 128, 512
        hidden_dim = 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        W_gate_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        W_up_np = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        W_gate_mlx = mx.array(W_gate_np)
        W_up_mlx = mx.array(W_up_np)

        # Fused
        fused_out = fused_geglu(x_mlx, W_gate_mlx, W_up_mlx)
        mx.eval(fused_out)

        # Separate (reference implementation using tanh approximation)
        gate = x_mlx @ W_gate_mlx.T
        up = x_mlx @ W_up_mlx.T
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715
        gelu_gate = 0.5 * gate * (1.0 + mx.tanh(sqrt_2_over_pi * (gate + coeff * gate**3)))
        separate_out = gelu_gate * up
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate GeGLU mismatch",
        )


# =============================================================================
# Fused RoPE + Attention Parity Tests
# =============================================================================


class TestFusedRoPEAttentionParity:
    """Fused RoPE + Attention parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused RoPE+Attention forward pass parity."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = (
            config["batch"],
            config["seq"],
            config["heads"],
            config["head_dim"],
        )
        scale = 1.0 / (head_dim**0.5)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        v_mlx = _convert_to_mlx(v_np, dtype)
        mlx_out = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(mlx_out)

        # PyTorch: apply RoPE then scaled_dot_product_attention
        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)
        v_torch = _convert_to_torch(v_np, dtype)

        # Compute RoPE
        positions = torch.arange(seq, device=q_torch.device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, head_dim, 2, device=q_torch.device, dtype=torch.float32) / head_dim
            )
        )
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = torch.cos(angles).to(q_torch.dtype)
        sin = torch.sin(angles).to(q_torch.dtype)

        def apply_rope(x, cos, sin):
            x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
            cos_expanded = cos[None, :, None, :]
            sin_expanded = sin[None, :, None, :]
            return torch.cat(
                [
                    x1 * cos_expanded - x2 * sin_expanded,
                    x1 * sin_expanded + x2 * cos_expanded,
                ],
                dim=-1,
            )

        q_rot = apply_rope(q_torch, cos, sin)
        k_rot = apply_rope(k_torch, cos, sin)

        # Transpose to (batch, heads, seq, dim) for SDPA
        q_t = q_rot.transpose(1, 2)
        k_t = k_rot.transpose(1, 2)
        v_t = v_torch.transpose(1, 2)

        torch_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
        torch_out = torch_out.transpose(1, 2)  # Back to (batch, seq, heads, dim)

        rtol, atol = get_tolerance("fused_ops", "fused_rope_attention", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out),
            _to_numpy(torch_out),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RoPE+Attention forward mismatch [{size}, {dtype}]",
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused RoPE+Attention backward pass parity."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        config = SIZE_CONFIGS[size]["attention"]
        batch, seq, heads, head_dim = (
            config["batch"],
            config["seq"],
            config["heads"],
            config["head_dim"],
        )
        scale = 1.0 / (head_dim**0.5)
        dtype = "fp32"

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX backward
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        def mlx_loss_fn(q):
            return mx.sum(fused_rope_attention(q, k_mlx, v_mlx, scale=scale, causal=True))

        q_mlx = mx.array(q_np)
        mlx_grad = mx.grad(mlx_loss_fn)(q_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        q_torch = torch.from_numpy(q_np).requires_grad_(True)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)

        positions = torch.arange(seq, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        def apply_rope(x, cos, sin):
            x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
            cos_expanded = cos[None, :, None, :]
            sin_expanded = sin[None, :, None, :]
            return torch.cat(
                [
                    x1 * cos_expanded - x2 * sin_expanded,
                    x1 * sin_expanded + x2 * cos_expanded,
                ],
                dim=-1,
            )

        q_rot = apply_rope(q_torch, cos, sin)
        k_rot = apply_rope(k_torch, cos, sin)

        q_t = q_rot.transpose(1, 2)
        k_t = k_rot.transpose(1, 2)
        v_t = v_torch.transpose(1, 2)

        torch_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("fused_ops", "fused_rope_attention", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad),
            q_torch.grad.numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Fused RoPE+Attention backward mismatch [{size}]",
        )

    @pytest.mark.forward_parity
    def test_vs_separate_ops(self):
        """Test that fused op matches separate RoPE and Attention."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention
        from mlx_primitives.attention.flash import flash_attention

        batch, seq, heads, head_dim = 2, 128, 8, 64
        scale = 1.0 / (head_dim**0.5)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        # Fused
        fused_out = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(fused_out)

        # Separate: apply RoPE manually + flash_attention
        positions = mx.arange(seq)
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        angles = positions[:, None] * inv_freq[None, :]
        cos = mx.cos(angles)
        sin = mx.sin(angles)

        def apply_rope_mlx(x, cos, sin):
            x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
            cos_expanded = cos[None, :, None, :]
            sin_expanded = sin[None, :, None, :]
            return mx.concatenate(
                [
                    x1 * cos_expanded - x2 * sin_expanded,
                    x1 * sin_expanded + x2 * cos_expanded,
                ],
                axis=-1,
            )

        q_rot = apply_rope_mlx(q_mlx, cos, sin)
        k_rot = apply_rope_mlx(k_mlx, cos, sin)
        separate_out = flash_attention(q_rot, k_rot, v_mlx, scale=scale, causal=True)
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out),
            _to_numpy(separate_out),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Fused vs separate RoPE+Attention mismatch",
        )

    @pytest.mark.forward_parity
    def test_causal_masking(self):
        """Test fused RoPE+Attention with causal masking."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        batch, seq, heads, head_dim = 2, 64, 4, 32
        scale = 1.0 / (head_dim**0.5)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)

        # Causal
        out_causal = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=True)
        mx.eval(out_causal)

        # Non-causal
        out_non_causal = fused_rope_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=False)
        mx.eval(out_non_causal)

        causal_np = _to_numpy(out_causal)
        non_causal_np = _to_numpy(out_non_causal)

        # Both outputs should have valid values (no NaN/Inf)
        assert not np.any(np.isnan(causal_np)), "NaN in causal output"
        assert not np.any(np.isinf(causal_np)), "Inf in causal output"
        assert not np.any(np.isnan(non_causal_np)), "NaN in non-causal output"
        assert not np.any(np.isinf(non_causal_np)), "Inf in non-causal output"

        # Causal and non-causal should produce different results
        # (they should differ because causal masks future tokens)
        assert not np.allclose(
            causal_np, non_causal_np, rtol=1e-3
        ), "Causal and non-causal outputs should differ"

        # Verify outputs have reasonable magnitudes
        assert np.abs(causal_np).mean() > 0.01, "Causal output seems too small"
        assert np.abs(non_causal_np).mean() > 0.01, "Non-causal output seems too small"
