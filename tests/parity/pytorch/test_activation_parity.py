"""PyTorch parity tests for activation functions."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from tests.parity.shared.input_generators import activation_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close, get_gradient_tolerance
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
    # bf16 isn't supported well in numpy, so we convert fp32 -> bf16 in torch
    x_torch = torch.from_numpy(x_np.astype(np.float32))
    torch_dtype = get_pytorch_dtype(dtype)
    return x_torch.to(torch_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        # Force evaluation and convert to float32 for comparison
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# SwiGLU Parity Tests
# =============================================================================

class TestSwiGLUParity:
    """SwiGLU activation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test SwiGLU forward pass parity."""
        from mlx_primitives.layers import SwiGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        # Generate inputs
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # Initialize MLX SwiGLU
        swiglu_mlx = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu_mlx.parameters())

        # Get MLX weights
        w1_np = np.array(swiglu_mlx.w1.weight)
        w2_np = np.array(swiglu_mlx.w2.weight)
        w_gate_np = np.array(swiglu_mlx.w_gate.weight)

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = swiglu_mlx(x_mlx)

        # PyTorch reference: SwiGLU(x) = (x @ W1) * SiLU(x @ W_gate) @ W2
        x_torch = _convert_to_torch(x_np, dtype)
        w1_torch = torch.from_numpy(w1_np.T).to(x_torch.dtype)  # Transpose for mm
        w2_torch = torch.from_numpy(w2_np.T).to(x_torch.dtype)
        w_gate_torch = torch.from_numpy(w_gate_np.T).to(x_torch.dtype)

        gate = F.silu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch

        rtol, atol = get_tolerance("activations", "swiglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"SwiGLU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test SwiGLU backward pass parity."""
        from mlx_primitives.layers import SwiGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX forward and backward
        swiglu_mlx = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(swiglu_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        w1_np = np.array(swiglu_mlx.w1.weight)
        w2_np = np.array(swiglu_mlx.w2.weight)
        w_gate_np = np.array(swiglu_mlx.w_gate.weight)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        w1_torch = torch.from_numpy(w1_np.T)
        w2_torch = torch.from_numpy(w2_np.T)
        w_gate_torch = torch.from_numpy(w_gate_np.T)

        gate = F.silu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "swiglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SwiGLU backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test edge cases: zeros, large values, negative values."""
        from mlx_primitives.layers import SwiGLU

        dim, hidden_dim = 64, 256
        swiglu_mlx = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu_mlx.parameters())

        # Zero input
        x_zeros = mx.zeros((1, 4, dim))
        out_zeros = swiglu_mlx(x_zeros)
        mx.eval(out_zeros)
        assert not np.any(np.isnan(_to_numpy(out_zeros))), "NaN in zero input output"

        # Large input
        x_large = mx.full((1, 4, dim), 10.0)
        out_large = swiglu_mlx(x_large)
        mx.eval(out_large)
        assert not np.any(np.isnan(_to_numpy(out_large))), "NaN in large input output"
        assert not np.any(np.isinf(_to_numpy(out_large))), "Inf in large input output"


# =============================================================================
# GeGLU Parity Tests
# =============================================================================

class TestGeGLUParity:
    """GeGLU activation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test GeGLU forward pass parity."""
        from mlx_primitives.layers import GeGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX GeGLU
        geglu_mlx = GeGLU(dim, hidden_dim)
        mx.eval(geglu_mlx.parameters())

        w1_np = np.array(geglu_mlx.w1.weight)
        w2_np = np.array(geglu_mlx.w2.weight)
        w_gate_np = np.array(geglu_mlx.w_gate.weight)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = geglu_mlx(x_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        w1_torch = torch.from_numpy(w1_np.T).to(x_torch.dtype)
        w2_torch = torch.from_numpy(w2_np.T).to(x_torch.dtype)
        w_gate_torch = torch.from_numpy(w_gate_np.T).to(x_torch.dtype)

        gate = F.gelu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch

        rtol, atol = get_tolerance("activations", "geglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GeGLU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test GeGLU backward pass parity."""
        from mlx_primitives.layers import GeGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        geglu_mlx = GeGLU(dim, hidden_dim)
        mx.eval(geglu_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(geglu_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch
        w1_np = np.array(geglu_mlx.w1.weight)
        w2_np = np.array(geglu_mlx.w2.weight)
        w_gate_np = np.array(geglu_mlx.w_gate.weight)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        w1_torch = torch.from_numpy(w1_np.T)
        w2_torch = torch.from_numpy(w2_np.T)
        w_gate_torch = torch.from_numpy(w_gate_np.T)

        gate = F.gelu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "geglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GeGLU backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test edge cases."""
        from mlx_primitives.layers import GeGLU

        dim, hidden_dim = 64, 256
        geglu_mlx = GeGLU(dim, hidden_dim)
        mx.eval(geglu_mlx.parameters())

        x_zeros = mx.zeros((1, 4, dim))
        out_zeros = geglu_mlx(x_zeros)
        mx.eval(out_zeros)
        assert not np.any(np.isnan(_to_numpy(out_zeros)))


# =============================================================================
# ReGLU Parity Tests
# =============================================================================

class TestReGLUParity:
    """ReGLU activation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test ReGLU forward pass parity."""
        from mlx_primitives.layers import ReGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        reglu_mlx = ReGLU(dim, hidden_dim)
        mx.eval(reglu_mlx.parameters())

        w1_np = np.array(reglu_mlx.w1.weight)
        w2_np = np.array(reglu_mlx.w2.weight)
        w_gate_np = np.array(reglu_mlx.w_gate.weight)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = reglu_mlx(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        w1_torch = torch.from_numpy(w1_np.T).to(x_torch.dtype)
        w2_torch = torch.from_numpy(w2_np.T).to(x_torch.dtype)
        w_gate_torch = torch.from_numpy(w_gate_np.T).to(x_torch.dtype)

        gate = F.relu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch

        rtol, atol = get_tolerance("activations", "reglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"ReGLU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test ReGLU backward pass parity."""
        from mlx_primitives.layers import ReGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        reglu_mlx = ReGLU(dim, hidden_dim)
        mx.eval(reglu_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(reglu_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        w1_np = np.array(reglu_mlx.w1.weight)
        w2_np = np.array(reglu_mlx.w2.weight)
        w_gate_np = np.array(reglu_mlx.w_gate.weight)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        w1_torch = torch.from_numpy(w1_np.T)
        w2_torch = torch.from_numpy(w2_np.T)
        w_gate_torch = torch.from_numpy(w_gate_np.T)

        gate = F.relu(x_torch @ w_gate_torch)
        up = x_torch @ w1_torch
        torch_out = (gate * up) @ w2_torch
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "reglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"ReGLU backward mismatch [{size}]"
        )


# =============================================================================
# GELU Exact Parity Tests
# =============================================================================

class TestGELUExactParity:
    """GELU (exact) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test exact GELU forward pass parity."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = nn.gelu(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.gelu(x_torch, approximate='none')

        rtol, atol = get_tolerance("activations", "gelu_exact", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU exact forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test exact GELU backward pass parity."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(nn.gelu(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.gelu(x_torch, approximate='none')
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "gelu_exact", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GELU exact backward mismatch [{size}]"
        )


# =============================================================================
# GELU Approximate Parity Tests
# =============================================================================

class TestGELUApproximateParity:
    """GELU (tanh approximation) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test approximate GELU forward pass parity."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = gelu_tanh(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.gelu(x_torch, approximate='tanh')

        rtol, atol = get_tolerance("activations", "gelu_approx", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU approx forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test approximate GELU backward pass parity."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(gelu_tanh(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.gelu(x_torch, approximate='tanh')
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "gelu_approx", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GELU approx backward mismatch [{size}]"
        )


# =============================================================================
# SiLU Parity Tests
# =============================================================================

class TestSiLUParity:
    """SiLU (Swish with beta=1) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test SiLU forward pass parity."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = nn.silu(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.silu(x_torch)

        rtol, atol = get_tolerance("activations", "silu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"SiLU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test SiLU backward pass parity."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(nn.silu(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.silu(x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "silu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SiLU backward mismatch [{size}]"
        )


# =============================================================================
# QuickGELU Parity Tests
# =============================================================================

class TestQuickGELUParity:
    """QuickGELU (x * sigmoid(1.702 * x)) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test QuickGELU forward pass parity."""
        from mlx_primitives.layers import quick_gelu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = quick_gelu(x_mlx)

        # PyTorch reference: x * sigmoid(1.702 * x)
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = x_torch * torch.sigmoid(1.702 * x_torch)

        rtol, atol = get_tolerance("activations", "quick_gelu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"QuickGELU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test QuickGELU backward pass parity."""
        from mlx_primitives.layers import quick_gelu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(quick_gelu(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = x_torch * torch.sigmoid(1.702 * x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "quick_gelu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"QuickGELU backward mismatch [{size}]"
        )


# =============================================================================
# GELU Tanh Parity Tests
# =============================================================================

class TestGELUTanhParity:
    """GELU with tanh approximation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test GELU tanh forward pass parity."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = gelu_tanh(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.gelu(x_torch, approximate='tanh')

        rtol, atol = get_tolerance("activations", "gelu_tanh", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU tanh forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test GELU tanh backward pass parity."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(gelu_tanh(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.gelu(x_torch, approximate='tanh')
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "gelu_tanh", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GELU tanh backward mismatch [{size}]"
        )


# =============================================================================
# Mish Parity Tests
# =============================================================================

class TestMishParity:
    """Mish activation (x * tanh(softplus(x))) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test Mish forward pass parity."""
        from mlx_primitives.layers import mish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = mish(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.mish(x_torch)

        rtol, atol = get_tolerance("activations", "mish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Mish forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test Mish backward pass parity."""
        from mlx_primitives.layers import mish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(mish(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.mish(x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "mish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Mish backward mismatch [{size}]"
        )


# =============================================================================
# Squared ReLU Parity Tests
# =============================================================================

class TestSquaredReLUParity:
    """Squared ReLU (relu(x)^2) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test Squared ReLU forward pass parity."""
        from mlx_primitives.layers import squared_relu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = squared_relu(x_mlx)

        # PyTorch reference: relu(x)^2
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.relu(x_torch) ** 2

        rtol, atol = get_tolerance("activations", "squared_relu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Squared ReLU forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test Squared ReLU backward pass parity."""
        from mlx_primitives.layers import squared_relu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(squared_relu(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.relu(x_torch) ** 2
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "squared_relu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Squared ReLU backward mismatch [{size}]"
        )


# =============================================================================
# Swish Parity Tests
# =============================================================================

class TestSwishParity:
    """Swish (x * sigmoid(beta * x)) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    def test_forward_parity(self, size, dtype, beta, skip_without_pytorch):
        """Test Swish forward pass parity."""
        from mlx_primitives.layers import swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = swish(x_mlx, beta=beta)

        # PyTorch reference: x * sigmoid(beta * x)
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = x_torch * torch.sigmoid(beta * x_torch)

        rtol, atol = get_tolerance("activations", "swish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Swish forward mismatch [{size}, {dtype}, beta={beta}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test Swish backward pass parity."""
        from mlx_primitives.layers import swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"
        beta = 1.0

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(swish(x, beta=beta))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = x_torch * torch.sigmoid(beta * x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "swish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Swish backward mismatch [{size}]"
        )


# =============================================================================
# Hard Swish Parity Tests
# =============================================================================

class TestHardSwishParity:
    """Hard Swish (x * relu6(x + 3) / 6) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test Hard Swish forward pass parity."""
        from mlx_primitives.layers import hard_swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = hard_swish(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.hardswish(x_torch)

        rtol, atol = get_tolerance("activations", "hard_swish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Swish forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test Hard Swish backward pass parity."""
        from mlx_primitives.layers import hard_swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(hard_swish(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.hardswish(x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "hard_swish", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Swish backward mismatch [{size}]"
        )


# =============================================================================
# Hard Sigmoid Parity Tests
# =============================================================================

class TestHardSigmoidParity:
    """Hard Sigmoid (relu6(x + 3) / 6) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test Hard Sigmoid forward pass parity."""
        from mlx_primitives.layers import hard_sigmoid

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = hard_sigmoid(x_mlx)

        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.hardsigmoid(x_torch)

        rtol, atol = get_tolerance("activations", "hard_sigmoid", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Sigmoid forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test Hard Sigmoid backward pass parity."""
        from mlx_primitives.layers import hard_sigmoid

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        def mlx_loss_fn(x):
            return mx.sum(hard_sigmoid(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.hardsigmoid(x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("activations", "hard_sigmoid", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Sigmoid backward mismatch [{size}]"
        )
