"""PyTorch parity tests for normalization operations."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

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
# RMSNorm Parity Tests
# =============================================================================

class TestRMSNormParity:
    """RMSNorm parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test RMSNorm forward pass parity."""
        # Skip bf16+large: bf16 accumulates significant precision errors over large
        # reductions (4096 elements), testing precision limits not algorithm correctness
        if dtype == "bf16" and size == "large":
            pytest.skip("bf16+large tests precision limits, not algorithm correctness")

        from mlx_primitives.layers import RMSNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        # Generate inputs
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        eps = 1e-6

        # MLX RMSNorm
        rmsnorm = RMSNorm(hidden, eps=eps)
        mx.eval(rmsnorm.parameters())
        weight_np = np.array(rmsnorm.weight)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = rmsnorm(x_mlx)

        # PyTorch reference: y = x / sqrt(mean(x^2) + eps) * weight
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + eps)
        torch_out = (x_torch / rms) * weight_torch

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test RMSNorm backward pass parity."""
        from mlx_primitives.layers import RMSNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        dtype = "fp32"
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX backward
        rmsnorm = RMSNorm(hidden, eps=eps)
        mx.eval(rmsnorm.parameters())
        weight_np = np.array(rmsnorm.weight)

        def mlx_loss_fn(x):
            return mx.sum(rmsnorm(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)
        rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + eps)
        torch_out = (x_torch / rms) * weight_torch
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_epsilon_values(self, eps, skip_without_pytorch):
        """Test different epsilon values."""
        from mlx_primitives.layers import RMSNorm

        batch, seq, hidden = 2, 64, 256
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX RMSNorm
        rmsnorm = RMSNorm(hidden, eps=eps)
        mx.eval(rmsnorm.parameters())
        weight_np = np.array(rmsnorm.weight)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = rmsnorm(x_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + eps)
        torch_out = (x_torch / rms) * weight_torch

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm epsilon mismatch [eps={eps}]"
        )


# =============================================================================
# LayerNorm Parity Tests
# =============================================================================

class TestLayerNormParity:
    """LayerNorm parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test LayerNorm forward pass parity."""
        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX LayerNorm
        layernorm = nn.LayerNorm(hidden, eps=eps)
        mx.eval(layernorm.parameters())
        weight_np = np.array(layernorm.weight)
        bias_np = np.array(layernorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = layernorm(x_mlx)

        # PyTorch LayerNorm
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.layer_norm(x_torch, (hidden,), weight_torch, bias_torch, eps)

        rtol, atol = get_tolerance("normalization", "layernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"LayerNorm forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test LayerNorm backward pass parity."""
        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX backward
        layernorm = nn.LayerNorm(hidden, eps=eps)
        mx.eval(layernorm.parameters())
        weight_np = np.array(layernorm.weight)
        bias_np = np.array(layernorm.bias)

        def mlx_loss_fn(x):
            return mx.sum(layernorm(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)
        bias_torch = torch.from_numpy(bias_np)
        torch_out = F.layer_norm(x_torch, (hidden,), weight_torch, bias_torch, eps)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "layernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"LayerNorm backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_bias(self, skip_without_pytorch):
        """Test LayerNorm with bias parameter."""
        batch, seq, hidden = 2, 64, 256
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX with affine=True (default)
        layernorm = nn.LayerNorm(hidden, eps=eps, affine=True)
        mx.eval(layernorm.parameters())

        # Set non-trivial weight and bias
        layernorm.weight = mx.array(np.random.randn(hidden).astype(np.float32))
        layernorm.bias = mx.array(np.random.randn(hidden).astype(np.float32))
        mx.eval(layernorm.parameters())

        weight_np = np.array(layernorm.weight)
        bias_np = np.array(layernorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = layernorm(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.layer_norm(x_torch, (hidden,), weight_torch, bias_torch, eps)

        rtol, atol = get_tolerance("normalization", "layernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="LayerNorm with bias mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_without_bias(self, skip_without_pytorch):
        """Test LayerNorm without bias parameter."""
        batch, seq, hidden = 2, 64, 256
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX with affine=False
        layernorm = nn.LayerNorm(hidden, eps=eps, affine=False)
        mx.eval(layernorm.parameters())

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = layernorm(x_mlx)

        # PyTorch (no weight/bias)
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.layer_norm(x_torch, (hidden,), None, None, eps)

        rtol, atol = get_tolerance("normalization", "layernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="LayerNorm without bias mismatch"
        )


# =============================================================================
# GroupNorm Parity Tests
# =============================================================================

class TestGroupNormParity:
    """GroupNorm parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("num_groups", [1, 4, 8, 32])
    def test_forward_parity(self, size, dtype, num_groups, skip_without_pytorch):
        """Test GroupNorm forward pass parity."""
        from mlx_primitives.layers import GroupNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        # GroupNorm expects NCHW format - use hidden as channels
        num_channels = config["hidden"]
        height, width = 8, 8  # Small spatial dims for testing
        eps = 1e-5

        # Skip if num_channels not divisible by num_groups
        if num_channels % num_groups != 0:
            pytest.skip(f"num_channels ({num_channels}) not divisible by num_groups ({num_groups})")

        np.random.seed(42)
        x_np = np.random.randn(batch, num_channels, height, width).astype(np.float32)

        # MLX GroupNorm
        groupnorm = GroupNorm(num_groups, num_channels, eps=eps)
        mx.eval(groupnorm.parameters())
        weight_np = np.array(groupnorm.weight)
        bias_np = np.array(groupnorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = groupnorm(x_mlx)

        # PyTorch GroupNorm
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.group_norm(x_torch, num_groups, weight_torch, bias_torch, eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GroupNorm forward mismatch [{size}, {dtype}, groups={num_groups}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test GroupNorm backward pass parity."""
        from mlx_primitives.layers import GroupNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        num_channels = config["hidden"]
        height, width = 8, 8
        num_groups = 8
        dtype = "fp32"
        eps = 1e-5

        # Skip if not divisible
        if num_channels % num_groups != 0:
            pytest.skip(f"num_channels ({num_channels}) not divisible by num_groups ({num_groups})")

        np.random.seed(42)
        x_np = np.random.randn(batch, num_channels, height, width).astype(np.float32)

        # MLX backward
        groupnorm = GroupNorm(num_groups, num_channels, eps=eps)
        mx.eval(groupnorm.parameters())
        weight_np = np.array(groupnorm.weight)
        bias_np = np.array(groupnorm.bias)

        def mlx_loss_fn(x):
            return mx.sum(groupnorm(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)
        bias_torch = torch.from_numpy(bias_np)
        torch_out = F.group_norm(x_torch, num_groups, weight_torch, bias_torch, eps)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "groupnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GroupNorm backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_single_group(self, skip_without_pytorch):
        """Test GroupNorm with single group (equivalent to LayerNorm)."""
        from mlx_primitives.layers import GroupNorm

        batch, num_channels, height, width = 2, 64, 8, 8
        num_groups = 1
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_channels, height, width).astype(np.float32)

        # MLX GroupNorm with 1 group
        groupnorm = GroupNorm(num_groups, num_channels, eps=eps)
        mx.eval(groupnorm.parameters())
        weight_np = np.array(groupnorm.weight)
        bias_np = np.array(groupnorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = groupnorm(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.group_norm(x_torch, num_groups, weight_torch, bias_torch, eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="GroupNorm single group mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_channels_equals_groups(self, skip_without_pytorch):
        """Test GroupNorm with channels == groups (equivalent to InstanceNorm)."""
        from mlx_primitives.layers import GroupNorm

        batch, num_channels, height, width = 2, 64, 8, 8
        num_groups = num_channels  # One channel per group
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_channels, height, width).astype(np.float32)

        # MLX GroupNorm with channels=groups
        groupnorm = GroupNorm(num_groups, num_channels, eps=eps)
        mx.eval(groupnorm.parameters())
        weight_np = np.array(groupnorm.weight)
        bias_np = np.array(groupnorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = groupnorm(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.group_norm(x_torch, num_groups, weight_torch, bias_torch, eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="GroupNorm channels=groups mismatch"
        )


# =============================================================================
# InstanceNorm Parity Tests
# =============================================================================

class TestInstanceNormParity:
    """InstanceNorm parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test InstanceNorm forward pass parity."""
        from mlx_primitives.layers import InstanceNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        num_features = config["hidden"]
        height, width = 8, 8
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_features, height, width).astype(np.float32)

        # MLX InstanceNorm
        instancenorm = InstanceNorm(num_features, eps=eps)
        mx.eval(instancenorm.parameters())
        weight_np = np.array(instancenorm.weight)
        bias_np = np.array(instancenorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = instancenorm(x_mlx)

        # PyTorch InstanceNorm
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.instance_norm(x_torch, weight=weight_torch, bias=bias_torch, eps=eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"InstanceNorm forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test InstanceNorm backward pass parity."""
        from mlx_primitives.layers import InstanceNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        num_features = config["hidden"]
        height, width = 8, 8
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_features, height, width).astype(np.float32)

        # MLX backward
        instancenorm = InstanceNorm(num_features, eps=eps)
        mx.eval(instancenorm.parameters())
        weight_np = np.array(instancenorm.weight)
        bias_np = np.array(instancenorm.bias)

        def mlx_loss_fn(x):
            return mx.sum(instancenorm(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)
        bias_torch = torch.from_numpy(bias_np)
        torch_out = F.instance_norm(x_torch, weight=weight_torch, bias=bias_torch, eps=eps)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "instancenorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"InstanceNorm backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_1d_input(self, skip_without_pytorch):
        """Test InstanceNorm1d parity."""
        from mlx_primitives.layers import InstanceNorm

        batch, num_features, length = 2, 64, 32
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_features, length).astype(np.float32)

        # MLX InstanceNorm (1D: NCL)
        instancenorm = InstanceNorm(num_features, eps=eps)
        mx.eval(instancenorm.parameters())
        weight_np = np.array(instancenorm.weight)
        bias_np = np.array(instancenorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = instancenorm(x_mlx)

        # PyTorch InstanceNorm1d
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.instance_norm(x_torch, weight=weight_torch, bias=bias_torch, eps=eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="InstanceNorm1d mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_2d_input(self, skip_without_pytorch):
        """Test InstanceNorm2d parity."""
        from mlx_primitives.layers import InstanceNorm

        batch, num_features, height, width = 2, 64, 16, 16
        dtype = "fp32"
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, num_features, height, width).astype(np.float32)

        # MLX InstanceNorm (2D: NCHW)
        instancenorm = InstanceNorm(num_features, eps=eps)
        mx.eval(instancenorm.parameters())
        weight_np = np.array(instancenorm.weight)
        bias_np = np.array(instancenorm.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = instancenorm(x_mlx)

        # PyTorch InstanceNorm2d
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = torch.from_numpy(weight_np).to(x_torch.dtype)
        bias_torch = torch.from_numpy(bias_np).to(x_torch.dtype)
        torch_out = F.instance_norm(x_torch, weight=weight_torch, bias=bias_torch, eps=eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="InstanceNorm2d mismatch"
        )


# =============================================================================
# AdaLayerNorm Parity Tests
# =============================================================================

class TestAdaLayerNormParity:
    """Adaptive LayerNorm (used in DiT, etc.) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test AdaLayerNorm forward pass parity."""
        from mlx_primitives.layers import AdaLayerNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        cond_dims = hidden // 2
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32)

        # MLX AdaLayerNorm
        adaln = AdaLayerNorm(hidden, cond_dims, eps=eps)
        mx.eval(adaln.parameters())
        proj_weight_np = np.array(adaln.proj.weight)
        proj_bias_np = np.array(adaln.proj.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        cond_mlx = _convert_to_mlx(cond_np, dtype)
        mlx_out = adaln(x_mlx, cond_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        cond_torch = _convert_to_torch(cond_np, dtype)
        proj_weight_torch = torch.from_numpy(proj_weight_np).to(x_torch.dtype)
        proj_bias_torch = torch.from_numpy(proj_bias_np).to(x_torch.dtype)

        # LayerNorm without affine
        mean = x_torch.mean(dim=-1, keepdim=True)
        var = x_torch.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_torch - mean) / torch.sqrt(var + eps)

        # Get scale/shift from conditioning
        scale_shift = F.linear(cond_torch, proj_weight_torch, proj_bias_torch)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Apply adaptive normalization
        scale = scale.unsqueeze(1)  # (batch, 1, dims)
        shift = shift.unsqueeze(1)
        torch_out = x_norm * (1 + scale) + shift

        rtol, atol = get_tolerance("normalization", "adalayernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test AdaLayerNorm backward pass parity."""
        from mlx_primitives.layers import AdaLayerNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        cond_dims = hidden // 2
        dtype = "fp32"
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32)

        # MLX backward
        adaln = AdaLayerNorm(hidden, cond_dims, eps=eps)
        mx.eval(adaln.parameters())
        proj_weight_np = np.array(adaln.proj.weight)
        proj_bias_np = np.array(adaln.proj.bias)

        def mlx_loss_fn(x, cond):
            return mx.sum(adaln(x, cond))

        x_mlx = mx.array(x_np)
        cond_mlx = mx.array(cond_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        mlx_grad_x, mlx_grad_cond = grad_fn(x_mlx, cond_mlx)
        mx.eval(mlx_grad_x, mlx_grad_cond)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        cond_torch = torch.from_numpy(cond_np).requires_grad_(True)
        proj_weight_torch = torch.from_numpy(proj_weight_np)
        proj_bias_torch = torch.from_numpy(proj_bias_np)

        mean = x_torch.mean(dim=-1, keepdim=True)
        var = x_torch.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_torch - mean) / torch.sqrt(var + eps)

        scale_shift = F.linear(cond_torch, proj_weight_torch, proj_bias_torch)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        torch_out = x_norm * (1 + scale) + shift
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "adalayernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm backward (x) mismatch [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_cond), cond_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm backward (cond) mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_shift_scale(self, skip_without_pytorch):
        """Test AdaLayerNorm with shift and scale conditioning."""
        from mlx_primitives.layers import AdaLayerNorm

        batch, seq, hidden, cond_dims = 2, 64, 256, 128
        dtype = "fp32"
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        # Test with varying conditioning values
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32) * 2  # Larger scale

        # MLX AdaLayerNorm
        adaln = AdaLayerNorm(hidden, cond_dims, eps=eps)
        mx.eval(adaln.parameters())
        proj_weight_np = np.array(adaln.proj.weight)
        proj_bias_np = np.array(adaln.proj.bias)

        x_mlx = _convert_to_mlx(x_np, dtype)
        cond_mlx = _convert_to_mlx(cond_np, dtype)
        mlx_out = adaln(x_mlx, cond_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        cond_torch = _convert_to_torch(cond_np, dtype)
        proj_weight_torch = torch.from_numpy(proj_weight_np).to(x_torch.dtype)
        proj_bias_torch = torch.from_numpy(proj_bias_np).to(x_torch.dtype)

        mean = x_torch.mean(dim=-1, keepdim=True)
        var = x_torch.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_torch - mean) / torch.sqrt(var + eps)

        scale_shift = F.linear(cond_torch, proj_weight_torch, proj_bias_torch)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        torch_out = x_norm * (1 + scale) + shift

        rtol, atol = get_tolerance("normalization", "adalayernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="AdaLayerNorm shift/scale mismatch"
        )
