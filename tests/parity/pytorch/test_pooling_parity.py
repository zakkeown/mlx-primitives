"""PyTorch parity tests for pooling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import pooling_inputs, SIZE_CONFIGS
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
# Adaptive Average Pooling 1D Parity Tests
# =============================================================================

class TestAdaptiveAvgPool1dParity:
    """AdaptiveAvgPool1d parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("output_size", [1, 4, 16])
    def test_forward_parity(self, size, dtype, output_size, skip_without_pytorch):
        """Test AdaptiveAvgPool1d forward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveAvgPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        # Use width as 1D length
        length = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX
        pool_mlx = AdaptiveAvgPool1d(output_size)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = pool_mlx(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.adaptive_avg_pool1d(x_torch, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool1d forward mismatch [{size}, {dtype}, out={output_size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test AdaptiveAvgPool1d backward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveAvgPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]
        output_size = 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX backward
        pool_mlx = AdaptiveAvgPool1d(output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.adaptive_avg_pool1d(x_torch, output_size)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "adaptive_avg_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool1d backward mismatch [{size}]"
        )


# =============================================================================
# Adaptive Average Pooling 2D Parity Tests
# =============================================================================

class TestAdaptiveAvgPool2dParity:
    """AdaptiveAvgPool2d parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4), (7, 7)])
    def test_forward_parity(self, size, dtype, output_size, skip_without_pytorch):
        """Test AdaptiveAvgPool2d forward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveAvgPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX
        pool_mlx = AdaptiveAvgPool2d(output_size)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = pool_mlx(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.adaptive_avg_pool2d(x_torch, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool2d forward mismatch [{size}, {dtype}, out={output_size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test AdaptiveAvgPool2d backward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveAvgPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        output_size = (4, 4)
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX backward
        pool_mlx = AdaptiveAvgPool2d(output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.adaptive_avg_pool2d(x_torch, output_size)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "adaptive_avg_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool2d backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_global_pooling(self, skip_without_pytorch):
        """Test global pooling special case (1, 1)."""
        from mlx_primitives.layers.pooling import AdaptiveAvgPool2d

        batch, channels, height, width = 2, 64, 32, 32
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        pool_mlx = AdaptiveAvgPool2d((1, 1))
        x_mlx = mx.array(x_np)
        mlx_out = pool_mlx(x_mlx)

        x_torch = torch.from_numpy(x_np)
        torch_out = F.adaptive_avg_pool2d(x_torch, (1, 1))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-5, atol=1e-6,
            err_msg="Global avg pooling mismatch"
        )


# =============================================================================
# Adaptive Max Pooling 1D Parity Tests
# =============================================================================

class TestAdaptiveMaxPool1dParity:
    """AdaptiveMaxPool1d parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("output_size", [1, 4, 16])
    def test_forward_parity(self, size, dtype, output_size, skip_without_pytorch):
        """Test AdaptiveMaxPool1d forward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveMaxPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX
        pool_mlx = AdaptiveMaxPool1d(output_size)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = pool_mlx(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.adaptive_max_pool1d(x_torch, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool1d forward mismatch [{size}, {dtype}, out={output_size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test AdaptiveMaxPool1d backward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveMaxPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]
        output_size = 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX backward
        pool_mlx = AdaptiveMaxPool1d(output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.adaptive_max_pool1d(x_torch, output_size)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "adaptive_max_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool1d backward mismatch [{size}]"
        )


# =============================================================================
# Adaptive Max Pooling 2D Parity Tests
# =============================================================================

class TestAdaptiveMaxPool2dParity:
    """AdaptiveMaxPool2d parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4), (7, 7)])
    def test_forward_parity(self, size, dtype, output_size, skip_without_pytorch):
        """Test AdaptiveMaxPool2d forward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveMaxPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX
        pool_mlx = AdaptiveMaxPool2d(output_size)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = pool_mlx(x_mlx)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        torch_out = F.adaptive_max_pool2d(x_torch, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool2d forward mismatch [{size}, {dtype}, out={output_size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test AdaptiveMaxPool2d backward pass parity."""
        from mlx_primitives.layers.pooling import AdaptiveMaxPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        output_size = (4, 4)
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX backward
        pool_mlx = AdaptiveMaxPool2d(output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = F.adaptive_max_pool2d(x_torch, output_size)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "adaptive_max_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool2d backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_global_pooling(self, skip_without_pytorch):
        """Test global max pooling special case (1, 1)."""
        from mlx_primitives.layers.pooling import AdaptiveMaxPool2d

        batch, channels, height, width = 2, 64, 32, 32
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        pool_mlx = AdaptiveMaxPool2d((1, 1))
        x_mlx = mx.array(x_np)
        mlx_out = pool_mlx(x_mlx)

        x_torch = torch.from_numpy(x_np)
        torch_out = F.adaptive_max_pool2d(x_torch, (1, 1))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-5, atol=1e-6,
            err_msg="Global max pooling mismatch"
        )


# =============================================================================
# Global Attention Pooling Parity Tests
# =============================================================================

class TestGlobalAttentionPoolingParity:
    """Global attention pooling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test global attention pooling forward pass parity."""
        from mlx_primitives.layers.pooling import GlobalAttentionPooling

        # Use activation config for sequence-based pooling
        config = SIZE_CONFIGS[size]["activation"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dim"]
        hidden_dims = dims // 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Initialize MLX pooling
        pool_mlx = GlobalAttentionPooling(dims, hidden_dims)
        mx.eval(pool_mlx.parameters())

        # Get MLX weights for PyTorch reference
        attention_layers = pool_mlx.attention.layers
        W1_np = np.array(attention_layers[0].weight).T  # (dims, hidden_dims)
        b1_np = np.array(attention_layers[0].bias)
        W2_np = np.array(attention_layers[2].weight).T  # (hidden_dims, 1)

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = pool_mlx(x_mlx)

        # PyTorch reference implementation
        x_torch = _convert_to_torch(x_np, dtype)
        W1 = torch.from_numpy(W1_np).to(x_torch.dtype)
        b1 = torch.from_numpy(b1_np).to(x_torch.dtype)
        W2 = torch.from_numpy(W2_np).to(x_torch.dtype)

        hidden = torch.tanh(x_torch @ W1 + b1)
        scores = (hidden @ W2).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        torch_out = torch.sum(x_torch * weights.unsqueeze(-1), dim=1)

        rtol, atol = get_tolerance("pooling", "global_attention_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GlobalAttentionPooling forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test global attention pooling backward pass parity."""
        from mlx_primitives.layers.pooling import GlobalAttentionPooling

        config = SIZE_CONFIGS[size]["activation"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dim"]
        hidden_dims = dims // 4
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # MLX
        pool_mlx = GlobalAttentionPooling(dims, hidden_dims)
        mx.eval(pool_mlx.parameters())

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch
        attention_layers = pool_mlx.attention.layers
        W1_np = np.array(attention_layers[0].weight).T
        b1_np = np.array(attention_layers[0].bias)
        W2_np = np.array(attention_layers[2].weight).T

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        W1 = torch.from_numpy(W1_np)
        b1 = torch.from_numpy(b1_np)
        W2 = torch.from_numpy(W2_np)

        hidden = torch.tanh(x_torch @ W1 + b1)
        scores = (hidden @ W2).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        torch_out = torch.sum(x_torch * weights.unsqueeze(-1), dim=1)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "global_attention_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GlobalAttentionPooling backward mismatch [{size}]"
        )


# =============================================================================
# GeM (Generalized Mean) Pooling Parity Tests
# =============================================================================

class TestGeMParity:
    """GeM (Generalized Mean) pooling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 5.0])
    def test_forward_parity(self, size, dtype, p, skip_without_pytorch):
        """Test GeM pooling forward pass parity."""
        from mlx_primitives.layers.pooling import GeM

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        eps = 1e-6

        np.random.seed(42)
        # Use positive values for GeM (it clamps to eps internally)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + eps

        # MLX with non-learnable p for comparison
        gem_mlx = GeM(p=p, eps=eps, learnable=False)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = gem_mlx(x_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        x_clamped = x_torch.clamp(min=eps)
        x_pow = x_clamped.pow(p)
        mean_pow = x_pow.mean(dim=(2, 3), keepdim=True)
        torch_out = mean_pow.pow(1.0 / p)

        rtol, atol = get_tolerance("pooling", "gem", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"GeM forward mismatch [{size}, {dtype}, p={p}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test GeM pooling backward pass parity."""
        from mlx_primitives.layers.pooling import GeM

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        p = 3.0
        eps = 1e-6
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + eps

        # MLX backward
        gem_mlx = GeM(p=p, eps=eps, learnable=False)

        def mlx_loss_fn(x):
            return mx.sum(gem_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        x_clamped = x_torch.clamp(min=eps)
        x_pow = x_clamped.pow(p)
        mean_pow = x_pow.mean(dim=(2, 3), keepdim=True)
        torch_out = mean_pow.pow(1.0 / p)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "gem", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"GeM backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_p_equals_1(self, skip_without_pytorch):
        """Test GeM with p=1 (equivalent to average pooling)."""
        from mlx_primitives.layers.pooling import GeM

        batch, channels, height, width = 2, 64, 8, 8
        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + 1e-6

        gem_mlx = GeM(p=1.0, eps=1e-6, learnable=False)
        x_mlx = mx.array(x_np)
        mlx_out = gem_mlx(x_mlx)

        # p=1 should be equivalent to average pooling
        x_torch = torch.from_numpy(x_np)
        torch_out = x_torch.mean(dim=(2, 3), keepdim=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-4, atol=1e-5,
            err_msg="GeM p=1 should equal average pooling"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_large_p(self, skip_without_pytorch):
        """Test GeM with large p (approaches max pooling)."""
        from mlx_primitives.layers.pooling import GeM

        batch, channels, height, width = 2, 64, 8, 8
        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + 1e-6

        gem_mlx = GeM(p=10.0, eps=1e-6, learnable=False)
        x_mlx = mx.array(x_np)
        mlx_out = gem_mlx(x_mlx)

        # Large p approaches max pooling
        x_torch = torch.from_numpy(x_np)
        torch_max = x_torch.amax(dim=(2, 3), keepdim=True)

        # Should be close to max but not exact
        mlx_np = _to_numpy(mlx_out)
        torch_np = _to_numpy(torch_max)

        # GeM with large p should be <= max (and close to it)
        assert np.all(mlx_np <= torch_np * 1.01), "GeM large p should be close to max"


# =============================================================================
# Spatial Pyramid Pooling Parity Tests
# =============================================================================

class TestSpatialPyramidPoolingParity:
    """Spatial Pyramid Pooling parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("levels", [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]])
    def test_forward_parity(self, size, dtype, levels, skip_without_pytorch):
        """Test SPP forward pass parity."""
        from mlx_primitives.layers.pooling import SpatialPyramidPooling

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX
        spp_mlx = SpatialPyramidPooling(output_sizes=levels)
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = spp_mlx(x_mlx)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        pooled = []
        for level in levels:
            p = F.adaptive_avg_pool2d(x_torch, (level, level))
            p = p.reshape(batch, -1)
            pooled.append(p)
        torch_out = torch.cat(pooled, dim=1)

        rtol, atol = get_tolerance("pooling", "spatial_pyramid_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"SPP forward mismatch [{size}, {dtype}, levels={levels}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test SPP backward pass parity."""
        from mlx_primitives.layers.pooling import SpatialPyramidPooling

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        levels = [1, 2, 4]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX backward
        spp_mlx = SpatialPyramidPooling(output_sizes=levels)

        def mlx_loss_fn(x):
            return mx.sum(spp_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        pooled = []
        for level in levels:
            p = F.adaptive_avg_pool2d(x_torch, (level, level))
            p = p.reshape(batch, -1)
            pooled.append(p)
        torch_out = torch.cat(pooled, dim=1)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("pooling", "spatial_pyramid_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SPP backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_output_shape(self, skip_without_pytorch):
        """Test SPP output shape matches expected."""
        from mlx_primitives.layers.pooling import SpatialPyramidPooling

        batch, channels, height, width = 2, 256, 13, 13
        levels = [1, 2, 4]

        # Expected output: channels * (1 + 4 + 16) = channels * 21
        expected_features = channels * sum(level**2 for level in levels)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        spp_mlx = SpatialPyramidPooling(output_sizes=levels)
        x_mlx = mx.array(x_np)
        mlx_out = spp_mlx(x_mlx)

        assert mlx_out.shape == (batch, expected_features), \
            f"SPP output shape mismatch: expected {(batch, expected_features)}, got {mlx_out.shape}"
