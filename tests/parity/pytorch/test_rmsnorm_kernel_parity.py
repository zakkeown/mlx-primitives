"""PyTorch parity tests for RMSNorm kernel implementations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch


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
# PyTorch Reference Implementations
# =============================================================================

def pytorch_rmsnorm(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    eps: float = 1e-6,
) -> "torch.Tensor":
    """PyTorch reference for RMSNorm.

    Formula: x / sqrt(mean(x^2) + eps) * weight
    """
    # Compute in float32 for numerical stability
    orig_dtype = x.dtype
    x_fp32 = x.float()
    rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + eps)
    normed = (x_fp32 / rms).to(orig_dtype)
    return normed * weight


def pytorch_rmsnorm_residual(
    x: "torch.Tensor",
    residual: "torch.Tensor",
    weight: "torch.Tensor",
    eps: float = 1e-6,
) -> "torch.Tensor":
    """PyTorch reference for fused RMSNorm(x + residual).

    Formula: RMSNorm(x + residual)
    """
    combined = x + residual
    return pytorch_rmsnorm(combined, weight, eps)


# =============================================================================
# Fast RMSNorm Parity Tests
# =============================================================================

class TestFastRMSNormParity:
    """Tests for fast_rmsnorm() Metal kernel parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fast_rmsnorm forward pass matches PyTorch."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx, eps=1e-6)
        mx.eval(mlx_out)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = _convert_to_torch(weight_np, dtype)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch, eps=1e-6)

        # Use normalization-specific tolerances
        tol_key = "rmsnorm" if "rmsnorm" in SIZE_CONFIGS.get("tiny", {}).get("normalization", {}) else "default"
        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_2d_input(self, skip_without_pytorch):
        """Test fast_rmsnorm with 2D input (seq, hidden)."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        seq, hidden = 256, 1024

        np.random.seed(42)
        x_np = np.random.randn(seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        # PyTorch
        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="fast_rmsnorm 2D input mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
    def test_various_eps(self, eps, skip_without_pytorch):
        """Test fast_rmsnorm with different epsilon values."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        batch, seq, hidden = 2, 128, 512

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx, eps=eps)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch, eps=eps)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm with eps={eps} mismatch"
        )


# =============================================================================
# Fast RMSNorm Residual Parity Tests
# =============================================================================

class TestFastRMSNormResidualParity:
    """Tests for fast_rmsnorm_residual() fused kernel parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fast_rmsnorm_residual matches PyTorch."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm_residual

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        residual_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        residual_mlx = _convert_to_mlx(residual_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = fast_rmsnorm_residual(x_mlx, residual_mlx, weight_mlx, eps=1e-6)
        mx.eval(mlx_out)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        residual_torch = _convert_to_torch(residual_np, dtype)
        weight_torch = _convert_to_torch(weight_np, dtype)
        torch_out = pytorch_rmsnorm_residual(x_torch, residual_torch, weight_torch, eps=1e-6)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"fast_rmsnorm_residual mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_consistency_with_separate_ops(self, skip_without_pytorch):
        """Verify fused kernel matches separate add + rmsnorm."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm, fast_rmsnorm_residual

        batch, seq, hidden = 4, 256, 1024

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        residual_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        residual_mlx = mx.array(residual_np)
        weight_mlx = mx.array(weight_np)

        # Fused
        fused_out = fast_rmsnorm_residual(x_mlx, residual_mlx, weight_mlx)
        mx.eval(fused_out)

        # Separate
        combined = x_mlx + residual_mlx
        separate_out = fast_rmsnorm(combined, weight_mlx)
        mx.eval(separate_out)

        np.testing.assert_allclose(
            _to_numpy(fused_out), _to_numpy(separate_out),
            rtol=1e-5, atol=1e-6,
            err_msg="Fused rmsnorm_residual differs from separate ops"
        )


# =============================================================================
# RMSNorm Auto-Select Path Parity Tests
# =============================================================================

class TestRMSNormAutoSelectParity:
    """Tests for rmsnorm() with automatic path selection."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test rmsnorm() auto-selection matches PyTorch."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX (auto-select)
        x_mlx = _convert_to_mlx(x_np, dtype)
        weight_mlx = _convert_to_mlx(weight_np, dtype)
        mlx_out = rmsnorm(x_mlx, weight_mlx, use_metal=True)
        mx.eval(mlx_out)

        # PyTorch
        x_torch = _convert_to_torch(x_np, dtype)
        weight_torch = _convert_to_torch(weight_np, dtype)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"rmsnorm auto-select mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    def test_metal_vs_fallback_consistency(self, skip_without_pytorch):
        """Verify Metal kernel matches Python fallback."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        # Use size that triggers Metal kernel
        batch, seq, hidden = 8, 512, 2048

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)

        # Metal path
        out_metal = rmsnorm(x_mlx, weight_mlx, use_metal=True)
        mx.eval(out_metal)

        # Fallback path
        out_fallback = rmsnorm(x_mlx, weight_mlx, use_metal=False)
        mx.eval(out_fallback)

        np.testing.assert_allclose(
            _to_numpy(out_metal), _to_numpy(out_fallback),
            rtol=1e-5, atol=1e-6,
            err_msg="Metal vs fallback RMSNorm mismatch"
        )


# =============================================================================
# Backward Pass Parity Tests
# =============================================================================

class TestRMSNormBackwardParity:
    """Tests for RMSNorm gradient computation parity."""

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test RMSNorm backward pass matches PyTorch."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        # MLX backward
        weight_mlx = mx.array(weight_np)

        def mlx_loss_fn(x):
            return mx.sum(rmsnorm(x, weight_mlx, use_metal=False))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm backward mismatch [{size}]"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRMSNormEdgeCases:
    """Edge case tests for RMSNorm implementations."""

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_small_values(self, skip_without_pytorch):
        """Test RMSNorm with very small input values."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        batch, seq, hidden = 2, 64, 256

        np.random.seed(42)
        # Scale down inputs to test numerical stability
        x_np = (np.random.randn(batch, seq, hidden) * 1e-4).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="RMSNorm with small values mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_large_values(self, skip_without_pytorch):
        """Test RMSNorm with large input values."""
        from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

        batch, seq, hidden = 2, 64, 256

        np.random.seed(42)
        # Scale up inputs
        x_np = (np.random.randn(batch, seq, hidden) * 100).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        mlx_out = fast_rmsnorm(x_mlx, weight_mlx)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="RMSNorm with large values mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_single_element(self, skip_without_pytorch):
        """Test RMSNorm with minimal dimensions."""
        from mlx_primitives.kernels.rmsnorm import rmsnorm

        batch, seq, hidden = 1, 1, 64

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        weight_np = np.random.randn(hidden).astype(np.float32)

        x_mlx = mx.array(x_np)
        weight_mlx = mx.array(weight_np)
        # Use fallback for small inputs
        mlx_out = rmsnorm(x_mlx, weight_mlx, use_metal=False)
        mx.eval(mlx_out)

        x_torch = torch.from_numpy(x_np)
        weight_torch = torch.from_numpy(weight_np)
        torch_out = pytorch_rmsnorm(x_torch, weight_torch)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), torch_out.numpy(),
            rtol=rtol, atol=atol,
            err_msg="RMSNorm single element mismatch"
        )
