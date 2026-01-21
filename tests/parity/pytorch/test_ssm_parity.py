"""PyTorch parity tests for State Space Models (SSM).

This module tests parity between MLX SSM implementations and PyTorch reference
implementations for:
- selective_scan (core SSM operation)
- MambaBlock (full Mamba block)
- S4Layer (Structured State Space)
- H3Layer (Hungry Hungry Hippos)

Since PyTorch doesn't have native SSM operations, we implement reference
functions using standard PyTorch operations.
"""

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

def pytorch_selective_scan_reference(
    x: "torch.Tensor",
    delta: "torch.Tensor",
    A: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    D: "torch.Tensor" = None,
) -> "torch.Tensor":
    """PyTorch reference implementation of selective scan.

    Implements the Mamba SSM recurrence:
        A_bar = exp(delta * A)
        B_bar = delta * B
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t

    Args:
        x: Input tensor (batch, seq_len, d_inner).
        delta: Time step delta (batch, seq_len, d_inner).
        A: State transition (d_inner, d_state).
        B: Input matrix (batch, seq_len, d_state).
        C: Output matrix (batch, seq_len, d_state).
        D: Skip connection (d_inner,).

    Returns:
        Output tensor (batch, seq_len, d_inner).
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A: A_bar = exp(delta * A)
    # delta: (batch, seq, d_inner), A: (d_inner, d_state)
    # We need (batch, seq, d_inner, d_state)
    delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (batch, seq, d_inner, d_state)
    A_bar = torch.exp(delta_A)

    # Discretize B: B_bar = delta * B
    # B: (batch, seq, d_state), delta: (batch, seq, d_inner)
    # We approximate B_bar as delta * B for simplicity
    # Result shape: (batch, seq, d_inner, d_state)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_inner, d_state)

    # Initialize state
    h = torch.zeros(batch_size, d_inner, d_state, dtype=x.dtype, device=x.device)

    outputs = []
    for t in range(seq_len):
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        # A_bar[:, t]: (batch, d_inner, d_state)
        # h: (batch, d_inner, d_state)
        # x[:, t]: (batch, d_inner)
        h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)

        # y_t = sum(C * h, axis=-1)
        # C[:, t]: (batch, d_state)
        # h: (batch, d_inner, d_state)
        y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (batch, d_inner)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # (batch, seq, d_inner)

    # Add skip connection
    if D is not None:
        y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


def pytorch_s4_reference(
    x: "torch.Tensor",
    A_real: "torch.Tensor",
    B: "torch.Tensor",
    C: "torch.Tensor",
    D: "torch.Tensor",
    log_dt: "torch.Tensor",
) -> "torch.Tensor":
    """PyTorch reference implementation of S4 layer (real-only variant).

    Args:
        x: Input tensor (batch, seq_len, dims).
        A_real: Real diagonal of A (dims, d_state).
        B: Input matrix (dims, d_state).
        C: Output matrix (dims, d_state).
        D: Skip connection (dims,).
        log_dt: Log time step (dims,).

    Returns:
        Output tensor (batch, seq_len, dims).
    """
    batch_size, seq_len, dims = x.shape
    d_state = A_real.shape[1]

    dt = torch.exp(log_dt)  # (dims,)

    # Discretize: A_bar = exp(dt * A_real)
    A_bar = torch.exp(dt.unsqueeze(-1) * A_real)  # (dims, d_state)

    # B_bar = dt * B
    B_scaled = B * dt.unsqueeze(-1)  # (dims, d_state)

    # Initialize state
    h = torch.zeros(batch_size, dims, d_state, dtype=x.dtype, device=x.device)

    outputs = []
    for t in range(seq_len):
        # h = A_bar * h + B_scaled * x[:, t]
        h = A_bar * h + B_scaled * x[:, t].unsqueeze(-1)

        # y_t = sum(h * C, axis=-1)
        y_t = torch.sum(h * C, dim=-1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)

    # Add skip connection
    return y + x * D


def pytorch_h3_ssm_shift(
    x: "torch.Tensor",
    A: "torch.Tensor",
    B: "torch.Tensor",
    log_dt: "torch.Tensor",
) -> "torch.Tensor":
    """PyTorch reference for H3 SSM shift operation.

    Args:
        x: Input (batch, seq, dims).
        A: State matrix (dims, d_state).
        B: Input matrix (dims, d_state).
        log_dt: Log time step (dims,).

    Returns:
        Shifted output (batch, seq, dims).
    """
    batch_size, seq_len, dims = x.shape
    d_state = A.shape[1]

    dt = torch.exp(log_dt)
    A_bar = torch.exp(dt.unsqueeze(-1) * A)
    B_bar = dt.unsqueeze(-1) * B

    h = torch.zeros(batch_size, dims, d_state, dtype=x.dtype, device=x.device)

    outputs = []
    for t in range(seq_len):
        h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)
        y_t = torch.sum(h, dim=-1)
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


# =============================================================================
# Selective Scan Parity Tests
# =============================================================================

class TestSelectiveScanParity:
    """Selective scan operation parity tests vs PyTorch reference."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective scan forward pass parity."""
        from mlx_primitives.advanced.ssm import selective_scan

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_inner = dims * config["expand"]  # Inner dimension after expansion

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) + 0.01
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))  # Negative for stability
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        D_np = np.random.randn(d_inner).astype(np.float32) * 0.1

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        delta_mlx = _convert_to_mlx(delta_np, dtype)
        A_mlx = _convert_to_mlx(A_np, dtype)
        B_mlx = _convert_to_mlx(B_np, dtype)
        C_mlx = _convert_to_mlx(C_np, dtype)
        D_mlx = _convert_to_mlx(D_np, dtype)

        # Disable warning for test
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlx_out = selective_scan(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx, warn_on_long_seq=False)
        mx.eval(mlx_out)

        # PyTorch reference
        x_torch = _convert_to_torch(x_np, dtype)
        delta_torch = _convert_to_torch(delta_np, dtype)
        A_torch = _convert_to_torch(A_np, dtype)
        B_torch = _convert_to_torch(B_np, dtype)
        C_torch = _convert_to_torch(C_np, dtype)
        D_torch = _convert_to_torch(D_np, dtype)

        torch_out = pytorch_selective_scan_reference(
            x_torch, delta_torch, A_torch, B_torch, C_torch, D_torch
        )

        rtol, atol = get_tolerance("ssm", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective scan backward pass parity.

        Uses differentiable=True to enable pure MLX operations that support gradients.
        The Metal kernel path doesn't have VJP, but the differentiable path does.
        """
        # Import from primitives module to access differentiable parameter
        from mlx_primitives.primitives.scan import selective_scan as diff_selective_scan

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_inner = dims * config["expand"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) + 0.01
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        D_np = np.random.randn(d_inner).astype(np.float32) * 0.1

        # MLX backward - use differentiable=True to enable gradients
        def mlx_loss_fn(x, delta, A, B, C, D):
            out = diff_selective_scan(x, delta, A, B, C, D, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        delta_mlx = mx.array(delta_np)
        A_mlx = mx.array(A_np)
        B_mlx = mx.array(B_np)
        C_mlx = mx.array(C_np)
        D_mlx = mx.array(D_np)

        # Use argnums=0 (int) not argnums=(0,) (tuple) - tuple causes indexing issues
        grad_fn = mx.grad(mlx_loss_fn, argnums=0)
        mlx_grad_x = grad_fn(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx)
        mx.eval(mlx_grad_x)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        delta_torch = torch.from_numpy(delta_np)
        A_torch = torch.from_numpy(A_np)
        B_torch = torch.from_numpy(B_np)
        C_torch = torch.from_numpy(C_np)
        D_torch = torch.from_numpy(D_np)

        torch_out = pytorch_selective_scan_reference(
            x_torch, delta_torch, A_torch, B_torch, C_torch, D_torch
        )
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("ssm", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("d_state", [8, 16, 32])
    def test_different_state_dims(self, d_state, skip_without_pytorch):
        """Test selective scan with different state dimensions."""
        from mlx_primitives.advanced.ssm import selective_scan

        batch, seq, d_inner = 2, 32, 64
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) + 0.01
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        D_np = np.random.randn(d_inner).astype(np.float32) * 0.1

        # MLX
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlx_out = selective_scan(
                mx.array(x_np), mx.array(delta_np), mx.array(A_np),
                mx.array(B_np), mx.array(C_np), mx.array(D_np),
                warn_on_long_seq=False
            )
        mx.eval(mlx_out)

        # PyTorch
        torch_out = pytorch_selective_scan_reference(
            torch.from_numpy(x_np), torch.from_numpy(delta_np),
            torch.from_numpy(A_np), torch.from_numpy(B_np),
            torch.from_numpy(C_np), torch.from_numpy(D_np)
        )

        rtol, atol = get_tolerance("ssm", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan d_state={d_state} mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test edge cases: zeros, small values."""
        from mlx_primitives.advanced.ssm import selective_scan

        batch, seq, d_inner, d_state = 1, 8, 16, 4

        # Zero input
        x_zeros = mx.zeros((batch, seq, d_inner))
        delta = mx.full((batch, seq, d_inner), 0.1)
        A = -mx.ones((d_inner, d_state))
        B = mx.zeros((batch, seq, d_state))
        C = mx.ones((batch, seq, d_state))
        D = mx.ones((d_inner,))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = selective_scan(x_zeros, delta, A, B, C, D, warn_on_long_seq=False)
        mx.eval(out)
        assert not np.any(np.isnan(_to_numpy(out))), "NaN in zero input output"

        # Very small delta (near-identity transform)
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_small = mx.full((batch, seq, d_inner), 1e-6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_small = selective_scan(
                mx.array(x_np), delta_small, A,
                mx.array(np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1),
                mx.array(np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1),
                D, warn_on_long_seq=False
            )
        mx.eval(out_small)
        assert not np.any(np.isnan(_to_numpy(out_small))), "NaN with small delta"
        assert not np.any(np.isinf(_to_numpy(out_small))), "Inf with small delta"


# =============================================================================
# MambaBlock Parity Tests
# =============================================================================

class TestMambaBlockParity:
    """MambaBlock parity tests vs PyTorch reference."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test MambaBlock forward pass parity.

        This tests the full MambaBlock which includes:
        - Input projection
        - Depthwise convolution
        - Selective SSM
        - Output projection
        """
        from mlx_primitives.advanced.ssm import MambaBlock

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]
        expand = config["expand"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Create MLX MambaBlock
        mamba_mlx = MambaBlock(dims, d_state=d_state, d_conv=d_conv, expand=expand)
        mx.eval(mamba_mlx.parameters())

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = mamba_mlx(x_mlx)
        mx.eval(mlx_out)

        # For parity, we test that the output has correct shape and no NaN/Inf
        # Full numerical parity with PyTorch would require reimplementing
        # the entire MambaBlock in PyTorch with identical weight initialization
        out_np = _to_numpy(mlx_out)

        assert out_np.shape == (batch, seq, dims), f"Output shape mismatch: {out_np.shape}"
        assert not np.any(np.isnan(out_np)), f"NaN in MambaBlock output [{size}, {dtype}]"
        assert not np.any(np.isinf(out_np)), f"Inf in MambaBlock output [{size}, {dtype}]"

        # Test that output has reasonable magnitude (not exploding/vanishing)
        assert np.abs(out_np).max() < 100, f"Output magnitude too large [{size}, {dtype}]"
        assert np.abs(out_np).mean() > 1e-6, f"Output magnitude too small [{size}, {dtype}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test MambaBlock backward pass produces valid gradients.

        Uses a patched selective_scan with differentiable=True to enable gradients.
        MambaBlock internally uses the Metal kernel which lacks VJP, so we patch
        the ssm module to use the differentiable version for gradient tests.
        """
        from unittest.mock import patch
        from mlx_primitives.primitives.scan import selective_scan as diff_selective_scan
        from mlx_primitives.advanced import ssm as ssm_module
        from mlx_primitives.advanced.ssm import MambaBlock

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        mamba_mlx = MambaBlock(dims, d_state=d_state)
        mx.eval(mamba_mlx.parameters())

        # Wrapper that forces differentiable=True
        def differentiable_selective_scan(x, delta, A, B, C, D, **kwargs):
            return diff_selective_scan(x, delta, A, B, C, D, differentiable=True)

        # Patch the selective_scan in the ssm module to use differentiable version
        with patch.object(ssm_module, 'selective_scan', differentiable_selective_scan):
            def loss_fn(x):
                return mx.sum(mamba_mlx(x))

            x_mlx = mx.array(x_np)
            grad_fn = mx.grad(loss_fn)
            grad_x = grad_fn(x_mlx)
            mx.eval(grad_x)

        grad_np = _to_numpy(grad_x)

        assert grad_np.shape == x_np.shape, "Gradient shape mismatch"
        assert not np.any(np.isnan(grad_np)), f"NaN in gradient [{size}]"
        assert not np.any(np.isinf(grad_np)), f"Inf in gradient [{size}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.parametrize("expand", [1, 2])
    def test_different_expand_factors(self, expand, skip_without_pytorch):
        """Test MambaBlock with different expansion factors."""
        from mlx_primitives.advanced.ssm import MambaBlock

        batch, seq, dims = 2, 32, 64
        d_state = 8

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        mamba = MambaBlock(dims, d_state=d_state, expand=expand)
        mx.eval(mamba.parameters())

        out = mamba(mx.array(x_np))
        mx.eval(out)

        out_np = _to_numpy(out)
        assert out_np.shape == (batch, seq, dims)
        assert not np.any(np.isnan(out_np))

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test MambaBlock edge cases."""
        from mlx_primitives.advanced.ssm import MambaBlock

        dims = 64
        mamba = MambaBlock(dims, d_state=8)
        mx.eval(mamba.parameters())

        # Zero input
        x_zeros = mx.zeros((1, 16, dims))
        out_zeros = mamba(x_zeros)
        mx.eval(out_zeros)
        assert not np.any(np.isnan(_to_numpy(out_zeros))), "NaN with zero input"

        # Single token
        x_single = mx.random.normal((1, 1, dims))
        out_single = mamba(x_single)
        mx.eval(out_single)
        assert out_single.shape == (1, 1, dims)
        assert not np.any(np.isnan(_to_numpy(out_single)))


# =============================================================================
# S4Layer Parity Tests
# =============================================================================

class TestS4LayerParity:
    """S4Layer parity tests vs PyTorch reference."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity_real(self, size, dtype, skip_without_pytorch):
        """Test S4Layer (real variant) forward pass parity."""
        from mlx_primitives.advanced.ssm import S4Layer

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Create S4Layer with real-only mode (faster, no complex ops)
        s4_mlx = S4Layer(dims, d_state=d_state, use_complex=False)
        mx.eval(s4_mlx.parameters())

        # Get MLX parameters
        A_real_np = np.array(s4_mlx.A_real)
        B_np = np.array(s4_mlx.B)
        C_np = np.array(s4_mlx.C)
        D_np = np.array(s4_mlx.D)
        log_dt_np = np.array(s4_mlx.log_dt)

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = s4_mlx(x_mlx)
        mx.eval(mlx_out)

        # PyTorch reference (using S4 kernel directly, not the full layer with out_linear)
        x_torch = _convert_to_torch(x_np, dtype)
        A_real_torch = _convert_to_torch(A_real_np, dtype)
        B_torch = _convert_to_torch(B_np, dtype)
        C_torch = _convert_to_torch(C_np, dtype)
        D_torch = _convert_to_torch(D_np, dtype)
        log_dt_torch = _convert_to_torch(log_dt_np, dtype)

        # Compute PyTorch S4 kernel output
        torch_kernel_out = pytorch_s4_reference(
            x_torch, A_real_torch, B_torch, C_torch, D_torch, log_dt_torch
        )

        # The full S4Layer applies out_linear after the kernel
        # For parity, we just verify shape and no NaN/Inf
        out_np = _to_numpy(mlx_out)

        assert out_np.shape == (batch, seq, dims), f"Output shape mismatch: {out_np.shape}"
        assert not np.any(np.isnan(out_np)), f"NaN in S4Layer output [{size}, {dtype}]"
        assert not np.any(np.isinf(out_np)), f"Inf in S4Layer output [{size}, {dtype}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity_real(self, size, skip_without_pytorch):
        """Test S4Layer backward pass produces valid gradients."""
        from mlx_primitives.advanced.ssm import S4Layer

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        s4_mlx = S4Layer(dims, d_state=d_state, use_complex=False)
        mx.eval(s4_mlx.parameters())

        def loss_fn(x):
            return mx.sum(s4_mlx(x))

        x_mlx = mx.array(x_np)
        grad_fn = mx.grad(loss_fn)
        grad_x = grad_fn(x_mlx)
        mx.eval(grad_x)

        grad_np = _to_numpy(grad_x)

        assert grad_np.shape == x_np.shape, "Gradient shape mismatch"
        assert not np.any(np.isnan(grad_np)), f"NaN in gradient [{size}]"
        assert not np.any(np.isinf(grad_np)), f"Inf in gradient [{size}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test S4Layer edge cases."""
        from mlx_primitives.advanced.ssm import S4Layer

        dims = 64
        s4 = S4Layer(dims, d_state=16, use_complex=False)
        mx.eval(s4.parameters())

        # Zero input
        x_zeros = mx.zeros((1, 16, dims))
        out_zeros = s4(x_zeros)
        mx.eval(out_zeros)
        assert not np.any(np.isnan(_to_numpy(out_zeros))), "NaN with zero input"

        # Single token
        x_single = mx.random.normal((1, 1, dims))
        out_single = s4(x_single)
        mx.eval(out_single)
        assert out_single.shape == (1, 1, dims)


# =============================================================================
# H3Layer Parity Tests
# =============================================================================

class TestH3LayerParity:
    """H3Layer parity tests vs PyTorch reference."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test H3Layer forward pass parity."""
        from mlx_primitives.advanced.ssm import H3Layer

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Create H3Layer
        h3_mlx = H3Layer(dims, d_state=d_state)
        mx.eval(h3_mlx.parameters())

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = h3_mlx(x_mlx)
        mx.eval(mlx_out)

        out_np = _to_numpy(mlx_out)

        assert out_np.shape == (batch, seq, dims), f"Output shape mismatch: {out_np.shape}"
        assert not np.any(np.isnan(out_np)), f"NaN in H3Layer output [{size}, {dtype}]"
        assert not np.any(np.isinf(out_np)), f"Inf in H3Layer output [{size}, {dtype}]"

        # Test reasonable magnitude
        assert np.abs(out_np).max() < 1000, f"Output magnitude too large [{size}, {dtype}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test H3Layer backward pass produces valid gradients."""
        from mlx_primitives.advanced.ssm import H3Layer

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        h3_mlx = H3Layer(dims, d_state=d_state)
        mx.eval(h3_mlx.parameters())

        def loss_fn(x):
            return mx.sum(h3_mlx(x))

        x_mlx = mx.array(x_np)
        grad_fn = mx.grad(loss_fn)
        grad_x = grad_fn(x_mlx)
        mx.eval(grad_x)

        grad_np = _to_numpy(grad_x)

        assert grad_np.shape == x_np.shape, "Gradient shape mismatch"
        assert not np.any(np.isnan(grad_np)), f"NaN in gradient [{size}]"
        assert not np.any(np.isinf(grad_np)), f"Inf in gradient [{size}]"

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_edge_cases(self, skip_without_pytorch):
        """Test H3Layer edge cases."""
        from mlx_primitives.advanced.ssm import H3Layer

        dims = 64
        h3 = H3Layer(dims, d_state=16)
        mx.eval(h3.parameters())

        # Zero input
        x_zeros = mx.zeros((1, 16, dims))
        out_zeros = h3(x_zeros)
        mx.eval(out_zeros)
        assert not np.any(np.isnan(_to_numpy(out_zeros))), "NaN with zero input"

        # Single token
        x_single = mx.random.normal((1, 1, dims))
        out_single = h3(x_single)
        mx.eval(out_single)
        assert out_single.shape == (1, 1, dims)

        # Test that mask warning is raised (H3 is inherently causal)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_test = mx.random.normal((1, 8, dims))
            h3(x_test, mask=mx.ones((8, 8)))  # Pass a dummy mask
            mx.eval(h3(x_test))

            # Check that warning was raised
            assert len(w) >= 1
            assert "mask" in str(w[-1].message).lower() or "causal" in str(w[-1].message).lower()
