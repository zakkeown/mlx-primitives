"""PyTorch parity tests for parallel primitives (scan, gather, scatter)."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import scan_inputs, SIZE_CONFIGS
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


def _pytorch_ssm_scan_reference(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of SSM scan: h[t] = A[t] * h[t-1] + x[t]."""
    batch, seq_len, dim = x.shape
    h = torch.zeros(batch, dim, dtype=x.dtype)
    outputs = []
    for t in range(seq_len):
        h = A[:, t, :] * h + x[:, t, :]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def _pytorch_selective_scan_reference(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch reference implementation of Mamba-style selective scan."""
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretization
    # delta: (batch, seq, d_inner) -> (batch, seq, d_inner, 1)
    # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
    delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (batch, seq, d_inner, d_state)
    A_bar = torch.exp(delta_A)

    # B_bar = delta * B
    # B: (batch, seq, d_state) -> (batch, seq, 1, d_state)
    B_x = B.unsqueeze(2) * x.unsqueeze(-1)  # (batch, seq, d_inner, d_state)
    B_bar_x = delta.unsqueeze(-1) * B_x

    # Sequential SSM scan
    h = torch.zeros(batch_size, d_inner, d_state, dtype=x.dtype)
    outputs = []
    for t in range(seq_len):
        h = A_bar[:, t, :, :] * h + B_bar_x[:, t, :, :]
        # Output projection: y = sum(C * h, axis=-1)
        y_t = torch.sum(C[:, t, :].unsqueeze(1) * h, dim=-1)  # (batch, d_inner)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # (batch, seq, d_inner)

    # Skip connection
    if D is not None:
        y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


def _pytorch_selective_scan_with_mlx_discretization(
    x_np: np.ndarray,
    delta_np: np.ndarray,
    A_np: np.ndarray,
    B_np: np.ndarray,
    C_np: np.ndarray,
    D_np: np.ndarray = None,
) -> np.ndarray:
    """PyTorch selective scan using MLX-computed discretization.

    This isolates the exp() precision difference by computing discretization
    in MLX, then running the sequential scan in PyTorch.
    """
    batch_size, seq_len, d_inner = x_np.shape
    d_state = A_np.shape[1]

    # Compute discretization using MLX (for consistent precision)
    delta_mlx = mx.array(delta_np)
    A_mlx = mx.array(A_np)
    x_mlx = mx.array(x_np)
    B_mlx = mx.array(B_np)

    # delta_A = delta[..., None] * A[None, None, :, :]
    delta_A = delta_mlx[..., None] * A_mlx[None, None, :, :]
    A_bar_mlx = mx.exp(delta_A)

    # B_bar_x = delta[..., None] * (B[:, :, None, :] * x[..., None])
    B_x = B_mlx[:, :, None, :] * x_mlx[..., None]
    B_bar_x_mlx = delta_mlx[..., None] * B_x

    mx.eval(A_bar_mlx, B_bar_x_mlx)
    A_bar_np = np.array(A_bar_mlx)
    B_bar_x_np = np.array(B_bar_x_mlx)

    # Sequential SSM scan using numpy/torch
    h = np.zeros((batch_size, d_inner, d_state), dtype=np.float32)
    outputs = []
    for t in range(seq_len):
        h = A_bar_np[:, t, :, :] * h + B_bar_x_np[:, t, :, :]
        # Output projection: y = sum(C * h, axis=-1)
        y_t = np.sum(C_np[:, t, :, None] * h.transpose(0, 2, 1), axis=1)  # (batch, d_inner)
        outputs.append(y_t)

    y = np.stack(outputs, axis=1)  # (batch, seq, d_inner)

    # Skip connection
    if D_np is not None:
        y = y + x_np * D_np[None, None, :]

    return y


# =============================================================================
# Associative Scan (Add) Parity Tests
# =============================================================================

class TestAssociativeScanAddParity:
    """Associative scan with add operator (cumsum) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan add forward pass against float64 ground truth.

        Parallel prefix sum algorithms accumulate rounding errors of O(log n * eps).
        We compare against float64 truth to verify mathematical correctness.
        Tolerances account for the expected precision of parallel algorithms.

        For fp16/bf16, we first quantize the input to match what MLX uses.

        Reference: https://ieeexplore.ieee.org/document/9286240
        """
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX - convert to target dtype
        x_mlx = _convert_to_mlx(x_np, dtype)

        # Our Metal kernel implementation
        mlx_out = associative_scan(x_mlx, operator="add", axis=1)
        mx.eval(mlx_out)

        # Get quantized input values (for fp16/bf16 consistency)
        x_quantized = np.array(x_mlx.astype(mx.float32))

        # Ground truth: float64 cumsum of quantized input
        truth = np.cumsum(x_quantized.astype(np.float64), axis=1).astype(np.float32)

        rtol, atol = get_tolerance("primitives", "associative_scan_add", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), truth,
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan add forward mismatch vs float64 truth [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan add backward pass parity."""
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX backward (use differentiable=True for gradient support)
        def mlx_loss_fn(x):
            out = associative_scan(x, operator="add", axis=1, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = torch.cumsum(x_torch, dim=1)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "associative_scan_add", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan add backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_different_axes(self, axis, skip_without_pytorch):
        """Test associative scan add on different axes."""
        from mlx_primitives.primitives.scan import associative_scan

        batch, seq, dim = 4, 32, 16

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = associative_scan(x_mlx, operator="add", axis=axis)

        # Reference: MLX builtin cumsum
        mlx_ref = mx.cumsum(x_mlx, axis=axis)
        mx.eval(mlx_ref)

        rtol, atol = get_tolerance("primitives", "associative_scan_add", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(mlx_ref),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan add axis={axis} mismatch vs mx.cumsum"
        )


# =============================================================================
# Associative Scan (Mul) Parity Tests
# =============================================================================

class TestAssociativeScanMulParity:
    """Associative scan with multiply operator (cumprod) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan mul forward pass against float64 ground truth.

        Parallel prefix product algorithms accumulate rounding errors of O(log n * eps).
        We compare against float64 truth to verify mathematical correctness.
        Tolerances account for the expected precision of parallel algorithms.

        For fp16/bf16, we first quantize the input to match what MLX uses.
        """
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]

        np.random.seed(42)
        # Use values close to 1 to avoid numerical instability in cumprod
        x_np = np.random.uniform(0.9, 1.1, (batch, seq, dim)).astype(np.float32)

        # MLX - convert to target dtype
        x_mlx = _convert_to_mlx(x_np, dtype)

        # Our Metal kernel implementation
        mlx_out = associative_scan(x_mlx, operator="mul", axis=1)
        mx.eval(mlx_out)

        # Get quantized input values (for fp16/bf16 consistency)
        x_quantized = np.array(x_mlx.astype(mx.float32))

        # Ground truth: float64 cumprod of quantized input
        truth = np.cumprod(x_quantized.astype(np.float64), axis=1).astype(np.float32)

        rtol, atol = get_tolerance("primitives", "associative_scan_mul", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), truth,
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan mul forward mismatch vs float64 truth [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan mul backward pass parity."""
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.uniform(0.9, 1.1, (batch, seq, dim)).astype(np.float32)

        # MLX backward (use differentiable=True for gradient support)
        def mlx_loss_fn(x):
            out = associative_scan(x, operator="mul", axis=1, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        torch_out = torch.cumprod(x_torch, dim=1)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "associative_scan_mul", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan mul backward mismatch [{size}]"
        )


# =============================================================================
# Associative Scan (SSM) Parity Tests
# =============================================================================

class TestAssociativeScanSSMParity:
    """Associative scan with SSM operator parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan SSM forward pass parity."""
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        # A is typically small (decay factors between 0 and 1)
        A_np = np.random.uniform(0.8, 0.99, (batch, seq, dim)).astype(np.float32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        A_mlx = _convert_to_mlx(A_np, dtype)
        mlx_out = associative_scan(x_mlx, operator="ssm", A=A_mlx, axis=1)

        # PyTorch reference (sequential implementation)
        x_torch = _convert_to_torch(x_np, dtype)
        A_torch = _convert_to_torch(A_np, dtype)
        torch_out = _pytorch_ssm_scan_reference(A_torch.float(), x_torch.float())

        rtol, atol = get_tolerance("primitives", "associative_scan_ssm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan SSM forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan SSM backward pass parity."""
        from mlx_primitives.primitives.scan import associative_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        A_np = np.random.uniform(0.8, 0.99, (batch, seq, dim)).astype(np.float32)

        # MLX backward (use differentiable=True for gradient support)
        def mlx_loss_fn(x, A):
            out = associative_scan(x, operator="ssm", A=A, axis=1, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        A_mlx = mx.array(A_np)
        mlx_grad_x, mlx_grad_A = mx.grad(mlx_loss_fn, argnums=(0, 1))(x_mlx, A_mlx)
        mx.eval(mlx_grad_x, mlx_grad_A)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        A_torch = torch.from_numpy(A_np).requires_grad_(True)
        torch_out = _pytorch_ssm_scan_reference(A_torch, x_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "associative_scan_ssm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan SSM x gradient mismatch [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_A), A_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"AssociativeScan SSM A gradient mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_mamba_style(self, skip_without_pytorch):
        """Test Mamba-style selective scan."""
        from mlx_primitives.primitives.scan import associative_scan

        # Mamba uses diagonal A with decay factors close to 1
        batch, seq, d_inner = 2, 64, 32

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        # Mamba-style: A is typically log-space initialized
        A_np = np.exp(np.random.uniform(-0.2, -0.01, (batch, seq, d_inner))).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        A_mlx = mx.array(A_np)
        mlx_out = associative_scan(x_mlx, operator="ssm", A=A_mlx, axis=1)

        # PyTorch reference
        x_torch = torch.from_numpy(x_np)
        A_torch = torch.from_numpy(A_np)
        torch_out = _pytorch_ssm_scan_reference(A_torch, x_torch)

        rtol, atol = get_tolerance("primitives", "associative_scan_ssm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="Mamba-style SSM scan mismatch"
        )


# =============================================================================
# Selective Scan Parity Tests
# =============================================================================

class TestSelectiveScanParity:
    """Selective scan (Mamba-style) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective scan forward pass parity.

        Uses MLX-computed discretization as reference to isolate exp()
        precision differences between Metal and CPU implementations.
        """
        from mlx_primitives.primitives.scan import selective_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        d_inner = config["dim"]
        d_state = 16

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) * 0.1 + 0.01
        # A must be negative for stable SSM dynamics (decay, not growth)
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32)) * 0.1 - 0.01
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1

        # Get quantized inputs for reference (critical for fp16/bf16)
        x_mlx = _convert_to_mlx(x_np, dtype)
        delta_mlx = _convert_to_mlx(delta_np, dtype)
        A_mlx = _convert_to_mlx(A_np, dtype)
        B_mlx = _convert_to_mlx(B_np, dtype)
        C_mlx = _convert_to_mlx(C_np, dtype)
        mx.eval(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx)

        # Get quantized values for reference computation
        x_q = np.array(x_mlx.astype(mx.float32))
        delta_q = np.array(delta_mlx.astype(mx.float32))
        A_q = np.array(A_mlx.astype(mx.float32))
        B_q = np.array(B_mlx.astype(mx.float32))
        C_q = np.array(C_mlx.astype(mx.float32))

        mlx_out = selective_scan(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D=None)

        # Reference using MLX-computed discretization with quantized inputs
        ref_out = _pytorch_selective_scan_with_mlx_discretization(
            x_q, delta_q, A_q, B_q, C_q, D_np=None
        )

        rtol, atol = get_tolerance("primitives", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveScan forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective scan backward pass parity."""
        from mlx_primitives.primitives.scan import selective_scan

        config = SIZE_CONFIGS[size]["scan"]
        batch = config["batch"]
        seq = config["seq"]
        d_inner = config["dim"]
        d_state = 16
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) * 0.1 + 0.01
        # A must be negative for stable SSM dynamics (decay, not growth)
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32)) * 0.1 - 0.01
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1

        # MLX backward (use differentiable=True for gradient support)
        def mlx_loss_fn(x, delta, A, B, C):
            out = selective_scan(x, delta, A, B, C, D=None, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        delta_mlx = mx.array(delta_np)
        A_mlx = mx.array(A_np)
        B_mlx = mx.array(B_np)
        C_mlx = mx.array(C_np)

        mlx_grad_x = mx.grad(mlx_loss_fn, argnums=0)(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx)
        mx.eval(mlx_grad_x)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        delta_torch = torch.from_numpy(delta_np).requires_grad_(True)
        A_torch = torch.from_numpy(A_np).requires_grad_(True)
        B_torch = torch.from_numpy(B_np).requires_grad_(True)
        C_torch = torch.from_numpy(C_np).requires_grad_(True)

        torch_out = _pytorch_selective_scan_reference(
            x_torch, delta_torch, A_torch, B_torch, C_torch, D=None
        )
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveScan x gradient mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_d_parameter(self, skip_without_pytorch):
        """Test selective scan with D (skip connection) parameter."""
        from mlx_primitives.primitives.scan import selective_scan

        batch, seq, d_inner, d_state = 2, 32, 16, 8

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) * 0.1 + 0.01
        # A must be negative for stable SSM dynamics
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32)) * 0.1 - 0.01
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        D_np = np.random.randn(d_inner).astype(np.float32)

        # MLX
        mlx_out = selective_scan(
            mx.array(x_np), mx.array(delta_np), mx.array(A_np),
            mx.array(B_np), mx.array(C_np), D=mx.array(D_np)
        )

        # PyTorch reference
        torch_out = _pytorch_selective_scan_reference(
            torch.from_numpy(x_np), torch.from_numpy(delta_np),
            torch.from_numpy(A_np), torch.from_numpy(B_np),
            torch.from_numpy(C_np), D=torch.from_numpy(D_np)
        )

        rtol, atol = get_tolerance("primitives", "selective_scan", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg="SelectiveScan with D mismatch"
        )


# =============================================================================
# Selective Gather Parity Tests
# =============================================================================

class TestSelectiveGatherParity:
    """Selective gather operation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective gather forward pass parity."""
        from mlx_primitives.primitives.gather_scatter import selective_gather

        config = SIZE_CONFIGS[size]["scan"]
        n_tokens = config["batch"] * config["seq"]
        dim = config["dim"]
        capacity = n_tokens // 2

        np.random.seed(42)
        x_np = np.random.randn(n_tokens, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)

        # MLX
        x_mlx = _convert_to_mlx(x_np, dtype)
        indices_mlx = mx.array(indices_np.astype(np.uint32))
        mlx_out = selective_gather(x_mlx, indices_mlx)

        # PyTorch reference (simple indexing)
        x_torch = _convert_to_torch(x_np, dtype)
        indices_torch = torch.from_numpy(indices_np).long()
        torch_out = x_torch[indices_torch]

        rtol, atol = get_tolerance("primitives", "selective_gather", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveGather forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective gather backward pass parity."""
        from mlx_primitives.primitives.gather_scatter import selective_gather

        config = SIZE_CONFIGS[size]["scan"]
        n_tokens = config["batch"] * config["seq"]
        dim = config["dim"]
        capacity = n_tokens // 2
        dtype = "fp32"

        np.random.seed(42)
        x_np = np.random.randn(n_tokens, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)

        # MLX backward (use differentiable=True for gradient support)
        def mlx_loss_fn(x):
            indices = mx.array(indices_np.astype(np.uint32))
            out = selective_gather(x, indices, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        indices_torch = torch.from_numpy(indices_np).long()
        torch_out = x_torch[indices_torch]
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "selective_gather", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), x_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveGather backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_torch_gather(self, skip_without_pytorch):
        """Test selective gather vs torch.gather."""
        from mlx_primitives.primitives.gather_scatter import selective_gather

        n_tokens, dim = 100, 64

        np.random.seed(42)
        x_np = np.random.randn(n_tokens, dim).astype(np.float32)
        # Select specific rows
        indices_np = np.array([0, 10, 20, 50, 99]).astype(np.int32)

        # MLX
        mlx_out = selective_gather(mx.array(x_np), mx.array(indices_np.astype(np.uint32)))

        # PyTorch indexing (equivalent to gather for row selection)
        x_torch = torch.from_numpy(x_np)
        torch_out = x_torch[torch.from_numpy(indices_np).long()]

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-6, atol=1e-7,
            err_msg="SelectiveGather vs torch indexing mismatch"
        )


# =============================================================================
# Selective Scatter Add Parity Tests
# =============================================================================

class TestSelectiveScatterAddParity:
    """Selective scatter-add operation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective scatter-add forward pass parity."""
        from mlx_primitives.primitives.gather_scatter import selective_scatter_add

        config = SIZE_CONFIGS[size]["scan"]
        n_tokens = config["batch"] * config["seq"]
        dim = config["dim"]
        capacity = n_tokens // 2

        np.random.seed(42)
        output_np = np.zeros((n_tokens, dim), dtype=np.float32)
        values_np = np.random.randn(capacity, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)
        weights_np = np.random.uniform(0.5, 1.0, capacity).astype(np.float32)

        # MLX
        output_mlx = _convert_to_mlx(output_np, dtype)
        values_mlx = _convert_to_mlx(values_np, dtype)
        indices_mlx = mx.array(indices_np.astype(np.uint32))
        weights_mlx = _convert_to_mlx(weights_np, dtype)
        mlx_out = selective_scatter_add(output_mlx, values_mlx, indices_mlx, weights_mlx)

        # PyTorch reference using scatter_add
        output_torch = _convert_to_torch(output_np, dtype)
        values_torch = _convert_to_torch(values_np, dtype)
        indices_torch = torch.from_numpy(indices_np).long()
        weights_torch = _convert_to_torch(weights_np, dtype)

        # weighted_values = values * weights[:, None]
        weighted_values = values_torch.float() * weights_torch.float().unsqueeze(1)

        # Scatter add: for each i, output[indices[i]] += weighted_values[i]
        torch_out = output_torch.float().clone()
        for i in range(capacity):
            torch_out[indices_torch[i]] += weighted_values[i]

        rtol, atol = get_tolerance("primitives", "selective_scatter_add", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveScatterAdd forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective scatter-add backward pass parity."""
        from mlx_primitives.primitives.gather_scatter import selective_scatter_add

        config = SIZE_CONFIGS[size]["scan"]
        n_tokens = config["batch"] * config["seq"]
        dim = config["dim"]
        capacity = n_tokens // 2
        dtype = "fp32"

        np.random.seed(42)
        output_np = np.zeros((n_tokens, dim), dtype=np.float32)
        values_np = np.random.randn(capacity, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)
        weights_np = np.random.uniform(0.5, 1.0, capacity).astype(np.float32)

        # MLX backward (gradient w.r.t. values, use differentiable=True for gradient support)
        def mlx_loss_fn(values):
            output = mx.array(output_np)
            indices = mx.array(indices_np.astype(np.uint32))
            weights = mx.array(weights_np)
            out = selective_scatter_add(output, values, indices, weights, differentiable=True)
            return mx.sum(out)

        values_mlx = mx.array(values_np)
        mlx_grad = mx.grad(mlx_loss_fn)(values_mlx)
        mx.eval(mlx_grad)

        # PyTorch backward
        values_torch = torch.from_numpy(values_np).requires_grad_(True)
        indices_torch = torch.from_numpy(indices_np).long()
        weights_torch = torch.from_numpy(weights_np)

        weighted_values = values_torch * weights_torch.unsqueeze(1)
        output_torch = torch.from_numpy(output_np).clone()
        for i in range(capacity):
            output_torch[indices_torch[i]] += weighted_values[i]
        output_torch.sum().backward()

        rtol, atol = get_gradient_tolerance("primitives", "selective_scatter_add", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), values_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"SelectiveScatterAdd backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_torch_scatter_add(self, skip_without_pytorch):
        """Test selective scatter-add vs torch.scatter_add."""
        from mlx_primitives.primitives.gather_scatter import selective_scatter_add

        n_tokens, dim = 100, 64
        capacity = 20

        np.random.seed(42)
        output_np = np.zeros((n_tokens, dim), dtype=np.float32)
        values_np = np.random.randn(capacity, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)
        # Use weights of 1.0 for direct comparison
        weights_np = np.ones(capacity, dtype=np.float32)

        # MLX
        mlx_out = selective_scatter_add(
            mx.array(output_np), mx.array(values_np),
            mx.array(indices_np.astype(np.uint32)), mx.array(weights_np)
        )

        # PyTorch scatter_add
        output_torch = torch.from_numpy(output_np.copy())
        values_torch = torch.from_numpy(values_np)
        # Expand indices to match values shape for scatter_add
        indices_expanded = torch.from_numpy(indices_np).long().unsqueeze(1).expand(-1, dim)
        torch_out = output_torch.scatter_add(0, indices_expanded, values_torch)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-5, atol=1e-6,
            err_msg="SelectiveScatterAdd vs torch.scatter_add mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_weights(self, skip_without_pytorch):
        """Test selective scatter-add with weighted values."""
        from mlx_primitives.primitives.gather_scatter import selective_scatter_add

        n_tokens, dim = 50, 32
        capacity = 10

        np.random.seed(42)
        output_np = np.zeros((n_tokens, dim), dtype=np.float32)
        values_np = np.random.randn(capacity, dim).astype(np.float32)
        indices_np = np.random.choice(n_tokens, size=capacity, replace=False).astype(np.int32)
        weights_np = np.random.uniform(0.1, 0.9, capacity).astype(np.float32)

        # MLX
        mlx_out = selective_scatter_add(
            mx.array(output_np), mx.array(values_np),
            mx.array(indices_np.astype(np.uint32)), mx.array(weights_np)
        )

        # Manual reference computation
        expected = output_np.copy()
        for i in range(capacity):
            expected[indices_np[i]] += values_np[i] * weights_np[i]

        np.testing.assert_allclose(
            _to_numpy(mlx_out), expected,
            rtol=1e-5, atol=1e-6,
            err_msg="SelectiveScatterAdd with weights mismatch"
        )
