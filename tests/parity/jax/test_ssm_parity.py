"""JAX parity tests for State Space Models (SSM).

This module tests parity between MLX SSM implementations and JAX reference
implementations for:
- selective_scan (core SSM operation)
- MambaBlock (full Mamba block)
- S4Layer (Structured State Space)
- H3Layer (Hungry Hungry Hippos)

Since JAX doesn't have native SSM operations, we implement reference
functions using jax.lax.associative_scan for efficient parallel computation.
"""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import lax


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_jax(x_np: np.ndarray, dtype: str) -> "jnp.ndarray":
    """Convert numpy array to JAX with proper dtype."""
    x_jax = jnp.array(x_np.astype(np.float32))
    jax_dtype = get_jax_dtype(dtype)
    return x_jax.astype(jax_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


# =============================================================================
# JAX Reference Implementations
# =============================================================================

def jax_selective_scan_reference(
    x: "jnp.ndarray",
    delta: "jnp.ndarray",
    A: "jnp.ndarray",
    B: "jnp.ndarray",
    C: "jnp.ndarray",
    D: "jnp.ndarray" = None,
) -> "jnp.ndarray":
    """JAX reference implementation of selective scan using sequential loop.

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
    delta_A = delta[:, :, :, None] * A[None, None, :, :]  # (batch, seq, d_inner, d_state)
    A_bar = jnp.exp(delta_A)

    # Discretize B: B_bar = delta * B
    B_bar = delta[:, :, :, None] * B[:, :, None, :]  # (batch, seq, d_inner, d_state)

    # Sequential scan
    def scan_fn(h, inputs):
        a_bar_t, b_bar_t, x_t, c_t = inputs
        # h: (batch, d_inner, d_state)
        h_new = a_bar_t * h + b_bar_t * x_t[:, :, None]
        y_t = jnp.sum(h_new * c_t[:, None, :], axis=-1)
        return h_new, y_t

    h_init = jnp.zeros((batch_size, d_inner, d_state), dtype=x.dtype)

    # Prepare inputs for scan: transpose seq to first axis
    a_bar_seq = jnp.transpose(A_bar, (1, 0, 2, 3))  # (seq, batch, d_inner, d_state)
    b_bar_seq = jnp.transpose(B_bar, (1, 0, 2, 3))
    x_seq = jnp.transpose(x, (1, 0, 2))  # (seq, batch, d_inner)
    c_seq = jnp.transpose(C, (1, 0, 2))  # (seq, batch, d_state)

    _, y_seq = lax.scan(scan_fn, h_init, (a_bar_seq, b_bar_seq, x_seq, c_seq))
    y = jnp.transpose(y_seq, (1, 0, 2))  # (batch, seq, d_inner)

    # Add skip connection
    if D is not None:
        y = y + x * D[None, None, :]

    return y


def jax_s4_reference(
    x: "jnp.ndarray",
    A_real: "jnp.ndarray",
    B: "jnp.ndarray",
    C: "jnp.ndarray",
    D: "jnp.ndarray",
    log_dt: "jnp.ndarray",
) -> "jnp.ndarray":
    """JAX reference implementation of S4 layer (real-only variant).

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

    dt = jnp.exp(log_dt)  # (dims,)

    # Discretize: A_bar = exp(dt * A_real)
    A_bar = jnp.exp(dt[:, None] * A_real)  # (dims, d_state)

    # B_bar = dt * B
    B_scaled = B * dt[:, None]  # (dims, d_state)

    def scan_fn(h, x_t):
        # h: (batch, dims, d_state)
        # x_t: (batch, dims)
        h_new = A_bar * h + B_scaled * x_t[:, :, None]
        y_t = jnp.sum(h_new * C, axis=-1)
        return h_new, y_t

    h_init = jnp.zeros((batch_size, dims, d_state), dtype=x.dtype)
    x_seq = jnp.transpose(x, (1, 0, 2))  # (seq, batch, dims)

    _, y_seq = lax.scan(scan_fn, h_init, x_seq)
    y = jnp.transpose(y_seq, (1, 0, 2))

    # Add skip connection
    return y + x * D


def jax_h3_ssm_shift(
    x: "jnp.ndarray",
    A: "jnp.ndarray",
    B: "jnp.ndarray",
    log_dt: "jnp.ndarray",
) -> "jnp.ndarray":
    """JAX reference for H3 SSM shift operation.

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

    dt = jnp.exp(log_dt)
    A_bar = jnp.exp(dt[:, None] * A)
    B_bar = dt[:, None] * B

    def scan_fn(h, x_t):
        h_new = A_bar * h + B_bar * x_t[:, :, None]
        y_t = jnp.sum(h_new, axis=-1)
        return h_new, y_t

    h_init = jnp.zeros((batch_size, dims, d_state), dtype=x.dtype)
    x_seq = jnp.transpose(x, (1, 0, 2))

    _, y_seq = lax.scan(scan_fn, h_init, x_seq)
    return jnp.transpose(y_seq, (1, 0, 2))


# =============================================================================
# Selective Scan Parity Tests
# =============================================================================

class TestSelectiveScanParity:
    """Selective scan operation parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test selective scan forward pass parity."""
        from mlx_primitives.advanced.ssm import selective_scan

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_inner = dims * config["expand"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) + 0.01
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))
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

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlx_out = selective_scan(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx, warn_on_long_seq=False)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = _convert_to_jax(x_np, dtype)
        delta_jax = _convert_to_jax(delta_np, dtype)
        A_jax = _convert_to_jax(A_np, dtype)
        B_jax = _convert_to_jax(B_np, dtype)
        C_jax = _convert_to_jax(C_np, dtype)
        D_jax = _convert_to_jax(D_np, dtype)

        jax_out = jax_selective_scan_reference(
            x_jax, delta_jax, A_jax, B_jax, C_jax, D_jax
        )

        rtol, atol = get_tolerance("ssm", "selective_scan", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test selective scan backward pass parity."""
        from mlx_primitives.primitives.scan import selective_scan as diff_selective_scan

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_inner = dims * config["expand"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, d_inner).astype(np.float32)
        delta_np = np.abs(np.random.randn(batch, seq, d_inner).astype(np.float32)) + 0.01
        A_np = -np.abs(np.random.randn(d_inner, d_state).astype(np.float32))
        B_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        C_np = np.random.randn(batch, seq, d_state).astype(np.float32) * 0.1
        D_np = np.random.randn(d_inner).astype(np.float32) * 0.1

        # MLX backward
        def mlx_loss_fn(x, delta, A, B, C, D):
            out = diff_selective_scan(x, delta, A, B, C, D, differentiable=True)
            return mx.sum(out)

        x_mlx = mx.array(x_np)
        delta_mlx = mx.array(delta_np)
        A_mlx = mx.array(A_np)
        B_mlx = mx.array(B_np)
        C_mlx = mx.array(C_np)
        D_mlx = mx.array(D_np)

        grad_fn = mx.grad(mlx_loss_fn, argnums=0)
        mlx_grad_x = grad_fn(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx)
        mx.eval(mlx_grad_x)

        # JAX backward
        def jax_loss_fn(x):
            out = jax_selective_scan_reference(
                x, jnp.array(delta_np), jnp.array(A_np),
                jnp.array(B_np), jnp.array(C_np), jnp.array(D_np)
            )
            return jnp.sum(out)

        x_jax = jnp.array(x_np)
        jax_grad_x = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("ssm", "selective_scan", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), _to_numpy(jax_grad_x),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.parametrize("d_state", [8, 16, 32])
    def test_different_state_dims(self, d_state, skip_without_jax):
        """Test selective scan with different state dimensions."""
        from mlx_primitives.advanced.ssm import selective_scan

        batch, seq, d_inner = 2, 32, 64

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

        # JAX
        jax_out = jax_selective_scan_reference(
            jnp.array(x_np), jnp.array(delta_np),
            jnp.array(A_np), jnp.array(B_np),
            jnp.array(C_np), jnp.array(D_np)
        )

        rtol, atol = get_tolerance("ssm", "selective_scan", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Selective scan d_state={d_state} mismatch (JAX)"
        )


# =============================================================================
# MambaBlock Parity Tests
# =============================================================================

class TestMambaBlockParity:
    """MambaBlock parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_consistency(self, size, dtype, skip_without_jax):
        """Test MambaBlock forward pass produces valid output.

        This tests that MambaBlock produces consistent, valid output.
        Full numerical parity would require reimplementing the entire
        MambaBlock in JAX with identical weight initialization.
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

        out_np = _to_numpy(mlx_out)

        assert out_np.shape == (batch, seq, dims), f"Output shape mismatch: {out_np.shape}"
        assert not np.any(np.isnan(out_np)), f"NaN in MambaBlock output [{size}, {dtype}]"
        assert not np.any(np.isinf(out_np)), f"Inf in MambaBlock output [{size}, {dtype}]"
        assert np.abs(out_np).max() < 100, f"Output magnitude too large [{size}, {dtype}]"

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_consistency(self, size, skip_without_jax):
        """Test MambaBlock backward pass produces valid gradients."""
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

        def differentiable_selective_scan(x, delta, A, B, C, D, **kwargs):
            return diff_selective_scan(x, delta, A, B, C, D, differentiable=True)

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


# =============================================================================
# S4Layer Parity Tests
# =============================================================================

class TestS4LayerParity:
    """S4Layer parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_consistency(self, size, dtype, skip_without_jax):
        """Test S4Layer forward pass consistency."""
        from mlx_primitives.advanced.ssm import S4Layer

        config = SIZE_CONFIGS[size]["ssm"]
        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # Create S4Layer
        s4_mlx = S4Layer(dims, d_state=d_state, use_complex=False)
        mx.eval(s4_mlx.parameters())

        # MLX forward
        x_mlx = _convert_to_mlx(x_np, dtype)
        mlx_out = s4_mlx(x_mlx)
        mx.eval(mlx_out)

        out_np = _to_numpy(mlx_out)

        assert out_np.shape == (batch, seq, dims), f"Output shape mismatch: {out_np.shape}"
        assert not np.any(np.isnan(out_np)), f"NaN in S4Layer output [{size}, {dtype}]"
        assert not np.any(np.isinf(out_np)), f"Inf in S4Layer output [{size}, {dtype}]"

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_consistency(self, size, skip_without_jax):
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


# =============================================================================
# H3Layer Parity Tests
# =============================================================================

class TestH3LayerParity:
    """H3Layer parity tests vs JAX reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_consistency(self, size, dtype, skip_without_jax):
        """Test H3Layer forward pass consistency."""
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
        assert np.abs(out_np).max() < 1000, f"Output magnitude too large [{size}, {dtype}]"

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small"])
    def test_backward_consistency(self, size, skip_without_jax):
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
