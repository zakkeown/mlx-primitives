"""NumPy reference implementations for State Space Models (SSM)."""

import numpy as np


def selective_scan(
    x: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray = None,
) -> np.ndarray:
    """Selective Scan (S6) operation.

    The core SSM recurrence with input-dependent parameters:
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t * h_t + D * x_t

    where A_t = exp(delta_t * A)

    Args:
        x: Input tensor, shape (batch, seq_len, d_inner)
        delta: Time step tensor, shape (batch, seq_len, d_inner)
        A: State matrix, shape (d_inner, d_state)
        B: Input matrix, shape (batch, seq_len, d_state)
        C: Output matrix, shape (batch, seq_len, d_state)
        D: Skip connection, shape (d_inner,) or None

    Returns:
        Output tensor, shape (batch, seq_len, d_inner)
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A: delta_A = exp(delta * A)
    # delta: (batch, seq_len, d_inner), A: (d_inner, d_state)
    # delta_A: (batch, seq_len, d_inner, d_state)
    delta_A = np.exp(delta[..., None] * A[None, None, :, :])

    # Discretize B: delta_B = delta * B
    # delta: (batch, seq_len, d_inner), B: (batch, seq_len, d_state)
    # delta_B: (batch, seq_len, d_inner, d_state)
    delta_B = delta[..., None] * B[:, :, None, :]

    # Initialize hidden state
    h = np.zeros((batch_size, d_inner, d_state), dtype=x.dtype)

    # Sequential scan
    outputs = []
    for t in range(seq_len):
        # h_t = A_t * h_{t-1} + B_t * x_t
        # delta_A[:, t]: (batch, d_inner, d_state)
        # h: (batch, d_inner, d_state)
        # delta_B[:, t]: (batch, d_inner, d_state)
        # x[:, t]: (batch, d_inner)
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t, :, None]

        # y_t = C_t * h_t
        # C[:, t]: (batch, d_state)
        # h: (batch, d_inner, d_state)
        y = (h * C[:, t, None, :]).sum(axis=-1)  # (batch, d_inner)
        outputs.append(y)

    # Stack outputs: (batch, seq_len, d_inner)
    out = np.stack(outputs, axis=1)

    # Add skip connection
    if D is not None:
        out = out + x * D

    return out


def mamba_block(
    x: np.ndarray,
    in_proj_weight: np.ndarray,
    conv1d_weight: np.ndarray,
    conv1d_bias: np.ndarray,
    x_proj_weight: np.ndarray,
    dt_proj_weight: np.ndarray,
    dt_proj_bias: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    out_proj_weight: np.ndarray,
    d_inner: int,
    d_state: int,
    d_conv: int = 4,
    dt_rank: int = None,
) -> np.ndarray:
    """Mamba block forward pass.

    Combines SSM with gating, similar to a gated linear unit but with
    selective state space dynamics.

    Args:
        x: Input tensor, shape (batch, seq_len, d_model)
        in_proj_weight: Input projection, shape (2 * d_inner, d_model)
        conv1d_weight: Conv1d weight, shape (d_inner, 1, d_conv)
        conv1d_bias: Conv1d bias, shape (d_inner,)
        x_proj_weight: SSM input projection, shape (dt_rank + 2*d_state, d_inner)
        dt_proj_weight: Delta projection, shape (d_inner, dt_rank)
        dt_proj_bias: Delta bias, shape (d_inner,)
        A: SSM A matrix (log space), shape (d_inner, d_state)
        D: Skip connection, shape (d_inner,)
        out_proj_weight: Output projection, shape (d_model, d_inner)
        d_inner: Inner dimension
        d_state: State dimension
        d_conv: Convolution kernel size
        dt_rank: Rank for delta projection

    Returns:
        Output tensor, shape (batch, seq_len, d_model)
    """
    batch, seq_len, d_model = x.shape

    if dt_rank is None:
        dt_rank = d_model // 16

    # Input projection: x -> (x_z, z) where x_z goes to SSM, z is gate
    xz = np.matmul(x, in_proj_weight.T)  # (batch, seq_len, 2 * d_inner)
    x_ssm = xz[..., :d_inner]  # (batch, seq_len, d_inner)
    z = xz[..., d_inner:]  # (batch, seq_len, d_inner)

    # Conv1d (causal)
    # Transpose to (batch, d_inner, seq_len) for conv
    x_conv = np.transpose(x_ssm, (0, 2, 1))

    # Pad for causal conv
    x_padded = np.pad(x_conv, ((0, 0), (0, 0), (d_conv - 1, 0)), mode='constant')

    # Manual conv1d
    x_conv_out = np.zeros_like(x_conv)
    for i in range(seq_len):
        window = x_padded[:, :, i:i + d_conv]  # (batch, d_inner, d_conv)
        x_conv_out[:, :, i] = (window * conv1d_weight[:, 0, :]).sum(axis=-1) + conv1d_bias

    x_conv_out = np.transpose(x_conv_out, (0, 2, 1))  # (batch, seq_len, d_inner)

    # SiLU activation
    x_conv_out = x_conv_out * (1 / (1 + np.exp(-x_conv_out)))

    # SSM input projection: x -> (delta, B, C)
    x_dbc = np.matmul(x_conv_out, x_proj_weight.T)  # (batch, seq_len, dt_rank + 2*d_state)

    delta_proj = x_dbc[..., :dt_rank]
    B = x_dbc[..., dt_rank:dt_rank + d_state]
    C = x_dbc[..., dt_rank + d_state:]

    # Delta projection with softplus
    delta = np.matmul(delta_proj, dt_proj_weight.T) + dt_proj_bias
    delta = np.log1p(np.exp(delta))  # softplus

    # A is stored in log space, convert
    A_real = -np.exp(A)

    # Run selective scan
    y = selective_scan(x_conv_out, delta, A_real, B, C, D)

    # Gate and output projection
    y = y * (z * (1 / (1 + np.exp(-z))))  # y * SiLU(z)

    out = np.matmul(y, out_proj_weight.T)

    return out
