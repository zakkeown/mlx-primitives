"""Associative scan primitive for MLX.

The associative scan (parallel prefix sum) is a fundamental parallel primitive
that enables O(log n) parallel computation of recurrences like:
    y[t] = f(y[t-1], x[t])

where f is an associative binary operator.

This is the key missing primitive for efficient SSM (State Space Model)
implementations like Mamba on MLX.

Performance Notes:
    - For seq_len > MIN_SEQ_FOR_METAL (default 8): Uses parallel Metal kernel
      with O(log n) complexity via SIMD warp-level intrinsics.
    - For seq_len <= MIN_SEQ_FOR_METAL: Uses MLX builtins (cumsum/cumprod) or
      sequential fallback. Still GPU-accelerated but O(n).
    - For seq_len > 1024: Falls back to sequential (multi-block not yet implemented).

    Configure threshold via MLX_PRIMITIVES_MIN_SEQ_FOR_METAL environment variable.
"""

import os
from typing import Literal, Optional

import mlx.core as mx

from mlx_primitives.constants import MIN_SEQ_FOR_METAL

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")


def _get_min_seq_for_metal() -> int:
    """Get minimum sequence length for Metal dispatch, configurable via env var."""
    env_val = os.environ.get("MLX_PRIMITIVES_MIN_SEQ_FOR_METAL")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return MIN_SEQ_FOR_METAL


def _reverse_along_axis(x: mx.array, axis: int) -> mx.array:
    """Reverse array along specified axis using slicing."""
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # Build slice tuple with ::-1 for the target axis
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, -1)
    return x[tuple(slices)]


def associative_scan(
    x: mx.array,
    operator: Literal["add", "mul", "ssm"] = "add",
    A: Optional[mx.array] = None,
    axis: int = -1,
    reverse: bool = False,
    inclusive: bool = True,
    use_metal: bool = True,
) -> mx.array:
    """Parallel associative scan.

    Computes a parallel prefix operation along the specified axis.
    For operator="add", this is equivalent to cumsum.
    For operator="mul", this is equivalent to cumprod.
    For operator="ssm", this computes h[t] = A[t] * h[t-1] + x[t].

    Args:
        x: Input tensor.
        operator: Scan operator:
            - "add": Cumulative sum (like mx.cumsum)
            - "mul": Cumulative product (like mx.cumprod)
            - "ssm": State space model recurrence h[t] = A[t] * h[t-1] + x[t]
        A: State transition for SSM mode. Required when operator="ssm".
           For diagonal A (Mamba), shape should match x: (batch, seq, d_inner).
        axis: Axis along which to scan (default: -1, last axis).
        reverse: If True, scan from end to beginning.
        inclusive: If True, include current element in result (default).
                  If False, compute exclusive scan (shifted by 1).
        use_metal: Use Metal kernel if available (default True).

    Returns:
        Scanned output tensor of same shape as x.

    Examples:
        >>> # Cumulative sum
        >>> x = mx.array([1, 2, 3, 4, 5])
        >>> associative_scan(x, operator="add")
        array([1, 3, 6, 10, 15])

        >>> # Cumulative product
        >>> x = mx.array([1, 2, 3, 4, 5])
        >>> associative_scan(x, operator="mul")
        array([1, 2, 6, 24, 120])

        >>> # SSM recurrence: h[t] = A[t] * h[t-1] + x[t]
        >>> A = mx.array([[0.9, 0.9, 0.9]])  # (1, 3, 1) decay factors
        >>> x = mx.array([[1.0, 1.0, 1.0]])  # (1, 3, 1) inputs
        >>> associative_scan(x, operator="ssm", A=A)
        array([[1.0, 1.9, 2.71]])  # 1, 0.9*1+1, 0.9*1.9+1
    """
    # Handle reverse scan
    if reverse:
        x = _reverse_along_axis(x, axis=axis)
        if A is not None:
            A = _reverse_along_axis(A, axis=axis)

    # Dispatch based on operator
    if operator == "ssm":
        if A is None:
            raise ValueError("A is required for SSM scan (operator='ssm')")
        result = _ssm_scan(x, A, axis, use_metal)
    elif operator in ("add", "mul"):
        result = _simple_scan(x, operator, axis, inclusive, use_metal)
    else:
        raise ValueError(f"Unknown operator: {operator}. Use 'add', 'mul', or 'ssm'.")

    # Reverse back if needed
    if reverse:
        result = _reverse_along_axis(result, axis=axis)

    return result


def _simple_scan(
    x: mx.array,
    operator: Literal["add", "mul"],
    axis: int,
    inclusive: bool,
    use_metal: bool,
) -> mx.array:
    """Simple additive or multiplicative scan."""
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # For short sequences or when Metal not available, use MLX builtins
    seq_len = x.shape[axis]
    threshold = _get_min_seq_for_metal()
    if not use_metal or not _HAS_METAL or seq_len <= threshold:
        if operator == "add":
            return mx.cumsum(x, axis=axis)
        else:
            return mx.cumprod(x, axis=axis)

    # Try Metal kernel
    try:
        from mlx_primitives.primitives._metal.scan_kernels import (
            metal_associative_scan,
        )

        return metal_associative_scan(x, operator, axis, inclusive)
    except Exception as e:
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("associative_scan", e)
        # Fall back to MLX builtins
        if operator == "add":
            return mx.cumsum(x, axis=axis)
        else:
            return mx.cumprod(x, axis=axis)


def _ssm_scan(
    x: mx.array,
    A: mx.array,
    axis: int,
    use_metal: bool,
) -> mx.array:
    """SSM scan: h[t] = A[t] * h[t-1] + x[t].

    This is the key operation for Mamba and other SSMs.

    Note:
        Currently requires 3D input (batch, seq, d_inner). For 4D+ tensors,
        reshape before calling:

        >>> # (batch, seq, heads, state) -> (batch*heads, seq, state)
        >>> x_3d = x.reshape(batch * heads, seq, state)
        >>> A_3d = A.reshape(batch * heads, seq, state)
        >>> result_3d = associative_scan(x_3d, operator="ssm", A=A_3d, axis=1)
        >>> result = result_3d.reshape(batch, heads, seq, state)

    Performance:
        - seq_len > threshold: Parallel Metal kernel O(log n)
        - seq_len <= threshold: Sequential fallback O(n)
        - seq_len > 1024: Sequential fallback (multi-block not implemented)
    """
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    if x.shape != A.shape:
        raise ValueError(f"x and A must have same shape, got {x.shape} and {A.shape}")

    # Move scan axis to position 1 (batch, seq, features)
    if axis != 1:
        # Transpose to (batch_dims..., seq, feature_dims...)
        # For now, assume 3D input (batch, seq, d_inner)
        if x.ndim != 3:
            raise ValueError(
                "SSM scan currently requires 3D input (batch, seq, d_inner)"
            )
        if axis == 0:
            x = mx.transpose(x, (1, 0, 2))
            A = mx.transpose(A, (1, 0, 2))
        elif axis == 2:
            x = mx.transpose(x, (0, 2, 1))
            A = mx.transpose(A, (0, 2, 1))

    batch_size, seq_len, d_inner = x.shape

    # For short sequences, use sequential implementation
    threshold = _get_min_seq_for_metal()
    if not use_metal or not _HAS_METAL or seq_len <= threshold:
        return _sequential_ssm_scan(A, x)

    # Try Metal kernel
    try:
        from mlx_primitives.primitives._metal.scan_kernels import metal_ssm_scan

        result = metal_ssm_scan(A, x)

        # Transpose back if needed
        if axis == 0:
            result = mx.transpose(result, (1, 0, 2))
        elif axis == 2:
            result = mx.transpose(result, (0, 2, 1))

        return result
    except Exception as e:
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("ssm_scan", e)
        # Fall back to sequential
        return _sequential_ssm_scan(A, x)


def _sequential_ssm_scan(A: mx.array, x: mx.array) -> mx.array:
    """Sequential SSM scan fallback.

    Computes h[t] = A[t] * h[t-1] + x[t] sequentially.
    This is O(n) but serves as reference and fallback.
    """
    batch_size, seq_len, d_inner = x.shape

    # Initialize hidden state
    h_prev = mx.zeros((batch_size, d_inner), dtype=x.dtype)

    # Sequential recurrence
    outputs = []
    for t in range(seq_len):
        h_t = A[:, t, :] * h_prev + x[:, t, :]
        outputs.append(h_t)
        h_prev = h_t

    return mx.stack(outputs, axis=1)


def selective_scan(
    x: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: Optional[mx.array] = None,
    use_metal: bool = True,
) -> mx.array:
    """Selective scan for Mamba-style SSMs.

    This is the full Mamba selective scan operation that combines:
    1. Discretization: A_bar = exp(delta * A), B_bar = delta * B
    2. SSM recurrence: h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
    3. Output projection: y[t] = C[t] @ h[t] + D * x[t]

    Args:
        x: Input tensor (batch, seq_len, d_inner).
        delta: Time step tensor (batch, seq_len, d_inner).
        A: State matrix diagonal (d_inner, d_state). Typically negative.
        B: Input projection (batch, seq_len, d_state).
        C: Output projection (batch, seq_len, d_state).
        D: Skip connection (d_inner,). Optional.
        use_metal: Use Metal kernels if available.

    Returns:
        Output tensor y of shape (batch, seq_len, d_inner).
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretization
    # A_bar = exp(delta * A) for each (batch, seq, d_inner, d_state)
    # delta: (batch, seq, d_inner) -> (batch, seq, d_inner, 1)
    # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
    delta_A = delta[..., None] * A[None, None, :, :]  # (batch, seq, d_inner, d_state)
    A_bar = mx.exp(delta_A)  # (batch, seq, d_inner, d_state)

    # B_bar = delta * B
    # B: (batch, seq, d_state) -> (batch, seq, 1, d_state)
    # x: (batch, seq, d_inner) -> (batch, seq, d_inner, 1)
    B_x = B[:, :, None, :] * x[..., None]  # (batch, seq, d_inner, d_state)
    B_bar_x = delta[..., None] * B_x  # (batch, seq, d_inner, d_state)

    # For parallel scan, we need to process each (d_inner, d_state) dimension
    # Reshape to (batch * d_inner, seq, d_state) for the scan
    A_bar_flat = A_bar.transpose(0, 2, 1, 3).reshape(
        batch_size * d_inner, seq_len, d_state
    )
    B_bar_x_flat = B_bar_x.transpose(0, 2, 1, 3).reshape(
        batch_size * d_inner, seq_len, d_state
    )

    # SSM scan: h[t] = A_bar[t] * h[t-1] + B_bar_x[t]
    h_flat = associative_scan(
        B_bar_x_flat, operator="ssm", A=A_bar_flat, axis=1, use_metal=use_metal
    )

    # Reshape back: (batch, d_inner, seq, d_state) -> (batch, seq, d_inner, d_state)
    h = h_flat.reshape(batch_size, d_inner, seq_len, d_state).transpose(0, 2, 1, 3)

    # Output projection: y = sum(C * h, axis=-1)
    # C: (batch, seq, d_state) -> (batch, seq, 1, d_state)
    y = mx.sum(C[:, :, None, :] * h, axis=-1)  # (batch, seq, d_inner)

    # Skip connection
    if D is not None:
        y = y + x * D[None, None, :]

    return y
