"""Fast RMSNorm Metal kernel implementation.

This module provides a Metal-accelerated RMSNorm implementation that fuses
the normalization computation into a single kernel pass.
"""

from pathlib import Path
from typing import Optional

import mlx.core as mx

# Cache compiled kernels
_rmsnorm_kernel = None
_rmsnorm_residual_kernel = None


def _get_rmsnorm_kernel():
    """Get or create the RMSNorm kernel."""
    global _rmsnorm_kernel

    if _rmsnorm_kernel is None:
        # Simple element-wise kernel - each thread handles one row
        # MLX auto-provides x_shape, x_strides, x_ndim when referenced
        source = """
            // Get row index from grid position
            uint row_idx = thread_position_in_grid.x;

            // Get dimensions from input shape (auto-provided by MLX)
            uint batch_size = x_shape[0];
            uint seq_len = x_shape[1];
            uint hidden_dim = x_shape[2];
            uint total_rows = batch_size * seq_len;

            if (row_idx >= total_rows) return;

            uint row_offset = row_idx * hidden_dim;

            // Compute sum of squares
            float sum_sq = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                float val = x[row_offset + i];
                sum_sq += val * val;
            }

            // Compute RMS inverse
            float rms_inv = metal::rsqrt(sum_sq / float(hidden_dim) + 1e-6f);

            // Normalize and scale
            for (uint i = 0; i < hidden_dim; i++) {
                float val = x[row_offset + i];
                out[row_offset + i] = val * rms_inv * weight[i];
            }
        """

        _rmsnorm_kernel = mx.fast.metal_kernel(
            name="rmsnorm_forward",
            input_names=["x", "weight"],
            output_names=["out"],
            source=source,
        )

    return _rmsnorm_kernel


def _get_rmsnorm_residual_kernel():
    """Get or create the fused RMSNorm + residual kernel."""
    global _rmsnorm_residual_kernel

    if _rmsnorm_residual_kernel is None:
        source = """
            uint row_idx = thread_position_in_grid.x;

            uint batch_size = x_shape[0];
            uint seq_len = x_shape[1];
            uint hidden_dim = x_shape[2];
            uint total_rows = batch_size * seq_len;

            if (row_idx >= total_rows) return;

            uint row_offset = row_idx * hidden_dim;

            // Compute sum of squares of (x + residual)
            float sum_sq = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                float val = x[row_offset + i] + residual[row_offset + i];
                sum_sq += val * val;
            }

            float rms_inv = metal::rsqrt(sum_sq / float(hidden_dim) + 1e-6f);

            // Normalize and scale
            for (uint i = 0; i < hidden_dim; i++) {
                float val = x[row_offset + i] + residual[row_offset + i];
                out[row_offset + i] = val * rms_inv * weight[i];
            }
        """

        _rmsnorm_residual_kernel = mx.fast.metal_kernel(
            name="rmsnorm_residual",
            input_names=["x", "residual", "weight"],
            output_names=["out"],
            source=source,
        )

    return _rmsnorm_residual_kernel


def _reference_rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Reference implementation using MLX ops (used when Metal overhead isn't worth it)."""
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def fast_rmsnorm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Fast Metal-accelerated RMSNorm.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of the same shape as input.
    """
    # Handle 2D input
    was_2d = x.ndim == 2
    if was_2d:
        x = x[None, :, :]

    batch_size, seq_len, hidden_dim = x.shape
    total_rows = batch_size * seq_len

    # Metal kernel overhead not amortized at small batch sizes or short sequences
    # (see RCA report - benchmark showed 0.27x-0.78x slowdown for these cases)
    # For small workloads, MLX's fused ops are faster than custom Metal kernels
    if batch_size <= 2 or (batch_size <= 4 and seq_len <= 512):
        result = _reference_rmsnorm(x, weight, eps)
        return result[0] if was_2d else result

    kernel = _get_rmsnorm_kernel()

    # One thread per row
    threadgroup_size = min(256, total_rows)
    num_groups = (total_rows + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[x, weight],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )

    result = outputs[0]

    if was_2d:
        result = result[0]

    return result


def fast_rmsnorm_residual(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Fused RMSNorm with residual connection.

    Computes: RMSNorm(x + residual)

    This is more efficient than separate add and normalize operations.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim).
        residual: Residual tensor of the same shape.
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor: RMSNorm(x + residual).
    """
    was_2d = x.ndim == 2
    if was_2d:
        x = x[None, :, :]
        residual = residual[None, :, :]

    batch_size, seq_len, hidden_dim = x.shape
    total_rows = batch_size * seq_len

    kernel = _get_rmsnorm_residual_kernel()

    threadgroup_size = min(256, total_rows)
    num_groups = (total_rows + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[x, residual, weight],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )

    result = outputs[0]

    if was_2d:
        result = result[0]

    return result


# Try to enable fast path when available
_USE_METAL_KERNELS = True

try:
    # Test if metal kernels work
    _test = mx.zeros((1, 1, 64))
    _weight = mx.ones((64,))
    # This will fail if metal_kernel isn't available
    if hasattr(mx.fast, 'metal_kernel'):
        pass
    else:
        _USE_METAL_KERNELS = False
except Exception:
    _USE_METAL_KERNELS = False


def rmsnorm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
    use_metal: bool = True,
) -> mx.array:
    """RMSNorm with automatic Metal acceleration.

    Uses Metal kernels when available, falls back to pure MLX otherwise.

    Args:
        x: Input tensor.
        weight: Scale parameter.
        eps: Numerical stability constant.
        use_metal: Whether to use Metal kernels (default True).

    Returns:
        Normalized tensor.
    """
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_rmsnorm(x, weight, eps)
        except Exception:
            pass

    # Fallback to pure MLX
    rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight
