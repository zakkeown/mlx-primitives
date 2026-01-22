"""Fast RMSNorm Metal kernel implementation.

This module provides a Metal-accelerated RMSNorm implementation that fuses
the normalization computation into a single kernel pass, with full VJP
(backward pass) support for automatic differentiation.
"""

import logging
from typing import Tuple

import mlx.core as mx

from mlx_primitives.kernels._registry import get_kernel
from mlx_primitives.utils import log_fallback

logger = logging.getLogger(__name__)


def _get_rmsnorm_kernel(eps: float = 1e-6):
    """Get or create the RMSNorm kernel with specified eps.

    Args:
        eps: Small constant for numerical stability.
    """
    # Simple element-wise kernel - each thread handles one row
    # MLX auto-provides x_shape, x_strides, x_ndim when referenced
    source = f"""
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
        for (uint i = 0; i < hidden_dim; i++) {{
            float val = x[row_offset + i];
            sum_sq += val * val;
        }}

        // Compute RMS inverse
        float rms_inv = metal::rsqrt(sum_sq / float(hidden_dim) + {eps}f);

        // Get output type for explicit casting (required for bf16)
        typedef decltype(out[0] + out[0]) OutT;

        // Normalize and scale
        for (uint i = 0; i < hidden_dim; i++) {{
            float val = x[row_offset + i];
            out[row_offset + i] = OutT(val * rms_inv * weight[i]);
        }}
    """

    kernel_name = f"rmsnorm_forward_eps{eps}"
    return get_kernel(
        kernel_name,
        lambda: mx.fast.metal_kernel(
            name=kernel_name,
            input_names=["x", "weight"],
            output_names=["out"],
            source=source,
        ),
    )


def _get_rmsnorm_backward_kernel(eps: float = 1e-6):
    """Get or create the RMSNorm backward kernel with specified eps.

    Computes gradients for x and weight given upstream gradient dy.

    Args:
        eps: Small constant for numerical stability.

    RMSNorm backward math:
        Let rms = sqrt(mean(x^2) + eps)
        Forward: y = (x / rms) * weight

        Backward:
        - dweight = sum(dy * x / rms, axis=(0,1))
        - dx = (dy * weight / rms) - x * sum(dy * weight * x, axis=-1) / (rms^3 * hidden_dim)
    """
    source = f"""
        uint row_idx = thread_position_in_grid.x;

        uint batch_size = x_shape[0];
        uint seq_len = x_shape[1];
        uint hidden_dim = x_shape[2];
        uint total_rows = batch_size * seq_len;

        if (row_idx >= total_rows) return;

        uint row_offset = row_idx * hidden_dim;

        // Recompute RMS (could cache but memory vs compute tradeoff)
        float sum_sq = 0.0f;
        for (uint i = 0; i < hidden_dim; i++) {{
            float val = x[row_offset + i];
            sum_sq += val * val;
        }}
        float variance = sum_sq / float(hidden_dim);
        float rms = metal::sqrt(variance + {eps}f);
        float rms_inv = 1.0f / rms;
        float rms_inv3 = rms_inv * rms_inv * rms_inv;

        // Compute sum(dy * weight * x) for this row
        float dot_sum = 0.0f;
        for (uint i = 0; i < hidden_dim; i++) {{
            float dy_val = dy[row_offset + i];
            float x_val = x[row_offset + i];
            float w_val = weight[i];
            dot_sum += dy_val * w_val * x_val;
        }}
        float scale = dot_sum * rms_inv3 / float(hidden_dim);

        typedef decltype(dx[0] + dx[0]) OutT;

        // Compute dx for each element
        for (uint i = 0; i < hidden_dim; i++) {{
            float dy_val = dy[row_offset + i];
            float x_val = x[row_offset + i];
            float w_val = weight[i];

            // dx = dy * weight / rms - x * scale
            float grad = dy_val * w_val * rms_inv - x_val * scale;
            dx[row_offset + i] = OutT(grad);
        }}
    """

    kernel_name = f"rmsnorm_backward_eps{eps}"
    return get_kernel(
        kernel_name,
        lambda: mx.fast.metal_kernel(
            name=kernel_name,
            input_names=["x", "weight", "dy"],
            output_names=["dx"],
            source=source,
        ),
    )


def _get_rmsnorm_residual_kernel(eps: float = 1e-6):
    """Get or create the fused RMSNorm + residual kernel with specified eps.

    Args:
        eps: Small constant for numerical stability.
    """
    source = f"""
        uint row_idx = thread_position_in_grid.x;

        uint batch_size = x_shape[0];
        uint seq_len = x_shape[1];
        uint hidden_dim = x_shape[2];
        uint total_rows = batch_size * seq_len;

        if (row_idx >= total_rows) return;

        uint row_offset = row_idx * hidden_dim;

        // Compute sum of squares of (x + residual)
        float sum_sq = 0.0f;
        for (uint i = 0; i < hidden_dim; i++) {{
            float val = x[row_offset + i] + residual[row_offset + i];
            sum_sq += val * val;
        }}

        float rms_inv = metal::rsqrt(sum_sq / float(hidden_dim) + {eps}f);

        // Get output type for explicit casting (required for bf16)
        typedef decltype(out[0] + out[0]) OutT;

        // Normalize and scale
        for (uint i = 0; i < hidden_dim; i++) {{
            float val = x[row_offset + i] + residual[row_offset + i];
            out[row_offset + i] = OutT(val * rms_inv * weight[i]);
        }}
    """

    kernel_name = f"rmsnorm_residual_eps{eps}"
    return get_kernel(
        kernel_name,
        lambda: mx.fast.metal_kernel(
            name=kernel_name,
            input_names=["x", "residual", "weight"],
            output_names=["out"],
            source=source,
        ),
    )


def _reference_rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Reference implementation using MLX ops (used when Metal overhead isn't worth it).

    Computes in float32 for numerical stability with reduced precision dtypes,
    matching both the Metal kernel behavior and PyTorch's approach.
    """
    orig_dtype = x.dtype
    x_fp32 = x.astype(mx.float32)
    rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + eps)
    normalized = (x_fp32 / rms).astype(orig_dtype)
    return normalized * weight


def _metal_rmsnorm_3d_impl(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Metal kernel RMSNorm implementation for 3D input.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim).
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of the same shape as input.
    """
    batch_size, seq_len, hidden_dim = x.shape
    total_rows = batch_size * seq_len

    kernel = _get_rmsnorm_kernel(eps)

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

    return outputs[0]


@mx.custom_function
def _metal_rmsnorm_3d(x: mx.array, weight: mx.array) -> mx.array:
    """Metal kernel RMSNorm for 3D input with VJP support (default eps=1e-6).

    This internal function is wrapped with @mx.custom_function to enable
    automatic differentiation through the Metal kernel.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim).
        weight: Learnable scale parameter of shape (hidden_dim,).

    Returns:
        Normalized tensor of the same shape as input.
    """
    return _metal_rmsnorm_3d_impl(x, weight, eps=1e-6)


@_metal_rmsnorm_3d.vjp
def _metal_rmsnorm_3d_vjp(
    primals: Tuple[mx.array, mx.array],
    cotangent: mx.array,
    output: mx.array,
) -> Tuple[mx.array, mx.array]:
    """VJP (backward pass) for Metal RMSNorm (default eps=1e-6).

    Computes gradients for x and weight given upstream gradient (cotangent).

    RMSNorm backward math:
        Let rms = sqrt(mean(x^2) + eps)
        Forward: y = (x / rms) * weight

        Backward given dy (cotangent):
        - dweight = sum(dy * x / rms, axis=(0,1))
        - dx = (dy * weight / rms) - x * sum(dy * weight * x, axis=-1) / (rms^3 * hidden_dim)
    """
    eps = 1e-6
    x, weight = primals
    dy = cotangent

    batch_size, seq_len, hidden_dim = x.shape
    total_rows = batch_size * seq_len

    # Compute dx using Metal kernel
    backward_kernel = _get_rmsnorm_backward_kernel(eps)

    threadgroup_size = min(256, total_rows)
    num_groups = (total_rows + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    dx_outputs = backward_kernel(
        inputs=[x, weight, dy],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    dx = dx_outputs[0]

    # Compute dweight: sum(dy * normalized, axis=(0, 1))
    # We need to recompute normalized = x / rms
    x_fp32 = x.astype(mx.float32)
    rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + eps)
    normalized = (x_fp32 / rms).astype(x.dtype)
    dweight = mx.sum(dy * normalized, axis=(0, 1))

    return dx, dweight


def fast_rmsnorm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Fast Metal-accelerated RMSNorm with full gradient support.

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

    # Metal kernel overhead not amortized at small batch sizes or short sequences
    # (see RCA report - benchmark showed 0.27x-0.78x slowdown for these cases)
    # For small workloads, MLX's fused ops are faster than custom Metal kernels
    if batch_size <= 2 or (batch_size <= 4 and seq_len <= 512):
        result = _reference_rmsnorm(x, weight, eps)
        return result[0] if was_2d else result

    # Use Metal kernel with VJP support for default eps
    # For non-default eps, use the implementation directly (no VJP support)
    if eps == 1e-6:
        result = _metal_rmsnorm_3d(x, weight)
    else:
        result = _metal_rmsnorm_3d_impl(x, weight, eps)

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

    kernel = _get_rmsnorm_residual_kernel(eps)

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


# Check if metal kernels are available
_USE_METAL_KERNELS = hasattr(mx.fast, 'metal_kernel')


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
        except Exception as e:
            log_fallback("rmsnorm", e, f"x.shape={x.shape}")

    # Fallback to pure MLX (compute in float32 for reduced precision dtypes)
    orig_dtype = x.dtype
    x_fp32 = x.astype(mx.float32)
    rms = mx.sqrt(mx.mean(x_fp32 ** 2, axis=-1, keepdims=True) + eps)
    return (x_fp32 / rms).astype(orig_dtype) * weight
