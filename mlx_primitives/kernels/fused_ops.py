"""Fused operations Metal kernels.

Common operation patterns that benefit from fusion:
- bias + activation (saves memory roundtrip)
- dropout + scale
- add + multiply (residual patterns)
"""

import logging
import warnings

import mlx.core as mx

from mlx_primitives.constants import GELU_SQRT_2_OVER_PI, GELU_TANH_COEFF
from mlx_primitives.kernels._registry import get_kernel
from mlx_primitives.utils import has_metal_kernels, RAISE_ON_METAL_FAILURE

logger = logging.getLogger(__name__)


def _compute_1d_grid(size: int, threadgroup_size: int = 256) -> tuple:
    """Compute 1D Metal grid and threadgroup for given size.

    Args:
        size: Total number of elements to process.
        threadgroup_size: Number of threads per threadgroup.

    Returns:
        Tuple of (grid, threadgroup) for Metal kernel dispatch.
    """
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)
    return grid, threadgroup


def _get_bias_gelu_kernel():
    """Get or create the bias + GELU kernel."""
    source = """
        uint idx = thread_position_in_grid.x;
        uint size = x_shape[0];  // Flattened size
        if (idx >= size) return;

        // bias is broadcast along last dim, need to compute bias index
        uint hidden_dim = bias_shape[0];
        uint bias_idx = idx % hidden_dim;

        float val = x[idx] + bias[bias_idx];

        // GELU tanh approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float gelu = 0.5f * val * (1.0f + metal::tanh(sqrt_2_over_pi * (val + coeff * val * val * val)));

        out[idx] = gelu;
    """

    return get_kernel(
        "bias_gelu",
        lambda: mx.fast.metal_kernel(
            name="bias_gelu",
            input_names=["x", "bias"],
            output_names=["out"],
            source=source,
        ),
    )


def _get_bias_silu_kernel():
    """Get or create the bias + SiLU kernel."""
    source = """
        uint idx = thread_position_in_grid.x;
        uint size = x_shape[0];
        if (idx >= size) return;

        uint hidden_dim = bias_shape[0];
        uint bias_idx = idx % hidden_dim;

        float val = x[idx] + bias[bias_idx];

        // SiLU: x * sigmoid(x)
        float silu = val / (1.0f + metal::exp(-val));

        out[idx] = silu;
    """

    return get_kernel(
        "bias_silu",
        lambda: mx.fast.metal_kernel(
            name="bias_silu",
            input_names=["x", "bias"],
            output_names=["out"],
            source=source,
        ),
    )


def _get_add_mul_kernel():
    """Get or create the add + multiply kernel (residual * scale)."""
    source = """
        uint idx = thread_position_in_grid.x;
        uint size = x_shape[0];
        if (idx >= size) return;

        out[idx] = (x[idx] + y[idx]) * scale[0];
    """

    return get_kernel(
        "add_mul",
        lambda: mx.fast.metal_kernel(
            name="add_mul",
            input_names=["x", "y", "scale"],
            output_names=["out"],
            source=source,
        ),
    )


def fast_bias_gelu(x: mx.array, bias: mx.array) -> mx.array:
    """Fused bias addition + GELU activation.

    Computes: GELU(x + bias)

    Args:
        x: Input tensor of shape (..., hidden_dim).
        bias: Bias tensor of shape (hidden_dim,).

    Returns:
        Activated tensor of same shape as x.

    Raises:
        ValueError: If last dim of x doesn't match bias dim.
    """
    original_shape = x.shape
    hidden_dim = bias.shape[0]
    if x.shape[-1] != hidden_dim:
        raise ValueError(f"Last dim of x ({x.shape[-1]}) must match bias dim ({hidden_dim})")

    x_flat = x.reshape(-1)
    size = x_flat.size

    kernel = _get_bias_gelu_kernel()
    grid, threadgroup = _compute_1d_grid(size)

    outputs = kernel(
        inputs=[x_flat, bias],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(size,)],
        output_dtypes=[x.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0].reshape(original_shape)


def fast_bias_silu(x: mx.array, bias: mx.array) -> mx.array:
    """Fused bias addition + SiLU activation.

    Computes: SiLU(x + bias) = (x + bias) * sigmoid(x + bias)

    Args:
        x: Input tensor of shape (..., hidden_dim).
        bias: Bias tensor of shape (hidden_dim,).

    Returns:
        Activated tensor of same shape as x.

    Raises:
        ValueError: If last dim of x doesn't match bias dim.
    """
    original_shape = x.shape
    hidden_dim = bias.shape[0]
    if x.shape[-1] != hidden_dim:
        raise ValueError(f"Last dim of x ({x.shape[-1]}) must match bias dim ({hidden_dim})")

    x_flat = x.reshape(-1)
    size = x_flat.size

    kernel = _get_bias_silu_kernel()
    grid, threadgroup = _compute_1d_grid(size)

    outputs = kernel(
        inputs=[x_flat, bias],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(size,)],
        output_dtypes=[x.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0].reshape(original_shape)


def fast_add_scale(x: mx.array, y: mx.array, scale: float) -> mx.array:
    """Fused addition + scale.

    Computes: (x + y) * scale

    Common in residual connections with layer scale.

    Args:
        x: First tensor.
        y: Second tensor (same shape as x).
        scale: Scalar multiplier.

    Returns:
        Scaled sum tensor.

    Raises:
        ValueError: If x and y shapes don't match.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes must match, got {x.shape} and {y.shape}")
    original_shape = x.shape

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    scale_arr = mx.array([scale], dtype=x.dtype)
    size = x_flat.size

    kernel = _get_add_mul_kernel()
    grid, threadgroup = _compute_1d_grid(size)

    outputs = kernel(
        inputs=[x_flat, y_flat, scale_arr],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(size,)],
        output_dtypes=[x.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0].reshape(original_shape)


def bias_gelu(x: mx.array, bias: mx.array, use_metal: bool = True) -> mx.array:
    """Bias + GELU with automatic Metal acceleration."""
    if use_metal and has_metal_kernels():
        try:
            return fast_bias_gelu(x, bias)
        except Exception as e:
            if RAISE_ON_METAL_FAILURE:
                raise
            warnings.warn(
                f"Metal kernel 'bias_gelu' failed, falling back to reference implementation: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Fallback
    val = x + bias
    return 0.5 * val * (1 + mx.tanh(GELU_SQRT_2_OVER_PI * (val + GELU_TANH_COEFF * val ** 3)))


def bias_silu(x: mx.array, bias: mx.array, use_metal: bool = True) -> mx.array:
    """Bias + SiLU with automatic Metal acceleration."""
    if use_metal and has_metal_kernels():
        try:
            return fast_bias_silu(x, bias)
        except Exception as e:
            if RAISE_ON_METAL_FAILURE:
                raise
            warnings.warn(
                f"Metal kernel 'bias_silu' failed, falling back to reference implementation: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Fallback
    val = x + bias
    return val * mx.sigmoid(val)


def add_scale(
    x: mx.array,
    y: mx.array,
    scale: float,
    use_metal: bool = True,
) -> mx.array:
    """Add + scale with automatic Metal acceleration."""
    if use_metal and has_metal_kernels():
        try:
            return fast_add_scale(x, y, scale)
        except Exception as e:
            if RAISE_ON_METAL_FAILURE:
                raise
            warnings.warn(
                f"Metal kernel 'add_scale' failed, falling back to reference implementation: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return (x + y) * scale
