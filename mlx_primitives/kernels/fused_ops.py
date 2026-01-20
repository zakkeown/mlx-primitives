"""Fused operations Metal kernels.

Common operation patterns that benefit from fusion:
- bias + activation (saves memory roundtrip)
- dropout + scale
- add + multiply (residual patterns)
"""

import warnings

import mlx.core as mx

from mlx_primitives.utils import has_metal_kernels, RAISE_ON_METAL_FAILURE

# Cache compiled kernels
_bias_gelu_kernel = None
_bias_silu_kernel = None
_dropout_kernel = None
_add_mul_kernel = None


def _get_bias_gelu_kernel():
    """Get or create the bias + GELU kernel."""
    global _bias_gelu_kernel

    if _bias_gelu_kernel is None:
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

        _bias_gelu_kernel = mx.fast.metal_kernel(
            name="bias_gelu",
            input_names=["x", "bias"],
            output_names=["out"],
            source=source,
        )

    return _bias_gelu_kernel


def _get_bias_silu_kernel():
    """Get or create the bias + SiLU kernel."""
    global _bias_silu_kernel

    if _bias_silu_kernel is None:
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

        _bias_silu_kernel = mx.fast.metal_kernel(
            name="bias_silu",
            input_names=["x", "bias"],
            output_names=["out"],
            source=source,
        )

    return _bias_silu_kernel


def _get_add_mul_kernel():
    """Get or create the add + multiply kernel (residual * scale)."""
    global _add_mul_kernel

    if _add_mul_kernel is None:
        source = """
            uint idx = thread_position_in_grid.x;
            uint size = x_shape[0];
            if (idx >= size) return;

            out[idx] = (x[idx] + y[idx]) * scale[0];
        """

        _add_mul_kernel = mx.fast.metal_kernel(
            name="add_mul",
            input_names=["x", "y", "scale"],
            output_names=["out"],
            source=source,
        )

    return _add_mul_kernel


def fast_bias_gelu(x: mx.array, bias: mx.array) -> mx.array:
    """Fused bias addition + GELU activation.

    Computes: GELU(x + bias)

    Args:
        x: Input tensor of shape (..., hidden_dim).
        bias: Bias tensor of shape (hidden_dim,).

    Returns:
        Activated tensor of same shape as x.
    """
    original_shape = x.shape
    hidden_dim = bias.shape[0]
    assert x.shape[-1] == hidden_dim, "Last dim must match bias"

    x_flat = x.reshape(-1)
    size = x_flat.size

    kernel = _get_bias_gelu_kernel()

    threadgroup_size = 256
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

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
    """
    original_shape = x.shape
    hidden_dim = bias.shape[0]
    assert x.shape[-1] == hidden_dim, "Last dim must match bias"

    x_flat = x.reshape(-1)
    size = x_flat.size

    kernel = _get_bias_silu_kernel()

    threadgroup_size = 256
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

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
    """
    assert x.shape == y.shape, "Shapes must match"
    original_shape = x.shape

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    scale_arr = mx.array([scale], dtype=x.dtype)
    size = x_flat.size

    kernel = _get_add_mul_kernel()

    threadgroup_size = 256
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

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
            if _RAISE_ON_METAL_FAILURE:
                raise
            warnings.warn(
                f"Metal kernel 'bias_gelu' failed, falling back to reference implementation: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Fallback
    val = x + bias
    sqrt_2_over_pi = 0.7978845608
    return 0.5 * val * (1 + mx.tanh(sqrt_2_over_pi * (val + 0.044715 * val ** 3)))


def bias_silu(x: mx.array, bias: mx.array, use_metal: bool = True) -> mx.array:
    """Bias + SiLU with automatic Metal acceleration."""
    if use_metal and has_metal_kernels():
        try:
            return fast_bias_silu(x, bias)
        except Exception as e:
            if _RAISE_ON_METAL_FAILURE:
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
            if _RAISE_ON_METAL_FAILURE:
                raise
            warnings.warn(
                f"Metal kernel 'add_scale' failed, falling back to reference implementation: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    return (x + y) * scale
