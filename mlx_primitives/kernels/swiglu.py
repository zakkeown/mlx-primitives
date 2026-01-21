"""Fast SwiGLU Metal kernel implementation.

Provides fused SiLU-gated linear unit activation that avoids
intermediate memory allocation.
"""

import mlx.core as mx

# Cache compiled kernels
_swiglu_kernel = None
_geglu_kernel = None


def _get_swiglu_kernel():
    """Get or create the SwiGLU kernel."""
    global _swiglu_kernel

    if _swiglu_kernel is None:
        # MLX auto-provides gate_shape when referenced
        source = """
            uint idx = thread_position_in_grid.x;
            uint size = gate_shape[0];  // Flattened array size
            if (idx >= size) return;

            float g = gate[idx];
            float u = up[idx];

            // SiLU: x * sigmoid(x)
            float silu_g = g / (1.0f + metal::exp(-g));
            out[idx] = silu_g * u;
        """

        _swiglu_kernel = mx.fast.metal_kernel(
            name="swiglu_forward",
            input_names=["gate", "up"],
            output_names=["out"],
            source=source,
        )

    return _swiglu_kernel


def _get_geglu_kernel():
    """Get or create the GeGLU kernel."""
    global _geglu_kernel

    if _geglu_kernel is None:
        # MLX auto-provides gate_shape when referenced
        source = """
            uint idx = thread_position_in_grid.x;
            uint size = gate_shape[0];  // Flattened array size
            if (idx >= size) return;

            float g = gate[idx];
            float u = up[idx];

            // GELU approximation (tanh version)
            const float sqrt_2_over_pi = 0.7978845608f;
            const float coeff = 0.044715f;
            float gelu = 0.5f * g * (1.0f + metal::tanh(sqrt_2_over_pi * (g + coeff * g * g * g)));

            out[idx] = gelu * u;
        """

        _geglu_kernel = mx.fast.metal_kernel(
            name="geglu_forward",
            input_names=["gate", "up"],
            output_names=["out"],
            source=source,
        )

    return _geglu_kernel


def _reference_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Reference implementation using MLX ops."""
    return mx.sigmoid(gate) * gate * up


def fast_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fast Metal-accelerated SwiGLU.

    Computes: silu(gate) * up

    Args:
        gate: Gate tensor (output of x @ W_gate).
        up: Up tensor (output of x @ W_up).

    Returns:
        SwiGLU activation result.
    """
    assert gate.shape == up.shape, "gate and up must have same shape"

    size = gate.size

    # Metal kernel overhead not amortized for small tensors (see RCA report)
    # Benchmark showed 0.89x at 2M elements, 1.28x at 8M elements
    if size < 4_000_000:
        return _reference_swiglu(gate, up)

    kernel = _get_swiglu_kernel()

    # Flatten for kernel
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)

    # Calculate grid
    threadgroup_size = 256
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[gate_flat, up_flat],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(size,)],
        output_dtypes=[gate.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0].reshape(gate.shape)


def fast_geglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fast Metal-accelerated GeGLU.

    Computes: gelu(gate) * up

    Args:
        gate: Gate tensor.
        up: Up tensor.

    Returns:
        GeGLU activation result.
    """
    assert gate.shape == up.shape, "gate and up must have same shape"

    kernel = _get_geglu_kernel()
    size = gate.size

    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)

    threadgroup_size = 256
    num_groups = (size + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[gate_flat, up_flat],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(size,)],
        output_dtypes=[gate.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0].reshape(gate.shape)


# Check if metal kernels are available
_USE_METAL_KERNELS = hasattr(mx.fast, 'metal_kernel')


def swiglu(
    gate: mx.array,
    up: mx.array,
    use_metal: bool = True,
) -> mx.array:
    """SwiGLU with automatic Metal acceleration.

    Args:
        gate: Gate tensor.
        up: Up tensor.
        use_metal: Whether to use Metal kernels.

    Returns:
        SwiGLU result.
    """
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_swiglu(gate, up)
        except Exception:
            pass

    # Fallback: silu(gate) * up
    return mx.sigmoid(gate) * gate * up


def geglu(
    gate: mx.array,
    up: mx.array,
    use_metal: bool = True,
) -> mx.array:
    """GeGLU with automatic Metal acceleration.

    Args:
        gate: Gate tensor.
        up: Up tensor.
        use_metal: Whether to use Metal kernels.

    Returns:
        GeGLU result.
    """
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_geglu(gate, up)
        except Exception:
            pass

    # Fallback: gelu(gate) * up
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608
    x3 = gate * gate * gate
    return 0.5 * gate * (1 + mx.tanh(sqrt_2_over_pi * (gate + 0.044715 * x3))) * up
