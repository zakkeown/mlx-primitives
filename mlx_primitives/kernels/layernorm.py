"""Fast LayerNorm Metal kernel implementation.

Provides fused layer normalization that computes mean, variance, and
normalization in a single pass per row.
"""

from typing import Optional
import mlx.core as mx

# Cache compiled kernels
_layernorm_kernel = None
_layernorm_affine_kernel = None


def _get_layernorm_kernel():
    """Get or create the LayerNorm kernel (no affine params)."""
    global _layernorm_kernel

    if _layernorm_kernel is None:
        # Each thread handles one row, computes mean+var in 2 passes
        source = """
            uint row_idx = thread_position_in_grid.x;

            uint batch_size = x_shape[0];
            uint seq_len = x_shape[1];
            uint hidden_dim = x_shape[2];
            uint total_rows = batch_size * seq_len;

            if (row_idx >= total_rows) return;

            uint row_offset = row_idx * hidden_dim;

            // Pass 1: Compute mean
            float sum = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                sum += x[row_offset + i];
            }
            float mean = sum / float(hidden_dim);

            // Pass 2: Compute variance
            float sum_sq = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                float diff = x[row_offset + i] - mean;
                sum_sq += diff * diff;
            }
            float var_inv = metal::rsqrt(sum_sq / float(hidden_dim) + 1e-6f);

            // Pass 3: Normalize
            for (uint i = 0; i < hidden_dim; i++) {
                out[row_offset + i] = (x[row_offset + i] - mean) * var_inv;
            }
        """

        _layernorm_kernel = mx.fast.metal_kernel(
            name="layernorm_forward",
            input_names=["x"],
            output_names=["out"],
            source=source,
        )

    return _layernorm_kernel


def _get_layernorm_affine_kernel():
    """Get or create the LayerNorm kernel with affine params (gamma, beta)."""
    global _layernorm_affine_kernel

    if _layernorm_affine_kernel is None:
        source = """
            uint row_idx = thread_position_in_grid.x;

            uint batch_size = x_shape[0];
            uint seq_len = x_shape[1];
            uint hidden_dim = x_shape[2];
            uint total_rows = batch_size * seq_len;

            if (row_idx >= total_rows) return;

            uint row_offset = row_idx * hidden_dim;

            // Pass 1: Compute mean
            float sum = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                sum += x[row_offset + i];
            }
            float mean = sum / float(hidden_dim);

            // Pass 2: Compute variance
            float sum_sq = 0.0f;
            for (uint i = 0; i < hidden_dim; i++) {
                float diff = x[row_offset + i] - mean;
                sum_sq += diff * diff;
            }
            float var_inv = metal::rsqrt(sum_sq / float(hidden_dim) + 1e-6f);

            // Pass 3: Normalize with affine transform
            for (uint i = 0; i < hidden_dim; i++) {
                float normalized = (x[row_offset + i] - mean) * var_inv;
                out[row_offset + i] = normalized * gamma[i] + beta[i];
            }
        """

        _layernorm_affine_kernel = mx.fast.metal_kernel(
            name="layernorm_affine_forward",
            input_names=["x", "gamma", "beta"],
            output_names=["out"],
            source=source,
        )

    return _layernorm_affine_kernel


def _reference_layernorm(
    x: mx.array,
    gamma: Optional[mx.array] = None,
    beta: Optional[mx.array] = None,
    eps: float = 1e-6,
) -> mx.array:
    """Reference implementation using MLX ops."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / mx.sqrt(var + eps)

    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta

    return normalized


def fast_layernorm(
    x: mx.array,
    gamma: Optional[mx.array] = None,
    beta: Optional[mx.array] = None,
    eps: float = 1e-6,
) -> mx.array:
    """Fast Metal-accelerated LayerNorm.

    Computes: (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
        gamma: Optional scale parameter of shape (hidden_dim,).
        beta: Optional bias parameter of shape (hidden_dim,).
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
    # (see RCA report - benchmark showed 0.82x-0.86x slowdown for b=4, s=512)
    # For small workloads, MLX's fused ops are faster than custom Metal kernels
    if batch_size <= 2 or (batch_size <= 4 and seq_len <= 512):
        result = _reference_layernorm(x, gamma, beta, eps)
        return result[0] if was_2d else result

    total_rows = batch_size * seq_len

    # One thread per row
    threadgroup_size = min(256, total_rows)
    num_groups = (total_rows + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    if gamma is not None and beta is not None:
        kernel = _get_layernorm_affine_kernel()
        outputs = kernel(
            inputs=[x, gamma, beta],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            stream=mx.default_stream(mx.default_device()),
        )
    else:
        kernel = _get_layernorm_kernel()
        outputs = kernel(
            inputs=[x],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            stream=mx.default_stream(mx.default_device()),
        )

    result = outputs[0]

    if was_2d:
        result = result[0]

    return result


# Check if metal kernels are available
_USE_METAL_KERNELS = hasattr(mx.fast, 'metal_kernel')


def layernorm(
    x: mx.array,
    gamma: Optional[mx.array] = None,
    beta: Optional[mx.array] = None,
    eps: float = 1e-6,
    use_metal: bool = True,
) -> mx.array:
    """LayerNorm with automatic Metal acceleration.

    Args:
        x: Input tensor.
        gamma: Optional scale parameter.
        beta: Optional bias parameter.
        eps: Numerical stability constant.
        use_metal: Whether to use Metal kernels.

    Returns:
        Normalized tensor.
    """
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_layernorm(x, gamma, beta, eps)
        except Exception:
            pass

    # Fallback to pure MLX
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / mx.sqrt(var + eps)

    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta

    return normalized
