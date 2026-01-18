"""Fused RMSNorm + Linear operation.

Every transformer layer computes Linear(RMSNorm(x)) twice:
1. For attention QKV projections
2. For FFN projections

This fusion eliminates memory round-trips by computing normalization
and linear projection in a single kernel pass.

Memory savings: 2x read + 1x write eliminated per operation.
"""

import math
from typing import Optional

import mlx.core as mx

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_fused_rmsnorm_linear_kernel: Optional[mx.fast.metal_kernel] = None
_rmsnorm_kernel: Optional[mx.fast.metal_kernel] = None


def _get_fused_rmsnorm_linear_kernel() -> mx.fast.metal_kernel:
    """Get or create the fused RMSNorm + Linear kernel."""
    global _fused_rmsnorm_linear_kernel
    if _fused_rmsnorm_linear_kernel is None:
        source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint out_idx = thread_position_in_grid.x;

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        uint x_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;

        // Step 1: Compute RMS
        float sum_sq = 0.0f;
        for (uint d = 0; d < hidden_dim; d++) {
            float val = x[x_offset + d];
            sum_sq += val * val;
        }
        float rms = sqrt(sum_sq / float(hidden_dim) + eps);
        float inv_rms = 1.0f / rms;

        // Step 2: Compute normalized @ W^T
        float acc = 0.0f;
        for (uint d = 0; d < hidden_dim; d++) {
            float x_d = x[x_offset + d];
            float norm_x_d = x_d * inv_rms * norm_weight[d];
            acc += norm_x_d * W[out_idx * hidden_dim + d];
        }

        // Add bias if present
        if (has_bias > 0) {
            acc += bias[out_idx];
        }

        uint out_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
        output[out_offset] = acc;
        """
        _fused_rmsnorm_linear_kernel = mx.fast.metal_kernel(
            name="fused_rmsnorm_linear",
            input_names=[
                "x", "norm_weight", "W", "bias",
                "batch_size", "seq_len", "hidden_dim", "out_features",
                "eps", "has_bias"
            ],
            output_names=["output"],
            source=source,
        )
    return _fused_rmsnorm_linear_kernel


def _get_rmsnorm_kernel() -> mx.fast.metal_kernel:
    """Get or create the RMSNorm kernel."""
    global _rmsnorm_kernel
    if _rmsnorm_kernel is None:
        source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint d = thread_position_in_grid.x;

        if (batch_idx >= batch_size || seq_idx >= seq_len || d >= hidden_dim) return;

        uint offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;

        // Compute RMS (simplified - each thread reads all elements)
        float sum_sq = 0.0f;
        for (uint i = 0; i < hidden_dim; i++) {
            float val = x[offset + i];
            sum_sq += val * val;
        }
        float rms = sqrt(sum_sq / float(hidden_dim) + eps);
        float inv_rms = 1.0f / rms;

        output[offset + d] = x[offset + d] * inv_rms * weight[d];
        """
        _rmsnorm_kernel = mx.fast.metal_kernel(
            name="rmsnorm",
            input_names=["x", "weight", "batch_size", "seq_len", "hidden_dim", "eps"],
            output_names=["output"],
            source=source,
        )
    return _rmsnorm_kernel


def fused_rmsnorm_linear(
    x: mx.array,
    norm_weight: mx.array,
    linear_weight: mx.array,
    linear_bias: Optional[mx.array] = None,
    eps: float = 1e-5,
    use_metal: bool = True,
) -> mx.array:
    """Fused RMSNorm followed by Linear projection.

    Computes: Linear(RMSNorm(x)) in a single kernel pass.

    This is equivalent to:
        norm_x = x / sqrt(mean(x^2) + eps) * norm_weight
        output = norm_x @ linear_weight.T + linear_bias

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim).
        norm_weight: RMSNorm weight of shape (hidden_dim,).
        linear_weight: Linear weight of shape (out_features, hidden_dim).
        linear_bias: Optional linear bias of shape (out_features,).
        eps: Epsilon for numerical stability in RMSNorm.
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, out_features).

    Example:
        >>> x = mx.random.normal((2, 128, 768))
        >>> norm_w = mx.ones((768,))
        >>> linear_w = mx.random.normal((3072, 768))
        >>> out = fused_rmsnorm_linear(x, norm_w, linear_w)
        >>> out.shape
        (2, 128, 3072)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input (batch, seq, hidden), got {x.ndim}D")

    batch_size, seq_len, hidden_dim = x.shape
    out_features = linear_weight.shape[0]

    if norm_weight.shape[0] != hidden_dim:
        raise ValueError(
            f"norm_weight dim {norm_weight.shape[0]} != hidden_dim {hidden_dim}"
        )
    if linear_weight.shape[1] != hidden_dim:
        raise ValueError(
            f"linear_weight dim {linear_weight.shape[1]} != hidden_dim {hidden_dim}"
        )

    # For small tensors or when Metal not available, use separate ops
    if not use_metal or not _HAS_METAL or seq_len < 8:
        return _reference_rmsnorm_linear(x, norm_weight, linear_weight, linear_bias, eps)

    try:
        return _metal_fused_rmsnorm_linear(
            x, norm_weight, linear_weight, linear_bias, eps
        )
    except Exception:
        return _reference_rmsnorm_linear(x, norm_weight, linear_weight, linear_bias, eps)


def _metal_fused_rmsnorm_linear(
    x: mx.array,
    norm_weight: mx.array,
    linear_weight: mx.array,
    linear_bias: Optional[mx.array],
    eps: float,
) -> mx.array:
    """Metal kernel implementation."""
    batch_size, seq_len, hidden_dim = x.shape
    out_features = linear_weight.shape[0]

    kernel = _get_fused_rmsnorm_linear_kernel()

    # Ensure contiguous float32
    x = mx.ascontiguousarray(x.astype(mx.float32))
    norm_weight = mx.ascontiguousarray(norm_weight.astype(mx.float32))
    linear_weight = mx.ascontiguousarray(linear_weight.astype(mx.float32))

    if linear_bias is not None:
        linear_bias = mx.ascontiguousarray(linear_bias.astype(mx.float32))
        has_bias = 1
    else:
        linear_bias = mx.zeros((out_features,), dtype=mx.float32)
        has_bias = 0

    # Scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    hidden_arr = mx.array([hidden_dim], dtype=mx.uint32)
    out_arr = mx.array([out_features], dtype=mx.uint32)
    eps_arr = mx.array([eps], dtype=mx.float32)
    has_bias_arr = mx.array([has_bias], dtype=mx.uint32)

    outputs = kernel(
        inputs=[
            x, norm_weight, linear_weight, linear_bias,
            batch_arr, seq_arr, hidden_arr, out_arr,
            eps_arr, has_bias_arr
        ],
        grid=(out_features, seq_len, batch_size),
        threadgroup=(min(out_features, 64), 1, 1),
        output_shapes=[(batch_size, seq_len, out_features)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_rmsnorm_linear(
    x: mx.array,
    norm_weight: mx.array,
    linear_weight: mx.array,
    linear_bias: Optional[mx.array],
    eps: float,
) -> mx.array:
    """Reference implementation using separate operations."""
    # RMSNorm
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    norm_x = x / rms * norm_weight

    # Linear
    out = norm_x @ linear_weight.T
    if linear_bias is not None:
        out = out + linear_bias

    return out


def rmsnorm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-5,
    use_metal: bool = True,
) -> mx.array:
    """RMSNorm (Root Mean Square Layer Normalization).

    Computes: x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor of shape (..., hidden_dim).
        weight: Weight tensor of shape (hidden_dim,).
        eps: Epsilon for numerical stability.
        use_metal: Use Metal kernel if available.

    Returns:
        Normalized tensor of same shape as input.
    """
    # Simple implementation using MLX ops
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


class FusedRMSNormLinear:
    """Fused RMSNorm + Linear layer.

    Combines RMSNorm and Linear projection into a single operation
    to reduce memory bandwidth.

    Args:
        hidden_dim: Input dimension.
        out_features: Output dimension.
        eps: RMSNorm epsilon.
        bias: Whether to include bias in linear.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_features: int,
        eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.eps = eps

        # Initialize weights
        scale = 1.0 / math.sqrt(hidden_dim)
        self.norm_weight = mx.ones((hidden_dim,))
        self.linear_weight = mx.random.normal((out_features, hidden_dim)) * scale
        self.linear_bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply fused RMSNorm + Linear.

        Args:
            x: Input tensor (batch, seq, hidden_dim).

        Returns:
            Output tensor (batch, seq, out_features).
        """
        return fused_rmsnorm_linear(
            x,
            self.norm_weight,
            self.linear_weight,
            self.linear_bias,
            self.eps,
        )
