"""Fused RMSNorm + Linear operation.

Every transformer layer computes Linear(RMSNorm(x)) twice:
1. For attention QKV projections
2. For FFN projections

This fusion eliminates memory round-trips by computing normalization
and linear projection in a single kernel pass.

Memory savings: 2x read + 1x write eliminated per operation.
"""

import math
import threading
from typing import Optional

import mlx.core as mx

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache with thread safety
_fused_rmsnorm_linear_kernel: Optional[mx.fast.metal_kernel] = None
_rmsnorm_kernel: Optional[mx.fast.metal_kernel] = None
_fused_kernel_lock = threading.Lock()
_rmsnorm_kernel_lock = threading.Lock()


def _get_fused_rmsnorm_linear_kernel() -> mx.fast.metal_kernel:
    """Get or create the fused RMSNorm + Linear kernel with parallel reduction (thread-safe).

    Optimization: Instead of each thread independently computing sum_sq (redundant reads),
    all threads cooperatively compute inv_rms using SIMD reduction, then each thread
    computes its output features.
    """
    global _fused_rmsnorm_linear_kernel
    if _fused_rmsnorm_linear_kernel is None:
        with _fused_kernel_lock:
            # Double-check after acquiring lock
            if _fused_rmsnorm_linear_kernel is not None:
                return _fused_rmsnorm_linear_kernel
            source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint num_threads = threads_per_threadgroup.x;

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _hidden_dim = hidden_dim[0];
        uint _out_features = out_features[0];
        float _eps = eps[0];
        uint _has_bias = has_bias[0];

        if (batch_idx >= _batch_size || seq_idx >= _seq_len) return;

        uint x_offset = batch_idx * _seq_len * _hidden_dim + seq_idx * _hidden_dim;

        // Phase 1: Cooperative RMS computation using parallel reduction
        // Each thread computes partial sum_sq with strided access
        float partial_sum = 0.0f;
        for (uint d = tid; d < _hidden_dim; d += num_threads) {
            float val = x[x_offset + d];
            partial_sum += val * val;
        }

        // SIMD reduction within each simdgroup
        float simd_total = simd_sum(partial_sum);

        // Cross-SIMD reduction via threadgroup memory
        threadgroup float simd_sums[32];

        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;
        uint num_simd_groups = (num_threads + 31) / 32;

        if (simd_lane == 0) {
            simd_sums[simd_group_id] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First SIMD group performs final reduction
        float final_sum = 0.0f;
        if (simd_group_id == 0) {
            if (simd_lane < num_simd_groups) {
                final_sum = simd_sums[simd_lane];
            }
            final_sum = simd_sum(final_sum);
            if (simd_lane == 0) {
                simd_sums[0] = final_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Broadcast inv_rms to all threads
        float sum_sq = simd_sums[0];
        float inv_rms = rsqrt(sum_sq / float(_hidden_dim) + _eps);

        // Phase 2: Each thread computes its assigned output features
        // Thread tid handles output features: tid, tid + num_threads, ...
        for (uint out_idx = tid; out_idx < _out_features; out_idx += num_threads) {
            float acc = 0.0f;
            for (uint d = 0; d < _hidden_dim; d++) {
                float x_d = x[x_offset + d];
                float norm_x_d = x_d * inv_rms * norm_weight[d];
                acc += norm_x_d * W[out_idx * _hidden_dim + d];
            }

            // Add bias if present
            if (_has_bias > 0) {
                acc += bias[out_idx];
            }

            uint out_offset = batch_idx * _seq_len * _out_features + seq_idx * _out_features + out_idx;
            output[out_offset] = acc;
        }
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
    """Get or create the RMSNorm kernel with parallel reduction (thread-safe).

    Uses SIMD and threadgroup reduction for efficient sum-of-squares computation:
    1. Phase 1: Each thread computes partial sum with strided access
    2. Phase 2: SIMD reduction within each simdgroup using simd_sum()
    3. Phase 3: Cross-SIMD reduction via threadgroup memory
    4. Phase 4: Broadcast inv_rms and compute output
    """
    global _rmsnorm_kernel
    if _rmsnorm_kernel is None:
        with _rmsnorm_kernel_lock:
            # Double-check after acquiring lock
            if _rmsnorm_kernel is not None:
                return _rmsnorm_kernel
            source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint num_threads = threads_per_threadgroup.x;

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _hidden_dim = hidden_dim[0];
        float _eps = eps[0];

        if (batch_idx >= _batch_size || seq_idx >= _seq_len) return;

        uint offset = batch_idx * _seq_len * _hidden_dim + seq_idx * _hidden_dim;

        // Phase 1: Compute partial sum-of-squares with strided access
        // Each thread handles elements: tid, tid + num_threads, tid + 2*num_threads, ...
        float partial_sum = 0.0f;
        for (uint i = tid; i < _hidden_dim; i += num_threads) {
            float val = x[offset + i];
            partial_sum += val * val;
        }

        // Phase 2: SIMD reduction within each simdgroup (32 threads on Apple Silicon)
        float simd_total = simd_sum(partial_sum);

        // Phase 3: Cross-SIMD reduction via threadgroup memory
        // Max 32 SIMD groups (1024 threads), typically we use 256 or less
        threadgroup float simd_sums[32];

        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;
        uint num_simd_groups = (num_threads + 31) / 32;

        // First lane of each SIMD group writes its sum to shared memory
        if (simd_lane == 0) {
            simd_sums[simd_group_id] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First SIMD group performs final reduction across all SIMD groups
        float final_sum = 0.0f;
        if (simd_group_id == 0) {
            if (simd_lane < num_simd_groups) {
                final_sum = simd_sums[simd_lane];
            }
            final_sum = simd_sum(final_sum);
            // Store result for broadcast
            if (simd_lane == 0) {
                simd_sums[0] = final_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Broadcast inv_rms and compute normalized output
        float sum_sq = simd_sums[0];
        float inv_rms = rsqrt(sum_sq / float(_hidden_dim) + _eps);

        // Each thread writes its assigned elements
        for (uint i = tid; i < _hidden_dim; i += num_threads) {
            output[offset + i] = x[offset + i] * inv_rms * weight[i];
        }
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
    except RuntimeError as e:
        # Catch Metal kernel errors, but let programming bugs propagate
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("fused_rmsnorm_linear", e)
        return _reference_rmsnorm_linear(x, norm_weight, linear_weight, linear_bias, eps)


def _metal_fused_rmsnorm_linear(
    x: mx.array,
    norm_weight: mx.array,
    linear_weight: mx.array,
    linear_bias: Optional[mx.array],
    eps: float,
) -> mx.array:
    """Metal kernel implementation with parallel reduction for RMS computation."""
    batch_size, seq_len, hidden_dim = x.shape
    out_features = linear_weight.shape[0]

    kernel = _get_fused_rmsnorm_linear_kernel()

    # Ensure contiguous float32
    x = mx.contiguous(x.astype(mx.float32))
    norm_weight = mx.contiguous(norm_weight.astype(mx.float32))
    linear_weight = mx.contiguous(linear_weight.astype(mx.float32))

    if linear_bias is not None:
        linear_bias = mx.contiguous(linear_bias.astype(mx.float32))
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

    # Configure threadgroup for efficient parallel reduction
    # Use up to 256 threads per threadgroup (8 SIMD groups)
    # Must be at least 32 for SIMD operations
    threads_per_group = min(256, max(32, hidden_dim))
    # Round up to multiple of 32 (SIMD width)
    threads_per_group = ((threads_per_group + 31) // 32) * 32

    outputs = kernel(
        inputs=[
            x, norm_weight, linear_weight, linear_bias,
            batch_arr, seq_arr, hidden_arr, out_arr,
            eps_arr, has_bias_arr
        ],
        # Grid: one threadgroup per (batch, seq) position
        grid=(threads_per_group, seq_len, batch_size),
        threadgroup=(threads_per_group, 1, 1),
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

    Uses parallel reduction with SIMD operations when Metal is available
    for efficient sum-of-squares computation.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim).
        weight: Weight tensor of shape (hidden_dim,).
        eps: Epsilon for numerical stability.
        use_metal: Use Metal kernel if available.

    Returns:
        Normalized tensor of same shape as input.
    """
    # Metal kernel requires 3D input
    if x.ndim != 3 or not use_metal or not _HAS_METAL:
        return _reference_rmsnorm(x, weight, eps)

    try:
        return _metal_rmsnorm(x, weight, eps)
    except RuntimeError as e:
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("rmsnorm", e)
        return _reference_rmsnorm(x, weight, eps)


def _reference_rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Reference implementation using MLX ops."""
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def _metal_rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Metal kernel implementation with parallel reduction."""
    batch_size, seq_len, hidden_dim = x.shape

    kernel = _get_rmsnorm_kernel()

    # Ensure contiguous float32
    x = mx.contiguous(x.astype(mx.float32))
    weight = mx.contiguous(weight.astype(mx.float32))

    # Scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    hidden_arr = mx.array([hidden_dim], dtype=mx.uint32)
    eps_arr = mx.array([eps], dtype=mx.float32)

    # Configure threadgroup size for efficient parallel reduction
    # Use up to 256 threads per threadgroup (8 SIMD groups)
    threads_per_group = min(256, hidden_dim)
    # Round up to multiple of 32 (SIMD width)
    threads_per_group = ((threads_per_group + 31) // 32) * 32

    outputs = kernel(
        inputs=[x, weight, batch_arr, seq_arr, hidden_arr, eps_arr],
        # Grid: one threadgroup per (batch, seq) position
        grid=(threads_per_group, seq_len, batch_size),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(batch_size, seq_len, hidden_dim)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


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
