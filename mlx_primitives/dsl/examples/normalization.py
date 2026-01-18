"""Normalization kernels using Metal-Triton DSL.

Includes:
- LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
- RMSNorm: x / sqrt(mean(x^2) + eps) * weight
- Fused Add + LayerNorm: LayerNorm(x + residual)
- Fused Add + RMSNorm: RMSNorm(x + residual)

These kernels demonstrate:
- SIMD reduction patterns
- Shared memory for cross-SIMD accumulation
- Multi-pass algorithms (mean, variance, normalize)
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def layer_norm(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: constexpr,  # batch * seq_len (number of rows)
    D: constexpr,  # hidden dimension
    eps: mt.float32 = 1e-5,
):
    """LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias.

    Each threadgroup processes one row (normalization dimension).

    Args:
        x_ptr: Input tensor (N, D)
        weight_ptr: Scale weights (D,)
        bias_ptr: Bias weights (D,) - can be None
        out_ptr: Output tensor (N, D)
        N: Number of rows (batch * seq_len)
        D: Hidden dimension
        eps: Epsilon for numerical stability

    Grid: (N,)
    Threadgroup: 256
    """
    row_idx = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()
    simd_group = mt.simd_group_id()
    block_size = mt.threads_per_threadgroup()

    if row_idx >= N:
        return

    row_offset = row_idx * D

    # Shared memory for cross-SIMD reduction
    # 8 SIMD groups max (256 threads / 32 = 8)
    shared = mt.shared_memory(8, dtype=mt.float32)

    # Phase 1: Compute mean
    local_sum = 0.0
    for d in range(tid, D, block_size):
        local_sum = local_sum + mt.load(x_ptr + row_offset + d)

    # SIMD reduction for sum
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 16)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 8)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 4)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 2)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 1)

    # Store SIMD group result to shared memory
    if simd_lane == 0:
        shared[simd_group] = local_sum
    mt.threadgroup_barrier()

    # Final reduction in first SIMD group
    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            shared[0] = total / D  # Store mean
    mt.threadgroup_barrier()

    mean = mt.load(shared + 0)

    # Phase 2: Compute variance
    local_var = 0.0
    for d in range(tid, D, block_size):
        diff = mt.load(x_ptr + row_offset + d) - mean
        local_var = local_var + diff * diff

    # SIMD reduction for variance
    local_var = local_var + mt.simd_shuffle_down(local_var, 16)
    local_var = local_var + mt.simd_shuffle_down(local_var, 8)
    local_var = local_var + mt.simd_shuffle_down(local_var, 4)
    local_var = local_var + mt.simd_shuffle_down(local_var, 2)
    local_var = local_var + mt.simd_shuffle_down(local_var, 1)

    if simd_lane == 0:
        shared[simd_group] = local_var
    mt.threadgroup_barrier()

    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            var = total / D
            shared[0] = 1.0 / mt.sqrt(var + eps)  # Store inv_std
    mt.threadgroup_barrier()

    inv_std = mt.load(shared + 0)

    # Phase 3: Normalize and apply affine transform
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        norm_val = (x_val - mean) * inv_std

        # Apply weight and bias
        w = mt.load(weight_ptr + d)
        b = mt.load(bias_ptr + d)
        out_val = norm_val * w + b

        mt.store(out_ptr + row_offset + d, out_val)


@metal_kernel
def rms_norm(
    x_ptr,
    weight_ptr,
    out_ptr,
    N: constexpr,  # batch * seq_len (number of rows)
    D: constexpr,  # hidden dimension
    eps: mt.float32 = 1e-5,
):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Simpler than LayerNorm - no mean subtraction or bias.
    Used in LLaMA, Mistral, and other models.

    Args:
        x_ptr: Input tensor (N, D)
        weight_ptr: Scale weights (D,)
        out_ptr: Output tensor (N, D)
        N: Number of rows (batch * seq_len)
        D: Hidden dimension
        eps: Epsilon for numerical stability

    Grid: (N,)
    Threadgroup: 256
    """
    row_idx = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()
    simd_group = mt.simd_group_id()
    block_size = mt.threads_per_threadgroup()

    if row_idx >= N:
        return

    row_offset = row_idx * D

    # Shared memory for cross-SIMD reduction
    shared = mt.shared_memory(8, dtype=mt.float32)

    # Compute sum of squares
    local_ss = 0.0
    for d in range(tid, D, block_size):
        val = mt.load(x_ptr + row_offset + d)
        local_ss = local_ss + val * val

    # SIMD reduction
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 16)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 8)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 4)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 2)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 1)

    if simd_lane == 0:
        shared[simd_group] = local_ss
    mt.threadgroup_barrier()

    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            rms = mt.sqrt(total / D + eps)
            shared[0] = 1.0 / rms  # Store inv_rms
    mt.threadgroup_barrier()

    inv_rms = mt.load(shared + 0)

    # Normalize and apply weight
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        w = mt.load(weight_ptr + d)
        out_val = x_val * inv_rms * w
        mt.store(out_ptr + row_offset + d, out_val)


@metal_kernel
def fused_add_layer_norm(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: constexpr,  # batch * seq_len
    D: constexpr,  # hidden dimension
    eps: mt.float32 = 1e-5,
):
    """Fused residual add + LayerNorm.

    y = LayerNorm(x + residual)

    Saves one memory round-trip by computing add inline.

    Args:
        x_ptr: Input tensor (N, D)
        residual_ptr: Residual tensor (N, D)
        weight_ptr: Scale weights (D,)
        bias_ptr: Bias weights (D,)
        out_ptr: Output tensor (N, D)
        N: Number of rows
        D: Hidden dimension
        eps: Epsilon for numerical stability

    Grid: (N,)
    Threadgroup: 256
    """
    row_idx = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()
    simd_group = mt.simd_group_id()
    block_size = mt.threads_per_threadgroup()

    if row_idx >= N:
        return

    row_offset = row_idx * D

    shared = mt.shared_memory(8, dtype=mt.float32)

    # Phase 1: Compute mean of (x + residual)
    local_sum = 0.0
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        res_val = mt.load(residual_ptr + row_offset + d)
        local_sum = local_sum + (x_val + res_val)

    # SIMD reduction
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 16)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 8)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 4)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 2)
    local_sum = local_sum + mt.simd_shuffle_down(local_sum, 1)

    if simd_lane == 0:
        shared[simd_group] = local_sum
    mt.threadgroup_barrier()

    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            shared[0] = total / D
    mt.threadgroup_barrier()

    mean = mt.load(shared + 0)

    # Phase 2: Compute variance
    local_var = 0.0
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        res_val = mt.load(residual_ptr + row_offset + d)
        fused = x_val + res_val
        diff = fused - mean
        local_var = local_var + diff * diff

    local_var = local_var + mt.simd_shuffle_down(local_var, 16)
    local_var = local_var + mt.simd_shuffle_down(local_var, 8)
    local_var = local_var + mt.simd_shuffle_down(local_var, 4)
    local_var = local_var + mt.simd_shuffle_down(local_var, 2)
    local_var = local_var + mt.simd_shuffle_down(local_var, 1)

    if simd_lane == 0:
        shared[simd_group] = local_var
    mt.threadgroup_barrier()

    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            var = total / D
            shared[0] = 1.0 / mt.sqrt(var + eps)
    mt.threadgroup_barrier()

    inv_std = mt.load(shared + 0)

    # Phase 3: Normalize and apply affine
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        res_val = mt.load(residual_ptr + row_offset + d)
        fused = x_val + res_val
        norm_val = (fused - mean) * inv_std

        w = mt.load(weight_ptr + d)
        b = mt.load(bias_ptr + d)
        out_val = norm_val * w + b

        mt.store(out_ptr + row_offset + d, out_val)


@metal_kernel
def fused_add_rms_norm(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    N: constexpr,
    D: constexpr,
    eps: mt.float32 = 1e-5,
):
    """Fused residual add + RMSNorm.

    y = RMSNorm(x + residual)

    Common in LLaMA and similar architectures.

    Args:
        x_ptr: Input tensor (N, D)
        residual_ptr: Residual tensor (N, D)
        weight_ptr: Scale weights (D,)
        out_ptr: Output tensor (N, D)
        N: Number of rows
        D: Hidden dimension
        eps: Epsilon for numerical stability

    Grid: (N,)
    Threadgroup: 256
    """
    row_idx = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()
    simd_group = mt.simd_group_id()
    block_size = mt.threads_per_threadgroup()

    if row_idx >= N:
        return

    row_offset = row_idx * D

    shared = mt.shared_memory(8, dtype=mt.float32)

    # Compute sum of squares of (x + residual)
    local_ss = 0.0
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        res_val = mt.load(residual_ptr + row_offset + d)
        fused = x_val + res_val
        local_ss = local_ss + fused * fused

    # SIMD reduction
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 16)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 8)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 4)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 2)
    local_ss = local_ss + mt.simd_shuffle_down(local_ss, 1)

    if simd_lane == 0:
        shared[simd_group] = local_ss
    mt.threadgroup_barrier()

    if tid < 8:
        total = shared[tid]
        total = total + mt.simd_shuffle_down(total, 4)
        total = total + mt.simd_shuffle_down(total, 2)
        total = total + mt.simd_shuffle_down(total, 1)
        if tid == 0:
            rms = mt.sqrt(total / D + eps)
            shared[0] = 1.0 / rms
    mt.threadgroup_barrier()

    inv_rms = mt.load(shared + 0)

    # Normalize and apply weight
    for d in range(tid, D, block_size):
        x_val = mt.load(x_ptr + row_offset + d)
        res_val = mt.load(residual_ptr + row_offset + d)
        fused = x_val + res_val
        w = mt.load(weight_ptr + d)
        out_val = fused * inv_rms * w
        mt.store(out_ptr + row_offset + d, out_val)
