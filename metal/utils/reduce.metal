// Reduction Operations Metal Kernels
// Efficient parallel reduction primitives for sum, max, mean, etc.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// Sum Reductions
// ============================================================================

// Sum reduction along the last dimension
kernel void reduce_sum_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],      // Product of all dims except last
    constant uint& reduce_size [[buffer(3)]],     // Size of last dimension
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    // Each thread sums a portion of the reduction dimension
    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_sum += x[base_offset + i];
    }

    // SIMD reduction
    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroups
    float total = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total = shared_mem[simd_lane_id];
    }
    total = simd_sum(total);

    if (tid == 0) {
        out[outer_idx] = total;
    }
}

// Sum reduction along a specified axis
kernel void reduce_sum_axis(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],      // Product of dims before axis
    constant uint& reduce_size [[buffer(3)]],     // Size of axis dimension
    constant uint& inner_size [[buffer(4)]],      // Product of dims after axis
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid.y;
    uint inner_idx = bid.x * 256 + tid.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Sum along the reduction axis
    float local_sum = 0.0f;
    for (uint r = 0; r < reduce_size; r++) {
        uint idx = outer_idx * (reduce_size * inner_size) + r * inner_size + inner_idx;
        local_sum += x[idx];
    }

    // Write result
    out[outer_idx * inner_size + inner_idx] = local_sum;
}

// ============================================================================
// Max Reductions
// ============================================================================

// Max reduction along the last dimension
kernel void reduce_max_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    float local_max = -INFINITY;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_max = max(local_max, x[base_offset + i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_max = -INFINITY;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_max = shared_mem[simd_lane_id];
    }
    total_max = simd_max(total_max);

    if (tid == 0) {
        out[outer_idx] = total_max;
    }
}

// Max reduction with argmax (returns both max value and index)
kernel void reduce_max_argmax(
    device const float* x [[buffer(0)]],
    device float* max_out [[buffer(1)]],
    device uint* argmax_out [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& reduce_size [[buffer(4)]],
    threadgroup float* shared_vals [[threadgroup(0)]],
    threadgroup uint* shared_idxs [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    float local_max = -INFINITY;
    uint local_argmax = 0;

    for (uint i = tid; i < reduce_size; i += local_size) {
        float val = x[base_offset + i];
        if (val > local_max) {
            local_max = val;
            local_argmax = i;
        }
    }

    // Store to shared memory
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_argmax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        max_out[outer_idx] = shared_vals[0];
        argmax_out[outer_idx] = shared_idxs[0];
    }
}

// ============================================================================
// Min Reductions
// ============================================================================

// Min reduction along the last dimension
kernel void reduce_min_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    float local_min = INFINITY;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_min = min(local_min, x[base_offset + i]);
    }

    local_min = simd_min(local_min);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_min;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_min = INFINITY;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_min = shared_mem[simd_lane_id];
    }
    total_min = simd_min(total_min);

    if (tid == 0) {
        out[outer_idx] = total_min;
    }
}

// ============================================================================
// Mean Reductions
// ============================================================================

// Mean reduction along the last dimension
kernel void reduce_mean_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_sum += x[base_offset + i];
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total = shared_mem[simd_lane_id];
    }
    total = simd_sum(total);

    if (tid == 0) {
        out[outer_idx] = total / float(reduce_size);
    }
}

// ============================================================================
// Variance/Std Reductions
// ============================================================================

// Variance reduction (uses Welford's algorithm for numerical stability)
kernel void reduce_var_last(
    device const float* x [[buffer(0)]],
    device float* var_out [[buffer(1)]],
    device float* mean_out [[buffer(2)]],         // Optional, can be null
    constant uint& outer_size [[buffer(3)]],
    constant uint& reduce_size [[buffer(4)]],
    constant bool& unbiased [[buffer(5)]],        // Use N-1 divisor if true
    threadgroup float* shared_sum [[threadgroup(0)]],
    threadgroup float* shared_sum_sq [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    // Two-pass for numerical stability
    // Pass 1: Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_sum += x[base_offset + i];
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum = shared_sum[simd_lane_id];
    }
    total_sum = simd_sum(total_sum);

    float mean = total_sum / float(reduce_size);

    if (tid == 0) {
        shared_sum[0] = mean;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    mean = shared_sum[0];

    // Pass 2: Compute sum of squared deviations
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        float diff = x[base_offset + i] - mean;
        local_sum_sq += diff * diff;
    }

    local_sum_sq = simd_sum(local_sum_sq);

    if (simd_lane_id == 0) {
        shared_sum_sq[simd_group_id] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum_sq = shared_sum_sq[simd_lane_id];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    if (tid == 0) {
        float divisor = unbiased ? float(reduce_size - 1) : float(reduce_size);
        var_out[outer_idx] = total_sum_sq / divisor;
        if (mean_out) {
            mean_out[outer_idx] = mean;
        }
    }
}

// ============================================================================
// Prod Reductions
// ============================================================================

// Product reduction along the last dimension
kernel void reduce_prod_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    float local_prod = 1.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_prod *= x[base_offset + i];
    }

    // Note: SIMD doesn't have built-in product, need custom reduction
    shared_mem[tid] = local_prod;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for product
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] *= shared_mem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[outer_idx] = shared_mem[0];
    }
}

// ============================================================================
// LogSumExp Reduction (numerically stable)
// ============================================================================

kernel void reduce_logsumexp_last(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& reduce_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * reduce_size;

    // Find max first
    float local_max = -INFINITY;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_max = max(local_max, x[base_offset + i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -INFINITY;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_max = shared_mem[simd_lane_id];
    }
    global_max = simd_max(global_max);

    if (tid == 0) {
        shared_mem[0] = global_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_max = shared_mem[0];

    // Sum exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += local_size) {
        local_sum += exp(x[base_offset + i] - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum = shared_mem[simd_lane_id];
    }
    total_sum = simd_sum(total_sum);

    if (tid == 0) {
        out[outer_idx] = global_max + log(total_sum);
    }
}
