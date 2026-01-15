// Softmax Operations Metal Kernels
// Numerically stable softmax and variants

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// Standard Softmax
// ============================================================================

// Softmax along the last dimension (most common case)
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
kernel void softmax_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],      // Product of all dims except last
    constant uint& softmax_size [[buffer(3)]],    // Size of softmax dimension
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Step 1: Find max for numerical stability
    float local_max = -INFINITY;
    for (uint i = tid; i < softmax_size; i += local_size) {
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

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_sum += exp(x[base_offset + i] - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float inv_sum = 1.0f / global_sum;

    // Step 3: Normalize
    for (uint i = tid; i < softmax_size; i += local_size) {
        out[base_offset + i] = exp(x[base_offset + i] - global_max) * inv_sum;
    }
}

// Softmax backward pass
// grad_x = softmax * (grad_out - sum(grad_out * softmax))
kernel void softmax_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* softmax_out [[buffer(1)]],  // Forward output
    device float* grad_x [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& softmax_size [[buffer(4)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Compute sum(grad_out * softmax)
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_sum += grad_out[base_offset + i] * softmax_out[base_offset + i];
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];

    // Compute gradient
    for (uint i = tid; i < softmax_size; i += local_size) {
        float s = softmax_out[base_offset + i];
        grad_x[base_offset + i] = s * (grad_out[base_offset + i] - global_sum);
    }
}

// In-place softmax
kernel void softmax_inplace(
    device float* x [[buffer(0)]],
    constant uint& outer_size [[buffer(1)]],
    constant uint& softmax_size [[buffer(2)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < softmax_size; i += local_size) {
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

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        float exp_val = exp(x[base_offset + i] - global_max);
        x[base_offset + i] = exp_val;  // Store exp temporarily
        local_sum += exp_val;
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float inv_sum = 1.0f / global_sum;

    // Normalize in-place
    for (uint i = tid; i < softmax_size; i += local_size) {
        x[base_offset + i] *= inv_sum;
    }
}

// ============================================================================
// Log Softmax
// ============================================================================

// Log softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
kernel void log_softmax_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& softmax_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < softmax_size; i += local_size) {
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

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_sum += exp(x[base_offset + i] - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float log_sum_exp = global_max + log(global_sum);

    // Compute log softmax
    for (uint i = tid; i < softmax_size; i += local_size) {
        out[base_offset + i] = x[base_offset + i] - log_sum_exp;
    }
}

// ============================================================================
// Softmax with Masking (for attention)
// ============================================================================

// Masked softmax: apply mask before softmax
kernel void softmax_masked(
    device const float* x [[buffer(0)]],
    device const float* mask [[buffer(1)]],       // 0 = keep, -inf = mask
    device float* out [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_k [[buffer(6)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_head_idx = bid.y;
    uint query_idx = bid.x;
    uint batch_idx = batch_head_idx / num_heads;
    uint head_idx = batch_head_idx % num_heads;

    if (batch_idx >= batch_size || query_idx >= seq_q) return;

    uint local_size = 256;
    uint row_offset = batch_head_idx * (seq_q * seq_k) + query_idx * seq_k;
    uint mask_offset = query_idx * seq_k;  // Broadcast mask across batch/heads

    // Find max (respecting mask)
    float local_max = -INFINITY;
    for (uint i = tid.x; i < seq_k; i += local_size) {
        float val = x[row_offset + i] + mask[mask_offset + i];
        local_max = max(local_max, val);
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

    if (tid.x == 0) {
        shared_mem[0] = global_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_max = shared_mem[0];

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint i = tid.x; i < seq_k; i += local_size) {
        float val = x[row_offset + i] + mask[mask_offset + i];
        local_sum += exp(val - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid.x == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float inv_sum = 1.0f / global_sum;

    // Normalize
    for (uint i = tid.x; i < seq_k; i += local_size) {
        float val = x[row_offset + i] + mask[mask_offset + i];
        out[row_offset + i] = exp(val - global_max) * inv_sum;
    }
}

// ============================================================================
// Half-Precision Softmax
// ============================================================================

kernel void softmax_forward_half(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& softmax_size [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Find max (using float for precision)
    float local_max = -INFINITY;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_max = max(local_max, float(x[base_offset + i]));
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

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_sum += exp(float(x[base_offset + i]) - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float inv_sum = 1.0f / global_sum;

    // Normalize (convert back to half)
    for (uint i = tid; i < softmax_size; i += local_size) {
        out[base_offset + i] = half(exp(float(x[base_offset + i]) - global_max) * inv_sum);
    }
}

// ============================================================================
// Fused Softmax + Scale (for attention scores)
// ============================================================================

// scale * softmax(x) in one pass
kernel void softmax_scale_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& softmax_size [[buffer(3)]],
    constant float& scale [[buffer(4)]],          // Usually 1/sqrt(d_k)
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint outer_idx = bid;
    if (outer_idx >= outer_size) return;

    uint local_size = 256;
    uint base_offset = outer_idx * softmax_size;

    // Find max of scaled values
    float local_max = -INFINITY;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_max = max(local_max, x[base_offset + i] * scale);
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

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint i = tid; i < softmax_size; i += local_size) {
        local_sum += exp(x[base_offset + i] * scale - global_max);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        global_sum = shared_mem[simd_lane_id];
    }
    global_sum = simd_sum(global_sum);

    if (tid == 0) {
        shared_mem[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    global_sum = shared_mem[0];
    float inv_sum = 1.0f / global_sum;

    // Normalize
    for (uint i = tid; i < softmax_size; i += local_size) {
        out[base_offset + i] = exp(x[base_offset + i] * scale - global_max) * inv_sum;
    }
}
