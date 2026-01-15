// RMSNorm (Root Mean Square Layer Normalization) Metal Kernel
// Single-pass fused implementation for maximum efficiency
//
// Based on "Root Mean Square Layer Normalization" by Zhang & Sennrich, 2019

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Single-pass RMSNorm kernel
// Computes RMS in parallel using SIMD reductions, then normalizes in-place
kernel void rmsnorm_forward(
    device const float* x [[buffer(0)]],          // [batch, seq_len, hidden_dim]
    device float* out [[buffer(1)]],
    device const float* weight [[buffer(2)]],     // [hidden_dim] - learnable scale
    constant uint& batch_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = bid.y;
    uint seq_idx = bid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint local_id = tid.x;
    uint local_size = 256;  // Assumed threadgroup size

    // Step 1: Compute sum of squares in parallel
    float local_sum_sq = 0.0f;
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = x[row_offset + i];
        local_sum_sq += val * val;
    }

    // SIMD reduction within simdgroup
    local_sum_sq = simd_sum(local_sum_sq);

    // Store simdgroup results to shared memory
    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroups (only first simdgroup)
    float total_sum_sq = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum_sq = shared_mem[simd_lane_id];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    // Broadcast RMS to all threads
    if (local_id == 0) {
        shared_mem[0] = rsqrt(total_sum_sq / float(hidden_dim) + eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_mem[0];

    // Step 2: Normalize and scale
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = x[row_offset + i];
        out[row_offset + i] = val * rms_inv * weight[i];
    }
}

// In-place RMSNorm (modifies input directly)
kernel void rmsnorm_inplace(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = bid.y;
    uint seq_idx = bid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint local_id = tid.x;
    uint local_size = 256;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = x[row_offset + i];
        local_sum_sq += val * val;
    }

    local_sum_sq = simd_sum(local_sum_sq);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum_sq = shared_mem[simd_lane_id];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    if (local_id == 0) {
        shared_mem[0] = rsqrt(total_sum_sq / float(hidden_dim) + eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_mem[0];

    // Normalize in-place
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        x[row_offset + i] = x[row_offset + i] * rms_inv * weight[i];
    }
}

// RMSNorm backward pass - compute gradients
kernel void rmsnorm_backward(
    device const float* grad_out [[buffer(0)]],   // [batch, seq_len, hidden_dim]
    device const float* x [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* rms_inv [[buffer(3)]],    // [batch, seq_len] - cached 1/rms from forward
    device float* grad_x [[buffer(4)]],
    device float* grad_weight [[buffer(5)]],      // [hidden_dim] - accumulated
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = bid.y;
    uint seq_idx = bid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint rms_idx = batch_idx * seq_len + seq_idx;
    float rms_inv_val = rms_inv[rms_idx];

    uint local_id = tid.x;
    uint local_size = 256;

    // Compute partial sums for gradient
    // grad_x = (grad_out * weight - x * mean(grad_out * weight * x) / hidden_dim) * rms_inv
    float local_sum = 0.0f;
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float grad_val = grad_out[row_offset + i];
        float x_val = x[row_offset + i];
        float w_val = weight[i];
        local_sum += grad_val * w_val * x_val;
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum = shared_mem[simd_lane_id];
    }
    total_sum = simd_sum(total_sum);

    if (local_id == 0) {
        shared_mem[0] = total_sum * rms_inv_val * rms_inv_val / float(hidden_dim);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float coeff = shared_mem[0];

    // Compute grad_x
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float grad_val = grad_out[row_offset + i];
        float x_val = x[row_offset + i];
        float w_val = weight[i];

        grad_x[row_offset + i] = rms_inv_val * (grad_val * w_val - x_val * coeff);

        // Accumulate grad_weight (needs atomic or separate reduction)
        // Note: In practice, grad_weight would be computed separately
    }
}

// Half-precision RMSNorm for memory efficiency
kernel void rmsnorm_forward_half(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = bid.y;
    uint seq_idx = bid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint local_id = tid.x;
    uint local_size = 256;

    // Compute sum of squares in float for precision
    float local_sum_sq = 0.0f;
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = float(x[row_offset + i]);
        local_sum_sq += val * val;
    }

    local_sum_sq = simd_sum(local_sum_sq);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum_sq = shared_mem[simd_lane_id];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    if (local_id == 0) {
        shared_mem[0] = rsqrt(total_sum_sq / float(hidden_dim) + eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_mem[0];

    // Normalize and scale
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = float(x[row_offset + i]);
        float w = float(weight[i]);
        out[row_offset + i] = half(val * rms_inv * w);
    }
}

// Fused RMSNorm + residual connection
// Computes: out = RMSNorm(x + residual)
kernel void rmsnorm_residual(
    device const float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_idx = bid.y;
    uint seq_idx = bid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    uint row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim;
    uint local_id = tid.x;
    uint local_size = 256;

    // First pass: compute sum of squares of (x + residual)
    float local_sum_sq = 0.0f;
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = x[row_offset + i] + residual[row_offset + i];
        local_sum_sq += val * val;
    }

    local_sum_sq = simd_sum(local_sum_sq);

    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    uint num_simdgroups = (local_size + 31) / 32;
    if (simd_group_id == 0 && simd_lane_id < num_simdgroups) {
        total_sum_sq = shared_mem[simd_lane_id];
    }
    total_sum_sq = simd_sum(total_sum_sq);

    if (local_id == 0) {
        shared_mem[0] = rsqrt(total_sum_sq / float(hidden_dim) + eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = shared_mem[0];

    // Second pass: normalize and scale
    for (uint i = local_id; i < hidden_dim; i += local_size) {
        float val = x[row_offset + i] + residual[row_offset + i];
        out[row_offset + i] = val * rms_inv * weight[i];
    }
}
