// Selective Gather/Scatter Operations for Mixture of Experts
//
// These kernels enable efficient sparse MoE routing without materializing
// full tensors. Instead of running all tokens through all experts and masking,
// we gather only the tokens that route to each expert, compute, and scatter back.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Selective Gather
// Extract tokens at specified indices from input tensor
// ============================================================================

kernel void selective_gather(
    device const float* input [[buffer(0)]],      // (n_tokens, dim)
    device const uint* indices [[buffer(1)]],     // (capacity,) - which tokens to gather
    device float* output [[buffer(2)]],           // (capacity, dim)
    constant uint& capacity [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    uint idx = tid.x;      // Index in output (0..capacity-1)
    uint d = tid.y;        // Dimension index (0..dim-1)

    if (idx >= capacity || d >= dim) return;

    uint src_idx = indices[idx];
    output[idx * dim + d] = input[src_idx * dim + d];
}

// Batched version: gather for multiple experts in one kernel
kernel void selective_gather_batched(
    device const float* input [[buffer(0)]],          // (n_tokens, dim)
    device const uint* all_indices [[buffer(1)]],     // (num_experts, max_capacity)
    device const uint* expert_counts [[buffer(2)]],   // (num_experts,) - actual count per expert
    device float* all_outputs [[buffer(3)]],          // (num_experts, max_capacity, dim)
    constant uint& num_experts [[buffer(4)]],
    constant uint& max_capacity [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint expert_idx = tid.z;
    uint token_idx = tid.x;
    uint d = tid.y;

    if (expert_idx >= num_experts || d >= dim) return;

    uint count = expert_counts[expert_idx];
    if (token_idx >= count) return;

    uint src_idx = all_indices[expert_idx * max_capacity + token_idx];
    uint dst_offset = expert_idx * max_capacity * dim + token_idx * dim + d;
    output[dst_offset] = input[src_idx * dim + d];
}

// ============================================================================
// Selective Scatter Add
// Accumulate values back into output at specified indices with weights
// Uses atomic operations for thread-safe accumulation
// ============================================================================

kernel void selective_scatter_add(
    device float* output [[buffer(0)]],           // (n_tokens, dim) - accumulator
    device const float* values [[buffer(1)]],     // (capacity, dim) - values to scatter
    device const uint* indices [[buffer(2)]],     // (capacity,) - where to scatter
    device const float* weights [[buffer(3)]],    // (capacity,) - routing weights
    constant uint& capacity [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint idx = tid.x;      // Index in values (0..capacity-1)
    uint d = tid.y;        // Dimension index (0..dim-1)

    if (idx >= capacity || d >= dim) return;

    uint dst_idx = indices[idx];
    float weight = weights[idx];
    float value = values[idx * dim + d] * weight;

    // Atomic add for thread-safe accumulation
    // Note: atomic_float requires Metal 2.4+
    atomic_fetch_add_explicit(
        (device atomic_float*)&output[dst_idx * dim + d],
        value,
        memory_order_relaxed
    );
}

// Non-atomic version for when we know there are no write conflicts
// (e.g., top-1 routing where each token goes to exactly one expert)
kernel void selective_scatter_add_noatomic(
    device float* output [[buffer(0)]],           // (n_tokens, dim)
    device const float* values [[buffer(1)]],     // (capacity, dim)
    device const uint* indices [[buffer(2)]],     // (capacity,)
    device const float* weights [[buffer(3)]],    // (capacity,)
    constant uint& capacity [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint idx = tid.x;
    uint d = tid.y;

    if (idx >= capacity || d >= dim) return;

    uint dst_idx = indices[idx];
    float weight = weights[idx];
    output[dst_idx * dim + d] += values[idx * dim + d] * weight;
}

// ============================================================================
// Build Expert Dispatch Indices
// Given routing decisions (which expert each token goes to), build per-expert
// index lists for efficient gather operations
// ============================================================================

kernel void build_dispatch_indices(
    device const uint* expert_assignments [[buffer(0)]],  // (n_tokens, top_k) - expert idx per token
    device const float* gate_weights [[buffer(1)]],       // (n_tokens, top_k) - routing weights
    device uint* expert_indices [[buffer(2)]],            // (num_experts, max_capacity) - output indices
    device float* expert_weights [[buffer(3)]],           // (num_experts, max_capacity) - output weights
    device atomic_uint* expert_counts [[buffer(4)]],      // (num_experts,) - count per expert
    constant uint& n_tokens [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant uint& num_experts [[buffer(7)]],
    constant uint& max_capacity [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_tokens * top_k) return;

    uint token_idx = tid / top_k;
    uint k = tid % top_k;

    uint expert_idx = expert_assignments[tid];
    float weight = gate_weights[tid];

    // Atomic increment to get position in expert's list
    uint pos = atomic_fetch_add_explicit(
        &expert_counts[expert_idx],
        1u,
        memory_order_relaxed
    );

    // Check capacity (drop tokens if expert is full)
    if (pos < max_capacity) {
        uint offset = expert_idx * max_capacity + pos;
        expert_indices[offset] = token_idx;
        expert_weights[offset] = weight;
    }
}

// ============================================================================
// Fused MoE Expert
// For experts with the same architecture, fuse gather + MLP + scatter
// This reduces memory traffic by not writing intermediate gather results
// ============================================================================

// Simple fused version for 2-layer MLP: gate * up -> down
// SwiGLU: output = down(silu(gate(x)) * up(x))
kernel void fused_moe_swiglu(
    device const float* input [[buffer(0)]],          // (n_tokens, d_model)
    device float* output [[buffer(1)]],               // (n_tokens, d_model)
    device const float* gate_proj [[buffer(2)]],      // (d_hidden, d_model)
    device const float* up_proj [[buffer(3)]],        // (d_hidden, d_model)
    device const float* down_proj [[buffer(4)]],      // (d_model, d_hidden)
    device const uint* indices [[buffer(5)]],         // (capacity,) - token indices
    device const float* weights [[buffer(6)]],        // (capacity,) - routing weights
    constant uint& capacity [[buffer(7)]],
    constant uint& d_model [[buffer(8)]],
    constant uint& d_hidden [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],     // For tiled matmul
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_size [[threads_per_threadgroup]]
) {
    // This is a simplified version - full implementation would use
    // tiled matrix multiplication for efficiency

    uint token_idx_in_batch = group_id.x;
    if (token_idx_in_batch >= capacity) return;

    uint src_token = indices[token_idx_in_batch];
    float routing_weight = weights[token_idx_in_batch];

    uint local_tid = tid.x + tid.y * group_size.x;

    // Each thread handles one output dimension
    uint out_d = local_tid;
    if (out_d >= d_model) return;

    // Compute: output = down(silu(gate(x)) * up(x))
    // gate(x) and up(x) are (d_hidden,) each
    // Then element-wise silu(gate) * up
    // Then down projection to (d_model,)

    float acc = 0.0f;
    for (uint h = 0; h < d_hidden; h++) {
        // Compute gate[h] = sum_d(gate_proj[h,d] * x[d])
        float gate_h = 0.0f;
        float up_h = 0.0f;
        for (uint d = 0; d < d_model; d++) {
            float x_d = input[src_token * d_model + d];
            gate_h += gate_proj[h * d_model + d] * x_d;
            up_h += up_proj[h * d_model + d] * x_d;
        }

        // SiLU activation: x * sigmoid(x)
        float silu_gate = gate_h / (1.0f + exp(-gate_h));
        float hidden_h = silu_gate * up_h;

        // Accumulate down projection
        acc += down_proj[out_d * d_hidden + h] * hidden_h;
    }

    // Scatter with routing weight (atomic for top-k > 1)
    atomic_fetch_add_explicit(
        (device atomic_float*)&output[src_token * d_model + out_d],
        acc * routing_weight,
        memory_order_relaxed
    );
}
