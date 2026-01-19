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

// Tiled fused MoE SwiGLU with shared memory optimization
// SwiGLU: output = down(silu(gate(x)) * up(x))
//
// Optimizations:
// 1. Cooperative loading of input tokens into shared memory
// 2. Tiled computation of gate/up projections
// 3. Hidden activations cached in shared memory
// 4. Tiled down projection with shared memory tiles
// 5. Atomic scatter for multi-expert routing
//
// Layout:
// - Threadgroup handles one token
// - TILE_K threads cooperatively compute gate[h]/up[h] for each h
// - Uses shared memory for input caching and intermediate results

// Tile sizes (configurable based on register/shared memory constraints)
constant constexpr uint MOE_TILE_K = 64;    // Input features per tile
constant constexpr uint MOE_TILE_H = 64;    // Hidden features per tile

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
    uint tid [[thread_position_in_threadgroup]],
    uint num_threads [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Shared memory layout:
    // [0, d_model): input token x (cached once)
    // [d_model, d_model + d_hidden): hidden activations (after SiLU * up)
    threadgroup float x_shared[4096];      // Input token (up to 4096 d_model)
    threadgroup float hidden_shared[8192];  // Hidden activations (up to 8192 d_hidden)

    uint token_idx_in_batch = group_id;
    if (token_idx_in_batch >= capacity) return;

    uint src_token = indices[token_idx_in_batch];
    float routing_weight = weights[token_idx_in_batch];

    // ===== Phase 1: Cooperative load input token into shared memory =====
    uint input_base = src_token * d_model;
    for (uint i = tid; i < d_model; i += num_threads) {
        x_shared[i] = input[input_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== Phase 2: Compute gate and up projections, apply SiLU =====
    // Each thread handles multiple hidden dimensions
    for (uint h = tid; h < d_hidden; h += num_threads) {
        float gate_h = 0.0f;
        float up_h = 0.0f;

        // Tiled dot product for this hidden dimension
        for (uint k = 0; k < d_model; k += MOE_TILE_K) {
            uint tile_size = min(MOE_TILE_K, d_model - k);

            // Accumulate dot products
            for (uint kk = 0; kk < tile_size; kk++) {
                float x_val = x_shared[k + kk];
                gate_h += gate_proj[h * d_model + k + kk] * x_val;
                up_h += up_proj[h * d_model + k + kk] * x_val;
            }
        }

        // SiLU activation: gate * sigmoid(gate)
        float silu_gate = gate_h / (1.0f + exp(-gate_h));

        // Store hidden activation (silu(gate) * up)
        hidden_shared[h] = silu_gate * up_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===== Phase 3: Compute down projection and scatter =====
    // Each thread handles multiple output dimensions
    for (uint out_d = tid; out_d < d_model; out_d += num_threads) {
        float acc = 0.0f;

        // Tiled dot product with hidden activations
        for (uint h = 0; h < d_hidden; h += MOE_TILE_H) {
            uint tile_size = min(MOE_TILE_H, d_hidden - h);

            for (uint hh = 0; hh < tile_size; hh++) {
                acc += down_proj[out_d * d_hidden + h + hh] * hidden_shared[h + hh];
            }
        }

        // Atomic scatter with routing weight (for top-k > 1)
        atomic_fetch_add_explicit(
            (device atomic_float*)&output[src_token * d_model + out_d],
            acc * routing_weight,
            memory_order_relaxed
        );
    }
}


// Optimized version for top-1 routing (no atomic operations needed)
// Also uses SIMD reduction for better performance on small dimensions
kernel void fused_moe_swiglu_top1(
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
    uint tid [[thread_position_in_threadgroup]],
    uint num_threads [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float x_shared[4096];
    threadgroup float hidden_shared[8192];

    uint token_idx_in_batch = group_id;
    if (token_idx_in_batch >= capacity) return;

    uint src_token = indices[token_idx_in_batch];
    float routing_weight = weights[token_idx_in_batch];

    // Phase 1: Cooperative load
    uint input_base = src_token * d_model;
    for (uint i = tid; i < d_model; i += num_threads) {
        x_shared[i] = input[input_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute hidden activations
    for (uint h = tid; h < d_hidden; h += num_threads) {
        float gate_h = 0.0f;
        float up_h = 0.0f;

        for (uint k = 0; k < d_model; k++) {
            float x_val = x_shared[k];
            gate_h += gate_proj[h * d_model + k] * x_val;
            up_h += up_proj[h * d_model + k] * x_val;
        }

        float silu_gate = gate_h / (1.0f + exp(-gate_h));
        hidden_shared[h] = silu_gate * up_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Down projection (non-atomic, direct write)
    for (uint out_d = tid; out_d < d_model; out_d += num_threads) {
        float acc = 0.0f;

        for (uint h = 0; h < d_hidden; h++) {
            acc += down_proj[out_d * d_hidden + h] * hidden_shared[h];
        }

        // Direct write (no atomic needed for top-1)
        output[src_token * d_model + out_d] = acc * routing_weight;
    }
}


// Legacy simple version for reference/fallback
kernel void fused_moe_swiglu_simple(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gate_proj [[buffer(2)]],
    device const float* up_proj [[buffer(3)]],
    device const float* down_proj [[buffer(4)]],
    device const uint* indices [[buffer(5)]],
    device const float* weights [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    constant uint& d_model [[buffer(8)]],
    constant uint& d_hidden [[buffer(9)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_size [[threads_per_threadgroup]]
) {
    uint token_idx_in_batch = group_id.x;
    if (token_idx_in_batch >= capacity) return;

    uint src_token = indices[token_idx_in_batch];
    float routing_weight = weights[token_idx_in_batch];

    uint local_tid = tid.x + tid.y * group_size.x;
    uint out_d = local_tid;
    if (out_d >= d_model) return;

    float acc = 0.0f;
    for (uint h = 0; h < d_hidden; h++) {
        float gate_h = 0.0f;
        float up_h = 0.0f;
        for (uint d = 0; d < d_model; d++) {
            float x_d = input[src_token * d_model + d];
            gate_h += gate_proj[h * d_model + d] * x_d;
            up_h += up_proj[h * d_model + d] * x_d;
        }

        float silu_gate = gate_h / (1.0f + exp(-gate_h));
        float hidden_h = silu_gate * up_h;
        acc += down_proj[out_d * d_hidden + h] * hidden_h;
    }

    atomic_fetch_add_explicit(
        (device atomic_float*)&output[src_token * d_model + out_d],
        acc * routing_weight,
        memory_order_relaxed
    );
}
