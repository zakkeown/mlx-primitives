// Chunked Cross-Attention Metal Kernel
//
// For very long KV sequences that don't fit in memory.
// Processes KV in chunks while Q stays resident in registers/shared memory.
//
// Memory: O(seq_q * chunk_size) instead of O(seq_q * seq_kv)
// Use case: Attending to 100K+ token documents

#include <metal_stdlib>
using namespace metal;

// Maximum supported head dimension (with padding)
#define MAX_HEAD_DIM_PAD 132  // 128 + 4

// ============================================================================
// Chunked Cross-Attention - Simple Version
// Each thread handles one query, streams through KV chunks
// ============================================================================

kernel void chunked_cross_attention_simple(
    device const float* Q [[buffer(0)]],        // (batch, seq_q, heads, dim)
    device const float* K [[buffer(1)]],        // (batch, seq_kv, heads, dim)
    device const float* V [[buffer(2)]],        // (batch, seq_kv, heads, dim)
    device float* O [[buffer(3)]],              // (batch, seq_q, heads, dim)
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& chunk_size [[buffer(9)]],
    constant bool& causal [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint q_pos = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_q) return;

    // Compute base offsets
    uint q_stride = num_heads * head_dim;
    uint kv_stride = num_heads * head_dim;

    uint q_offset = batch_idx * seq_q * q_stride + q_pos * q_stride + head_idx * head_dim;
    uint kv_base = batch_idx * seq_kv * kv_stride + head_idx * head_dim;

    // Initialize online softmax state
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM_PAD];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Load Q into registers
    float q_reg[MAX_HEAD_DIM_PAD];
    for (uint d = 0; d < head_dim; d++) {
        q_reg[d] = Q[q_offset + d];
    }

    // KV limit for causal masking
    uint kv_limit = causal ? (q_pos + 1) : seq_kv;

    // Process all KV positions
    for (uint kv_pos = 0; kv_pos < kv_limit; kv_pos++) {
        uint k_offset = kv_base + kv_pos * kv_stride;
        uint v_offset = kv_base + kv_pos * kv_stride;

        // Compute attention score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_reg[d] * K[k_offset + d];
        }
        score *= scale;

        // Online softmax update
        if (score > max_val) {
            float ratio = exp(max_val - score);
            sum_exp = sum_exp * ratio + 1.0f;
            for (uint d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * ratio + V[v_offset + d];
            }
            max_val = score;
        } else {
            float weight = exp(score - max_val);
            sum_exp += weight;
            for (uint d = 0; d < head_dim; d++) {
                acc[d] += weight * V[v_offset + d];
            }
        }
    }

    // Normalize and write output
    uint o_offset = batch_idx * seq_q * q_stride + q_pos * q_stride + head_idx * head_dim;
    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (uint d = 0; d < head_dim; d++) {
        O[o_offset + d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Chunked Cross-Attention - Tiled Version with Shared Memory
// Q block stays in shared memory, KV streams through
// ============================================================================

kernel void chunked_cross_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& block_q [[buffer(9)]],
    constant uint& chunk_size [[buffer(10)]],
    constant bool& causal [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    threadgroup float* Q_shared [[threadgroup(0)]],   // block_q * head_dim_pad
    threadgroup float* K_shared [[threadgroup(1)]],   // chunk_size * head_dim_pad
    threadgroup float* V_shared [[threadgroup(2)]],   // chunk_size * head_dim_pad
    uint3 group_id [[threadgroup_position_in_grid]],   // (q_block, head, batch)
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.z;
    uint head_idx = group_id.y;
    uint q_block_start = group_id.x * block_q;
    uint local_tid = local_id.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Each thread in the block handles one query
    uint q_pos = q_block_start + local_tid;
    bool valid_q = q_pos < seq_q;

    // Padded dimension for bank-conflict-free access
    uint head_dim_pad = head_dim + 4;

    // Compute base offsets
    uint q_stride = num_heads * head_dim;
    uint kv_stride = num_heads * head_dim;
    uint kv_base = batch_idx * seq_kv * kv_stride + head_idx * head_dim;

    // Per-thread online softmax state
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM_PAD];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Load Q block into shared memory (for potential reuse)
    // Then load into registers for this thread
    uint q_offset = batch_idx * seq_q * q_stride + q_pos * q_stride + head_idx * head_dim;
    float q_reg[MAX_HEAD_DIM_PAD];

    // Cooperative load Q block
    for (uint i = local_tid; i < block_q * head_dim; i += group_size.x) {
        uint q_local = i / head_dim;
        uint d = i % head_dim;
        uint q_global = q_block_start + q_local;
        if (q_global < seq_q) {
            Q_shared[q_local * head_dim_pad + d] = Q[batch_idx * seq_q * q_stride + q_global * q_stride + head_idx * head_dim + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load my query into registers
    if (valid_q) {
        for (uint d = 0; d < head_dim; d++) {
            q_reg[d] = Q_shared[local_tid * head_dim_pad + d];
        }
    }

    // Process KV in chunks
    for (uint kv_start = 0; kv_start < seq_kv; kv_start += chunk_size) {
        uint kv_end = min(kv_start + chunk_size, seq_kv);
        uint tile_size = kv_end - kv_start;

        // Cooperative load K chunk
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            K_shared[kv_local * head_dim_pad + d] = K[kv_base + kv_global * kv_stride + d];
        }

        // Cooperative load V chunk
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            V_shared[kv_local * head_dim_pad + d] = V[kv_base + kv_global * kv_stride + d];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread processes its query against this KV chunk
        if (valid_q) {
            // Determine valid KV range for this query (causal masking)
            uint my_kv_start = kv_start;
            uint my_kv_end = causal ? min(kv_end, q_pos + 1) : kv_end;

            for (uint kv_global = my_kv_start; kv_global < my_kv_end; kv_global++) {
                uint kv_local = kv_global - kv_start;

                // Compute score
                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    score += q_reg[d] * K_shared[kv_local * head_dim_pad + d];
                }
                score *= scale;

                // Online softmax update
                if (score > max_val) {
                    float ratio = exp(max_val - score);
                    sum_exp = sum_exp * ratio + 1.0f;
                    for (uint d = 0; d < head_dim; d++) {
                        acc[d] = acc[d] * ratio + V_shared[kv_local * head_dim_pad + d];
                    }
                    max_val = score;
                } else {
                    float weight = exp(score - max_val);
                    sum_exp += weight;
                    for (uint d = 0; d < head_dim; d++) {
                        acc[d] += weight * V_shared[kv_local * head_dim_pad + d];
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    if (valid_q) {
        uint o_offset = batch_idx * seq_q * q_stride + q_pos * q_stride + head_idx * head_dim;
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (uint d = 0; d < head_dim; d++) {
            O[o_offset + d] = acc[d] * inv_sum;
        }
    }
}
