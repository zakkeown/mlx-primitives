// Flash Attention Metal Kernel
//
// Memory-efficient exact attention via tiled online softmax.
// Never materializes the O(n²) attention matrix.
//
// Memory: O(n * d) instead of O(n²)
// Algorithm: FlashAttention (Dao et al., 2022)

#include <metal_stdlib>
using namespace metal;

// Maximum supported head dimension (with padding)
#define MAX_HEAD_DIM_PAD 132  // 128 + 4

// Online softmax state for numerically stable incremental computation
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
    float output[MAX_HEAD_DIM_PAD];

    // Initialize to identity state
    void init(uint head_dim) {
        max_val = -INFINITY;
        sum_exp = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            output[d] = 0.0f;
        }
    }

    // Update state with a new (score, value) pair
    void update(float score, device const float* v, uint head_dim, uint head_dim_pad) {
        if (score > max_val) {
            // New maximum: rescale accumulated values
            float ratio = exp(max_val - score);
            sum_exp = sum_exp * ratio + 1.0f;
            for (uint d = 0; d < head_dim; d++) {
                output[d] = output[d] * ratio + v[d];
            }
            max_val = score;
        } else {
            // Existing maximum: just accumulate
            float weight = exp(score - max_val);
            sum_exp += weight;
            for (uint d = 0; d < head_dim; d++) {
                output[d] += weight * v[d];
            }
        }
    }

    // Update state with value from shared memory (padded indexing)
    void update_shared(float score, threadgroup const float* v_shared, uint kv_idx, uint head_dim, uint head_dim_pad) {
        if (score > max_val) {
            float ratio = exp(max_val - score);
            sum_exp = sum_exp * ratio + 1.0f;
            for (uint d = 0; d < head_dim; d++) {
                output[d] = output[d] * ratio + v_shared[kv_idx * head_dim_pad + d];
            }
            max_val = score;
        } else {
            float weight = exp(score - max_val);
            sum_exp += weight;
            for (uint d = 0; d < head_dim; d++) {
                output[d] += weight * v_shared[kv_idx * head_dim_pad + d];
            }
        }
    }

    // Normalize output by sum of exponentials
    void normalize(uint head_dim) {
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (uint d = 0; d < head_dim; d++) {
            output[d] *= inv_sum;
        }
    }
};

// ============================================================================
// Flash Attention - Simple Version
// Each thread handles one query position
// ============================================================================

kernel void flash_attention_simple(
    device const float* Q [[buffer(0)]],        // (batch, seq, heads, dim)
    device const float* K [[buffer(1)]],        // (batch, seq, heads, dim)
    device const float* V [[buffer(2)]],        // (batch, seq, heads, dim)
    device float* O [[buffer(3)]],              // (batch, seq, heads, dim)
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant bool& causal [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint q_pos = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len) return;

    // Compute base offsets
    uint qkv_stride = num_heads * head_dim;
    uint q_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
    uint kv_base = batch_idx * seq_len * qkv_stride + head_idx * head_dim;

    // KV limit for causal masking
    uint kv_limit = causal ? (q_pos + 1) : seq_len;

    // Initialize online softmax state
    OnlineSoftmaxState state;
    state.init(head_dim);

    // Load query into registers
    float q_reg[MAX_HEAD_DIM_PAD];
    for (uint d = 0; d < head_dim; d++) {
        q_reg[d] = Q[q_offset + d];
    }

    // Process all valid KV positions
    for (uint kv_pos = 0; kv_pos < kv_limit; kv_pos++) {
        uint k_offset = kv_base + kv_pos * qkv_stride;
        uint v_offset = kv_base + kv_pos * qkv_stride;

        // Compute attention score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_reg[d] * K[k_offset + d];
        }
        score *= scale;

        // Update online softmax
        state.update(score, &V[v_offset], head_dim, head_dim);
    }

    // Normalize and write output
    state.normalize(head_dim);

    uint o_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        O[o_offset + d] = state.output[d];
    }
}

// ============================================================================
// Flash Attention - Tiled Version with Shared Memory
// Each threadgroup handles one query block across all KV
// ============================================================================

kernel void flash_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& block_kv [[buffer(8)]],
    constant bool& causal [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    threadgroup float* K_shared [[threadgroup(0)]],   // block_kv * head_dim_pad
    threadgroup float* V_shared [[threadgroup(1)]],   // block_kv * head_dim_pad
    uint3 group_id [[threadgroup_position_in_grid]],   // (q_pos, head, batch)
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.z;
    uint head_idx = group_id.y;
    uint q_pos = group_id.x;
    uint local_tid = local_id.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len) return;

    // Padded dimension for bank-conflict-free access
    uint head_dim_pad = head_dim + 4;

    // Compute base offsets
    uint qkv_stride = num_heads * head_dim;
    uint q_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
    uint kv_base = batch_idx * seq_len * qkv_stride + head_idx * head_dim;

    // KV limit for causal masking
    uint kv_limit = causal ? (q_pos + 1) : seq_len;

    // Initialize online softmax state (in registers)
    OnlineSoftmaxState state;
    state.init(head_dim);

    // Load query into registers (only first thread needs full Q, others help with K/V loading)
    float q_reg[MAX_HEAD_DIM_PAD];
    if (local_tid == 0) {
        for (uint d = 0; d < head_dim; d++) {
            q_reg[d] = Q[q_offset + d];
        }
    }

    // Process KV in tiles
    for (uint kv_start = 0; kv_start < kv_limit; kv_start += block_kv) {
        uint kv_end = min(kv_start + block_kv, kv_limit);
        uint tile_size = kv_end - kv_start;

        // Cooperative load K tile into shared memory
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            K_shared[kv_local * head_dim_pad + d] = K[kv_base + kv_global * qkv_stride + d];
        }

        // Cooperative load V tile into shared memory
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            V_shared[kv_local * head_dim_pad + d] = V[kv_base + kv_global * qkv_stride + d];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First thread processes the tile
        if (local_tid == 0) {
            for (uint kv_local = 0; kv_local < tile_size; kv_local++) {
                // Compute attention score
                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    score += q_reg[d] * K_shared[kv_local * head_dim_pad + d];
                }
                score *= scale;

                // Update online softmax
                state.update_shared(score, V_shared, kv_local, head_dim, head_dim_pad);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output (first thread only)
    if (local_tid == 0) {
        state.normalize(head_dim);

        uint o_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            O[o_offset + d] = state.output[d];
        }
    }
}

// ============================================================================
// Flash Attention - Block Parallel Version
// Each threadgroup handles a block of queries, each thread handles one query
// ============================================================================

kernel void flash_attention_block_parallel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& block_q [[buffer(8)]],
    constant uint& block_kv [[buffer(9)]],
    constant bool& causal [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    threadgroup float* K_shared [[threadgroup(0)]],
    threadgroup float* V_shared [[threadgroup(1)]],
    uint3 group_id [[threadgroup_position_in_grid]],    // (q_block, head, batch)
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.z;
    uint head_idx = group_id.y;
    uint q_block_start = group_id.x * block_q;
    uint local_tid = local_id.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Each thread handles one query in the block
    uint q_pos = q_block_start + local_tid;
    bool valid_q = q_pos < seq_len;

    // Padded dimension
    uint head_dim_pad = head_dim + 4;

    // Compute base offsets
    uint qkv_stride = num_heads * head_dim;
    uint q_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
    uint kv_base = batch_idx * seq_len * qkv_stride + head_idx * head_dim;

    // Per-thread online softmax state
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM_PAD];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Load query into registers
    float q_reg[MAX_HEAD_DIM_PAD];
    if (valid_q) {
        for (uint d = 0; d < head_dim; d++) {
            q_reg[d] = Q[q_offset + d];
        }
    }

    // KV limit: for causal, need to process up to max(q_pos) in this block
    uint kv_limit_global = causal ? min(q_block_start + block_q, seq_len) : seq_len;

    // Process KV in tiles
    for (uint kv_start = 0; kv_start < kv_limit_global; kv_start += block_kv) {
        uint kv_end = min(kv_start + block_kv, kv_limit_global);
        uint tile_size = kv_end - kv_start;

        // Cooperative load K tile
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            K_shared[kv_local * head_dim_pad + d] = K[kv_base + kv_global * qkv_stride + d];
        }

        // Cooperative load V tile
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_local = i / head_dim;
            uint d = i % head_dim;
            uint kv_global = kv_start + kv_local;
            V_shared[kv_local * head_dim_pad + d] = V[kv_base + kv_global * qkv_stride + d];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread processes its query against this KV tile
        if (valid_q) {
            uint my_kv_limit = causal ? min(kv_end, q_pos + 1) : kv_end;

            for (uint kv_global = kv_start; kv_global < my_kv_limit; kv_global++) {
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
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        uint o_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            O[o_offset + d] = acc[d] * inv_sum;
        }
    }
}
