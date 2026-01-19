// Fused Sliding Window Attention
//
// Efficient attention where each query only attends to a local window of keys.
// Instead of computing full O(n²) attention and masking, we iterate only over
// the valid KV range for each query position.
//
// Memory: O(n * window_size) instead of O(n²)
// Compute: O(n * window_size * d) instead of O(n² * d)

#include <metal_stdlib>
using namespace metal;

// Online softmax state for numerically stable attention
struct OnlineSoftmax {
    float max_val;
    float sum_exp;

    OnlineSoftmax() : max_val(-INFINITY), sum_exp(0.0f) {}

    void update(float score) {
        if (score > max_val) {
            sum_exp = sum_exp * exp(max_val - score) + 1.0f;
            max_val = score;
        } else {
            sum_exp += exp(score - max_val);
        }
    }

    float normalize(float score) {
        return exp(score - max_val) / sum_exp;
    }
};

// ============================================================================
// Sliding Window Attention - Simple Version
// Each thread handles one (batch, head, query_pos) output row
// ============================================================================

kernel void sliding_window_attention_simple(
    device const float* Q [[buffer(0)]],        // (batch, seq_q, heads, dim)
    device const float* K [[buffer(1)]],        // (batch, seq_kv, heads, dim)
    device const float* V [[buffer(2)]],        // (batch, seq_kv, heads, dim)
    device float* O [[buffer(3)]],              // (batch, seq_q, heads, dim)
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& window_size [[buffer(9)]],   // One-sided window size
    constant bool& causal [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint q_pos = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_q) return;

    // Compute valid KV range for this query position
    uint kv_start, kv_end;
    if (causal) {
        // Causal: can only attend to positions <= q_pos, within window
        kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
        kv_end = q_pos + 1;  // Inclusive of current position
    } else {
        // Bidirectional: attend to window on both sides
        kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
        kv_end = min(q_pos + window_size + 1, seq_kv);
    }

    // Base offsets for this (batch, head)
    uint q_offset = batch_idx * seq_q * num_heads * head_dim +
                    q_pos * num_heads * head_dim +
                    head_idx * head_dim;
    uint kv_base = batch_idx * seq_kv * num_heads * head_dim +
                   head_idx * head_dim;

    // Online softmax accumulator
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Output accumulator (in registers)
    float acc[128];  // Assume head_dim <= 128
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // First pass: compute max for numerical stability
    for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        uint k_offset = kv_base + kv_pos * num_heads * head_dim;

        // Compute Q @ K^T for this position
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;

        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and accumulate output
    for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        uint k_offset = kv_base + kv_pos * num_heads * head_dim;
        uint v_offset = kv_base + kv_pos * num_heads * head_dim;

        // Compute score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;

        // Softmax weight
        float weight = exp(score - max_score);
        sum_exp += weight;

        // Accumulate weighted V
        for (uint d = 0; d < head_dim; d++) {
            acc[d] += weight * V[v_offset + d];
        }
    }

    // Normalize and write output
    uint o_offset = batch_idx * seq_q * num_heads * head_dim +
                    q_pos * num_heads * head_dim +
                    head_idx * head_dim;

    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (uint d = 0; d < head_dim; d++) {
        O[o_offset + d] = acc[d] * inv_sum;
    }
}

// ============================================================================
// Sliding Window Attention - Tiled Version with Shared Memory
// For longer sequences, uses tiled processing with shared memory for K/V
// ============================================================================

kernel void sliding_window_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& window_size [[buffer(9)]],
    constant bool& causal [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& BLOCK_SIZE [[buffer(12)]],
    threadgroup float* K_shared [[threadgroup(0)]],
    threadgroup float* V_shared [[threadgroup(1)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.z;
    uint head_idx = group_id.y;
    uint q_block_start = group_id.x * BLOCK_SIZE;
    uint local_tid = local_id.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    // Padded dimension for bank-conflict-free shared memory access
    uint head_dim_pad = head_dim + 4;

    // Each thread in the block handles one query position
    uint q_pos = q_block_start + local_tid;
    bool valid_q = q_pos < seq_q;

    // Compute valid KV range for this query block
    // We need to load K/V tiles that overlap with any query's window
    uint block_kv_start, block_kv_end;
    if (causal) {
        block_kv_start = (q_block_start >= window_size) ? (q_block_start - window_size) : 0;
        block_kv_end = min(q_block_start + BLOCK_SIZE, seq_kv);
    } else {
        block_kv_start = (q_block_start >= window_size) ? (q_block_start - window_size) : 0;
        block_kv_end = min(q_block_start + BLOCK_SIZE + window_size, seq_kv);
    }

    // Per-thread accumulators
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[128];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Load Q for this thread into registers
    float q_reg[128];
    uint q_offset = batch_idx * seq_q * num_heads * head_dim +
                    q_pos * num_heads * head_dim +
                    head_idx * head_dim;
    if (valid_q) {
        for (uint d = 0; d < head_dim; d++) {
            q_reg[d] = Q[q_offset + d];
        }
    }

    // Process KV in tiles
    uint kv_base = batch_idx * seq_kv * num_heads * head_dim + head_idx * head_dim;

    for (uint kv_tile_start = block_kv_start; kv_tile_start < block_kv_end; kv_tile_start += BLOCK_SIZE) {
        // Cooperative loading of K and V tiles into shared memory
        uint kv_tile_end = min(kv_tile_start + BLOCK_SIZE, seq_kv);
        uint tile_size = kv_tile_end - kv_tile_start;

        // Load K tile
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_idx = i / head_dim;
            uint d = i % head_dim;
            uint kv_pos = kv_tile_start + kv_idx;
            K_shared[kv_idx * head_dim_pad + d] = K[kv_base + kv_pos * num_heads * head_dim + d];
        }

        // Load V tile
        for (uint i = local_tid; i < tile_size * head_dim; i += group_size.x) {
            uint kv_idx = i / head_dim;
            uint d = i % head_dim;
            uint kv_pos = kv_tile_start + kv_idx;
            V_shared[kv_idx * head_dim_pad + d] = V[kv_base + kv_pos * num_heads * head_dim + d];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process this KV tile
        if (valid_q) {
            // Compute valid range for this query within this tile
            uint my_kv_start, my_kv_end;
            if (causal) {
                my_kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
                my_kv_end = q_pos + 1;
            } else {
                my_kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
                my_kv_end = min(q_pos + window_size + 1, seq_kv);
            }

            // Clamp to tile bounds
            uint tile_start_clamped = max(my_kv_start, kv_tile_start);
            uint tile_end_clamped = min(my_kv_end, kv_tile_end);

            for (uint kv_pos = tile_start_clamped; kv_pos < tile_end_clamped; kv_pos++) {
                uint kv_local = kv_pos - kv_tile_start;

                // Compute score
                float score = 0.0f;
                for (uint d = 0; d < head_dim; d++) {
                    score += q_reg[d] * K_shared[kv_local * head_dim_pad + d];
                }
                score *= scale;

                // Online softmax update
                if (score > max_score) {
                    float ratio = exp(max_score - score);
                    sum_exp = sum_exp * ratio + 1.0f;
                    for (uint d = 0; d < head_dim; d++) {
                        acc[d] *= ratio;
                    }
                    max_score = score;
                } else {
                    float weight = exp(score - max_score);
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
        uint o_offset = batch_idx * seq_q * num_heads * head_dim +
                        q_pos * num_heads * head_dim +
                        head_idx * head_dim;

        float inv_sum = 1.0f / (sum_exp + 1e-6f);
        for (uint d = 0; d < head_dim; d++) {
            O[o_offset + d] = acc[d] * inv_sum;
        }
    }
}

// ============================================================================
// Sliding Window Attention with RoPE
// Applies rotary position embeddings inside the attention kernel
// ============================================================================

kernel void sliding_window_attention_rope(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device const float* cos_cache [[buffer(4)]],  // (max_seq, dim/2)
    device const float* sin_cache [[buffer(5)]],  // (max_seq, dim/2)
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_q [[buffer(7)]],
    constant uint& seq_kv [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& window_size [[buffer(11)]],
    constant bool& causal [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint q_pos = tid.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_q) return;

    uint half_dim = head_dim / 2;

    // Load Q and apply RoPE
    uint q_offset = batch_idx * seq_q * num_heads * head_dim +
                    q_pos * num_heads * head_dim +
                    head_idx * head_dim;

    float q_rope[128];
    for (uint d = 0; d < half_dim; d++) {
        float q0 = Q[q_offset + d];
        float q1 = Q[q_offset + d + half_dim];
        float cos_val = cos_cache[q_pos * half_dim + d];
        float sin_val = sin_cache[q_pos * half_dim + d];

        q_rope[d] = q0 * cos_val - q1 * sin_val;
        q_rope[d + half_dim] = q0 * sin_val + q1 * cos_val;
    }

    // Compute valid KV range
    uint kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
    uint kv_end = causal ? (q_pos + 1) : min(q_pos + window_size + 1, seq_kv);

    // Attention computation
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[128];
    for (uint d = 0; d < head_dim; d++) acc[d] = 0.0f;

    uint kv_base = batch_idx * seq_kv * num_heads * head_dim + head_idx * head_dim;

    for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        uint k_offset = kv_base + kv_pos * num_heads * head_dim;

        // Apply RoPE to K and compute score
        float score = 0.0f;
        for (uint d = 0; d < half_dim; d++) {
            float k0 = K[k_offset + d];
            float k1 = K[k_offset + d + half_dim];
            float cos_val = cos_cache[kv_pos * half_dim + d];
            float sin_val = sin_cache[kv_pos * half_dim + d];

            float k_rope_d = k0 * cos_val - k1 * sin_val;
            float k_rope_d_half = k0 * sin_val + k1 * cos_val;

            score += q_rope[d] * k_rope_d + q_rope[d + half_dim] * k_rope_d_half;
        }
        score *= scale;

        // Update max
        max_score = max(max_score, score);
    }

    // Second pass with stable softmax
    for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        uint k_offset = kv_base + kv_pos * num_heads * head_dim;
        uint v_offset = kv_base + kv_pos * num_heads * head_dim;

        // Recompute score with RoPE
        float score = 0.0f;
        for (uint d = 0; d < half_dim; d++) {
            float k0 = K[k_offset + d];
            float k1 = K[k_offset + d + half_dim];
            float cos_val = cos_cache[kv_pos * half_dim + d];
            float sin_val = sin_cache[kv_pos * half_dim + d];

            float k_rope_d = k0 * cos_val - k1 * sin_val;
            float k_rope_d_half = k0 * sin_val + k1 * cos_val;

            score += q_rope[d] * k_rope_d + q_rope[d + half_dim] * k_rope_d_half;
        }
        score *= scale;

        float weight = exp(score - max_score);
        sum_exp += weight;

        for (uint d = 0; d < head_dim; d++) {
            acc[d] += weight * V[v_offset + d];
        }
    }

    // Write output
    uint o_offset = batch_idx * seq_q * num_heads * head_dim +
                    q_pos * num_heads * head_dim +
                    head_idx * head_dim;

    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (uint d = 0; d < head_dim; d++) {
        O[o_offset + d] = acc[d] * inv_sum;
    }
}
