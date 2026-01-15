// FlashAttention Metal Kernel
// Memory-efficient attention using tiling and shared memory
//
// Based on "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
// by Dao et al., 2022

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Configuration constants
constant uint BLOCK_M [[function_constant(0)]];  // Query block size (default 64)
constant uint BLOCK_N [[function_constant(1)]];  // Key/Value block size (default 64)
constant uint HEAD_DIM [[function_constant(2)]]; // Head dimension

// Softmax scaling factor (1/sqrt(d_k))
constant float SOFTMAX_SCALE [[function_constant(3)]];

// Shared memory tile for loading Q, K, V blocks
struct AttentionTile {
    float data[64][64];
};

// Online softmax helper - maintains running max and sum
struct OnlineSoftmax {
    float max_val;
    float sum;

    void update(float new_max, float new_sum) {
        if (new_max > max_val) {
            float scale = exp(max_val - new_max);
            sum = sum * scale + new_sum;
            max_val = new_max;
        } else {
            sum += new_sum * exp(new_max - max_val);
        }
    }
};

// Flash Attention forward kernel
// Implements Algorithm 1 from FlashAttention paper
kernel void flash_attention_forward(
    device const float* Q [[buffer(0)]],          // [batch, seq_len, num_heads, head_dim]
    device const float* K [[buffer(1)]],          // [batch, seq_len, num_heads, head_dim]
    device const float* V [[buffer(2)]],          // [batch, seq_len, num_heads, head_dim]
    device float* O [[buffer(3)]],                // [batch, seq_len, num_heads, head_dim]
    device float* L [[buffer(4)]],                // [batch, seq_len, num_heads] - logsumexp for backward
    device float* M [[buffer(5)]],                // [batch, seq_len, num_heads] - max values for backward
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant bool& is_causal [[buffer(10)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Extract batch and head indices from block ID
    uint batch_idx = bid.z;
    uint head_idx = bid.y;
    uint block_row = bid.x;

    // Compute offsets into global memory
    uint batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Local thread position within the block
    uint local_row = tid.y;
    uint local_col = tid.x;

    // Shared memory layout:
    // [0, BLOCK_M * head_dim): Q tile
    // [BLOCK_M * head_dim, BLOCK_M * head_dim + BLOCK_N * head_dim): K tile
    // [BLOCK_M * head_dim + BLOCK_N * head_dim, ...): V tile
    threadgroup float* Q_tile = shared_mem;
    threadgroup float* K_tile = shared_mem + BLOCK_M * head_dim;
    threadgroup float* V_tile = K_tile + BLOCK_N * head_dim;
    threadgroup float* S_tile = V_tile + BLOCK_N * head_dim;

    // Initialize output accumulator and online softmax state
    float output_acc[64];  // Each thread accumulates part of the output
    OnlineSoftmax softmax_state;
    softmax_state.max_val = -INFINITY;
    softmax_state.sum = 0.0f;

    // Load Q block into shared memory
    uint q_row = block_row * BLOCK_M + local_row;
    for (uint d = local_col; d < head_dim; d += BLOCK_N) {
        if (q_row < seq_len) {
            Q_tile[local_row * head_dim + d] = Q[batch_head_offset + q_row * head_dim + d];
        } else {
            Q_tile[local_row * head_dim + d] = 0.0f;
        }
    }

    // Initialize output accumulator to zero
    for (uint d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over K/V blocks
    uint num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    uint max_kv_block = is_causal ? min((block_row + 1) * BLOCK_M / BLOCK_N + 1, num_kv_blocks) : num_kv_blocks;

    for (uint kv_block = 0; kv_block < max_kv_block; kv_block++) {
        // Load K block into shared memory
        uint k_col = kv_block * BLOCK_N + local_col;
        for (uint d = local_row; d < head_dim; d += BLOCK_M) {
            if (k_col < seq_len) {
                K_tile[local_col * head_dim + d] = K[batch_head_offset + k_col * head_dim + d];
            } else {
                K_tile[local_col * head_dim + d] = 0.0f;
            }
        }

        // Load V block into shared memory
        uint v_col = kv_block * BLOCK_N + local_col;
        for (uint d = local_row; d < head_dim; d += BLOCK_M) {
            if (v_col < seq_len) {
                V_tile[local_col * head_dim + d] = V[batch_head_offset + v_col * head_dim + d];
            } else {
                V_tile[local_col * head_dim + d] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Q @ K^T for this block
        float block_max = -INFINITY;
        float s_vals[64];

        for (uint j = 0; j < BLOCK_N; j++) {
            float dot_product = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot_product += Q_tile[local_row * head_dim + d] * K_tile[j * head_dim + d];
            }
            dot_product *= SOFTMAX_SCALE;

            // Apply causal mask if needed
            uint global_q_pos = block_row * BLOCK_M + local_row;
            uint global_k_pos = kv_block * BLOCK_N + j;
            if (is_causal && global_k_pos > global_q_pos) {
                dot_product = -INFINITY;
            }

            s_vals[j] = dot_product;
            block_max = max(block_max, dot_product);
        }

        // Compute softmax for this block with online correction
        float block_sum = 0.0f;
        for (uint j = 0; j < BLOCK_N; j++) {
            s_vals[j] = exp(s_vals[j] - block_max);
            block_sum += s_vals[j];
        }

        // Update online softmax state
        float old_max = softmax_state.max_val;
        softmax_state.update(block_max, block_sum);

        // Rescale previous output accumulator
        if (old_max != -INFINITY) {
            float scale = exp(old_max - softmax_state.max_val);
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] *= scale;
            }
        }

        // Compute attention @ V for this block and accumulate
        float attn_scale = exp(block_max - softmax_state.max_val);
        for (uint j = 0; j < BLOCK_N; j++) {
            float attn_weight = s_vals[j] * attn_scale;
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] += attn_weight * V_tile[j * head_dim + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize output by softmax sum
    float inv_sum = 1.0f / softmax_state.sum;
    for (uint d = 0; d < head_dim; d++) {
        output_acc[d] *= inv_sum;
    }

    // Write output to global memory
    uint out_row = block_row * BLOCK_M + local_row;
    if (out_row < seq_len) {
        for (uint d = 0; d < head_dim; d++) {
            O[batch_head_offset + out_row * head_dim + d] = output_acc[d];
        }

        // Store logsumexp for backward pass
        uint lm_offset = (batch_idx * num_heads + head_idx) * seq_len + out_row;
        L[lm_offset] = softmax_state.max_val + log(softmax_state.sum);
        M[lm_offset] = softmax_state.max_val;
    }
}

// Flash Attention backward kernel for dQ
kernel void flash_attention_backward_dq(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* O [[buffer(3)]],
    device const float* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device float* dQ [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant bool& is_causal [[buffer(11)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]]
) {
    // Similar structure to forward pass
    // Computes gradient with respect to Q
    uint batch_idx = bid.z;
    uint head_idx = bid.y;
    uint block_row = bid.x;

    uint batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    uint local_row = tid.y;

    // Load dO and O for this row
    uint row = block_row * BLOCK_M + local_row;
    if (row >= seq_len) return;

    float di = 0.0f;  // Sum of dO * O
    for (uint d = 0; d < head_dim; d++) {
        float do_val = dO[batch_head_offset + row * head_dim + d];
        float o_val = O[batch_head_offset + row * head_dim + d];
        di += do_val * o_val;
    }

    // Compute dQ by iterating over K/V blocks
    float dq_acc[64];
    for (uint d = 0; d < head_dim; d++) {
        dq_acc[d] = 0.0f;
    }

    float l_val = L[(batch_idx * num_heads + head_idx) * seq_len + row];

    uint num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        // Load K and V tiles (simplified - full implementation would tile)
        for (uint j = 0; j < BLOCK_N; j++) {
            uint k_pos = kv_block * BLOCK_N + j;
            if (k_pos >= seq_len) continue;
            if (is_causal && k_pos > row) continue;

            // Recompute attention weight
            float qk = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                qk += Q[batch_head_offset + row * head_dim + d] *
                      K[batch_head_offset + k_pos * head_dim + d];
            }
            qk *= SOFTMAX_SCALE;
            float p = exp(qk - l_val);

            // Compute dS = P * (dO @ V^T - Di)
            float dov = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dov += dO[batch_head_offset + row * head_dim + d] *
                       V[batch_head_offset + k_pos * head_dim + d];
            }
            float ds = p * (dov - di);

            // Accumulate dQ
            for (uint d = 0; d < head_dim; d++) {
                dq_acc[d] += ds * K[batch_head_offset + k_pos * head_dim + d] * SOFTMAX_SCALE;
            }
        }
    }

    // Write dQ to global memory
    for (uint d = 0; d < head_dim; d++) {
        dQ[batch_head_offset + row * head_dim + d] = dq_acc[d];
    }
}

// Half-precision version for memory efficiency
kernel void flash_attention_forward_half(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    device float* M [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant bool& is_causal [[buffer(10)]],
    threadgroup half* shared_mem [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Same algorithm as float version but uses half precision
    // for reduced memory bandwidth and increased throughput

    uint batch_idx = bid.z;
    uint head_idx = bid.y;
    uint block_row = bid.x;

    uint batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    uint local_row = tid.y;
    uint local_col = tid.x;

    // Shared memory for tiles
    threadgroup half* Q_tile = shared_mem;
    threadgroup half* K_tile = shared_mem + BLOCK_M * head_dim;
    threadgroup half* V_tile = K_tile + BLOCK_N * head_dim;

    // Use float for accumulation to maintain precision
    float output_acc[64];
    float max_val = -INFINITY;
    float sum_val = 0.0f;

    for (uint d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0f;
    }

    // Load Q tile
    uint q_row = block_row * BLOCK_M + local_row;
    for (uint d = local_col; d < head_dim; d += BLOCK_N) {
        if (q_row < seq_len) {
            Q_tile[local_row * head_dim + d] = Q[batch_head_offset + q_row * head_dim + d];
        } else {
            Q_tile[local_row * head_dim + d] = half(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process K/V blocks
    uint num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    uint max_kv_block = is_causal ? min((block_row + 1) * BLOCK_M / BLOCK_N + 1, num_kv_blocks) : num_kv_blocks;

    for (uint kv_block = 0; kv_block < max_kv_block; kv_block++) {
        // Load K and V tiles
        uint kv_col = kv_block * BLOCK_N + local_col;
        for (uint d = local_row; d < head_dim; d += BLOCK_M) {
            if (kv_col < seq_len) {
                K_tile[local_col * head_dim + d] = K[batch_head_offset + kv_col * head_dim + d];
                V_tile[local_col * head_dim + d] = V[batch_head_offset + kv_col * head_dim + d];
            } else {
                K_tile[local_col * head_dim + d] = half(0.0f);
                V_tile[local_col * head_dim + d] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for this block
        float block_max = -INFINITY;
        float s_vals[64];

        for (uint j = 0; j < BLOCK_N; j++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += float(Q_tile[local_row * head_dim + d]) * float(K_tile[j * head_dim + d]);
            }
            dot *= SOFTMAX_SCALE;

            uint global_q = block_row * BLOCK_M + local_row;
            uint global_k = kv_block * BLOCK_N + j;
            if (is_causal && global_k > global_q) {
                dot = -INFINITY;
            }

            s_vals[j] = dot;
            block_max = max(block_max, dot);
        }

        // Online softmax update
        float block_sum = 0.0f;
        for (uint j = 0; j < BLOCK_N; j++) {
            s_vals[j] = exp(s_vals[j] - block_max);
            block_sum += s_vals[j];
        }

        float old_max = max_val;
        if (block_max > max_val) {
            float scale = exp(max_val - block_max);
            sum_val = sum_val * scale + block_sum;
            max_val = block_max;
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] *= scale;
            }
        } else {
            sum_val += block_sum * exp(block_max - max_val);
        }

        // Accumulate output
        float attn_scale = exp(block_max - max_val);
        for (uint j = 0; j < BLOCK_N; j++) {
            float w = s_vals[j] * attn_scale;
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] += w * float(V_tile[j * head_dim + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float inv_sum = 1.0f / sum_val;
    uint out_row = block_row * BLOCK_M + local_row;
    if (out_row < seq_len) {
        for (uint d = 0; d < head_dim; d++) {
            O[batch_head_offset + out_row * head_dim + d] = half(output_acc[d] * inv_sum);
        }

        uint lm_offset = (batch_idx * num_heads + head_idx) * seq_len + out_row;
        L[lm_offset] = max_val + log(sum_val);
        M[lm_offset] = max_val;
    }
}
