// Fused RoPE + Flash Attention Metal Kernel
// Applies rotary position embeddings during tile load, eliminating separate RoPE pass
//
// Based on:
// - FlashAttention (Dao et al., 2022)
// - RoFormer (Su et al., 2021)

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

// Configuration constants (set via function constants or template)
constant uint BLOCK_M [[function_constant(0)]];   // Query block size (default 64)
constant uint BLOCK_N [[function_constant(1)]];   // KV block size (default 64)
constant uint HEAD_DIM [[function_constant(2)]];  // Head dimension
constant float SOFTMAX_SCALE [[function_constant(3)]];  // 1/sqrt(head_dim)
constant float ROPE_BASE [[function_constant(4)]];      // RoPE base (default 10000)

// Online softmax state for numerically stable streaming computation
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

// Compute RoPE theta for a dimension pair
// theta_i = base^(-2i/d) where i is dimension pair index
inline float compute_rope_theta(uint dim_pair, uint head_dim, float base) {
    return pow(base, -2.0f * float(dim_pair) / float(head_dim));
}

// Apply RoPE rotation to a pair of values
// Returns (x1*cos - x2*sin, x1*sin + x2*cos)
inline float2 apply_rope_rotation(float x1, float x2, float cos_val, float sin_val) {
    return float2(
        x1 * cos_val - x2 * sin_val,
        x1 * sin_val + x2 * cos_val
    );
}

// Main fused kernel: RoPE + Flash Attention forward pass
// Tensor layouts:
//   Q, K, V: [batch, seq_len, num_heads, head_dim]
//   O: [batch, seq_len, num_heads, head_dim]
//   cos_cache, sin_cache: [max_seq, head_dim/2] (optional, can be nullptr)
kernel void fused_rope_flash_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* L [[buffer(4)]],                  // logsumexp for backward
    device float* M [[buffer(5)]],                  // max values for backward
    device const float* cos_cache [[buffer(6)]],   // precomputed cos (optional)
    device const float* sin_cache [[buffer(7)]],   // precomputed sin (optional)
    constant uint& batch_size [[buffer(8)]],
    constant uint& seq_len_q [[buffer(9)]],
    constant uint& seq_len_kv [[buffer(10)]],
    constant uint& num_heads [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant bool& is_causal [[buffer(13)]],
    constant uint& q_offset [[buffer(14)]],        // position offset for Q
    constant uint& kv_offset [[buffer(15)]],       // position offset for K
    constant bool& use_cache [[buffer(16)]],       // use precomputed cos/sin
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Block indices: (q_block, head, batch)
    uint batch_idx = bid.z;
    uint head_idx = bid.y;
    uint block_row = bid.x;  // which Q block

    // Local thread position within block
    uint local_row = tid.y;
    uint local_col = tid.x;

    // Shared memory layout
    threadgroup float* Q_tile = shared_mem;                         // [BLOCK_M, head_dim]
    threadgroup float* K_tile = shared_mem + BLOCK_M * head_dim;    // [BLOCK_N, head_dim]
    threadgroup float* V_tile = K_tile + BLOCK_N * head_dim;        // [BLOCK_N, head_dim]

    // Global memory offsets for Q and KV (different seq lengths possible)
    uint q_batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    uint kv_batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len_kv * head_dim;

    // Global Q row for this thread
    uint q_row_global = block_row * BLOCK_M + local_row;
    uint q_position = q_row_global + q_offset;  // position for RoPE

    // Initialize output accumulator and softmax state
    float output_acc[64];  // sized for max head_dim
    for (uint d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0f;
    }

    OnlineSoftmax softmax_state;
    softmax_state.max_val = -INFINITY;
    softmax_state.sum = 0.0f;

    // ============ Load Q tile with fused RoPE ============
    // Each thread loads and rotates part of the Q row
    uint half_dim = head_dim / 2;

    if (q_row_global < seq_len_q) {
        for (uint dp = local_col; dp < half_dim; dp += BLOCK_N) {
            uint d1 = dp;           // first half dimension
            uint d2 = dp + half_dim; // second half dimension

            // Load raw Q values
            float q1 = Q[q_batch_head_offset + q_row_global * head_dim + d1];
            float q2 = Q[q_batch_head_offset + q_row_global * head_dim + d2];

            // Get cos/sin (from cache or compute on-the-fly)
            float cos_val, sin_val;
            if (use_cache) {
                cos_val = cos_cache[q_position * half_dim + dp];
                sin_val = sin_cache[q_position * half_dim + dp];
            } else {
                float theta = compute_rope_theta(dp, head_dim, ROPE_BASE);
                float angle = float(q_position) * theta;
                cos_val = cos(angle);
                sin_val = sin(angle);
            }

            // Apply RoPE and store to shared memory
            float2 rotated = apply_rope_rotation(q1, q2, cos_val, sin_val);
            Q_tile[local_row * head_dim + d1] = rotated.x;
            Q_tile[local_row * head_dim + d2] = rotated.y;
        }
    } else {
        // Pad with zeros for out-of-bounds
        for (uint d = local_col; d < head_dim; d += BLOCK_N) {
            Q_tile[local_row * head_dim + d] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ Iterate over KV blocks ============
    uint num_kv_blocks = (seq_len_kv + BLOCK_N - 1) / BLOCK_N;

    // For causal masking, limit KV blocks based on Q position
    uint max_kv_block;
    if (is_causal) {
        // Only process KV positions <= max Q position in this block
        uint max_q_pos = (block_row + 1) * BLOCK_M - 1 + q_offset;
        max_kv_block = min((max_q_pos + 1 - kv_offset + BLOCK_N - 1) / BLOCK_N, num_kv_blocks);
    } else {
        max_kv_block = num_kv_blocks;
    }

    for (uint kv_block = 0; kv_block < max_kv_block; kv_block++) {
        uint kv_col_global = kv_block * BLOCK_N + local_col;
        uint kv_position = kv_col_global + kv_offset;  // position for RoPE

        // ============ Load K tile with fused RoPE ============
        if (kv_col_global < seq_len_kv) {
            for (uint dp = local_row; dp < half_dim; dp += BLOCK_M) {
                uint d1 = dp;
                uint d2 = dp + half_dim;

                float k1 = K[kv_batch_head_offset + kv_col_global * head_dim + d1];
                float k2 = K[kv_batch_head_offset + kv_col_global * head_dim + d2];

                float cos_val, sin_val;
                if (use_cache) {
                    cos_val = cos_cache[kv_position * half_dim + dp];
                    sin_val = sin_cache[kv_position * half_dim + dp];
                } else {
                    float theta = compute_rope_theta(dp, head_dim, ROPE_BASE);
                    float angle = float(kv_position) * theta;
                    cos_val = cos(angle);
                    sin_val = sin(angle);
                }

                float2 rotated = apply_rope_rotation(k1, k2, cos_val, sin_val);
                K_tile[local_col * head_dim + d1] = rotated.x;
                K_tile[local_col * head_dim + d2] = rotated.y;
            }
        } else {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                K_tile[local_col * head_dim + d] = 0.0f;
            }
        }

        // ============ Load V tile (no RoPE needed) ============
        if (kv_col_global < seq_len_kv) {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                V_tile[local_col * head_dim + d] = V[kv_batch_head_offset + kv_col_global * head_dim + d];
            }
        } else {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                V_tile[local_col * head_dim + d] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ Compute Q @ K^T for this block ============
        float block_max = -INFINITY;
        float s_vals[64];  // attention scores for this KV block

        for (uint j = 0; j < BLOCK_N; j++) {
            float dot_product = 0.0f;

            // Q and K already have RoPE applied in shared memory
            for (uint d = 0; d < head_dim; d++) {
                dot_product += Q_tile[local_row * head_dim + d] * K_tile[j * head_dim + d];
            }

            dot_product *= SOFTMAX_SCALE;

            // Apply causal mask
            uint global_q_pos = q_position;
            uint global_k_pos = kv_block * BLOCK_N + j + kv_offset;
            if (is_causal && global_k_pos > global_q_pos) {
                dot_product = -INFINITY;
            }

            // Bounds check for padding
            if (kv_block * BLOCK_N + j >= seq_len_kv) {
                dot_product = -INFINITY;
            }

            s_vals[j] = dot_product;
            block_max = max(block_max, dot_product);
        }

        // ============ Online softmax update ============
        float block_sum = 0.0f;
        for (uint j = 0; j < BLOCK_N; j++) {
            s_vals[j] = exp(s_vals[j] - block_max);
            block_sum += s_vals[j];
        }

        float old_max = softmax_state.max_val;
        softmax_state.update(block_max, block_sum);

        // Rescale previous output accumulator
        if (old_max != -INFINITY) {
            float scale = exp(old_max - softmax_state.max_val);
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] *= scale;
            }
        }

        // ============ Accumulate attention @ V ============
        float attn_scale = exp(block_max - softmax_state.max_val);
        for (uint j = 0; j < BLOCK_N; j++) {
            float attn_weight = s_vals[j] * attn_scale;
            for (uint d = 0; d < head_dim; d++) {
                output_acc[d] += attn_weight * V_tile[j * head_dim + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============ Normalize and write output ============
    if (q_row_global < seq_len_q) {
        float inv_sum = 1.0f / softmax_state.sum;
        for (uint d = 0; d < head_dim; d++) {
            O[q_batch_head_offset + q_row_global * head_dim + d] = output_acc[d] * inv_sum;
        }

        // Store logsumexp and max for backward pass
        uint lm_offset = (batch_idx * num_heads + head_idx) * seq_len_q + q_row_global;
        L[lm_offset] = softmax_state.max_val + log(softmax_state.sum);
        M[lm_offset] = softmax_state.max_val;
    }
}

// Half-precision variant for memory efficiency
kernel void fused_rope_flash_attention_forward_half(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],                  // keep in float for precision
    device float* M [[buffer(5)]],
    device const half* cos_cache [[buffer(6)]],
    device const half* sin_cache [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& seq_len_q [[buffer(9)]],
    constant uint& seq_len_kv [[buffer(10)]],
    constant uint& num_heads [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant bool& is_causal [[buffer(13)]],
    constant uint& q_offset [[buffer(14)]],
    constant uint& kv_offset [[buffer(15)]],
    constant bool& use_cache [[buffer(16)]],
    threadgroup half* shared_mem [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Same algorithm as float version but uses half for storage
    // and float for accumulation to maintain numerical stability

    uint batch_idx = bid.z;
    uint head_idx = bid.y;
    uint block_row = bid.x;

    uint local_row = tid.y;
    uint local_col = tid.x;

    // Shared memory
    threadgroup half* Q_tile = shared_mem;
    threadgroup half* K_tile = shared_mem + BLOCK_M * head_dim;
    threadgroup half* V_tile = K_tile + BLOCK_N * head_dim;

    uint q_batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    uint kv_batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len_kv * head_dim;

    uint q_row_global = block_row * BLOCK_M + local_row;
    uint q_position = q_row_global + q_offset;

    // Use float for accumulation
    float output_acc[64];
    for (uint d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0f;
    }

    float max_val = -INFINITY;
    float sum_val = 0.0f;

    uint half_dim = head_dim / 2;

    // Load Q with RoPE
    if (q_row_global < seq_len_q) {
        for (uint dp = local_col; dp < half_dim; dp += BLOCK_N) {
            uint d1 = dp;
            uint d2 = dp + half_dim;

            float q1 = float(Q[q_batch_head_offset + q_row_global * head_dim + d1]);
            float q2 = float(Q[q_batch_head_offset + q_row_global * head_dim + d2]);

            float cos_val, sin_val;
            if (use_cache) {
                cos_val = float(cos_cache[q_position * half_dim + dp]);
                sin_val = float(sin_cache[q_position * half_dim + dp]);
            } else {
                float theta = compute_rope_theta(dp, head_dim, ROPE_BASE);
                float angle = float(q_position) * theta;
                cos_val = cos(angle);
                sin_val = sin(angle);
            }

            float2 rotated = apply_rope_rotation(q1, q2, cos_val, sin_val);
            Q_tile[local_row * head_dim + d1] = half(rotated.x);
            Q_tile[local_row * head_dim + d2] = half(rotated.y);
        }
    } else {
        for (uint d = local_col; d < head_dim; d += BLOCK_N) {
            Q_tile[local_row * head_dim + d] = half(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (seq_len_kv + BLOCK_N - 1) / BLOCK_N;
    uint max_kv_block;
    if (is_causal) {
        uint max_q_pos = (block_row + 1) * BLOCK_M - 1 + q_offset;
        max_kv_block = min((max_q_pos + 1 - kv_offset + BLOCK_N - 1) / BLOCK_N, num_kv_blocks);
    } else {
        max_kv_block = num_kv_blocks;
    }

    for (uint kv_block = 0; kv_block < max_kv_block; kv_block++) {
        uint kv_col_global = kv_block * BLOCK_N + local_col;
        uint kv_position = kv_col_global + kv_offset;

        // Load K with RoPE
        if (kv_col_global < seq_len_kv) {
            for (uint dp = local_row; dp < half_dim; dp += BLOCK_M) {
                uint d1 = dp;
                uint d2 = dp + half_dim;

                float k1 = float(K[kv_batch_head_offset + kv_col_global * head_dim + d1]);
                float k2 = float(K[kv_batch_head_offset + kv_col_global * head_dim + d2]);

                float cos_val, sin_val;
                if (use_cache) {
                    cos_val = float(cos_cache[kv_position * half_dim + dp]);
                    sin_val = float(sin_cache[kv_position * half_dim + dp]);
                } else {
                    float theta = compute_rope_theta(dp, head_dim, ROPE_BASE);
                    float angle = float(kv_position) * theta;
                    cos_val = cos(angle);
                    sin_val = sin(angle);
                }

                float2 rotated = apply_rope_rotation(k1, k2, cos_val, sin_val);
                K_tile[local_col * head_dim + d1] = half(rotated.x);
                K_tile[local_col * head_dim + d2] = half(rotated.y);
            }
        } else {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                K_tile[local_col * head_dim + d] = half(0.0f);
            }
        }

        // Load V
        if (kv_col_global < seq_len_kv) {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                V_tile[local_col * head_dim + d] = V[kv_batch_head_offset + kv_col_global * head_dim + d];
            }
        } else {
            for (uint d = local_row; d < head_dim; d += BLOCK_M) {
                V_tile[local_col * head_dim + d] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores
        float block_max = -INFINITY;
        float s_vals[64];

        for (uint j = 0; j < BLOCK_N; j++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += float(Q_tile[local_row * head_dim + d]) * float(K_tile[j * head_dim + d]);
            }
            dot *= SOFTMAX_SCALE;

            uint global_q_pos = q_position;
            uint global_k_pos = kv_block * BLOCK_N + j + kv_offset;
            if (is_causal && global_k_pos > global_q_pos) {
                dot = -INFINITY;
            }
            if (kv_block * BLOCK_N + j >= seq_len_kv) {
                dot = -INFINITY;
            }

            s_vals[j] = dot;
            block_max = max(block_max, dot);
        }

        // Online softmax
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

    // Write output
    if (q_row_global < seq_len_q) {
        float inv_sum = 1.0f / sum_val;
        for (uint d = 0; d < head_dim; d++) {
            O[q_batch_head_offset + q_row_global * head_dim + d] = half(output_acc[d] * inv_sum);
        }

        uint lm_offset = (batch_idx * num_heads + head_idx) * seq_len_q + q_row_global;
        L[lm_offset] = max_val + log(sum_val);
        M[lm_offset] = max_val;
    }
}
