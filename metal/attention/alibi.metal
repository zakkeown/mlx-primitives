// ALiBi (Attention with Linear Biases) Metal Kernel
// Adds linear position-dependent biases to attention scores
//
// Based on "Train Short, Test Long: Attention with Linear Biases Enables
// Input Length Extrapolation" by Press et al., 2022

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Compute ALiBi slopes for each head
// Slopes are computed as 2^(-8/n * i) for head i
kernel void alibi_compute_slopes(
    device float* slopes [[buffer(0)]],
    constant uint& num_heads [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_heads) return;

    // Compute slope: 2^(-8/num_heads * (head_index + 1))
    // For efficiency, we use the formula: 2^(-8 * (i+1) / n)
    float exponent = -8.0f * float(tid + 1) / float(num_heads);
    slopes[tid] = pow(2.0f, exponent);
}

// Apply ALiBi bias to attention scores
// Adds -slope * |i - j| to each attention score
kernel void alibi_add_bias(
    device float* attn_scores [[buffer(0)]],      // [batch, num_heads, seq_q, seq_k]
    device const float* slopes [[buffer(1)]],     // [num_heads]
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& seq_q [[buffer(4)]],
    constant uint& seq_k [[buffer(5)]],
    constant uint& q_offset [[buffer(6)]],        // Offset for KV cache scenarios
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint q_idx = tid.y;
    uint k_idx = tid.x;

    if (batch_idx >= batch_size || q_idx >= seq_q || k_idx >= seq_k) {
        return;
    }

    // Compute global index into attention scores
    uint global_idx = batch_idx * (num_heads * seq_q * seq_k) +
                      head_idx * (seq_q * seq_k) +
                      q_idx * seq_k +
                      k_idx;

    // Get the slope for this head
    float slope = slopes[head_idx];

    // Compute position difference with offset
    // In autoregressive generation, q_offset indicates the current position
    int q_pos = int(q_idx + q_offset);
    int k_pos = int(k_idx);

    // Compute relative position (always <= 0 for causal attention)
    // ALiBi uses negative values: -slope * (k_pos - q_pos)
    // For causal, k_pos <= q_pos, so this is negative
    float bias = slope * float(k_pos - q_pos);

    // Add bias to attention score
    attn_scores[global_idx] += bias;
}

// Fused attention + ALiBi computation
// More efficient than separate attention and bias addition
kernel void attention_with_alibi(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* attn_scores [[buffer(2)]],
    device const float* slopes [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& seq_q [[buffer(6)]],
    constant uint& seq_k [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],          // 1/sqrt(d_k)
    constant uint& q_offset [[buffer(10)]],
    constant bool& causal [[buffer(11)]],
    threadgroup float* shared_Q [[threadgroup(0)]],
    threadgroup float* shared_K [[threadgroup(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Block dimensions
    constexpr uint BLOCK_SIZE = 32;

    uint batch_idx = bid.z / num_heads;
    uint head_idx = bid.z % num_heads;
    uint block_q = bid.y;
    uint block_k = bid.x;

    uint local_q = tid.y;
    uint local_k = tid.x;

    uint global_q = block_q * BLOCK_SIZE + local_q;
    uint global_k = block_k * BLOCK_SIZE + local_k;

    if (batch_idx >= batch_size) return;

    // Load Q tile into shared memory
    uint q_offset_global = batch_idx * (num_heads * seq_q * head_dim) +
                           head_idx * (seq_q * head_dim) +
                           global_q * head_dim;

    uint k_offset_global = batch_idx * (num_heads * seq_k * head_dim) +
                           head_idx * (seq_k * head_dim) +
                           global_k * head_dim;

    // Each thread loads part of the Q and K vectors
    for (uint d = tid.x; d < head_dim; d += BLOCK_SIZE) {
        if (global_q < seq_q) {
            shared_Q[local_q * head_dim + d] = Q[q_offset_global + d];
        } else {
            shared_Q[local_q * head_dim + d] = 0.0f;
        }
    }

    for (uint d = tid.y; d < head_dim; d += BLOCK_SIZE) {
        if (global_k < seq_k) {
            shared_K[local_k * head_dim + d] = K[k_offset_global + d];
        } else {
            shared_K[local_k * head_dim + d] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute dot product
    float dot_product = 0.0f;
    if (global_q < seq_q && global_k < seq_k) {
        for (uint d = 0; d < head_dim; d++) {
            dot_product += shared_Q[local_q * head_dim + d] * shared_K[local_k * head_dim + d];
        }
        dot_product *= scale;

        // Apply causal mask if needed
        if (causal && global_k > global_q + q_offset) {
            dot_product = -INFINITY;
        } else {
            // Add ALiBi bias
            float slope = slopes[head_idx];
            int q_pos = int(global_q + q_offset);
            int k_pos = int(global_k);
            dot_product += slope * float(k_pos - q_pos);
        }
    } else {
        dot_product = -INFINITY;
    }

    // Write result
    if (global_q < seq_q && global_k < seq_k) {
        uint out_idx = batch_idx * (num_heads * seq_q * seq_k) +
                       head_idx * (seq_q * seq_k) +
                       global_q * seq_k +
                       global_k;
        attn_scores[out_idx] = dot_product;
    }
}

// Half-precision version
kernel void alibi_add_bias_half(
    device half* attn_scores [[buffer(0)]],
    device const float* slopes [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& seq_q [[buffer(4)]],
    constant uint& seq_k [[buffer(5)]],
    constant uint& q_offset [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint q_idx = tid.y;
    uint k_idx = tid.x;

    if (batch_idx >= batch_size || q_idx >= seq_q || k_idx >= seq_k) {
        return;
    }

    uint global_idx = batch_idx * (num_heads * seq_q * seq_k) +
                      head_idx * (seq_q * seq_k) +
                      q_idx * seq_k +
                      k_idx;

    float slope = slopes[head_idx];
    int q_pos = int(q_idx + q_offset);
    int k_pos = int(k_idx);
    float bias = slope * float(k_pos - q_pos);

    // Convert to float, add bias, convert back
    attn_scores[global_idx] = half(float(attn_scores[global_idx]) + bias);
}

// Compute ALiBi mask matrix (can be cached)
// Creates a [seq_q, seq_k] matrix of biases for a single head
kernel void alibi_compute_mask(
    device float* mask [[buffer(0)]],
    constant float& slope [[buffer(1)]],
    constant uint& seq_q [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    constant uint& q_offset [[buffer(4)]],
    constant bool& causal [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint q_idx = tid.y;
    uint k_idx = tid.x;

    if (q_idx >= seq_q || k_idx >= seq_k) return;

    uint idx = q_idx * seq_k + k_idx;

    int q_pos = int(q_idx + q_offset);
    int k_pos = int(k_idx);

    if (causal && k_pos > q_pos) {
        mask[idx] = -INFINITY;
    } else {
        mask[idx] = slope * float(k_pos - q_pos);
    }
}

// Batched ALiBi mask computation for all heads
kernel void alibi_compute_mask_all_heads(
    device float* masks [[buffer(0)]],            // [num_heads, seq_q, seq_k]
    device const float* slopes [[buffer(1)]],
    constant uint& num_heads [[buffer(2)]],
    constant uint& seq_q [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant uint& q_offset [[buffer(5)]],
    constant bool& causal [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint head_idx = tid.z;
    uint q_idx = tid.y;
    uint k_idx = tid.x;

    if (head_idx >= num_heads || q_idx >= seq_q || k_idx >= seq_k) return;

    uint idx = head_idx * (seq_q * seq_k) + q_idx * seq_k + k_idx;

    float slope = slopes[head_idx];
    int q_pos = int(q_idx + q_offset);
    int k_pos = int(k_idx);

    if (causal && k_pos > q_pos) {
        masks[idx] = -INFINITY;
    } else {
        masks[idx] = slope * float(k_pos - q_pos);
    }
}

// Symmetric ALiBi for bidirectional attention
// Uses |i - j| instead of (j - i)
kernel void alibi_add_bias_symmetric(
    device float* attn_scores [[buffer(0)]],
    device const float* slopes [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& seq_q [[buffer(4)]],
    constant uint& seq_k [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint q_idx = tid.y;
    uint k_idx = tid.x;

    if (batch_idx >= batch_size || q_idx >= seq_q || k_idx >= seq_k) {
        return;
    }

    uint global_idx = batch_idx * (num_heads * seq_q * seq_k) +
                      head_idx * (seq_q * seq_k) +
                      q_idx * seq_k +
                      k_idx;

    float slope = slopes[head_idx];

    // Use absolute distance for bidirectional
    float distance = abs(float(q_idx) - float(k_idx));
    float bias = -slope * distance;

    attn_scores[global_idx] += bias;
}
