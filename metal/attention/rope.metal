// Rotary Position Embedding (RoPE) Metal Kernel
// Fused implementation of rotary position embeddings
//
// Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// by Su et al., 2021

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

// Precomputed theta values can be passed or computed on-the-fly
// theta_i = base^(-2i/d) where i is the dimension index

// Apply RoPE to a single query or key tensor
// Rotates pairs of dimensions by position-dependent angles
kernel void rope_forward(
    device const float* x [[buffer(0)]],         // [batch, seq_len, num_heads, head_dim]
    device float* out [[buffer(1)]],             // [batch, seq_len, num_heads, head_dim]
    device const float* cos_cache [[buffer(2)]], // [max_seq, head_dim/2] precomputed cosines
    device const float* sin_cache [[buffer(3)]], // [max_seq, head_dim/2] precomputed sines
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& offset [[buffer(8)]],         // Position offset for KV cache
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;  // Processes pairs of dimensions

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    // Compute global index
    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    // Get the two elements that form a rotation pair
    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    float x1 = x[global_idx + d1];
    float x2 = x[global_idx + d2];

    // Get position with offset
    uint pos = seq_idx + offset;

    // Get cos/sin values for this position and dimension
    float cos_val = cos_cache[pos * (head_dim / 2) + dim_pair];
    float sin_val = sin_cache[pos * (head_dim / 2) + dim_pair];

    // Apply rotation: [cos, -sin; sin, cos] @ [x1, x2]
    out[global_idx + d1] = x1 * cos_val - x2 * sin_val;
    out[global_idx + d2] = x1 * sin_val + x2 * cos_val;
}

// Apply RoPE to Q and K simultaneously (more efficient)
kernel void rope_forward_qk(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* Q_out [[buffer(2)]],
    device float* K_out [[buffer(3)]],
    device const float* cos_cache [[buffer(4)]],
    device const float* sin_cache [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& offset [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    // Load Q values
    float q1 = Q[global_idx + d1];
    float q2 = Q[global_idx + d2];

    // Load K values
    float k1 = K[global_idx + d1];
    float k2 = K[global_idx + d2];

    uint pos = seq_idx + offset;
    float cos_val = cos_cache[pos * (head_dim / 2) + dim_pair];
    float sin_val = sin_cache[pos * (head_dim / 2) + dim_pair];

    // Apply rotation to Q
    Q_out[global_idx + d1] = q1 * cos_val - q2 * sin_val;
    Q_out[global_idx + d2] = q1 * sin_val + q2 * cos_val;

    // Apply rotation to K
    K_out[global_idx + d1] = k1 * cos_val - k2 * sin_val;
    K_out[global_idx + d2] = k1 * sin_val + k2 * cos_val;
}

// Generate cos/sin cache for given positions
// More efficient to precompute than compute per-token
kernel void rope_compute_cache(
    device float* cos_cache [[buffer(0)]],
    device float* sin_cache [[buffer(1)]],
    constant uint& max_seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& base [[buffer(4)]],          // Usually 10000
    uint2 tid [[thread_position_in_grid]]
) {
    uint pos = tid.x;
    uint dim = tid.y;

    if (pos >= max_seq_len || dim >= head_dim / 2) {
        return;
    }

    // Compute theta = base^(-2d/head_dim)
    float theta = pow(base, -2.0f * float(dim) / float(head_dim));

    // Compute position * theta
    float angle = float(pos) * theta;

    uint idx = pos * (head_dim / 2) + dim;
    cos_cache[idx] = cos(angle);
    sin_cache[idx] = sin(angle);
}

// Half-precision version for memory efficiency
kernel void rope_forward_half(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const half* cos_cache [[buffer(2)]],
    device const half* sin_cache [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& offset [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    // Use float for computation, half for storage
    float x1 = float(x[global_idx + d1]);
    float x2 = float(x[global_idx + d2]);

    uint pos = seq_idx + offset;
    float cos_val = float(cos_cache[pos * (head_dim / 2) + dim_pair]);
    float sin_val = float(sin_cache[pos * (head_dim / 2) + dim_pair]);

    out[global_idx + d1] = half(x1 * cos_val - x2 * sin_val);
    out[global_idx + d2] = half(x1 * sin_val + x2 * cos_val);
}

// In-place RoPE application (modifies input directly)
kernel void rope_inplace(
    device float* x [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& offset [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    float x1 = x[global_idx + d1];
    float x2 = x[global_idx + d2];

    uint pos = seq_idx + offset;
    float cos_val = cos_cache[pos * (head_dim / 2) + dim_pair];
    float sin_val = sin_cache[pos * (head_dim / 2) + dim_pair];

    // Write rotated values back in-place
    x[global_idx + d1] = x1 * cos_val - x2 * sin_val;
    x[global_idx + d2] = x1 * sin_val + x2 * cos_val;
}

// NTK-aware interpolation for extended context
// Implements NTK-aware scaled RoPE for long contexts
kernel void rope_ntk_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& base [[buffer(6)]],
    constant float& scale [[buffer(7)]],         // NTK scale factor (e.g., 2.0 for 2x context)
    constant uint& offset [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    float x1 = x[global_idx + d1];
    float x2 = x[global_idx + d2];

    // NTK-aware interpolation: scale the base
    float ntk_base = base * pow(scale, float(head_dim) / float(head_dim - 2));
    float theta = pow(ntk_base, -2.0f * float(dim_pair) / float(head_dim));

    uint pos = seq_idx + offset;
    float angle = float(pos) * theta;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    out[global_idx + d1] = x1 * cos_val - x2 * sin_val;
    out[global_idx + d2] = x1 * sin_val + x2 * cos_val;
}

// YaRN (Yet another RoPE extensioN) interpolation
// Combines NTK interpolation with attention scaling
kernel void rope_yarn_forward(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& base [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant float& beta_fast [[buffer(8)]],     // YaRN parameters
    constant float& beta_slow [[buffer(9)]],
    constant uint& original_max_pos [[buffer(10)]],
    constant uint& offset [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z / num_heads;
    uint head_idx = tid.z % num_heads;
    uint seq_idx = tid.y;
    uint dim_pair = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_pair >= head_dim / 2) {
        return;
    }

    uint global_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;

    uint d1 = dim_pair * 2;
    uint d2 = dim_pair * 2 + 1;

    float x1 = x[global_idx + d1];
    float x2 = x[global_idx + d2];

    // Compute dimension-dependent wavelength
    float dim_ratio = 2.0f * float(dim_pair) / float(head_dim);
    float wavelength = 2.0f * M_PI_F * pow(base, dim_ratio);

    // YaRN interpolation factor based on wavelength
    float low = max(0.0f, (wavelength / (beta_fast * float(original_max_pos)) - 1.0f));
    float high = min(1.0f, (wavelength / (beta_slow * float(original_max_pos)) - 1.0f));

    // Blend between original and scaled theta
    float interp = (high - low) / (1.0f - low + 1e-6f);
    float theta_original = pow(base, -dim_ratio);
    float theta_scaled = theta_original / scale;
    float theta = theta_original * (1.0f - interp) + theta_scaled * interp;

    uint pos = seq_idx + offset;
    float angle = float(pos) * theta;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    out[global_idx + d1] = x1 * cos_val - x2 * sin_val;
    out[global_idx + d2] = x1 * sin_val + x2 * cos_val;
}
