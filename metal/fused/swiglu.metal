// Fused SwiGLU Activation
//
// SwiGLU is used in many modern LLMs (LLaMA, Mistral, etc.):
//   output = SiLU(x @ W_gate) * (x @ W_up)
//
// Where SiLU(x) = x * sigmoid(x)
//
// Without fusion (3 kernels + 2 memory round-trips):
// 1. gate = x @ W_gate
// 2. up = x @ W_up
// 3. output = silu(gate) * up
//
// With fusion (1 kernel):
// - Read x, W_gate, W_up once
// - Compute everything in registers
// - Write output once

#include <metal_stdlib>
using namespace metal;

// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ============================================================================
// Fused SwiGLU: silu(x @ W_gate + b_gate) * (x @ W_up + b_up)
// Simple version where each thread computes one output element
// ============================================================================

kernel void fused_swiglu_simple(
    device const float* x [[buffer(0)]],           // (batch, seq, in_features)
    device const float* W_gate [[buffer(1)]],      // (out_features, in_features)
    device const float* W_up [[buffer(2)]],        // (out_features, in_features)
    device const float* b_gate [[buffer(3)]],      // (out_features,) or nullptr
    device const float* b_up [[buffer(4)]],        // (out_features,) or nullptr
    device float* output [[buffer(5)]],            // (batch, seq, out_features)
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& out_features [[buffer(9)]],
    constant bool& has_bias [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint seq_idx = tid.y;
    uint out_idx = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

    uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;

    // Compute gate = x @ W_gate^T
    float gate = 0.0f;
    float up = 0.0f;

    for (uint d = 0; d < in_features; d++) {
        float x_d = x[x_offset + d];
        gate += x_d * W_gate[out_idx * in_features + d];
        up += x_d * W_up[out_idx * in_features + d];
    }

    // Add biases
    if (has_bias) {
        gate += b_gate[out_idx];
        up += b_up[out_idx];
    }

    // SwiGLU: silu(gate) * up
    float result = silu(gate) * up;

    uint out_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
    output[out_offset] = result;
}

// ============================================================================
// Fused GeGLU: gelu(x @ W_gate + b_gate) * (x @ W_up + b_up)
// ============================================================================

// Approximate GELU using tanh
inline float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void fused_geglu(
    device const float* x [[buffer(0)]],
    device const float* W_gate [[buffer(1)]],
    device const float* W_up [[buffer(2)]],
    device const float* b_gate [[buffer(3)]],
    device const float* b_up [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& out_features [[buffer(9)]],
    constant bool& has_bias [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.z;
    uint seq_idx = tid.y;
    uint out_idx = tid.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

    uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;

    float gate = 0.0f;
    float up = 0.0f;

    for (uint d = 0; d < in_features; d++) {
        float x_d = x[x_offset + d];
        gate += x_d * W_gate[out_idx * in_features + d];
        up += x_d * W_up[out_idx * in_features + d];
    }

    if (has_bias) {
        gate += b_gate[out_idx];
        up += b_up[out_idx];
    }

    // GeGLU: gelu(gate) * up
    float result = gelu_tanh(gate) * up;

    uint out_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
    output[out_offset] = result;
}

// ============================================================================
// Standalone SiLU activation (for when projection is separate)
// ============================================================================

kernel void silu_kernel(
    device const float* x [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    output[tid] = silu(x[tid]);
}

// ============================================================================
// Fused bias + SiLU
// ============================================================================

kernel void fused_bias_silu(
    device const float* x [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_seq [[buffer(3)]],
    constant uint& features [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pos = tid.y;
    uint f = tid.x;

    if (pos >= batch_seq || f >= features) return;

    uint idx = pos * features + f;
    float val = x[idx] + bias[f];
    output[idx] = silu(val);
}

// ============================================================================
// Fused element-wise multiply (for gate * up after separate projections)
// ============================================================================

kernel void fused_gate_multiply(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    // Assumes gate already has activation applied
    output[tid] = gate[tid] * up[tid];
}
