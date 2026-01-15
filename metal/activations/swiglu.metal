// SwiGLU (Swish-Gated Linear Unit) Metal Kernel
// Fused implementation: silu(x @ W1) * (x @ W2)
//
// Based on "GLU Variants Improve Transformer" by Shazeer, 2020

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// SiLU (Swish) activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline half silu_half(half x) {
    return x / (half(1.0f) + exp(-x));
}

// Basic SwiGLU: silu(gate) * up
// Applied element-wise to pre-computed projections
kernel void swiglu_forward(
    device const float* gate [[buffer(0)]],       // [batch, seq_len, hidden_dim] = x @ W_gate
    device const float* up [[buffer(1)]],         // [batch, seq_len, hidden_dim] = x @ W_up
    device float* out [[buffer(2)]],              // [batch, seq_len, hidden_dim]
    constant uint& size [[buffer(3)]],            // Total number of elements
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];
    out[tid] = silu(g) * u;
}

// Fused SwiGLU with GEMM: x @ [W_gate; W_up] -> split -> silu(gate) * up
// More memory efficient as it avoids storing intermediate results
kernel void swiglu_fused(
    device const float* x [[buffer(0)]],          // [batch, seq_len, in_dim]
    device const float* W [[buffer(1)]],          // [in_dim, 2 * hidden_dim] - concatenated weights
    device float* out [[buffer(2)]],              // [batch, seq_len, hidden_dim]
    constant uint& batch_seq [[buffer(3)]],       // batch * seq_len
    constant uint& in_dim [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    threadgroup float* shared_x [[threadgroup(0)]],
    threadgroup float* shared_W [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    constexpr uint BLOCK_SIZE = 32;

    uint row = bid.y * BLOCK_SIZE + tid.y;  // batch_seq index
    uint col = bid.x * BLOCK_SIZE + tid.x;  // hidden_dim index

    if (row >= batch_seq || col >= hidden_dim) return;

    // Compute both gate and up projections
    float gate_val = 0.0f;
    float up_val = 0.0f;

    uint num_tiles = (in_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        // Load tile of x into shared memory
        uint x_col = tile * BLOCK_SIZE + tid.x;
        if (x_col < in_dim && row < batch_seq) {
            shared_x[tid.y * BLOCK_SIZE + tid.x] = x[row * in_dim + x_col];
        } else {
            shared_x[tid.y * BLOCK_SIZE + tid.x] = 0.0f;
        }

        // Load tiles of W (both gate and up columns) into shared memory
        uint w_row = tile * BLOCK_SIZE + tid.y;
        if (w_row < in_dim && col < hidden_dim) {
            // Gate weight is in first hidden_dim columns
            shared_W[tid.y * BLOCK_SIZE + tid.x] = W[w_row * (2 * hidden_dim) + col];
            // Up weight is in second hidden_dim columns
            shared_W[BLOCK_SIZE * BLOCK_SIZE + tid.y * BLOCK_SIZE + tid.x] =
                W[w_row * (2 * hidden_dim) + hidden_dim + col];
        } else {
            shared_W[tid.y * BLOCK_SIZE + tid.x] = 0.0f;
            shared_W[BLOCK_SIZE * BLOCK_SIZE + tid.y * BLOCK_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot products
        for (uint k = 0; k < BLOCK_SIZE; k++) {
            float x_val = shared_x[tid.y * BLOCK_SIZE + k];
            gate_val += x_val * shared_W[k * BLOCK_SIZE + tid.x];
            up_val += x_val * shared_W[BLOCK_SIZE * BLOCK_SIZE + k * BLOCK_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply SwiGLU activation and write output
    if (row < batch_seq && col < hidden_dim) {
        out[row * hidden_dim + col] = silu(gate_val) * up_val;
    }
}

// SwiGLU backward pass - compute gradients
kernel void swiglu_backward(
    device const float* grad_out [[buffer(0)]],   // [batch, seq_len, hidden_dim]
    device const float* gate [[buffer(1)]],       // [batch, seq_len, hidden_dim]
    device const float* up [[buffer(2)]],         // [batch, seq_len, hidden_dim]
    device float* grad_gate [[buffer(3)]],        // [batch, seq_len, hidden_dim]
    device float* grad_up [[buffer(4)]],          // [batch, seq_len, hidden_dim]
    constant uint& size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];
    float grad = grad_out[tid];

    // SiLU derivative: silu(x) + sigmoid(x) * (1 - silu(x))
    //                = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;
    float silu_deriv = sig * (1.0f + g * (1.0f - sig));

    // grad_gate = grad_out * up * silu'(gate)
    grad_gate[tid] = grad * u * silu_deriv;

    // grad_up = grad_out * silu(gate)
    grad_up[tid] = grad * silu_g;
}

// Half-precision SwiGLU
kernel void swiglu_forward_half(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    // Use float for computation, half for storage
    float g = float(gate[tid]);
    float u = float(up[tid]);
    out[tid] = half(silu(g) * u);
}

// GeGLU variant: gelu(gate) * up
kernel void geglu_forward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float gelu = 0.5f * g * (1.0f + tanh(sqrt_2_over_pi * (g + coeff * g * g * g)));

    out[tid] = gelu * u;
}

// ReGLU variant: relu(gate) * up
kernel void reglu_forward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];
    out[tid] = max(g, 0.0f) * u;
}

// Squared ReLU variant (from Primer paper)
kernel void sqrelu_glu_forward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];
    float relu_g = max(g, 0.0f);
    out[tid] = relu_g * relu_g * u;  // relu(x)^2 * up
}

// Vectorized SwiGLU using float4 for better memory throughput
kernel void swiglu_forward_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& vec_size [[buffer(3)]],        // size / 4
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= vec_size) return;

    float4 g = gate[tid];
    float4 u = up[tid];

    float4 result;
    result.x = silu(g.x) * u.x;
    result.y = silu(g.y) * u.y;
    result.z = silu(g.z) * u.z;
    result.w = silu(g.w) * u.w;

    out[tid] = result;
}

// SwiGLU with bias
kernel void swiglu_forward_bias(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* bias_gate [[buffer(2)]],  // [hidden_dim]
    device const float* bias_up [[buffer(3)]],    // [hidden_dim]
    device float* out [[buffer(4)]],
    constant uint& batch_seq [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;

    if (row >= batch_seq || col >= hidden_dim) return;

    uint idx = row * hidden_dim + col;
    float g = gate[idx] + bias_gate[col];
    float u = up[idx] + bias_up[col];

    out[idx] = silu(g) * u;
}

// In-place SwiGLU (gate contains result after, up is unchanged)
kernel void swiglu_inplace(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];
    gate[tid] = silu(g) * u;
}

// Mish-GLU variant: mish(gate) * up
// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
kernel void mishglu_forward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];

    // Mish: x * tanh(softplus(x))
    float sp = log(1.0f + exp(g));
    float mish = g * tanh(sp);

    out[tid] = mish * u;
}
