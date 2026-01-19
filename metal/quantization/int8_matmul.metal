/**
 * INT8 Weight-Only Quantized Matrix Multiplication for MLX.
 *
 * Apple Silicon has NO native INT8 matmul instructions, so we must:
 * 1. Load quantized weights (INT8)
 * 2. Dequantize to float (scale * (weight - zero_point))
 * 3. Compute in float32
 *
 * The benefit is reduced memory bandwidth - 4x smaller weights means faster
 * loading from memory, which dominates inference time.
 *
 * Supports:
 * - Per-tensor quantization: single scale/zero_point for entire weight matrix
 * - Per-channel quantization: scale/zero_point per output channel (row)
 */

#include <metal_stdlib>
using namespace metal;


/**
 * INT8 weight-only matmul with per-tensor quantization.
 *
 * Computes: Y = X @ dequant(W).T + bias
 * Where: dequant(w) = scale * (w - zero_point)
 *
 * Grid: (out_features, seq_len, batch_size)
 */
[[kernel]] void int8_matmul_per_tensor(
    device const float* X [[buffer(0)]],           // (batch, seq, in_features) float
    device const char* W_quant [[buffer(1)]],      // (out_features, in_features) int8
    device const float* scale [[buffer(2)]],       // scalar
    device const float* zero_point [[buffer(3)]],  // scalar
    device const float* bias [[buffer(4)]],        // (out_features,) or nullptr
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],
    device const uint* out_features [[buffer(8)]],
    device const uint* has_bias [[buffer(9)]],
    device float* Y [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.x;
    uint seq_idx = tid.y;
    uint batch_idx = tid.z;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len || out_idx >= *out_features) return;

    float s = *scale;
    float zp = *zero_point;
    uint in_dim = *in_features;

    // Input offset
    uint x_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;

    // Weight row offset
    uint w_offset = out_idx * in_dim;

    // Compute dot product with dequantization
    float acc = 0.0f;
    for (uint d = 0; d < in_dim; d++) {
        float x_val = X[x_offset + d];
        // Dequantize: scale * (weight - zero_point)
        float w_val = s * (float(W_quant[w_offset + d]) - zp);
        acc += x_val * w_val;
    }

    // Add bias if present
    if (*has_bias > 0) {
        acc += bias[out_idx];
    }

    // Write output
    uint y_offset = batch_idx * (*seq_len) * (*out_features) + seq_idx * (*out_features) + out_idx;
    Y[y_offset] = acc;
}


/**
 * INT8 weight-only matmul with per-channel quantization.
 *
 * Each output channel (row of W) has its own scale and zero_point.
 * This improves accuracy vs per-tensor quantization.
 *
 * Grid: (out_features, seq_len, batch_size)
 */
[[kernel]] void int8_matmul_per_channel(
    device const float* X [[buffer(0)]],           // (batch, seq, in_features) float
    device const char* W_quant [[buffer(1)]],      // (out_features, in_features) int8
    device const float* scales [[buffer(2)]],      // (out_features,)
    device const float* zero_points [[buffer(3)]],  // (out_features,)
    device const float* bias [[buffer(4)]],        // (out_features,) or nullptr
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],
    device const uint* out_features [[buffer(8)]],
    device const uint* has_bias [[buffer(9)]],
    device float* Y [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.x;
    uint seq_idx = tid.y;
    uint batch_idx = tid.z;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len || out_idx >= *out_features) return;

    // Per-channel scale and zero point
    float s = scales[out_idx];
    float zp = zero_points[out_idx];
    uint in_dim = *in_features;

    // Input offset
    uint x_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;

    // Weight row offset
    uint w_offset = out_idx * in_dim;

    // Compute dot product with dequantization
    float acc = 0.0f;
    for (uint d = 0; d < in_dim; d++) {
        float x_val = X[x_offset + d];
        float w_val = s * (float(W_quant[w_offset + d]) - zp);
        acc += x_val * w_val;
    }

    // Add bias if present
    if (*has_bias > 0) {
        acc += bias[out_idx];
    }

    // Write output
    uint y_offset = batch_idx * (*seq_len) * (*out_features) + seq_idx * (*out_features) + out_idx;
    Y[y_offset] = acc;
}


/**
 * Quantize FP32 weights to INT8 with per-tensor quantization.
 *
 * Computes scale and zero_point from min/max values:
 *   scale = (max - min) / 255
 *   zero_point = -round(min / scale)
 *   quantized = round(weight / scale) + zero_point
 *
 * Grid: (total_elements,)
 */
[[kernel]] void quantize_int8_per_tensor(
    device const float* weights [[buffer(0)]],     // (out_features, in_features)
    device const float* min_val [[buffer(1)]],     // scalar
    device const float* max_val [[buffer(2)]],     // scalar
    device char* W_quant [[buffer(3)]],            // (out_features, in_features)
    device float* scale_out [[buffer(4)]],         // scalar
    device float* zero_point_out [[buffer(5)]],    // scalar
    device const uint* total_elements [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= *total_elements) return;

    float mn = *min_val;
    float mx = *max_val;

    // Compute scale and zero point (symmetric quantization around 0)
    float s = (mx - mn) / 255.0f;
    float zp = -round(mn / s);

    // Clamp zero point to valid INT8 range
    zp = clamp(zp, 0.0f, 255.0f);

    // Store scale and zero point (only first thread)
    if (tid == 0) {
        *scale_out = s;
        *zero_point_out = zp;
    }

    // Quantize this element
    float w = weights[tid];
    int q = int(round(w / s) + zp);
    q = clamp(q, 0, 255);

    // Store as signed int8 (shifted by 128 for unsigned storage)
    W_quant[tid] = char(q - 128);
}


/**
 * Tiled INT8 matmul with cooperative loading for better memory access patterns.
 *
 * Uses shared memory to load tiles of X and W_quant cooperatively, reducing
 * global memory bandwidth. All threads in a threadgroup participate in loading.
 *
 * Key optimizations:
 * 1. Cooperative loading: All threads help load X and W tiles
 * 2. W stored as INT8 in shared memory (4x smaller), dequantized during compute
 * 3. char4 vectorized loads for aligned W data
 * 4. Each threadgroup handles TILE_M outputs for one (batch, seq) position
 *
 * Grid: (ceil(out_features/TILE_M), seq_len, batch_size)
 * Threadgroup: (TILE_M, 1, 1) where TILE_M = 32
 */
[[kernel]] void int8_matmul_tiled(
    device const float* X [[buffer(0)]],
    device const char* W_quant [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],
    device const uint* out_features [[buffer(8)]],
    device const uint* has_bias [[buffer(9)]],
    device float* Y [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Tile sizes
    constexpr uint TILE_M = 32;  // Outputs per threadgroup
    constexpr uint TILE_K = 32;  // Input elements per tile

    // Shared memory for tiles
    // X tile: TILE_K floats (shared across all outputs in threadgroup)
    // W tile: TILE_M * TILE_K INT8 values (one row per output, kept as INT8 to save space)
    threadgroup float x_tile[TILE_K];
    threadgroup char w_tile[TILE_M * TILE_K];

    uint out_base = tgid.x * TILE_M;  // First output index for this threadgroup
    uint seq_idx = tgid.y;
    uint batch_idx = tgid.z;
    uint tid_local = lid.x;  // Thread index within threadgroup (0 to TILE_M-1)

    // Early exit for out-of-bounds threadgroups
    if (batch_idx >= *batch_size || seq_idx >= *seq_len) return;

    float s = *scale;
    float zp = *zero_point;
    uint in_dim = *in_features;
    uint out_dim = *out_features;

    // Each thread computes one output (if within bounds)
    uint out_idx = out_base + tid_local;

    // Base offset for input row (same for all outputs in this threadgroup)
    uint x_base_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;

    // Accumulator for this thread's output
    float acc = 0.0f;

    // Process input dimension in tiles of TILE_K
    for (uint k = 0; k < in_dim; k += TILE_K) {
        uint tile_k_size = min(TILE_K, in_dim - k);

        // ===== COOPERATIVE LOADING PHASE =====

        // Phase 1: Load X tile (TILE_K elements)
        // All TILE_M threads cooperate - thread i loads element i (if i < TILE_K)
        if (tid_local < TILE_K && (k + tid_local) < in_dim) {
            x_tile[tid_local] = X[x_base_offset + k + tid_local];
        }

        // Phase 2: Load W tile (TILE_M * TILE_K INT8 elements)
        // Use all threads to load cooperatively with strided access
        // Each thread loads multiple elements: positions tid_local, tid_local + TILE_M, ...
        uint total_w_elements = TILE_M * TILE_K;
        for (uint i = tid_local; i < total_w_elements; i += TILE_M) {
            uint m = i / TILE_K;  // Row within tile (0 to TILE_M-1)
            uint kk = i % TILE_K;  // Column within tile (0 to TILE_K-1)
            uint global_out = out_base + m;
            uint global_k = k + kk;

            if (global_out < out_dim && global_k < in_dim) {
                // Store INT8 directly in shared memory (dequantize during compute)
                w_tile[m * TILE_K + kk] = W_quant[global_out * in_dim + global_k];
            } else {
                // Zero-fill for out-of-bounds (will be multiplied by zero anyway)
                w_tile[m * TILE_K + kk] = char(zp);  // Results in 0 after dequant
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== COMPUTE PHASE =====
        // Each thread computes dot product for its output row
        if (out_idx < out_dim) {
            // Get pointer to this thread's weight row in shared memory
            uint w_row_offset = tid_local * TILE_K;

            // Compute partial dot product with dequantization
            for (uint t = 0; t < tile_k_size; t++) {
                float x_val = x_tile[t];
                // Dequantize: scale * (weight - zero_point)
                float w_val = s * (float(w_tile[w_row_offset + t]) - zp);
                acc += x_val * w_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (out_idx < out_dim) {
        if (*has_bias > 0) {
            acc += bias[out_idx];
        }
        uint y_offset = batch_idx * (*seq_len) * out_dim + seq_idx * out_dim + out_idx;
        Y[y_offset] = acc;
    }
}


/**
 * Vectorized tiled INT8 matmul using char4 loads.
 *
 * Same as int8_matmul_tiled but uses char4 vectorized loads for W
 * when in_features is aligned to 4 bytes.
 *
 * Grid: (ceil(out_features/TILE_M), seq_len, batch_size)
 * Threadgroup: (TILE_M, 1, 1) where TILE_M = 32
 */
[[kernel]] void int8_matmul_tiled_vec4(
    device const float* X [[buffer(0)]],
    device const char4* W_quant_vec4 [[buffer(1)]],  // Reinterpreted as char4
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],  // Must be multiple of 4
    device const uint* out_features [[buffer(8)]],
    device const uint* has_bias [[buffer(9)]],
    device float* Y [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_K = 32;  // Must be multiple of 4

    threadgroup float x_tile[TILE_K];
    threadgroup char w_tile[TILE_M * TILE_K];

    uint out_base = tgid.x * TILE_M;
    uint seq_idx = tgid.y;
    uint batch_idx = tgid.z;
    uint tid_local = lid.x;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len) return;

    float s = *scale;
    float zp = *zero_point;
    uint in_dim = *in_features;
    uint out_dim = *out_features;
    uint in_dim_vec4 = in_dim / 4;

    uint out_idx = out_base + tid_local;
    uint x_base_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;

    float acc = 0.0f;

    for (uint k = 0; k < in_dim; k += TILE_K) {
        uint tile_k_size = min(TILE_K, in_dim - k);
        uint tile_k_vec4 = tile_k_size / 4;

        // Load X tile
        if (tid_local < TILE_K && (k + tid_local) < in_dim) {
            x_tile[tid_local] = X[x_base_offset + k + tid_local];
        }

        // Load W tile using char4 vectorized loads
        // Each thread loads 4 INT8 values at once
        uint total_vec4_elements = (TILE_M * TILE_K) / 4;
        for (uint i = tid_local; i < total_vec4_elements; i += TILE_M) {
            uint linear_byte_idx = i * 4;
            uint m = linear_byte_idx / TILE_K;
            uint kk_base = linear_byte_idx % TILE_K;
            uint global_out = out_base + m;
            uint global_k_base = k + kk_base;

            if (global_out < out_dim && global_k_base + 3 < in_dim) {
                // Vectorized load of 4 INT8 values
                uint w_vec_idx = global_out * in_dim_vec4 + global_k_base / 4;
                char4 w4 = W_quant_vec4[w_vec_idx];

                // Store in shared memory
                uint shared_idx = m * TILE_K + kk_base;
                w_tile[shared_idx + 0] = w4.x;
                w_tile[shared_idx + 1] = w4.y;
                w_tile[shared_idx + 2] = w4.z;
                w_tile[shared_idx + 3] = w4.w;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute with unrolled loop for better performance
        if (out_idx < out_dim) {
            uint w_row_offset = tid_local * TILE_K;

            // Process 4 elements at a time
            uint t = 0;
            for (; t + 3 < tile_k_size; t += 4) {
                float x0 = x_tile[t];
                float x1 = x_tile[t + 1];
                float x2 = x_tile[t + 2];
                float x3 = x_tile[t + 3];

                float w0 = s * (float(w_tile[w_row_offset + t]) - zp);
                float w1 = s * (float(w_tile[w_row_offset + t + 1]) - zp);
                float w2 = s * (float(w_tile[w_row_offset + t + 2]) - zp);
                float w3 = s * (float(w_tile[w_row_offset + t + 3]) - zp);

                acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
            }

            // Handle remaining elements
            for (; t < tile_k_size; t++) {
                acc += x_tile[t] * (s * (float(w_tile[w_row_offset + t]) - zp));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_idx < out_dim) {
        if (*has_bias > 0) {
            acc += bias[out_idx];
        }
        uint y_offset = batch_idx * (*seq_len) * out_dim + seq_idx * out_dim + out_idx;
        Y[y_offset] = acc;
    }
}
