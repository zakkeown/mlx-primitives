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
 * Tiled INT8 matmul for better memory access patterns.
 *
 * Uses shared memory to load tiles of X and W_quant, reducing
 * global memory bandwidth.
 *
 * Grid: (out_features, seq_len, batch_size)
 * Threadgroup: (TILE_SIZE, 1, 1) where TILE_SIZE typically 32 or 64
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
    // Tile size for shared memory loading
    constexpr uint TILE_K = 32;

    // Shared memory for tiles
    threadgroup float x_tile[TILE_K];
    threadgroup float w_tile[TILE_K];

    uint out_idx = tid.x;
    uint seq_idx = tid.y;
    uint batch_idx = tid.z;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len || out_idx >= *out_features) return;

    float s = *scale;
    float zp = *zero_point;
    uint in_dim = *in_features;

    uint x_base = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;
    uint w_base = out_idx * in_dim;

    float acc = 0.0f;

    // Process in tiles
    for (uint k = 0; k < in_dim; k += TILE_K) {
        // Load tile into shared memory (simplified - each thread loads one element)
        uint load_idx = lid.x;
        if (load_idx < TILE_K && (k + load_idx) < in_dim) {
            x_tile[load_idx] = X[x_base + k + load_idx];
            w_tile[load_idx] = s * (float(W_quant[w_base + k + load_idx]) - zp);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        uint tile_size = min(TILE_K, in_dim - k);
        for (uint t = 0; t < tile_size; t++) {
            acc += x_tile[t] * w_tile[t];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (*has_bias > 0) {
        acc += bias[out_idx];
    }

    uint y_offset = batch_idx * (*seq_len) * (*out_features) + seq_idx * (*out_features) + out_idx;
    Y[y_offset] = acc;
}
