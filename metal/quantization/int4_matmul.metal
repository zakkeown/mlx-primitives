/**
 * INT4 Grouped Quantized Matrix Multiplication for MLX.
 *
 * INT4 weights provide 8x memory reduction vs FP32 and 2x vs INT8.
 * Typical configuration: group_size=128 with per-group scale/zero_point.
 *
 * Packing: Two INT4 values packed per uint8 (byte):
 *   - Lower nibble (bits 0-3): first value
 *   - Upper nibble (bits 4-7): second value
 *
 * For weight matrix (out_features, in_features):
 *   - Packed shape: (out_features, in_features / 2)
 *   - Scales shape: (out_features, in_features / group_size)
 *   - Zero points shape: (out_features, in_features / group_size)
 */

#include <metal_stdlib>
using namespace metal;


/**
 * Unpack INT4 value from packed byte.
 *
 * @param packed The packed byte containing two INT4 values
 * @param idx 0 for lower nibble, 1 for upper nibble
 * @return The unpacked 4-bit value as signed int (-8 to 7)
 */
inline int unpack_int4(uchar packed, uint idx) {
    uchar nibble;
    if (idx == 0) {
        nibble = packed & 0x0F;  // Lower nibble
    } else {
        nibble = (packed >> 4) & 0x0F;  // Upper nibble
    }
    // Convert unsigned 4-bit to signed (-8 to 7)
    // Values 0-7 stay as-is, 8-15 become -8 to -1
    if (nibble >= 8) {
        return int(nibble) - 16;
    }
    return int(nibble);
}


/**
 * INT4 grouped quantized matmul.
 *
 * Computes: Y = X @ dequant(W_packed).T + bias
 * Where: dequant(w) = scale[group] * (w - zero_point[group])
 *
 * Grid: (out_features, seq_len, batch_size)
 */
[[kernel]] void int4_matmul_grouped(
    device const float* X [[buffer(0)]],           // (batch, seq, in_features) float
    device const uchar* W_packed [[buffer(1)]],    // (out_features, in_features/2) packed int4
    device const float* scales [[buffer(2)]],      // (out_features, num_groups)
    device const float* zero_points [[buffer(3)]], // (out_features, num_groups)
    device const float* bias [[buffer(4)]],        // (out_features,) or nullptr
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],
    device const uint* out_features [[buffer(8)]],
    device const uint* group_size [[buffer(9)]],
    device const uint* has_bias [[buffer(10)]],
    device float* Y [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.x;
    uint seq_idx = tid.y;
    uint batch_idx = tid.z;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len || out_idx >= *out_features) return;

    uint in_dim = *in_features;
    uint g_size = *group_size;
    uint num_groups = (in_dim + g_size - 1) / g_size;
    uint packed_in_dim = in_dim / 2;

    // Input offset
    uint x_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;

    // Weight row offset (packed)
    uint w_offset = out_idx * packed_in_dim;

    // Scale/zero point row offset
    uint sz_offset = out_idx * num_groups;

    float acc = 0.0f;

    // Process groups
    for (uint g = 0; g < num_groups; g++) {
        float s = scales[sz_offset + g];
        float zp = zero_points[sz_offset + g];

        uint group_start = g * g_size;
        uint group_end = min(group_start + g_size, in_dim);

        // Process pairs of elements within this group
        for (uint d = group_start; d < group_end; d += 2) {
            // Load packed byte
            uint pack_idx = d / 2;
            uchar packed = W_packed[w_offset + pack_idx];

            // Unpack and dequantize first element
            float x_val0 = X[x_offset + d];
            float w_val0 = s * (float(unpack_int4(packed, 0)) - zp);
            acc += x_val0 * w_val0;

            // Unpack and dequantize second element (if within bounds)
            if (d + 1 < group_end) {
                float x_val1 = X[x_offset + d + 1];
                float w_val1 = s * (float(unpack_int4(packed, 1)) - zp);
                acc += x_val1 * w_val1;
            }
        }
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
 * Pack FP32 weights to INT4 with grouped quantization.
 *
 * For each group, computes:
 *   scale = (max - min) / 15  (15 = 2^4 - 1)
 *   zero_point = -round(min / scale)
 *   quantized = clamp(round(weight / scale) + zero_point, 0, 15)
 *
 * Grid: (out_features, num_groups)
 */
[[kernel]] void quantize_int4_grouped(
    device const float* weights [[buffer(0)]],     // (out_features, in_features)
    device uchar* W_packed [[buffer(1)]],          // (out_features, in_features/2)
    device float* scales_out [[buffer(2)]],        // (out_features, num_groups)
    device float* zero_points_out [[buffer(3)]],   // (out_features, num_groups)
    device const uint* in_features [[buffer(4)]],
    device const uint* out_features [[buffer(5)]],
    device const uint* group_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.y;
    uint group_idx = tid.x;

    uint in_dim = *in_features;
    uint g_size = *group_size;
    uint num_groups = (in_dim + g_size - 1) / g_size;
    uint packed_in_dim = in_dim / 2;

    if (out_idx >= *out_features || group_idx >= num_groups) return;

    uint w_row_offset = out_idx * in_dim;
    uint packed_row_offset = out_idx * packed_in_dim;
    uint sz_offset = out_idx * num_groups + group_idx;

    uint group_start = group_idx * g_size;
    uint group_end = min(group_start + g_size, in_dim);

    // Find min/max in this group
    float mn = weights[w_row_offset + group_start];
    float mx = mn;
    for (uint d = group_start + 1; d < group_end; d++) {
        float w = weights[w_row_offset + d];
        mn = min(mn, w);
        mx = max(mx, w);
    }

    // Compute scale and zero point for unsigned 4-bit (0-15)
    float range = mx - mn;
    float s = (range > 1e-8f) ? (range / 15.0f) : 1e-8f;
    float zp = -round(mn / s);
    zp = clamp(zp, 0.0f, 15.0f);

    scales_out[sz_offset] = s;
    zero_points_out[sz_offset] = zp;

    // Quantize and pack elements in this group
    for (uint d = group_start; d < group_end; d += 2) {
        // Quantize first element
        float w0 = weights[w_row_offset + d];
        int q0 = int(round(w0 / s) + zp);
        q0 = clamp(q0, 0, 15);

        // Quantize second element (or use 0 if out of bounds)
        int q1 = 0;
        if (d + 1 < group_end) {
            float w1 = weights[w_row_offset + d + 1];
            q1 = int(round(w1 / s) + zp);
            q1 = clamp(q1, 0, 15);
        }

        // Pack into single byte: lower nibble = q0, upper nibble = q1
        uchar packed = uchar(q0) | (uchar(q1) << 4);

        uint pack_idx = d / 2;
        W_packed[packed_row_offset + pack_idx] = packed;
    }
}


/**
 * INT4 matmul with per-tensor quantization (simpler, less accurate).
 *
 * Uses a single scale/zero_point for the entire weight matrix.
 * Faster than grouped but lower accuracy.
 *
 * Grid: (out_features, seq_len, batch_size)
 */
[[kernel]] void int4_matmul_per_tensor(
    device const float* X [[buffer(0)]],
    device const uchar* W_packed [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    device const float* bias [[buffer(4)]],
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
    uint packed_in_dim = in_dim / 2;

    uint x_offset = batch_idx * (*seq_len) * in_dim + seq_idx * in_dim;
    uint w_offset = out_idx * packed_in_dim;

    float acc = 0.0f;

    // Process pairs of elements
    for (uint d = 0; d < in_dim; d += 2) {
        uint pack_idx = d / 2;
        uchar packed = W_packed[w_offset + pack_idx];

        // First element
        float x_val0 = X[x_offset + d];
        float w_val0 = s * (float(unpack_int4(packed, 0)) - zp);
        acc += x_val0 * w_val0;

        // Second element
        if (d + 1 < in_dim) {
            float x_val1 = X[x_offset + d + 1];
            float w_val1 = s * (float(unpack_int4(packed, 1)) - zp);
            acc += x_val1 * w_val1;
        }
    }

    if (*has_bias > 0) {
        acc += bias[out_idx];
    }

    uint y_offset = batch_idx * (*seq_len) * (*out_features) + seq_idx * (*out_features) + out_idx;
    Y[y_offset] = acc;
}


/**
 * Vectorized INT4 matmul using SIMD operations.
 *
 * Processes 8 INT4 values (4 bytes) at once for better throughput.
 *
 * Grid: (out_features, seq_len, batch_size)
 */
[[kernel]] void int4_matmul_vectorized(
    device const float4* X4 [[buffer(0)]],         // (batch, seq, in_features/4) float4
    device const uint* W_packed4 [[buffer(1)]],    // (out_features, in_features/8) uint (8 INT4s)
    device const float* scales [[buffer(2)]],
    device const float* zero_points [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const uint* batch_size [[buffer(5)]],
    device const uint* seq_len [[buffer(6)]],
    device const uint* in_features [[buffer(7)]],
    device const uint* out_features [[buffer(8)]],
    device const uint* group_size [[buffer(9)]],
    device const uint* has_bias [[buffer(10)]],
    device float* Y [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint out_idx = tid.x;
    uint seq_idx = tid.y;
    uint batch_idx = tid.z;

    if (batch_idx >= *batch_size || seq_idx >= *seq_len || out_idx >= *out_features) return;

    uint in_dim = *in_features;
    uint g_size = *group_size;
    uint num_groups = (in_dim + g_size - 1) / g_size;
    uint in_dim4 = in_dim / 4;
    uint packed_in_dim8 = in_dim / 8;

    uint x4_offset = batch_idx * (*seq_len) * in_dim4 + seq_idx * in_dim4;
    uint w_offset = out_idx * packed_in_dim8;
    uint sz_offset = out_idx * num_groups;

    float acc = 0.0f;

    // Process 8 elements at a time
    for (uint d8 = 0; d8 < packed_in_dim8; d8++) {
        // Determine group for first element of this chunk
        uint base_idx = d8 * 8;
        uint g = base_idx / g_size;
        float s = scales[sz_offset + g];
        float zp = zero_points[sz_offset + g];

        // Load packed uint containing 8 INT4 values
        uint packed = W_packed4[w_offset + d8];

        // Load two float4 vectors for 8 input elements
        float4 x0 = X4[x4_offset + d8 * 2];
        float4 x1 = X4[x4_offset + d8 * 2 + 1];

        // Unpack and dequantize 8 INT4 values
        float w0 = s * (float((packed >> 0) & 0xF) - zp);
        float w1 = s * (float((packed >> 4) & 0xF) - zp);
        float w2 = s * (float((packed >> 8) & 0xF) - zp);
        float w3 = s * (float((packed >> 12) & 0xF) - zp);
        float w4 = s * (float((packed >> 16) & 0xF) - zp);
        float w5 = s * (float((packed >> 20) & 0xF) - zp);
        float w6 = s * (float((packed >> 24) & 0xF) - zp);
        float w7 = s * (float((packed >> 28) & 0xF) - zp);

        // Accumulate
        acc += x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
        acc += x1.x * w4 + x1.y * w5 + x1.z * w6 + x1.w * w7;
    }

    if (*has_bias > 0) {
        acc += bias[out_idx];
    }

    uint y_offset = batch_idx * (*seq_len) * (*out_features) + seq_idx * (*out_features) + out_idx;
    Y[y_offset] = acc;
}
