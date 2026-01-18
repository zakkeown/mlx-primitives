"""Quantized linear operations for MLX.

Apple Silicon has NO native INT8/INT4 matmul instructions. The benefit of
quantization comes from reduced memory bandwidth - 4x smaller weights (INT8)
or 8x smaller (INT4) means faster loading from memory.

The computation still happens in float32:
1. Load quantized weights
2. Dequantize: scale * (weight - zero_point)
3. Compute matmul in float32

Supported formats:
- INT8: 8-bit signed integers, per-tensor or per-channel quantization
- INT4: 4-bit integers packed 2 per byte, grouped quantization (e.g., group_size=128)
"""

import math
from typing import Literal, Optional, Tuple

import mlx.core as mx

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_int8_matmul_kernel: Optional[mx.fast.metal_kernel] = None
_int8_matmul_per_channel_kernel: Optional[mx.fast.metal_kernel] = None
_int4_matmul_kernel: Optional[mx.fast.metal_kernel] = None


def _get_int8_matmul_kernel() -> mx.fast.metal_kernel:
    """Get or create INT8 per-tensor matmul kernel."""
    global _int8_matmul_kernel
    if _int8_matmul_kernel is None:
        source = """
        uint out_idx = thread_position_in_grid.x;
        uint seq_idx = thread_position_in_grid.y;
        uint batch_idx = thread_position_in_grid.z;

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        float s = scale[0];
        float zp = zero_point[0];

        uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;
        uint w_offset = out_idx * in_features;

        float acc = 0.0f;
        for (uint d = 0; d < in_features; d++) {
            float x_val = X[x_offset + d];
            // W_quant stored as int8 (-128 to 127), dequantize
            float w_val = s * (float(W_quant[w_offset + d]) - zp);
            acc += x_val * w_val;
        }

        if (has_bias > 0) {
            acc += bias[out_idx];
        }

        uint y_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
        Y[y_offset] = acc;
        """
        _int8_matmul_kernel = mx.fast.metal_kernel(
            name="int8_matmul",
            input_names=[
                "X", "W_quant", "scale", "zero_point", "bias",
                "batch_size", "seq_len", "in_features", "out_features", "has_bias"
            ],
            output_names=["Y"],
            source=source,
        )
    return _int8_matmul_kernel


def _get_int8_matmul_per_channel_kernel() -> mx.fast.metal_kernel:
    """Get or create INT8 per-channel matmul kernel."""
    global _int8_matmul_per_channel_kernel
    if _int8_matmul_per_channel_kernel is None:
        source = """
        uint out_idx = thread_position_in_grid.x;
        uint seq_idx = thread_position_in_grid.y;
        uint batch_idx = thread_position_in_grid.z;

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        float s = scales[out_idx];
        float zp = zero_points[out_idx];

        uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;
        uint w_offset = out_idx * in_features;

        float acc = 0.0f;
        for (uint d = 0; d < in_features; d++) {
            float x_val = X[x_offset + d];
            float w_val = s * (float(W_quant[w_offset + d]) - zp);
            acc += x_val * w_val;
        }

        if (has_bias > 0) {
            acc += bias[out_idx];
        }

        uint y_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
        Y[y_offset] = acc;
        """
        _int8_matmul_per_channel_kernel = mx.fast.metal_kernel(
            name="int8_matmul_per_channel",
            input_names=[
                "X", "W_quant", "scales", "zero_points", "bias",
                "batch_size", "seq_len", "in_features", "out_features", "has_bias"
            ],
            output_names=["Y"],
            source=source,
        )
    return _int8_matmul_per_channel_kernel


def _get_int4_matmul_kernel() -> mx.fast.metal_kernel:
    """Get or create INT4 grouped matmul kernel."""
    global _int4_matmul_kernel
    if _int4_matmul_kernel is None:
        source = """
        uint out_idx = thread_position_in_grid.x;
        uint seq_idx = thread_position_in_grid.y;
        uint batch_idx = thread_position_in_grid.z;

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        uint num_groups = (in_features + group_size - 1) / group_size;
        uint packed_in_dim = in_features / 2;

        uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;
        uint w_offset = out_idx * packed_in_dim;
        uint sz_offset = out_idx * num_groups;

        float acc = 0.0f;

        for (uint g = 0; g < num_groups; g++) {
            float s = scales[sz_offset + g];
            float zp = zero_points[sz_offset + g];

            uint group_start = g * group_size;
            uint group_end = min(group_start + group_size, in_features);

            for (uint d = group_start; d < group_end; d += 2) {
                uint pack_idx = d / 2;
                uint packed = W_packed[w_offset + pack_idx];

                // Unpack lower nibble
                int q0 = int(packed & 0xFFu);
                // Convert to signed if > 7
                if (q0 >= 8) q0 = q0 - 16;

                float x_val0 = X[x_offset + d];
                float w_val0 = s * (float(q0) - zp);
                acc += x_val0 * w_val0;

                if (d + 1 < group_end) {
                    // Unpack upper nibble (stored in next byte position)
                    int q1 = int((packed >> 8) & 0xFFu);
                    if (q1 >= 8) q1 = q1 - 16;

                    float x_val1 = X[x_offset + d + 1];
                    float w_val1 = s * (float(q1) - zp);
                    acc += x_val1 * w_val1;
                }
            }
        }

        if (has_bias > 0) {
            acc += bias[out_idx];
        }

        uint y_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
        Y[y_offset] = acc;
        """
        _int4_matmul_kernel = mx.fast.metal_kernel(
            name="int4_matmul_grouped",
            input_names=[
                "X", "W_packed", "scales", "zero_points", "bias",
                "batch_size", "seq_len", "in_features", "out_features",
                "group_size", "has_bias"
            ],
            output_names=["Y"],
            source=source,
        )
    return _int4_matmul_kernel


def quantize_int8(
    weights: mx.array,
    per_channel: bool = True,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Quantize FP32 weights to INT8.

    Args:
        weights: Weight tensor of shape (out_features, in_features).
        per_channel: If True, use per-channel (row) quantization.
            If False, use per-tensor quantization.

    Returns:
        Tuple of (quantized_weights, scales, zero_points):
        - quantized_weights: INT8 tensor of shape (out_features, in_features)
        - scales: FP32 tensor, shape (out_features,) if per_channel else (1,)
        - zero_points: FP32 tensor, same shape as scales
    """
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D weights, got {weights.ndim}D")

    out_features, in_features = weights.shape

    if per_channel:
        # Per-channel: compute min/max per row
        w_min = mx.min(weights, axis=1)  # (out_features,)
        w_max = mx.max(weights, axis=1)

        # Avoid division by zero
        scale = (w_max - w_min) / 255.0
        scale = mx.maximum(scale, 1e-8)

        # Zero point (for asymmetric quantization)
        zero_point = -mx.round(w_min / scale)
        zero_point = mx.clip(zero_point, 0, 255)

        # Quantize: q = round(w / scale) + zero_point
        # Broadcast scale and zero_point for element-wise ops
        quantized = mx.round(weights / scale[:, None]) + zero_point[:, None]
        quantized = mx.clip(quantized, 0, 255)

        # Convert to signed int8 (-128 to 127 range)
        quantized = (quantized - 128).astype(mx.int8)
        # Adjust zero_point to match signed representation
        zero_point = zero_point - 128

    else:
        # Per-tensor: single scale/zero_point for entire matrix
        w_min = mx.min(weights)
        w_max = mx.max(weights)

        scale = (w_max - w_min) / 255.0
        scale = mx.maximum(scale, mx.array(1e-8))

        zero_point = -mx.round(w_min / scale)
        zero_point = mx.clip(zero_point, 0, 255)

        quantized = mx.round(weights / scale) + zero_point
        quantized = mx.clip(quantized, 0, 255)

        # Convert to signed int8
        quantized = (quantized - 128).astype(mx.int8)
        zero_point = zero_point - 128

        # Keep as scalars (1,) shape
        scale = mx.array([float(scale)])
        zero_point = mx.array([float(zero_point)])

    mx.eval(quantized, scale, zero_point)
    return quantized, scale.astype(mx.float32), zero_point.astype(mx.float32)


def quantize_int4(
    weights: mx.array,
    group_size: int = 128,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Quantize FP32 weights to INT4 with grouped quantization.

    Two INT4 values are packed per byte (uint8):
    - Lower nibble: even indices
    - Upper nibble: odd indices

    Args:
        weights: Weight tensor of shape (out_features, in_features).
        group_size: Number of elements per quantization group.

    Returns:
        Tuple of (packed_weights, scales, zero_points):
        - packed_weights: UINT8 tensor of shape (out_features, in_features // 2)
        - scales: FP32 tensor of shape (out_features, num_groups)
        - zero_points: FP32 tensor of shape (out_features, num_groups)
    """
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D weights, got {weights.ndim}D")

    out_features, in_features = weights.shape

    if in_features % 2 != 0:
        raise ValueError(f"in_features must be even, got {in_features}")

    num_groups = (in_features + group_size - 1) // group_size

    # Reshape for group-wise operations
    # (out_features, num_groups, group_size)
    padded_in = num_groups * group_size
    if padded_in > in_features:
        # Pad weights if needed
        pad_amount = padded_in - in_features
        weights = mx.concatenate([
            weights,
            mx.zeros((out_features, pad_amount), dtype=weights.dtype)
        ], axis=1)

    weights_grouped = weights.reshape(out_features, num_groups, group_size)

    # Compute min/max per group
    w_min = mx.min(weights_grouped, axis=2)  # (out_features, num_groups)
    w_max = mx.max(weights_grouped, axis=2)

    # Scale and zero point for 4-bit (0-15 range)
    scale = (w_max - w_min) / 15.0
    scale = mx.maximum(scale, 1e-8)

    zero_point = -mx.round(w_min / scale)
    zero_point = mx.clip(zero_point, 0, 15)

    # Quantize to 0-15 range
    quantized = mx.round(weights_grouped / scale[:, :, None]) + zero_point[:, :, None]
    quantized = mx.clip(quantized, 0, 15).astype(mx.uint8)

    # Reshape back to 2D
    quantized = quantized.reshape(out_features, padded_in)

    # Trim to original size
    if padded_in > in_features:
        quantized = quantized[:, :in_features]

    # Pack pairs of INT4 into bytes
    # Even indices go to lower nibble, odd indices to upper nibble
    even_vals = quantized[:, 0::2]  # (out_features, in_features // 2)
    odd_vals = quantized[:, 1::2]
    packed = even_vals | (odd_vals << 4)  # Lower nibble | upper nibble

    mx.eval(packed, scale, zero_point)
    return packed.astype(mx.uint8), scale.astype(mx.float32), zero_point.astype(mx.float32)


def int8_linear(
    x: mx.array,
    W_quant: mx.array,
    scale: mx.array,
    zero_point: mx.array,
    bias: Optional[mx.array] = None,
    use_metal: bool = True,
) -> mx.array:
    """INT8 quantized linear layer.

    Computes: Y = X @ dequant(W_quant).T + bias
    Where: dequant(w) = scale * (w - zero_point)

    Args:
        x: Input tensor of shape (batch, seq_len, in_features).
        W_quant: Quantized weights of shape (out_features, in_features), INT8.
        scale: Scale tensor, (out_features,) for per-channel or (1,) for per-tensor.
        zero_point: Zero point tensor, same shape as scale.
        bias: Optional bias of shape (out_features,).
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, out_features).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input (batch, seq, features), got {x.ndim}D")

    batch_size, seq_len, in_features = x.shape
    out_features = W_quant.shape[0]

    # Determine if per-channel or per-tensor
    per_channel = scale.size > 1

    if not use_metal or not _HAS_METAL or seq_len < 8:
        return _reference_int8_linear(x, W_quant, scale, zero_point, bias, per_channel)

    try:
        return _metal_int8_linear(x, W_quant, scale, zero_point, bias, per_channel)
    except Exception:
        return _reference_int8_linear(x, W_quant, scale, zero_point, bias, per_channel)


def _metal_int8_linear(
    x: mx.array,
    W_quant: mx.array,
    scale: mx.array,
    zero_point: mx.array,
    bias: Optional[mx.array],
    per_channel: bool,
) -> mx.array:
    """Metal kernel INT8 linear."""
    batch_size, seq_len, in_features = x.shape
    out_features = W_quant.shape[0]

    if per_channel:
        kernel = _get_int8_matmul_per_channel_kernel()
    else:
        kernel = _get_int8_matmul_kernel()

    x = mx.ascontiguousarray(x.astype(mx.float32))
    W_quant = mx.ascontiguousarray(W_quant.astype(mx.int8))
    scale = mx.ascontiguousarray(scale.astype(mx.float32))
    zero_point = mx.ascontiguousarray(zero_point.astype(mx.float32))

    if bias is not None:
        bias = mx.ascontiguousarray(bias.astype(mx.float32))
        has_bias = 1
    else:
        bias = mx.zeros((out_features,), dtype=mx.float32)
        has_bias = 0

    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    in_arr = mx.array([in_features], dtype=mx.uint32)
    out_arr = mx.array([out_features], dtype=mx.uint32)
    has_bias_arr = mx.array([has_bias], dtype=mx.uint32)

    if per_channel:
        inputs = [x, W_quant, scale, zero_point, bias,
                  batch_arr, seq_arr, in_arr, out_arr, has_bias_arr]
    else:
        inputs = [x, W_quant, scale, zero_point, bias,
                  batch_arr, seq_arr, in_arr, out_arr, has_bias_arr]

    outputs = kernel(
        inputs=inputs,
        grid=(out_features, seq_len, batch_size),
        threadgroup=(min(out_features, 64), 1, 1),
        output_shapes=[(batch_size, seq_len, out_features)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_int8_linear(
    x: mx.array,
    W_quant: mx.array,
    scale: mx.array,
    zero_point: mx.array,
    bias: Optional[mx.array],
    per_channel: bool,
) -> mx.array:
    """Reference INT8 linear implementation."""
    # Dequantize weights
    W_float = W_quant.astype(mx.float32)
    if per_channel:
        W_dequant = scale[:, None] * (W_float - zero_point[:, None])
    else:
        W_dequant = scale * (W_float - zero_point)

    # Matmul
    out = x @ W_dequant.T

    if bias is not None:
        out = out + bias

    return out


def int4_linear(
    x: mx.array,
    W_packed: mx.array,
    scales: mx.array,
    zero_points: mx.array,
    bias: Optional[mx.array] = None,
    group_size: int = 128,
    use_metal: bool = True,
) -> mx.array:
    """INT4 quantized linear layer with grouped quantization.

    Args:
        x: Input tensor of shape (batch, seq_len, in_features).
        W_packed: Packed INT4 weights of shape (out_features, in_features // 2).
        scales: Per-group scales of shape (out_features, num_groups).
        zero_points: Per-group zero points of shape (out_features, num_groups).
        bias: Optional bias of shape (out_features,).
        group_size: Quantization group size.
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, out_features).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input (batch, seq, features), got {x.ndim}D")

    batch_size, seq_len, in_features = x.shape
    out_features = W_packed.shape[0]

    # For now, always use reference implementation as INT4 kernel is complex
    return _reference_int4_linear(
        x, W_packed, scales, zero_points, bias, group_size
    )


def _reference_int4_linear(
    x: mx.array,
    W_packed: mx.array,
    scales: mx.array,
    zero_points: mx.array,
    bias: Optional[mx.array],
    group_size: int,
) -> mx.array:
    """Reference INT4 linear implementation."""
    out_features, packed_in = W_packed.shape
    in_features = packed_in * 2

    # Unpack INT4 values
    W_packed_int = W_packed.astype(mx.int32)
    even_vals = W_packed_int & 0xF  # Lower nibble
    odd_vals = (W_packed_int >> 4) & 0xF  # Upper nibble

    # Interleave to get original order
    # Shape: (out_features, in_features)
    W_int4 = mx.zeros((out_features, in_features), dtype=mx.float32)

    # Even indices
    even_indices = mx.arange(0, in_features, 2)
    odd_indices = mx.arange(1, in_features, 2)

    # This is a simplified unpacking - in practice you'd do this more efficiently
    # For the reference implementation, we'll dequantize group by group
    num_groups = scales.shape[1]

    W_dequant = mx.zeros((out_features, in_features), dtype=mx.float32)

    for g in range(num_groups):
        group_start = g * group_size
        group_end = min(group_start + group_size, in_features)

        s = scales[:, g:g+1]  # (out_features, 1)
        zp = zero_points[:, g:g+1]

        # Get packed bytes for this group
        pack_start = group_start // 2
        pack_end = (group_end + 1) // 2

        for d in range(group_start, group_end, 2):
            pack_idx = d // 2
            # Get even value (lower nibble)
            q0 = (W_packed[:, pack_idx].astype(mx.float32)) % 16
            W_dequant[:, d] = s[:, 0] * (q0 - zp[:, 0])

            if d + 1 < group_end:
                # Get odd value (upper nibble)
                q1 = (W_packed[:, pack_idx].astype(mx.float32) // 16) % 16
                W_dequant[:, d + 1] = s[:, 0] * (q1 - zp[:, 0])

    # Matmul
    out = x @ W_dequant.T

    if bias is not None:
        out = out + bias

    return out


class QuantizedLinear:
    """Quantized linear layer wrapper.

    Supports INT8 and INT4 quantization with automatic weight quantization
    on initialization.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bits (4 or 8).
        group_size: Group size for INT4 grouped quantization.
        bias: Whether to include bias.

    Example:
        >>> layer = QuantizedLinear(768, 3072, bits=8)
        >>> layer.quantize_weights(pretrained_weights)
        >>> out = layer(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: Literal[4, 8] = 8,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Initialize with random weights (will typically be replaced)
        scale = 1.0 / math.sqrt(in_features)
        self._fp_weights = mx.random.normal((out_features, in_features)) * scale

        # Quantized weights (populated after quantize_weights call)
        self.W_quant: Optional[mx.array] = None
        self.scales: Optional[mx.array] = None
        self.zero_points: Optional[mx.array] = None

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Auto-quantize on init
        self.quantize_weights(self._fp_weights)

    def quantize_weights(self, weights: mx.array) -> None:
        """Quantize the provided weights.

        Args:
            weights: FP32 weights of shape (out_features, in_features).
        """
        if self.bits == 8:
            self.W_quant, self.scales, self.zero_points = quantize_int8(
                weights, per_channel=True
            )
        elif self.bits == 4:
            self.W_quant, self.scales, self.zero_points = quantize_int4(
                weights, group_size=self.group_size
            )
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor (batch, seq, in_features).

        Returns:
            Output tensor (batch, seq, out_features).
        """
        if self.W_quant is None:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        if self.bits == 8:
            return int8_linear(
                x, self.W_quant, self.scales, self.zero_points, self.bias
            )
        else:
            return int4_linear(
                x, self.W_quant, self.scales, self.zero_points,
                self.bias, self.group_size
            )


def dequantize_int8(
    W_quant: mx.array,
    scale: mx.array,
    zero_point: mx.array,
) -> mx.array:
    """Dequantize INT8 weights back to FP32.

    Args:
        W_quant: Quantized weights (out_features, in_features), INT8.
        scale: Scale tensor.
        zero_point: Zero point tensor.

    Returns:
        Dequantized FP32 weights.
    """
    W_float = W_quant.astype(mx.float32)
    if scale.size > 1:
        # Per-channel
        return scale[:, None] * (W_float - zero_point[:, None])
    else:
        # Per-tensor
        return scale * (W_float - zero_point)


def dequantize_int4(
    W_packed: mx.array,
    scales: mx.array,
    zero_points: mx.array,
    group_size: int = 128,
) -> mx.array:
    """Dequantize INT4 weights back to FP32.

    Args:
        W_packed: Packed weights (out_features, in_features // 2).
        scales: Per-group scales (out_features, num_groups).
        zero_points: Per-group zero points (out_features, num_groups).
        group_size: Quantization group size.

    Returns:
        Dequantized FP32 weights (out_features, in_features).
    """
    out_features, packed_in = W_packed.shape
    in_features = packed_in * 2
    num_groups = scales.shape[1]

    W_dequant = mx.zeros((out_features, in_features), dtype=mx.float32)

    for g in range(num_groups):
        group_start = g * group_size
        group_end = min(group_start + group_size, in_features)

        s = scales[:, g:g+1]
        zp = zero_points[:, g:g+1]

        for d in range(group_start, group_end, 2):
            pack_idx = d // 2
            q0 = (W_packed[:, pack_idx].astype(mx.float32)) % 16
            W_dequant[:, d] = s[:, 0] * (q0 - zp[:, 0])

            if d + 1 < group_end:
                q1 = (W_packed[:, pack_idx].astype(mx.float32) // 16) % 16
                W_dequant[:, d + 1] = s[:, 0] * (q1 - zp[:, 0])

    return W_dequant
