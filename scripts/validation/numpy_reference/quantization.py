"""NumPy reference implementations for quantization operations."""

import numpy as np


def quantize(
    x: np.ndarray,
    num_bits: int = 8,
    symmetric: bool = True,
    axis: int = -1,
) -> tuple:
    """Quantize floating-point values to integers.

    Args:
        x: Input tensor to quantize
        num_bits: Number of bits for quantization (4 or 8)
        symmetric: Whether to use symmetric quantization
        axis: Axis along which to compute scale

    Returns:
        Tuple of:
        - quantized: Quantized integer values
        - scale: Scale factors
        - zero_point: Zero points (None for symmetric)
    """
    if symmetric:
        # Symmetric quantization: [-qmax, qmax]
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax

        # Compute scale from max absolute value
        max_val = np.max(np.abs(x), axis=axis, keepdims=True)
        scale = max_val / qmax
        scale = np.where(scale == 0, 1.0, scale)  # Avoid division by zero

        # Quantize
        quantized = np.clip(np.round(x / scale), qmin, qmax).astype(np.int8 if num_bits == 8 else np.int8)

        return quantized, scale, None
    else:
        # Asymmetric quantization: [0, qmax]
        qmax = 2 ** num_bits - 1
        qmin = 0

        # Compute min/max
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)

        # Compute scale and zero point
        scale = (x_max - x_min) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(-x_min / scale).astype(np.int32)

        # Quantize
        quantized = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(np.uint8)

        return quantized, scale, zero_point


def dequantize(
    quantized: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray = None,
) -> np.ndarray:
    """Dequantize integer values back to floating-point.

    Args:
        quantized: Quantized integer values
        scale: Scale factors
        zero_point: Zero points (None for symmetric quantization)

    Returns:
        Dequantized floating-point values
    """
    quantized = quantized.astype(np.float32)

    if zero_point is None:
        # Symmetric
        return quantized * scale
    else:
        # Asymmetric
        return (quantized - zero_point) * scale


def quantize_per_token(
    x: np.ndarray,
    num_bits: int = 8,
) -> tuple:
    """Per-token quantization (common for activations).

    Quantizes each token independently.

    Args:
        x: Input tensor, shape (..., hidden_dim)
        num_bits: Number of bits

    Returns:
        Tuple of (quantized, scale)
    """
    return quantize(x, num_bits=num_bits, symmetric=True, axis=-1)


def quantize_per_channel(
    x: np.ndarray,
    num_bits: int = 8,
) -> tuple:
    """Per-channel quantization (common for weights).

    Quantizes each output channel independently.

    Args:
        x: Weight tensor, shape (out_features, in_features)
        num_bits: Number of bits

    Returns:
        Tuple of (quantized, scale)
    """
    return quantize(x, num_bits=num_bits, symmetric=True, axis=1)


def quantized_matmul(
    x_quant: np.ndarray,
    x_scale: np.ndarray,
    w_quant: np.ndarray,
    w_scale: np.ndarray,
) -> np.ndarray:
    """Quantized matrix multiplication.

    Performs matmul in integer domain and rescales.

    Args:
        x_quant: Quantized activations, shape (..., in_features)
        x_scale: Activation scales
        w_quant: Quantized weights, shape (out_features, in_features)
        w_scale: Weight scales

    Returns:
        Result tensor, shape (..., out_features)
    """
    # Integer matmul
    result_int = np.matmul(x_quant.astype(np.int32), w_quant.T.astype(np.int32))

    # Rescale
    combined_scale = x_scale * w_scale.T
    result = result_int.astype(np.float32) * combined_scale

    return result
