"""Tensor conversion between MLX and Core ML formats.

This module provides utilities for converting tensors between
MLX arrays and Core ML-compatible formats for ANE execution.
"""

from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np


def mlx_to_coreml_input(
    tensors: List[mx.array],
    names: List[str],
) -> Dict[str, np.ndarray]:
    """Convert MLX tensors to Core ML input dictionary.

    Core ML models expect inputs as a dictionary mapping
    input names to NumPy arrays.

    Note: On Apple Silicon unified memory, this conversion
    is efficient as data doesn't need to be copied between
    CPU and GPU memory spaces.

    Args:
        tensors: List of MLX arrays to convert.
        names: Corresponding input names for the Core ML model.

    Returns:
        Dictionary mapping names to NumPy arrays.

    Example:
        >>> inputs = mlx_to_coreml_input(
        ...     [query, key, value],
        ...     ["Q", "K", "V"]
        ... )
        >>> outputs = model.predict(inputs)
    """
    if len(tensors) != len(names):
        raise ValueError(
            f"Number of tensors ({len(tensors)}) must match "
            f"number of names ({len(names)})"
        )

    # Ensure all tensors are evaluated
    mx.eval(*tensors)

    result = {}
    for tensor, name in zip(tensors, names):
        # Convert to numpy
        np_array = np.array(tensor)

        # Core ML prefers float16 or float32
        # Convert other dtypes as needed
        if np_array.dtype not in (np.float32, np.float16):
            if np.issubdtype(np_array.dtype, np.floating):
                np_array = np_array.astype(np.float32)
            elif np.issubdtype(np_array.dtype, np.integer):
                # Keep integer types as-is for quantized models
                pass

        result[name] = np_array

    return result


def coreml_to_mlx(
    output: Dict[str, np.ndarray],
    output_name: str,
    target_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Convert Core ML output to MLX tensor.

    Args:
        output: Core ML model output dictionary.
        output_name: Name of the output to extract.
        target_dtype: Target MLX dtype for the result.

    Returns:
        MLX array with the output data.
    """
    if output_name not in output:
        available = list(output.keys())
        raise KeyError(
            f"Output '{output_name}' not found. Available: {available}"
        )

    np_array = output[output_name]
    mlx_array = mx.array(np_array)

    # Convert dtype if needed
    if mlx_array.dtype != target_dtype:
        mlx_array = mlx_array.astype(target_dtype)

    return mlx_array


def coreml_outputs_to_mlx(
    output: Dict[str, np.ndarray],
    output_names: List[str],
    target_dtype: mx.Dtype = mx.float32,
) -> List[mx.array]:
    """Convert multiple Core ML outputs to MLX tensors.

    Args:
        output: Core ML model output dictionary.
        output_names: Names of outputs to extract.
        target_dtype: Target MLX dtype for results.

    Returns:
        List of MLX arrays in the same order as output_names.
    """
    return [
        coreml_to_mlx(output, name, target_dtype)
        for name in output_names
    ]


def estimate_transfer_overhead_ms(
    shapes: List[Tuple[int, ...]],
    dtype_bytes: int = 4,
) -> float:
    """Estimate data transfer overhead between MLX and ANE.

    This helps dispatch decisions - if transfer overhead exceeds
    compute benefit, stay on GPU.

    Args:
        shapes: List of tensor shapes.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).

    Returns:
        Estimated transfer time in milliseconds.
    """
    total_bytes = sum(_product(shape) * dtype_bytes for shape in shapes)

    # On unified memory, there's no actual copy, but there's still
    # synchronization and cache invalidation overhead.
    # Estimate based on ~20 GB/s effective throughput.
    effective_bandwidth_gbps = 20.0

    time_s = total_bytes / (effective_bandwidth_gbps * 1e9)
    return time_s * 1000  # Convert to ms


def prepare_for_ane(
    tensor: mx.array,
    preferred_dtype: str = "float16",
) -> mx.array:
    """Prepare tensor for ANE execution.

    ANE typically performs best with float16 data. This function
    converts tensors to the preferred dtype and ensures they're
    in contiguous memory layout.

    Args:
        tensor: Input MLX array.
        preferred_dtype: Preferred dtype ("float16" or "float32").

    Returns:
        Tensor prepared for ANE execution.
    """
    # Determine target dtype
    target_dtype = mx.float16 if preferred_dtype == "float16" else mx.float32

    # Convert if needed
    if tensor.dtype != target_dtype:
        tensor = tensor.astype(target_dtype)

    # Ensure evaluated and contiguous
    mx.eval(tensor)

    return tensor


def batch_prepare_for_ane(
    tensors: List[mx.array],
    preferred_dtype: str = "float16",
) -> List[mx.array]:
    """Prepare multiple tensors for ANE execution.

    Args:
        tensors: List of input tensors.
        preferred_dtype: Preferred dtype for all tensors.

    Returns:
        List of prepared tensors.
    """
    prepared = [prepare_for_ane(t, preferred_dtype) for t in tensors]
    mx.eval(*prepared)
    return prepared


def _product(shape: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for dim in shape:
        result *= dim
    return result


# Type conversion utilities

_MLX_TO_NUMPY_DTYPE = {
    mx.float32: np.float32,
    mx.float16: np.float16,
    mx.int32: np.int32,
    mx.int16: np.int16,
    mx.int8: np.int8,
    mx.uint32: np.uint32,
    mx.uint16: np.uint16,
    mx.uint8: np.uint8,
    mx.bool_: np.bool_,
}

_NUMPY_TO_MLX_DTYPE = {v: k for k, v in _MLX_TO_NUMPY_DTYPE.items()}


def mlx_dtype_to_numpy(dtype: mx.Dtype) -> np.dtype:
    """Convert MLX dtype to NumPy dtype."""
    return _MLX_TO_NUMPY_DTYPE.get(dtype, np.float32)


def numpy_dtype_to_mlx(dtype: np.dtype) -> mx.Dtype:
    """Convert NumPy dtype to MLX dtype."""
    dtype = np.dtype(dtype)
    return _NUMPY_TO_MLX_DTYPE.get(dtype.type, mx.float32)
