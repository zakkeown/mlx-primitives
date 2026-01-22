"""Centralized dtype utilities for MLX arrays.

This module provides consistent dtype handling across the codebase,
avoiding duplicated implementations in multiple files.
"""

from typing import Dict

import mlx.core as mx
import numpy as np


# MLX dtype to size in bytes
_MLX_DTYPE_SIZES: Dict[mx.Dtype, int] = {
    mx.float32: 4,
    mx.float16: 2,
    mx.bfloat16: 2,
    mx.float64: 8,
    mx.int8: 1,
    mx.int16: 2,
    mx.int32: 4,
    mx.int64: 8,
    mx.uint8: 1,
    mx.uint16: 2,
    mx.uint32: 4,
    mx.uint64: 8,
    mx.bool_: 1,
    mx.complex64: 8,
}

# MLX dtype to NumPy dtype mapping
_MLX_TO_NUMPY: Dict[mx.Dtype, np.dtype] = {
    mx.float32: np.dtype(np.float32),
    mx.float16: np.dtype(np.float16),
    mx.float64: np.dtype(np.float64),
    mx.int8: np.dtype(np.int8),
    mx.int16: np.dtype(np.int16),
    mx.int32: np.dtype(np.int32),
    mx.int64: np.dtype(np.int64),
    mx.uint8: np.dtype(np.uint8),
    mx.uint16: np.dtype(np.uint16),
    mx.uint32: np.dtype(np.uint32),
    mx.uint64: np.dtype(np.uint64),
    # bfloat16 has no direct numpy equivalent, use float32 for conversion
    mx.bfloat16: np.dtype(np.float32),
    mx.bool_: np.dtype(np.bool_),
    mx.complex64: np.dtype(np.complex64),
}

# NumPy dtype to MLX dtype mapping
_NUMPY_TO_MLX: Dict[type, mx.Dtype] = {
    np.float32: mx.float32,
    np.float16: mx.float16,
    np.float64: mx.float64,
    np.int8: mx.int8,
    np.int16: mx.int16,
    np.int32: mx.int32,
    np.int64: mx.int64,
    np.uint8: mx.uint8,
    np.uint16: mx.uint16,
    np.uint32: mx.uint32,
    np.uint64: mx.uint64,
    np.bool_: mx.bool_,
    np.complex64: mx.complex64,
}


def get_dtype_size(dtype: mx.Dtype) -> int:
    """Get size in bytes for an MLX dtype.

    Args:
        dtype: MLX data type.

    Returns:
        Size in bytes. Returns 4 (float32 size) for unknown dtypes.

    Example:
        >>> get_dtype_size(mx.float16)
        2
        >>> get_dtype_size(mx.int64)
        8
    """
    return _MLX_DTYPE_SIZES.get(dtype, 4)


def mlx_to_numpy_dtype(dtype: mx.Dtype) -> np.dtype:
    """Convert MLX dtype to NumPy dtype.

    Args:
        dtype: MLX data type.

    Returns:
        Corresponding NumPy dtype. Returns np.float32 for unknown dtypes.

    Note:
        bfloat16 is mapped to float32 since NumPy doesn't support bfloat16.

    Example:
        >>> mlx_to_numpy_dtype(mx.int32)
        dtype('int32')
    """
    return _MLX_TO_NUMPY.get(dtype, np.dtype(np.float32))


def numpy_to_mlx_dtype(dtype: np.dtype) -> mx.Dtype:
    """Convert NumPy dtype to MLX dtype.

    Args:
        dtype: NumPy data type (or dtype object).

    Returns:
        Corresponding MLX dtype. Returns mx.float32 for unknown dtypes.

    Example:
        >>> numpy_to_mlx_dtype(np.dtype(np.int64))
        mlx.core.int64
    """
    # Handle numpy dtype objects
    dtype = np.dtype(dtype)
    return _NUMPY_TO_MLX.get(dtype.type, mx.float32)


def is_floating_point(dtype: mx.Dtype) -> bool:
    """Check if dtype is a floating point type.

    Args:
        dtype: MLX data type.

    Returns:
        True if dtype is float16, bfloat16, float32, or float64.
    """
    return dtype in (mx.float16, mx.bfloat16, mx.float32, mx.float64)


def is_integer(dtype: mx.Dtype) -> bool:
    """Check if dtype is an integer type.

    Args:
        dtype: MLX data type.

    Returns:
        True if dtype is any signed or unsigned integer type.
    """
    return dtype in (
        mx.int8, mx.int16, mx.int32, mx.int64,
        mx.uint8, mx.uint16, mx.uint32, mx.uint64,
    )


def get_dtype_info(dtype: mx.Dtype) -> dict:
    """Get comprehensive information about a dtype.

    Args:
        dtype: MLX data type.

    Returns:
        Dictionary with dtype information:
        - size_bytes: Size in bytes
        - is_floating: Whether it's a floating point type
        - is_integer: Whether it's an integer type
        - numpy_dtype: Corresponding NumPy dtype (if available)
    """
    return {
        "size_bytes": get_dtype_size(dtype),
        "is_floating": is_floating_point(dtype),
        "is_integer": is_integer(dtype),
        "numpy_dtype": _MLX_TO_NUMPY.get(dtype),
    }
