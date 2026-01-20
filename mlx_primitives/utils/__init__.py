"""Utility functions for MLX Primitives."""

from mlx_primitives.utils.array_ops import (
    argwhere,
    gather_where,
    nonzero,
    scatter_where,
)
from mlx_primitives.utils.dtypes import (
    get_dtype_info,
    get_dtype_size,
    is_floating_point,
    is_integer,
    mlx_to_numpy_dtype,
    numpy_to_mlx_dtype,
)
from mlx_primitives.utils.logging import (
    get_logger,
    has_metal_kernels,
    log_fallback,
    METAL_MIN_SEQ_LEN,
    RAISE_ON_METAL_FAILURE,
    should_use_metal,
    validate_dtype_for_metal,
)

__all__: list[str] = [
    # Array operations
    "argwhere",
    "gather_where",
    "nonzero",
    "scatter_where",
    # Dtype utilities
    "get_dtype_info",
    "get_dtype_size",
    "is_floating_point",
    "is_integer",
    "mlx_to_numpy_dtype",
    "numpy_to_mlx_dtype",
    # Logging
    "get_logger",
    "has_metal_kernels",
    "log_fallback",
    "METAL_MIN_SEQ_LEN",
    "RAISE_ON_METAL_FAILURE",
    "should_use_metal",
    "validate_dtype_for_metal",
]
