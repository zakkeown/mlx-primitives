"""Runtime support for Metal-Triton kernels.

Provides:
- Kernel caching
- MLX integration utilities
- Auto-tuning runtime
"""

from mlx_primitives.dsl.runtime.kernel_cache import KernelCache, get_kernel_cache
from mlx_primitives.dsl.runtime.mlx_wrapper import (
    create_mlx_kernel,
    execute_mlx_kernel,
    validate_inputs,
)

__all__ = [
    "KernelCache",
    "get_kernel_cache",
    "create_mlx_kernel",
    "execute_mlx_kernel",
    "validate_inputs",
]
