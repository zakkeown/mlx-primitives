"""ANE-accelerated primitive operations."""

from mlx_primitives.ane.primitives.matmul import ane_linear, ane_matmul

__all__ = [
    "ane_matmul",
    "ane_linear",
]
