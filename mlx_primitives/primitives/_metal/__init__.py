"""Metal kernel wrappers for primitives."""

from mlx_primitives.primitives._metal.scan_kernels import (
    metal_associative_scan,
    metal_ssm_scan,
)

__all__ = [
    "metal_associative_scan",
    "metal_ssm_scan",
]
