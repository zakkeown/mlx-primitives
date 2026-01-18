"""MLX integration utilities for Metal-Triton.

Handles the interface between DSL kernels and MLX's metal_kernel API.
"""

from __future__ import annotations
from typing import Any, Sequence, Optional
import warnings


def create_mlx_kernel(
    name: str,
    metal_source: str,
    input_names: Sequence[str],
    output_names: Sequence[str] = (),
) -> Any:
    """Create an MLX Metal kernel from source.

    Args:
        name: Kernel name
        metal_source: Metal shader source code
        input_names: Names of input parameters
        output_names: Names of output parameters

    Returns:
        MLX metal_kernel object
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError(
            "MLX not available. Install with: pip install mlx"
        )

    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        source=metal_source,
    )


def execute_mlx_kernel(
    mlx_kernel: Any,
    inputs: Sequence[Any],
    grid: tuple[int, ...],
    threadgroup: tuple[int, ...],
    output_shapes: Sequence[tuple[int, ...]] = (),
    output_dtypes: Sequence[Any] = (),
    stream: Optional[Any] = None,
) -> Any:
    """Execute an MLX Metal kernel.

    Args:
        mlx_kernel: Compiled MLX kernel
        inputs: Input tensors and scalars
        grid: Grid dimensions (threadgroups)
        threadgroup: Threadgroup dimensions
        output_shapes: Shapes for output tensors
        output_dtypes: Dtypes for output tensors
        stream: MLX stream (default stream if None)

    Returns:
        Output tensor(s) or empty tuple
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("MLX not available")

    if stream is None:
        stream = mx.default_stream(mx.default_device())

    # Normalize grid to 3D
    grid = _normalize_dims(grid, 3)
    threadgroup = _normalize_dims(threadgroup, 3)

    return mlx_kernel(
        inputs=list(inputs),
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=list(output_shapes),
        output_dtypes=list(output_dtypes),
        stream=stream,
    )


def _normalize_dims(dims: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Normalize dimensions to specified length by padding with 1s."""
    if len(dims) >= ndim:
        return dims[:ndim]
    return dims + (1,) * (ndim - len(dims))


def validate_inputs(
    inputs: Sequence[Any],
    expected_count: int,
    kernel_name: str,
) -> None:
    """Validate kernel inputs.

    Args:
        inputs: Provided inputs
        expected_count: Expected number of inputs
        kernel_name: Name for error messages

    Raises:
        ValueError: If validation fails
    """
    if len(inputs) != expected_count:
        raise ValueError(
            f"Kernel '{kernel_name}' expects {expected_count} inputs, "
            f"got {len(inputs)}"
        )


def compute_grid_size(
    total_elements: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Compute grid size for 1D parallelism.

    Args:
        total_elements: Total number of elements to process
        block_size: Elements per threadgroup

    Returns:
        (grid_x, grid_y, grid_z) tuple
    """
    num_blocks = (total_elements + block_size - 1) // block_size
    return (num_blocks, 1, 1)


def compute_threadgroup_size(
    num_warps: int = 4,
    warp_size: int = 32,
) -> tuple[int, int, int]:
    """Compute threadgroup size.

    Args:
        num_warps: Number of SIMD groups (warps)
        warp_size: Threads per SIMD group (32 for Apple Silicon)

    Returns:
        (threads_x, threads_y, threads_z) tuple
    """
    return (num_warps * warp_size, 1, 1)


def check_mlx_metal_available() -> bool:
    """Check if MLX Metal kernels are available."""
    try:
        import mlx.core as mx
        return hasattr(mx.fast, "metal_kernel")
    except ImportError:
        return False


def get_device_info() -> dict[str, Any]:
    """Get MLX device information."""
    try:
        import mlx.core as mx
        device = mx.default_device()
        return {
            "device": str(device),
            "metal_available": check_mlx_metal_available(),
        }
    except ImportError:
        return {
            "device": "unknown",
            "metal_available": False,
        }
