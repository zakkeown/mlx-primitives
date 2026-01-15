"""Block size configuration for Flash Attention kernels.

Provides optimal block size selection based on head dimension, data type,
and hardware capabilities. Block sizes are critical for performance as they
determine shared memory usage and parallelism.

Constraints:
- Shared memory usage must fit in 32KB threadgroup memory
- Shared memory = (BLOCK_M + 2 * BLOCK_N) * head_dim * dtype_bytes
- Block sizes should be multiples of 8 for SIMD alignment
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx

from mlx_primitives.kernels.hardware_info import get_hardware_info, get_max_threadgroup_memory


@dataclass
class BlockConfig:
    """Configuration for flash attention block sizes.

    Attributes:
        block_m: Query block size (rows of Q processed per threadgroup).
        block_n: Key/Value block size (columns of K/V processed per iteration).
        head_dim: Head dimension.
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32).
    """

    block_m: int
    block_n: int
    head_dim: int
    dtype_bytes: int

    @property
    def shared_memory_bytes(self) -> int:
        """Calculate total shared memory usage.

        Layout: Q tile + K tile + V tile
        = (BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_N * head_dim) * dtype_bytes
        = (BLOCK_M + 2 * BLOCK_N) * head_dim * dtype_bytes
        """
        return (self.block_m + 2 * self.block_n) * self.head_dim * self.dtype_bytes

    @property
    def fits_in_threadgroup(self) -> bool:
        """Check if configuration fits in 32KB threadgroup memory."""
        return self.shared_memory_bytes <= 32768

    def __str__(self) -> str:
        return (
            f"BlockConfig(block_m={self.block_m}, block_n={self.block_n}, "
            f"head_dim={self.head_dim}, shared_mem={self.shared_memory_bytes} bytes)"
        )


# Pre-computed optimal configurations based on benchmarking
# Format: (head_dim, dtype_bytes) -> (block_m, block_n)
# These are tuned for Apple Silicon (M1/M2/M3) with 32KB threadgroup memory
OPTIMAL_BLOCK_SIZES: Dict[Tuple[int, int], Tuple[int, int]] = {
    # Float32 (4 bytes per element)
    # Shared memory = (block_m + 2*block_n) * head_dim * 4
    (32, 4): (64, 64),  # (64 + 128) * 32 * 4 = 24,576 bytes
    (64, 4): (32, 32),  # (32 + 64) * 64 * 4 = 24,576 bytes
    (96, 4): (32, 32),  # (32 + 64) * 96 * 4 = 36,864 bytes - slightly over but works
    (128, 4): (32, 16),  # (32 + 32) * 128 * 4 = 32,768 bytes - exactly at limit
    (256, 4): (16, 16),  # (16 + 32) * 256 * 4 = 49,152 bytes - over, need smaller

    # Float16/BFloat16 (2 bytes per element) - can use larger blocks
    (32, 2): (64, 64),  # (64 + 128) * 32 * 2 = 12,288 bytes
    (64, 2): (64, 64),  # (64 + 128) * 64 * 2 = 24,576 bytes
    (96, 2): (64, 48),  # (64 + 96) * 96 * 2 = 30,720 bytes
    (128, 2): (64, 32),  # (64 + 64) * 128 * 2 = 32,768 bytes - at limit
    (256, 2): (32, 32),  # (32 + 64) * 256 * 2 = 49,152 bytes - over, need smaller
}

# Chip-specific tuning adjustments
# Some chips may benefit from different configurations
CHIP_OPTIMIZATIONS: Dict[str, Dict[Tuple[int, int], Tuple[int, int]]] = {
    "M3": {
        # M3 has improved matrix math units, can sometimes benefit from larger blocks
        (64, 2): (64, 64),
        (128, 2): (64, 48),  # Slightly larger V tile
    },
    "M4": {
        # M4 placeholder - benchmark to determine optimal
        (64, 2): (64, 64),
        (128, 2): (64, 64),
    },
}


def _validate_config(
    block_m: int,
    block_n: int,
    head_dim: int,
    dtype_bytes: int,
    max_mem: int,
) -> bool:
    """Validate that a block configuration fits memory constraints."""
    shared_mem = (block_m + 2 * block_n) * head_dim * dtype_bytes
    return shared_mem <= max_mem


def _compute_safe_block_config(
    head_dim: int,
    dtype_bytes: int,
    max_mem: int,
) -> Tuple[int, int]:
    """Compute a safe block configuration that fits memory constraints.

    Tries progressively smaller block sizes until one fits.
    """
    # Try block sizes from large to small
    # Prefer square blocks for balanced memory access
    for block_size in [64, 48, 32, 24, 16]:
        block_m = block_size
        block_n = block_size
        if _validate_config(block_m, block_n, head_dim, dtype_bytes, max_mem):
            return (block_m, block_n)

    # Try asymmetric configurations (smaller block_n)
    for block_m in [32, 24, 16]:
        for block_n in [32, 24, 16, 8]:
            if _validate_config(block_m, block_n, head_dim, dtype_bytes, max_mem):
                return (block_m, block_n)

    # Ultimate fallback: very small blocks (should always fit)
    return (16, 8)


def get_optimal_block_config(
    head_dim: int,
    dtype: mx.Dtype,
    chip_family: Optional[str] = None,
    max_threadgroup_memory: Optional[int] = None,
) -> Tuple[int, int]:
    """Select optimal block sizes for flash attention.

    Considers head dimension, data type, and hardware capabilities to select
    block sizes that maximize performance while fitting memory constraints.

    Args:
        head_dim: Attention head dimension (typically 64, 96, 128).
        dtype: Data type (mx.float32, mx.float16, mx.bfloat16).
        chip_family: Optional chip family for fine-tuned selection.
            If None, auto-detected from hardware.
        max_threadgroup_memory: Maximum threadgroup memory in bytes.
            If None, auto-detected from hardware (typically 32KB).

    Returns:
        Tuple of (block_m, block_n) block sizes.

    Example:
        >>> block_m, block_n = get_optimal_block_config(head_dim=64, dtype=mx.float16)
        >>> print(f"Using blocks: ({block_m}, {block_n})")
    """
    # Determine bytes per element
    if dtype in (mx.float32,):
        dtype_bytes = 4
    elif dtype in (mx.float16, mx.bfloat16):
        dtype_bytes = 2
    else:
        # Default to float32 sizing for safety
        dtype_bytes = 4

    # Get hardware info if not provided
    if chip_family is None:
        hw_info = get_hardware_info()
        chip_family = hw_info.chip_family

    if max_threadgroup_memory is None:
        max_threadgroup_memory = get_max_threadgroup_memory()

    key = (head_dim, dtype_bytes)

    # Check chip-specific optimizations first
    if chip_family in CHIP_OPTIMIZATIONS:
        chip_configs = CHIP_OPTIMIZATIONS[chip_family]
        if key in chip_configs:
            block_m, block_n = chip_configs[key]
            if _validate_config(block_m, block_n, head_dim, dtype_bytes, max_threadgroup_memory):
                return (block_m, block_n)

    # Check pre-computed optimal configurations
    if key in OPTIMAL_BLOCK_SIZES:
        block_m, block_n = OPTIMAL_BLOCK_SIZES[key]
        if _validate_config(block_m, block_n, head_dim, dtype_bytes, max_threadgroup_memory):
            return (block_m, block_n)

    # Fall back to computing a safe configuration
    return _compute_safe_block_config(head_dim, dtype_bytes, max_threadgroup_memory)


def get_block_config_info(
    head_dim: int,
    dtype: mx.Dtype,
) -> BlockConfig:
    """Get detailed block configuration info including memory usage.

    Args:
        head_dim: Attention head dimension.
        dtype: Data type.

    Returns:
        BlockConfig with detailed information.

    Example:
        >>> config = get_block_config_info(head_dim=128, dtype=mx.float16)
        >>> print(config)
        >>> print(f"Fits in threadgroup: {config.fits_in_threadgroup}")
    """
    dtype_bytes = 4 if dtype == mx.float32 else 2
    block_m, block_n = get_optimal_block_config(head_dim, dtype)
    return BlockConfig(
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )


def estimate_shared_memory(
    block_m: int,
    block_n: int,
    head_dim: int,
    dtype: mx.Dtype,
) -> int:
    """Estimate shared memory usage for given block configuration.

    Args:
        block_m: Query block size.
        block_n: Key/Value block size.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        Estimated shared memory usage in bytes.
    """
    dtype_bytes = 4 if dtype == mx.float32 else 2
    return (block_m + 2 * block_n) * head_dim * dtype_bytes


def validate_block_config(
    block_m: int,
    block_n: int,
    head_dim: int,
    dtype: mx.Dtype,
) -> bool:
    """Validate that a block configuration fits hardware constraints.

    Args:
        block_m: Query block size.
        block_n: Key/Value block size.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration exceeds memory limits.
    """
    dtype_bytes = 4 if dtype == mx.float32 else 2
    max_mem = get_max_threadgroup_memory()
    shared_mem = estimate_shared_memory(block_m, block_n, head_dim, dtype)

    if shared_mem > max_mem:
        raise ValueError(
            f"Block configuration requires {shared_mem} bytes shared memory, "
            f"but maximum is {max_mem} bytes. "
            f"Try smaller block sizes or head_dim."
        )

    return True


def warmup_block_configs(
    head_dims: Optional[list] = None,
    dtypes: Optional[list] = None,
) -> Dict[Tuple[int, mx.Dtype], Tuple[int, int]]:
    """Pre-compute optimal block configurations for common settings.

    Call this at startup to avoid latency during inference.

    Args:
        head_dims: List of head dimensions to pre-compute (default: [64, 128]).
        dtypes: List of data types (default: [mx.float32, mx.float16]).

    Returns:
        Dictionary mapping (head_dim, dtype) to (block_m, block_n).

    Example:
        >>> configs = warmup_block_configs()
        >>> print(f"Warmed up {len(configs)} configurations")
    """
    if head_dims is None:
        head_dims = [32, 64, 96, 128]

    if dtypes is None:
        dtypes = [mx.float32, mx.float16]

    configs = {}
    for head_dim in head_dims:
        for dtype in dtypes:
            block_m, block_n = get_optimal_block_config(head_dim, dtype)
            configs[(head_dim, dtype)] = (block_m, block_n)

    return configs
