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

    Note on shared memory layout:
        Flash Attention tiles K and V into shared memory, while Q elements are
        typically kept in registers or streamed from global memory. This is
        because Q is accessed sequentially in the outer loop while K/V are
        iterated in the inner loop with reuse.

        Shared memory = 2 * BLOCK_N * (head_dim + padding) * dtype_bytes
        where padding=4 elements avoids bank conflicts.
    """

    block_m: int
    block_n: int
    head_dim: int
    dtype_bytes: int

    @property
    def shared_memory_bytes(self) -> int:
        """Calculate total shared memory usage for K and V tiles.

        Layout: K tile + V tile with bank conflict padding
        = 2 * BLOCK_N * (head_dim + 4) * dtype_bytes

        Note: Q is NOT in shared memory - it's kept in registers or streamed.
        The +4 padding on head_dim avoids shared memory bank conflicts.
        """
        padded_dim = self.head_dim + 4  # Bank conflict avoidance padding
        return 2 * self.block_n * padded_dim * self.dtype_bytes

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
#
# Shared memory formula: 2 * block_n * (head_dim + 4) * dtype_bytes <= 32768
# The +4 is padding to avoid bank conflicts.
# block_m doesn't affect shared memory (Q is in registers, not shared memory).
OPTIMAL_BLOCK_SIZES: Dict[Tuple[int, int], Tuple[int, int]] = {
    # Float32 (4 bytes per element)
    # Constraint: 2 * block_n * (head_dim + 4) * 4 <= 32768
    (32, 4): (64, 64),   # 2 * 64 * 36 * 4 = 18,432 bytes
    (64, 4): (64, 48),   # 2 * 48 * 68 * 4 = 26,112 bytes
    (96, 4): (64, 32),   # 2 * 32 * 100 * 4 = 25,600 bytes
    (128, 4): (64, 24),  # 2 * 24 * 132 * 4 = 25,344 bytes
    (256, 4): (32, 12),  # 2 * 12 * 260 * 4 = 24,960 bytes

    # Float16/BFloat16 (2 bytes per element) - can use larger blocks
    # Constraint: 2 * block_n * (head_dim + 4) * 2 <= 32768
    (32, 2): (64, 64),   # 2 * 64 * 36 * 2 = 9,216 bytes
    (64, 2): (64, 64),   # 2 * 64 * 68 * 2 = 17,408 bytes
    (96, 2): (64, 64),   # 2 * 64 * 100 * 2 = 25,600 bytes
    (128, 2): (64, 48),  # 2 * 48 * 132 * 2 = 25,344 bytes
    (256, 2): (64, 24),  # 2 * 24 * 260 * 2 = 24,960 bytes
}

# Chip-specific tuning adjustments
# Some chips may benefit from different configurations
# Note: These use the same shared memory formula: 2 * block_n * (head_dim + 4) * dtype_bytes
CHIP_OPTIMIZATIONS: Dict[str, Dict[Tuple[int, int], Tuple[int, int]]] = {
    "M3": {
        # M3 has improved matrix math units, can benefit from larger blocks
        (64, 2): (64, 64),   # 2 * 64 * 68 * 2 = 17,408 bytes
        (128, 2): (64, 48),  # 2 * 48 * 132 * 2 = 25,344 bytes
    },
    "M4": {
        # M4 has further improvements - use larger blocks where possible
        (64, 2): (64, 64),   # 2 * 64 * 68 * 2 = 17,408 bytes
        (128, 2): (64, 48),  # 2 * 48 * 132 * 2 = 25,344 bytes
    },
    "M5": {
        # M5 has next-gen GPU with Neural Accelerators per core
        # 45% higher graphics performance vs M4
        (64, 2): (64, 64),   # 2 * 64 * 68 * 2 = 17,408 bytes
        (128, 2): (64, 48),  # 2 * 48 * 132 * 2 = 25,344 bytes
    },
}


def _validate_config(
    block_m: int,
    block_n: int,
    head_dim: int,
    dtype_bytes: int,
    max_mem: int,
) -> bool:
    """Validate that a block configuration fits memory constraints.

    Uses formula: 2 * block_n * (head_dim + 4) * dtype_bytes
    Note: block_m doesn't affect shared memory (Q is in registers).
    """
    padded_dim = head_dim + 4  # Bank conflict avoidance
    shared_mem = 2 * block_n * padded_dim * dtype_bytes
    return shared_mem <= max_mem


def _compute_safe_block_config(
    head_dim: int,
    dtype_bytes: int,
    max_mem: int,
) -> Tuple[int, int]:
    """Compute a safe block configuration that fits memory constraints.

    Tries progressively smaller block_n sizes until one fits.
    Note: block_m doesn't affect shared memory, so we prefer larger block_m.
    """
    # Calculate max block_n that fits: 2 * block_n * (head_dim + 4) * dtype_bytes <= max_mem
    padded_dim = head_dim + 4
    max_block_n = max_mem // (2 * padded_dim * dtype_bytes)

    # Try block_n sizes from large to small
    for block_n in [64, 48, 32, 24, 16, 8]:
        if block_n <= max_block_n:
            # block_m can be larger since it doesn't use shared memory
            # Use 64 as default for good parallelism
            return (64, block_n)

    # Ultimate fallback: very small blocks (should always fit)
    return (32, 8)


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
        # Default to float32 sizing for safety, but warn
        warnings.warn(
            f"Unknown dtype {dtype} for block config; assuming 4 bytes per element. "
            f"Block sizes may not be optimal for this dtype.",
            stacklevel=2,
        )
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

    Uses formula: 2 * block_n * (head_dim + 4) * dtype_bytes
    Note: block_m is accepted for API compatibility but doesn't affect
    the calculation (Q tiles are in registers, not shared memory).

    Args:
        block_m: Query block size (unused in calculation).
        block_n: Key/Value block size.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        Estimated shared memory usage in bytes.
    """
    _ = block_m  # Unused - Q is in registers
    dtype_bytes = 4 if dtype == mx.float32 else 2
    padded_dim = head_dim + 4  # Bank conflict avoidance
    return 2 * block_n * padded_dim * dtype_bytes


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
    """Pre-compute block configurations and trigger hardware detection.

    Note: The primary value of this function is triggering the hardware
    detection cache on first call. The block config lookups themselves
    are O(1) dict lookups and don't benefit from warming.

    Args:
        head_dims: List of head dimensions to pre-compute (default: [32, 64, 96, 128]).
        dtypes: List of data types (default: [mx.float32, mx.float16]).

    Returns:
        Dictionary mapping (head_dim, dtype) to (block_m, block_n).

    Example:
        >>> configs = warmup_block_configs()
        >>> print(f"Computed {len(configs)} configurations")
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
