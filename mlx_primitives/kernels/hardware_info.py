"""Hardware detection for Apple Silicon GPUs.

Provides runtime detection of hardware capabilities for optimal kernel configuration.

NOTE: This module delegates to mlx_primitives.hardware.detection for actual
hardware detection. The AppleSiliconInfo class is maintained for backwards
compatibility but wraps the authoritative ChipInfo from the hardware module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from mlx_primitives.hardware.detection import ChipFamily, get_chip_info


@dataclass
class AppleSiliconInfo:
    """Hardware capabilities for Apple Silicon GPUs.

    Attributes:
        device_name: Full device name (e.g., "Apple M3 Pro").
        chip_family: Chip family (e.g., "M1", "M2", "M3", "M4").
        architecture: GPU architecture string (e.g., "applegpu_g15g").
        memory_size_gb: Total GPU memory in GB.
        max_threadgroup_memory: Maximum threadgroup memory in bytes (typically 32KB).
        simd_width: SIMD lane count (typically 32).
        max_threads_per_threadgroup: Maximum threads per threadgroup (typically 1024).
    """

    device_name: str
    chip_family: str
    architecture: str
    memory_size_gb: float
    max_threadgroup_memory: int
    simd_width: int
    max_threads_per_threadgroup: int


# Known threadgroup memory limits per generation (in bytes)
# All current Apple Silicon GPUs have 32KB threadgroup memory
THREADGROUP_MEMORY: Dict[str, int] = {
    "M1": 32768,  # 32 KB
    "M2": 32768,  # 32 KB
    "M3": 32768,  # 32 KB
    "M4": 32768,  # 32 KB
    "M5": 32768,  # 32 KB
    "Unknown": 32768,  # Safe default
}


@lru_cache(maxsize=1)
def get_hardware_info() -> AppleSiliconInfo:
    """Query and cache Apple Silicon hardware information.

    This function delegates to the authoritative get_chip_info() from
    mlx_primitives.hardware.detection and wraps the result in an
    AppleSiliconInfo for backwards compatibility.

    Returns:
        AppleSiliconInfo with detected hardware capabilities.

    Example:
        >>> info = get_hardware_info()
        >>> print(f"Running on {info.device_name} ({info.chip_family})")
        >>> print(f"Threadgroup memory: {info.max_threadgroup_memory} bytes")
    """
    # Delegate to authoritative hardware detection
    chip_info = get_chip_info()

    # Map ChipFamily enum to string for backwards compatibility
    chip_family_str = chip_info.family.value if chip_info.family != ChipFamily.UNKNOWN else "Unknown"

    # Get threadgroup memory limit
    threadgroup_mem = THREADGROUP_MEMORY.get(chip_family_str, 32768)

    return AppleSiliconInfo(
        device_name=chip_info.device_name,
        chip_family=chip_family_str,
        architecture="",  # Not exposed by ChipInfo, leave empty
        memory_size_gb=chip_info.memory_gb,
        max_threadgroup_memory=threadgroup_mem,
        simd_width=chip_info.simd_width,
        max_threads_per_threadgroup=chip_info.max_threads_per_threadgroup,
    )


def get_chip_family() -> str:
    """Get the chip family (M1, M2, M3, M4, or Unknown).

    Convenience function for quick chip detection.

    Returns:
        Chip family string.

    Example:
        >>> if get_chip_family() == "M3":
        ...     print("Running on M3 series")
    """
    return get_hardware_info().chip_family


def get_max_threadgroup_memory() -> int:
    """Get maximum threadgroup (shared) memory in bytes.

    Returns:
        Maximum threadgroup memory (typically 32768 bytes / 32 KB).

    Example:
        >>> max_mem = get_max_threadgroup_memory()
        >>> # Calculate if tiles fit
        >>> tile_size = block_m * head_dim * 4  # float32
        >>> assert tile_size <= max_mem
    """
    return get_hardware_info().max_threadgroup_memory
