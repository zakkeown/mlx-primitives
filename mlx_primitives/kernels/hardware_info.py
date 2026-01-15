"""Hardware detection for Apple Silicon GPUs.

Provides runtime detection of hardware capabilities for optimal kernel configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import mlx.core as mx


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


# Map GPU architecture prefix to chip family
CHIP_FAMILY_MAP: Dict[str, str] = {
    "applegpu_g13": "M1",  # M1 series (M1, M1 Pro, M1 Max, M1 Ultra)
    "applegpu_g14": "M2",  # M2 series (M2, M2 Pro, M2 Max, M2 Ultra)
    "applegpu_g15": "M3",  # M3 series (M3, M3 Pro, M3 Max)
    "applegpu_g16": "M4",  # M4 series (M4, M4 Pro, M4 Max) - estimated
}

# Known threadgroup memory limits per generation (in bytes)
# All current Apple Silicon GPUs have 32KB threadgroup memory
THREADGROUP_MEMORY: Dict[str, int] = {
    "M1": 32768,  # 32 KB
    "M2": 32768,  # 32 KB
    "M3": 32768,  # 32 KB
    "M4": 32768,  # 32 KB (conservative default)
    "Unknown": 32768,  # Safe default
}


@lru_cache(maxsize=1)
def get_hardware_info() -> AppleSiliconInfo:
    """Query and cache Apple Silicon hardware information.

    Returns:
        AppleSiliconInfo with detected hardware capabilities.

    Example:
        >>> info = get_hardware_info()
        >>> print(f"Running on {info.device_name} ({info.chip_family})")
        >>> print(f"Threadgroup memory: {info.max_threadgroup_memory} bytes")
    """
    try:
        info = mx.metal.device_info()
    except Exception:
        # Fallback for systems without Metal
        return AppleSiliconInfo(
            device_name="Unknown",
            chip_family="Unknown",
            architecture="unknown",
            memory_size_gb=0.0,
            max_threadgroup_memory=32768,
            simd_width=32,
            max_threads_per_threadgroup=1024,
        )

    device_name = info.get("device_name", "Unknown")
    arch = info.get("architecture", "")
    memory_bytes = info.get("memory_size", 0)

    # Parse chip family from architecture string
    chip_family = "Unknown"
    for arch_prefix, family in CHIP_FAMILY_MAP.items():
        if arch.startswith(arch_prefix):
            chip_family = family
            break

    # Also try to parse from device name if architecture didn't match
    if chip_family == "Unknown":
        device_lower = device_name.lower()
        if "m1" in device_lower:
            chip_family = "M1"
        elif "m2" in device_lower:
            chip_family = "M2"
        elif "m3" in device_lower:
            chip_family = "M3"
        elif "m4" in device_lower:
            chip_family = "M4"

    # Get threadgroup memory limit
    threadgroup_mem = THREADGROUP_MEMORY.get(chip_family, 32768)

    return AppleSiliconInfo(
        device_name=device_name,
        chip_family=chip_family,
        architecture=arch,
        memory_size_gb=memory_bytes / (1024**3) if memory_bytes else 0.0,
        max_threadgroup_memory=threadgroup_mem,
        simd_width=32,  # Consistent across Apple Silicon
        max_threads_per_threadgroup=1024,  # Metal default
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
