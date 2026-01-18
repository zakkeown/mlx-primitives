"""Apple Silicon hardware detection and configuration.

Detects the specific Apple Silicon chip and provides optimal configurations
for Metal kernels based on hardware capabilities.
"""

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

import mlx.core as mx


class ChipFamily(Enum):
    """Apple Silicon chip families."""

    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    UNKNOWN = "UNKNOWN"


class ChipTier(Enum):
    """Apple Silicon chip tiers within a family."""

    BASE = "base"
    PRO = "Pro"
    MAX = "Max"
    ULTRA = "Ultra"


@dataclass(frozen=True)
class ChipInfo:
    """Information about the current Apple Silicon chip.

    Attributes:
        family: Chip family (M1, M2, M3, M4).
        tier: Chip tier (base, Pro, Max, Ultra).
        device_name: Full device name string.
        gpu_cores: Number of GPU cores (estimated).
        memory_gb: Total unified memory in GB.
        simd_width: SIMD width (typically 32 for Apple Silicon).
        max_threadgroup_memory: Maximum threadgroup memory in bytes (32KB).
        max_threads_per_threadgroup: Maximum threads per threadgroup.
        ane_tops: Neural Engine throughput in trillion ops/sec.
        l2_cache_mb: Estimated L2 cache size in MB.
        memory_bandwidth_gbps: Memory bandwidth in GB/s.
    """

    family: ChipFamily
    tier: ChipTier
    device_name: str
    gpu_cores: int
    memory_gb: float
    simd_width: int = 32
    max_threadgroup_memory: int = 32768  # 32KB for all Apple Silicon
    max_threads_per_threadgroup: int = 1024
    ane_tops: float = 11.0  # Neural Engine TOPS
    l2_cache_mb: int = 8  # L2 cache size
    memory_bandwidth_gbps: float = 100.0  # Memory bandwidth


@dataclass(frozen=True)
class KernelConfig:
    """Configuration for a Metal kernel.

    Attributes:
        block_m: Block size for M dimension (rows).
        block_n: Block size for N dimension (columns).
        block_k: Block size for K dimension (reduction).
        num_warps: Number of warps (SIMD groups).
        shared_memory: Shared memory size in bytes.
    """

    block_m: int
    block_n: int
    block_k: int = 32
    num_warps: int = 4
    shared_memory: int = 0


# GPU core counts for different chip variants (approximate)
_GPU_CORES = {
    (ChipFamily.M1, ChipTier.BASE): 8,
    (ChipFamily.M1, ChipTier.PRO): 16,
    (ChipFamily.M1, ChipTier.MAX): 32,
    (ChipFamily.M1, ChipTier.ULTRA): 64,
    (ChipFamily.M2, ChipTier.BASE): 10,
    (ChipFamily.M2, ChipTier.PRO): 19,
    (ChipFamily.M2, ChipTier.MAX): 38,
    (ChipFamily.M2, ChipTier.ULTRA): 76,
    (ChipFamily.M3, ChipTier.BASE): 10,
    (ChipFamily.M3, ChipTier.PRO): 18,
    (ChipFamily.M3, ChipTier.MAX): 40,
    (ChipFamily.M4, ChipTier.BASE): 10,
    (ChipFamily.M4, ChipTier.PRO): 20,
    (ChipFamily.M4, ChipTier.MAX): 40,
}

# ANE (Neural Engine) TOPS for different chip variants
_ANE_TOPS = {
    (ChipFamily.M1, ChipTier.BASE): 11.0,
    (ChipFamily.M1, ChipTier.PRO): 11.0,
    (ChipFamily.M1, ChipTier.MAX): 11.0,
    (ChipFamily.M1, ChipTier.ULTRA): 22.0,
    (ChipFamily.M2, ChipTier.BASE): 15.8,
    (ChipFamily.M2, ChipTier.PRO): 15.8,
    (ChipFamily.M2, ChipTier.MAX): 15.8,
    (ChipFamily.M2, ChipTier.ULTRA): 31.6,
    (ChipFamily.M3, ChipTier.BASE): 18.0,
    (ChipFamily.M3, ChipTier.PRO): 18.0,
    (ChipFamily.M3, ChipTier.MAX): 18.0,
    (ChipFamily.M4, ChipTier.BASE): 38.0,
    (ChipFamily.M4, ChipTier.PRO): 38.0,
    (ChipFamily.M4, ChipTier.MAX): 38.0,
}

# L2 cache sizes in MB (approximate)
_L2_CACHE_MB = {
    (ChipFamily.M1, ChipTier.BASE): 8,
    (ChipFamily.M1, ChipTier.PRO): 24,
    (ChipFamily.M1, ChipTier.MAX): 48,
    (ChipFamily.M1, ChipTier.ULTRA): 96,
    (ChipFamily.M2, ChipTier.BASE): 8,
    (ChipFamily.M2, ChipTier.PRO): 24,
    (ChipFamily.M2, ChipTier.MAX): 48,
    (ChipFamily.M2, ChipTier.ULTRA): 96,
    (ChipFamily.M3, ChipTier.BASE): 8,
    (ChipFamily.M3, ChipTier.PRO): 24,
    (ChipFamily.M3, ChipTier.MAX): 48,
    (ChipFamily.M4, ChipTier.BASE): 12,
    (ChipFamily.M4, ChipTier.PRO): 24,
    (ChipFamily.M4, ChipTier.MAX): 48,
}

# Memory bandwidth in GB/s
_MEMORY_BANDWIDTH = {
    (ChipFamily.M1, ChipTier.BASE): 68.25,
    (ChipFamily.M1, ChipTier.PRO): 200.0,
    (ChipFamily.M1, ChipTier.MAX): 400.0,
    (ChipFamily.M1, ChipTier.ULTRA): 800.0,
    (ChipFamily.M2, ChipTier.BASE): 100.0,
    (ChipFamily.M2, ChipTier.PRO): 200.0,
    (ChipFamily.M2, ChipTier.MAX): 400.0,
    (ChipFamily.M2, ChipTier.ULTRA): 800.0,
    (ChipFamily.M3, ChipTier.BASE): 100.0,
    (ChipFamily.M3, ChipTier.PRO): 150.0,
    (ChipFamily.M3, ChipTier.MAX): 400.0,
    (ChipFamily.M4, ChipTier.BASE): 120.0,
    (ChipFamily.M4, ChipTier.PRO): 273.0,
    (ChipFamily.M4, ChipTier.MAX): 546.0,
}

# Optimal block sizes for attention kernels per chip family
_ATTENTION_BLOCK_SIZES: dict[ChipFamily, dict[int, tuple[int, int]]] = {
    ChipFamily.M1: {
        64: (64, 64),  # head_dim -> (block_m, block_n)
        128: (64, 48),
    },
    ChipFamily.M2: {
        64: (64, 64),
        128: (64, 64),
    },
    ChipFamily.M3: {
        64: (64, 64),
        128: (64, 64),
    },
    ChipFamily.M4: {
        64: (64, 64),
        128: (64, 64),
    },
}


def _parse_device_name(device_name: str) -> tuple[ChipFamily, ChipTier]:
    """Parse device name to extract chip family and tier."""
    name_lower = device_name.lower()

    # Determine family
    family = ChipFamily.UNKNOWN
    if "m1" in name_lower:
        family = ChipFamily.M1
    elif "m2" in name_lower:
        family = ChipFamily.M2
    elif "m3" in name_lower:
        family = ChipFamily.M3
    elif "m4" in name_lower:
        family = ChipFamily.M4

    # Determine tier
    tier = ChipTier.BASE
    if "ultra" in name_lower:
        tier = ChipTier.ULTRA
    elif "max" in name_lower:
        tier = ChipTier.MAX
    elif "pro" in name_lower:
        tier = ChipTier.PRO

    return family, tier


@lru_cache(maxsize=1)
def get_chip_info() -> ChipInfo:
    """Get information about the current Apple Silicon chip.

    Returns:
        ChipInfo with hardware details.

    Example:
        >>> info = get_chip_info()
        >>> print(f"Running on {info.device_name}")
        >>> print(f"Family: {info.family.value}, Tier: {info.tier.value}")
        >>> print(f"GPU cores: {info.gpu_cores}, Memory: {info.memory_gb}GB")
    """
    try:
        device_info = mx.metal.device_info()
    except Exception:
        # Fallback if Metal not available
        return ChipInfo(
            family=ChipFamily.UNKNOWN,
            tier=ChipTier.BASE,
            device_name="Unknown",
            gpu_cores=8,
            memory_gb=8.0,
        )

    device_name = device_info.get("device_name", "Unknown")
    memory_size = device_info.get("memory_size", 8 * 1024 * 1024 * 1024)
    memory_gb = memory_size / (1024**3)

    family, tier = _parse_device_name(device_name)

    # Get GPU core count
    gpu_cores = _GPU_CORES.get((family, tier), 8)

    # Get ANE TOPS
    ane_tops = _ANE_TOPS.get((family, tier), 11.0)

    # Get L2 cache size
    l2_cache_mb = _L2_CACHE_MB.get((family, tier), 8)

    # Get memory bandwidth
    bandwidth = _MEMORY_BANDWIDTH.get((family, tier), 100.0)

    return ChipInfo(
        family=family,
        tier=tier,
        device_name=device_name,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        ane_tops=ane_tops,
        l2_cache_mb=l2_cache_mb,
        memory_bandwidth_gbps=bandwidth,
    )


def get_optimal_attention_config(head_dim: int) -> KernelConfig:
    """Get optimal kernel configuration for attention operations.

    Args:
        head_dim: Attention head dimension (typically 64 or 128).

    Returns:
        KernelConfig with optimal block sizes for the current hardware.
    """
    chip_info = get_chip_info()
    family = chip_info.family

    # Get block sizes for this chip family
    block_sizes = _ATTENTION_BLOCK_SIZES.get(family, _ATTENTION_BLOCK_SIZES[ChipFamily.M1])

    # Find best matching head_dim
    if head_dim in block_sizes:
        block_m, block_n = block_sizes[head_dim]
    else:
        # Default for unknown head_dim
        block_m, block_n = (64, 64)

    # Calculate shared memory for tiled attention:
    # - K tile: (block_n, head_dim + 4) for bank conflict avoidance
    # - V tile: (block_n, head_dim + 4)
    # Must fit within 32KB limit
    padded_dim = head_dim + 4
    shared_memory = 2 * block_n * padded_dim * 4  # 4 bytes per float

    # Ensure we don't exceed 32KB
    max_shared = 32768
    if shared_memory > max_shared:
        # Reduce block_n to fit
        block_n = max_shared // (2 * padded_dim * 4)
        shared_memory = 2 * block_n * padded_dim * 4

    return KernelConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=head_dim,
        num_warps=4,
        shared_memory=shared_memory,
    )


def get_optimal_scan_config(seq_len: int) -> KernelConfig:
    """Get optimal kernel configuration for scan operations.

    Args:
        seq_len: Sequence length to scan.

    Returns:
        KernelConfig with optimal block size.
    """
    # Scan uses power-of-2 block sizes
    # Prefer 256 or 512 for most cases
    if seq_len <= 256:
        block_size = 256
    elif seq_len <= 512:
        block_size = 512
    else:
        block_size = 1024

    # For SSM scan, need 2x shared memory (A and h arrays)
    shared_memory = 2 * block_size * 4  # 4 bytes per float

    return KernelConfig(
        block_m=block_size,
        block_n=1,
        block_k=1,
        num_warps=block_size // 32,
        shared_memory=shared_memory,
    )


def get_optimal_matmul_config(m: int, n: int, k: int) -> KernelConfig:
    """Get optimal kernel configuration for matrix multiplication.

    Args:
        m: Number of rows in output.
        n: Number of columns in output.
        k: Reduction dimension.

    Returns:
        KernelConfig with optimal block sizes.
    """
    chip_info = get_chip_info()

    # Base configuration
    block_m = 64
    block_n = 64
    block_k = 32

    # Adjust for small matrices
    if m < 64:
        block_m = 32
    if n < 64:
        block_n = 32
    if k < 32:
        block_k = 16

    # Calculate shared memory: A tile + B tile
    shared_memory = (block_m * block_k + block_k * block_n) * 4

    return KernelConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        shared_memory=shared_memory,
    )


def check_shared_memory(required_bytes: int) -> bool:
    """Check if required shared memory fits in hardware limits.

    Args:
        required_bytes: Required threadgroup memory in bytes.

    Returns:
        True if the memory requirement can be satisfied.
    """
    chip_info = get_chip_info()
    return required_bytes <= chip_info.max_threadgroup_memory


def get_memory_bandwidth_gbps() -> float:
    """Get estimated memory bandwidth in GB/s.

    Returns:
        Estimated memory bandwidth based on chip family.
    """
    chip_info = get_chip_info()
    return chip_info.memory_bandwidth_gbps


def estimate_kernel_time_ms(
    flops: int,
    memory_bytes: int,
    is_compute_bound: bool = False,
) -> float:
    """Estimate kernel execution time in milliseconds.

    This is a rough estimate useful for comparing different implementations.

    Args:
        flops: Number of floating point operations.
        memory_bytes: Bytes of memory accessed.
        is_compute_bound: If True, use compute throughput; else memory bandwidth.

    Returns:
        Estimated time in milliseconds.
    """
    chip_info = get_chip_info()
    bandwidth_gbps = get_memory_bandwidth_gbps()

    # Rough TFLOPS estimates
    tflops = {
        ChipFamily.M1: 2.6,
        ChipFamily.M2: 3.6,
        ChipFamily.M3: 4.1,
        ChipFamily.M4: 4.5,
    }

    if is_compute_bound:
        peak_tflops = tflops.get(chip_info.family, 2.6)
        time_s = flops / (peak_tflops * 1e12)
    else:
        time_s = memory_bytes / (bandwidth_gbps * 1e9)

    return time_s * 1000  # Convert to ms


def get_optimal_config(
    operation: "OperationType",
    problem_shape: tuple,
    dtype: "mx.Dtype" = None,
    auto_tune: bool = False,
) -> "TilingConfig":
    """Get optimal tiling configuration for an operation.

    This is the primary API for obtaining chip-specific tiling configurations.
    It queries the tiling database for the best configuration based on:
    - Current Apple Silicon chip (M1/M2/M3/M4)
    - Chip tier (BASE/PRO/MAX/ULTRA)
    - Operation type
    - Problem size
    - Data type

    Args:
        operation: Operation type from OperationType enum.
        problem_shape: Shape tuple for the problem (varies by operation).
        dtype: MLX data type. Defaults to float32.
        auto_tune: If True, run auto-tuning (future feature).

    Returns:
        TilingConfig with optimal block sizes and parameters.

    Example:
        >>> from mlx_primitives.hardware import get_optimal_config
        >>> from mlx_primitives.hardware.tiling import OperationType
        >>> config = get_optimal_config(
        ...     OperationType.FLASH_ATTENTION,
        ...     problem_shape=(1, 1024, 8, 64),  # (batch, seq, heads, dim)
        ... )
        >>> print(f"Block sizes: {config.block_m}x{config.block_n}")
    """
    from mlx_primitives.hardware.tiling import (
        DataType,
        OperationType as OpType,
        classify_problem_size,
        dtype_to_enum,
    )
    from mlx_primitives.hardware.tiling_database import get_tiling_database

    # Get current chip info
    chip_info = get_chip_info()

    # Convert dtype to enum
    if dtype is None:
        data_type = DataType.FP32
    else:
        data_type = dtype_to_enum(dtype)

    # Classify problem size
    problem_size = classify_problem_size(problem_shape, operation)

    # Get from database
    db = get_tiling_database()
    config = db.get_config(
        operation=operation,
        chip_family=chip_info.family,
        chip_tier=chip_info.tier,
        problem_size=problem_size,
        dtype=data_type,
    )

    return config


def get_ane_tops() -> float:
    """Get Neural Engine throughput in TOPS.

    Returns:
        Neural Engine trillion operations per second.
    """
    chip_info = get_chip_info()
    return chip_info.ane_tops
