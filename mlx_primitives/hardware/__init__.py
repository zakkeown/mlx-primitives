"""Hardware detection and optimization for Apple Silicon.

This module provides:
- Chip detection (M1/M2/M3/M4)
- Optimal configuration selection
- Auto-tuning infrastructure
"""

from mlx_primitives.hardware.detection import (
    ChipFamily,
    ChipInfo,
    ChipTier,
    KernelConfig,
    check_shared_memory,
    estimate_kernel_time_ms,
    get_chip_info,
    get_memory_bandwidth_gbps,
    get_optimal_attention_config,
    get_optimal_matmul_config,
    get_optimal_scan_config,
)

__all__ = [
    "ChipFamily",
    "ChipTier",
    "ChipInfo",
    "KernelConfig",
    "get_chip_info",
    "get_optimal_attention_config",
    "get_optimal_scan_config",
    "get_optimal_matmul_config",
    "check_shared_memory",
    "get_memory_bandwidth_gbps",
    "estimate_kernel_time_ms",
]
