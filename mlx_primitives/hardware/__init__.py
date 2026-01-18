"""Hardware detection and optimization for Apple Silicon.

This module provides:
- Chip detection (M1/M2/M3/M4)
- Optimal configuration selection
- Tiling configurations for different operations
- Auto-tuning infrastructure
"""

from mlx_primitives.hardware.detection import (
    ChipFamily,
    ChipInfo,
    ChipTier,
    KernelConfig,
    check_shared_memory,
    estimate_kernel_time_ms,
    get_ane_tops,
    get_chip_info,
    get_memory_bandwidth_gbps,
    get_optimal_attention_config,
    get_optimal_config,
    get_optimal_matmul_config,
    get_optimal_scan_config,
)
from mlx_primitives.hardware.tiling import (
    DataType,
    OperationType,
    ProblemSize,
    TilingConfig,
    classify_problem_size,
    dtype_size,
    dtype_to_enum,
)
from mlx_primitives.hardware.tiling_database import (
    TilingDatabase,
    get_tiling_database,
)
from mlx_primitives.hardware.autotuner import (
    AutoTuner,
    BenchmarkResult,
    auto_tune_for_workload,
    get_autotuner,
)

__all__ = [
    # Chip detection
    "ChipFamily",
    "ChipTier",
    "ChipInfo",
    "get_chip_info",
    # Legacy kernel config
    "KernelConfig",
    "get_optimal_attention_config",
    "get_optimal_scan_config",
    "get_optimal_matmul_config",
    # New tiling system
    "TilingConfig",
    "OperationType",
    "ProblemSize",
    "DataType",
    "get_optimal_config",
    "classify_problem_size",
    "dtype_to_enum",
    "dtype_size",
    # Tiling database
    "TilingDatabase",
    "get_tiling_database",
    # Hardware info
    "check_shared_memory",
    "get_memory_bandwidth_gbps",
    "get_ane_tops",
    "estimate_kernel_time_ms",
    # Auto-tuning
    "AutoTuner",
    "BenchmarkResult",
    "get_autotuner",
    "auto_tune_for_workload",
]
