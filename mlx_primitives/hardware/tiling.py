"""Tiling configuration types and enums for Apple Silicon optimization.

This module provides comprehensive tiling configurations that vary based on:
- Chip family (M1/M2/M3/M4)
- Chip tier (BASE/PRO/MAX/ULTRA)
- Operation type (attention, matmul, scan, etc.)
- Problem size (tiny to huge)
- Data type (fp32, fp16, int8, int4)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OperationType(Enum):
    """Operations requiring tiling configurations."""

    ATTENTION = "attention"
    FLASH_ATTENTION = "flash_attention"
    SLIDING_WINDOW = "sliding_window"
    CHUNKED_ATTENTION = "chunked_attention"
    MATMUL = "matmul"
    SCAN = "scan"
    SSM_SCAN = "ssm_scan"
    FUSED_RMSNORM_LINEAR = "fused_rmsnorm_linear"
    FUSED_SWIGLU = "fused_swiglu"
    INT8_LINEAR = "int8_linear"
    INT4_LINEAR = "int4_linear"
    GATHER = "gather"
    SCATTER = "scatter"


class ProblemSize(Enum):
    """Problem size categories for tiling decisions.

    The classification thresholds vary by operation type but generally:
    - TINY: Very small, may not benefit from GPU dispatch
    - SMALL: Small enough for single-block kernels
    - MEDIUM: Standard workloads, default optimization target
    - LARGE: Large workloads, may need multi-block strategies
    - HUGE: Very large, may need streaming/chunking
    """

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"


class DataType(Enum):
    """Supported data types for tiling configurations."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True)
class TilingConfig:
    """Comprehensive tiling configuration for a Metal kernel.

    This extends the basic KernelConfig with additional parameters
    for fine-grained performance tuning.

    Attributes:
        block_m: Block size for M dimension (rows/queries).
        block_n: Block size for N dimension (columns/keys).
        block_k: Block size for K dimension (reduction/head_dim).
        threads_per_threadgroup: Total threads in threadgroup.
        num_simd_groups: Number of SIMD groups (threads / 32).
        shared_memory_bytes: Threadgroup memory allocation.
        use_padding: Whether to pad for bank conflict avoidance.
        padding_elements: Number of padding elements (typically 4).
        grid_divisor_m: Grid division factor for M (load balancing).
        grid_divisor_n: Grid division factor for N.
        unroll_factor: Loop unrolling factor.
        prefetch_distance: Number of tiles to prefetch ahead.
        use_vector_loads: Whether to use vector load instructions.
        vector_width: Width of vector loads (e.g., 4 for float4).
        accumulate_in_fp32: Accumulate in fp32 for precision.
    """

    # Primary block dimensions
    block_m: int
    block_n: int
    block_k: int = 32

    # Threadgroup configuration
    threads_per_threadgroup: int = 256
    num_simd_groups: int = 8  # threads / 32

    # Shared memory layout
    shared_memory_bytes: int = 0
    use_padding: bool = True
    padding_elements: int = 4

    # Grid dispatch hints
    grid_divisor_m: int = 1
    grid_divisor_n: int = 1

    # Performance tuning
    unroll_factor: int = 1
    prefetch_distance: int = 1
    use_vector_loads: bool = True
    vector_width: int = 4

    # Precision
    accumulate_in_fp32: bool = True

    def shared_memory_for_tiles(
        self,
        dtype_size: int = 4,
        num_tiles: int = 2,
    ) -> int:
        """Calculate shared memory needed for tiled operations.

        For attention: K tile + V tile with padding.
        For matmul: A tile + B tile.

        Args:
            dtype_size: Bytes per element (4 for float32).
            num_tiles: Number of tiles to store (typically 2).

        Returns:
            Required shared memory in bytes.
        """
        if self.use_padding:
            padded_k = self.block_k + self.padding_elements
        else:
            padded_k = self.block_k

        tile_size = self.block_n * padded_k * dtype_size
        return num_tiles * tile_size

    def validate(self, max_shared_memory: int = 32768) -> bool:
        """Validate configuration against hardware constraints.

        Args:
            max_shared_memory: Maximum threadgroup memory (32KB for Apple Silicon).

        Returns:
            True if configuration is valid.
        """
        if self.block_m <= 0 or self.block_n <= 0 or self.block_k <= 0:
            return False

        if self.threads_per_threadgroup <= 0 or self.threads_per_threadgroup > 1024:
            return False

        if self.shared_memory_bytes > max_shared_memory:
            return False

        return True

    def to_kernel_config(self) -> "KernelConfig":
        """Convert to basic KernelConfig for backward compatibility.

        Returns:
            KernelConfig with block sizes and shared memory.
        """
        from mlx_primitives.hardware.detection import KernelConfig

        return KernelConfig(
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_warps=self.num_simd_groups,
            shared_memory=self.shared_memory_bytes,
        )


def classify_problem_size(
    shape: tuple,
    operation: OperationType,
) -> ProblemSize:
    """Classify problem size for tiling selection.

    Args:
        shape: Problem shape tuple (varies by operation).
        operation: Type of operation.

    Returns:
        ProblemSize classification.
    """
    if operation in (
        OperationType.ATTENTION,
        OperationType.FLASH_ATTENTION,
        OperationType.SLIDING_WINDOW,
        OperationType.CHUNKED_ATTENTION,
    ):
        # For attention: shape is (batch, seq_len, num_heads, head_dim)
        if len(shape) >= 2:
            seq_len = shape[1]
        else:
            seq_len = shape[0]

        if seq_len < 128:
            return ProblemSize.TINY
        elif seq_len < 512:
            return ProblemSize.SMALL
        elif seq_len < 2048:
            return ProblemSize.MEDIUM
        elif seq_len < 8192:
            return ProblemSize.LARGE
        else:
            return ProblemSize.HUGE

    elif operation == OperationType.MATMUL:
        # For matmul: shape is (M, N, K) or similar
        if len(shape) >= 2:
            total = shape[0] * shape[1]
        else:
            total = shape[0] * shape[0]

        if total < 128 * 128:
            return ProblemSize.TINY
        elif total < 512 * 512:
            return ProblemSize.SMALL
        elif total < 2048 * 2048:
            return ProblemSize.MEDIUM
        elif total < 8192 * 8192:
            return ProblemSize.LARGE
        else:
            return ProblemSize.HUGE

    elif operation in (OperationType.SCAN, OperationType.SSM_SCAN):
        # For scan: shape is (batch, seq_len, d_inner)
        if len(shape) >= 2:
            seq_len = shape[1]
        else:
            seq_len = shape[0]

        if seq_len <= 128:
            return ProblemSize.TINY
        elif seq_len <= 256:
            return ProblemSize.SMALL
        elif seq_len <= 1024:
            return ProblemSize.MEDIUM
        elif seq_len <= 4096:
            return ProblemSize.LARGE
        else:
            return ProblemSize.HUGE

    elif operation in (OperationType.INT8_LINEAR, OperationType.INT4_LINEAR):
        # For quantized linear: shape is (batch * seq, out_features, in_features)
        if len(shape) >= 2:
            total = shape[0] * shape[1]
        else:
            total = shape[0] * shape[0]

        if total < 256 * 256:
            return ProblemSize.TINY
        elif total < 1024 * 1024:
            return ProblemSize.SMALL
        elif total < 4096 * 4096:
            return ProblemSize.MEDIUM
        else:
            return ProblemSize.LARGE

    # Default classification
    return ProblemSize.MEDIUM


def dtype_to_enum(dtype) -> DataType:
    """Convert MLX dtype to DataType enum.

    Args:
        dtype: MLX data type.

    Returns:
        DataType enum value.
    """
    import mlx.core as mx

    dtype_map = {
        mx.float32: DataType.FP32,
        mx.float16: DataType.FP16,
        mx.bfloat16: DataType.BF16,
        mx.int8: DataType.INT8,
    }

    return dtype_map.get(dtype, DataType.FP32)


def dtype_size(dtype: DataType) -> int:
    """Get size in bytes for a data type.

    Args:
        dtype: DataType enum value.

    Returns:
        Size in bytes.
    """
    sizes = {
        DataType.FP32: 4,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.INT8: 1,
        DataType.INT4: 1,  # Packed, but use 1 for calculations
    }
    return sizes.get(dtype, 4)
