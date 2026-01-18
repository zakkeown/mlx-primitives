"""Intelligent dispatch between GPU (Metal) and ANE (Core ML).

This module provides routing logic to decide whether to execute
operations on the GPU (via Metal/MLX) or on the Neural Engine
(via Core ML).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from mlx_primitives.ane.detection import get_ane_info, is_ane_available
from mlx_primitives.hardware import get_chip_info


class ComputeTarget(Enum):
    """Target device for computation."""

    GPU = "gpu"  # Metal/MLX
    ANE = "ane"  # Neural Engine via Core ML
    AUTO = "auto"  # Let router decide


@dataclass
class DispatchDecision:
    """Result of dispatch decision.

    Attributes:
        target: Selected compute target.
        reason: Human-readable explanation of the decision.
        estimated_speedup: Estimated speedup factor (1.0 = same speed).
    """

    target: ComputeTarget
    reason: str
    estimated_speedup: float = 1.0


# Minimum speedup required to justify ANE dispatch overhead
# Below this threshold, stay on GPU to avoid transfer costs
_MIN_SPEEDUP_THRESHOLD = 1.2

# Minimum tensor size (elements) to consider ANE
# Very small tensors have too much overhead
_MIN_TENSOR_SIZE = 10000


def should_use_ane(
    operation: str,
    input_shapes: List[Tuple[int, ...]],
    is_training: bool = False,
    force_target: Optional[ComputeTarget] = None,
) -> DispatchDecision:
    """Decide whether to use ANE for an operation.

    Decision factors:
    1. ANE availability
    2. Operation type (ANE only supports specific ops)
    3. Tensor shapes (ANE prefers fixed, reasonably-sized tensors)
    4. Training vs inference (ANE is inference-only)
    5. Estimated speedup vs transfer overhead

    Args:
        operation: Operation name ("matmul", "conv2d", etc.).
        input_shapes: Shapes of input tensors.
        is_training: If True, never use ANE (no gradient support).
        force_target: Override automatic decision.

    Returns:
        DispatchDecision with target and reasoning.

    Example:
        >>> decision = should_use_ane("matmul", [(1024, 512), (512, 1024)])
        >>> if decision.target == ComputeTarget.ANE:
        ...     result = ane_matmul(a, b)
        ... else:
        ...     result = a @ b
    """
    # Handle explicit override
    if force_target == ComputeTarget.GPU:
        return DispatchDecision(ComputeTarget.GPU, "Explicitly requested GPU")

    if force_target == ComputeTarget.ANE:
        if not is_ane_available():
            return DispatchDecision(ComputeTarget.GPU, "ANE requested but unavailable")
        return DispatchDecision(ComputeTarget.ANE, "Explicitly requested ANE")

    # Training mode - always use GPU (ANE doesn't support gradients)
    if is_training:
        return DispatchDecision(
            ComputeTarget.GPU,
            "Training mode (ANE is inference-only)"
        )

    # Check ANE availability
    if not is_ane_available():
        return DispatchDecision(ComputeTarget.GPU, "ANE not available")

    ane_info = get_ane_info()

    # Check operation support
    supported_ops = {
        "matmul": ane_info.supports_matmul,
        "linear": ane_info.supports_matmul,
        "conv2d": ane_info.supports_conv2d,
        "depthwise_conv": ane_info.supports_depthwise_conv,
        "batch_norm": ane_info.supports_batch_norm,
        "layer_norm": ane_info.supports_layer_norm,
        "gelu": ane_info.supports_activations,
        "silu": ane_info.supports_activations,
        "relu": ane_info.supports_activations,
        "softmax": ane_info.supports_activations,
    }

    if operation not in supported_ops or not supported_ops[operation]:
        return DispatchDecision(
            ComputeTarget.GPU,
            f"Operation '{operation}' not supported on ANE"
        )

    # Check tensor sizes
    total_elements = sum(_product(shape) for shape in input_shapes)

    # Too small - overhead dominates
    if total_elements < _MIN_TENSOR_SIZE:
        return DispatchDecision(
            ComputeTarget.GPU,
            f"Tensor too small ({total_elements} elements)"
        )

    # Check against ANE max size
    total_mb = total_elements * 4 / (1024 * 1024)  # Assuming float32
    if total_mb > ane_info.max_tensor_size_mb:
        return DispatchDecision(
            ComputeTarget.GPU,
            f"Tensor too large for ANE ({total_mb:.1f}MB)"
        )

    # Check for dynamic shapes (not supported)
    for shape in input_shapes:
        if any(dim <= 0 for dim in shape):
            return DispatchDecision(
                ComputeTarget.GPU,
                "Dynamic shapes not supported on ANE"
            )

    # Estimate speedup
    speedup = _estimate_ane_speedup(operation, input_shapes)

    # Only use ANE if speedup exceeds threshold
    if speedup < _MIN_SPEEDUP_THRESHOLD:
        return DispatchDecision(
            ComputeTarget.GPU,
            f"Insufficient speedup ({speedup:.2f}x < {_MIN_SPEEDUP_THRESHOLD}x)",
            speedup,
        )

    return DispatchDecision(
        ComputeTarget.ANE,
        f"Estimated {speedup:.2f}x speedup on ANE",
        speedup,
    )


def _product(shape: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def _estimate_ane_speedup(
    operation: str,
    input_shapes: List[Tuple[int, ...]],
) -> float:
    """Estimate speedup from using ANE vs GPU.

    This is a heuristic based on:
    - ANE TOPS vs GPU TFLOPS
    - Memory transfer overhead
    - Operation characteristics

    Args:
        operation: Operation name.
        input_shapes: Input tensor shapes.

    Returns:
        Estimated speedup factor (>1 means ANE is faster).
    """
    ane_info = get_ane_info()
    chip_info = get_chip_info()

    # ANE TOPS
    ane_tops = ane_info.tops

    # GPU TFLOPS estimates (FP32)
    gpu_tflops = {
        "M1": 2.6,
        "M2": 3.6,
        "M3": 4.1,
        "M4": 4.5,
    }.get(chip_info.family.value, 3.0)

    # For PRO/MAX/ULTRA variants, scale up GPU power
    gpu_multipliers = {
        "base": 1.0,
        "Pro": 1.5,
        "Max": 2.5,
        "Ultra": 4.0,
    }
    gpu_tflops *= gpu_multipliers.get(chip_info.tier.value, 1.0)

    # Operation-specific ANE efficiency multipliers
    # ANE excels at certain patterns due to dedicated hardware
    op_multipliers = {
        "matmul": 1.5,  # ANE very efficient at matmul
        "linear": 1.5,
        "conv2d": 2.0,  # ANE optimized for convolutions
        "depthwise_conv": 2.5,  # Especially efficient
        "batch_norm": 1.2,
        "layer_norm": 1.3,
        "gelu": 0.8,  # Simple ops - less benefit
        "silu": 0.8,
        "relu": 0.5,  # Very simple - overhead dominates
        "softmax": 1.0,
    }

    op_mult = op_multipliers.get(operation, 1.0)

    # Base speedup from raw throughput
    # Note: TOPS vs TFLOPS isn't directly comparable, but gives rough estimate
    base_speedup = (ane_tops / gpu_tflops) * op_mult

    # Penalize for transfer overhead on small tensors
    total_elements = sum(_product(s) for s in input_shapes)

    if total_elements < 50000:  # Small
        base_speedup *= 0.5
    elif total_elements < 200000:  # Medium-small
        base_speedup *= 0.75
    # Large tensors get full benefit

    return base_speedup


def estimate_transfer_overhead_ms(
    shapes: List[Tuple[int, ...]],
    dtype_bytes: int = 4,
) -> float:
    """Estimate data transfer overhead between MLX and ANE.

    This helps dispatch decisions - if transfer overhead exceeds
    compute benefit, stay on GPU.

    Args:
        shapes: List of tensor shapes.
        dtype_bytes: Bytes per element (4 for float32).

    Returns:
        Estimated transfer time in milliseconds.
    """
    total_bytes = sum(_product(shape) * dtype_bytes for shape in shapes)

    # Unified memory means no actual copy, but there's still
    # cache invalidation and synchronization overhead.
    # Estimate ~20 GB/s effective throughput for small transfers.
    effective_bandwidth_gbps = 20.0

    time_s = total_bytes / (effective_bandwidth_gbps * 1e9)
    return time_s * 1000  # Convert to ms


def get_recommended_target(
    operation: str,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    is_training: bool = False,
) -> ComputeTarget:
    """Get recommended compute target for transformer-style operations.

    Convenience function for common transformer workloads.

    Args:
        operation: Operation type.
        batch_size: Batch size.
        seq_len: Sequence length.
        hidden_dim: Hidden dimension.
        is_training: Whether training or inference.

    Returns:
        Recommended ComputeTarget.
    """
    if is_training:
        return ComputeTarget.GPU

    # Build typical shapes for the operation
    if operation in ("matmul", "linear"):
        shapes = [(batch_size * seq_len, hidden_dim), (hidden_dim, hidden_dim)]
    elif operation == "layer_norm":
        shapes = [(batch_size, seq_len, hidden_dim)]
    elif operation in ("gelu", "silu", "relu"):
        shapes = [(batch_size, seq_len, hidden_dim)]
    else:
        shapes = [(batch_size, seq_len, hidden_dim)]

    decision = should_use_ane(operation, shapes, is_training)
    return decision.target
