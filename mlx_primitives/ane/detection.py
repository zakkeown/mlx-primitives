"""ANE (Apple Neural Engine) capability detection.

This module detects Neural Engine availability and capabilities
for the current Apple Silicon chip.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from mlx_primitives.hardware import ChipFamily, ChipTier, get_chip_info


@dataclass(frozen=True)
class ANECapabilities:
    """Neural Engine capabilities for the current device.

    Attributes:
        available: Whether ANE is available for use.
        tops: Throughput in trillion operations per second.
        supported_dtypes: Tuple of supported data type strings.
        max_tensor_size_mb: Maximum tensor size ANE can handle.
        supports_dynamic_shapes: Whether dynamic shapes are supported.
        supports_training: Whether training (gradients) is supported.
        supports_matmul: Whether matrix multiply is supported.
        supports_conv2d: Whether 2D convolution is supported.
        supports_depthwise_conv: Whether depthwise convolution is supported.
        supports_batch_norm: Whether batch normalization is supported.
        supports_layer_norm: Whether layer normalization is supported.
        supports_activations: Whether activation functions are supported.
    """

    available: bool
    tops: float
    supported_dtypes: Tuple[str, ...]
    max_tensor_size_mb: int
    supports_dynamic_shapes: bool = False
    supports_training: bool = False  # ANE is inference-only
    supports_matmul: bool = True
    supports_conv2d: bool = True
    supports_depthwise_conv: bool = True
    supports_batch_norm: bool = True
    supports_layer_norm: bool = True
    supports_activations: bool = True


# ANE TOPS specifications per chip family
# Ultra variants have doubled ANE capacity
_ANE_TOPS: dict[Tuple[ChipFamily, ChipTier], float] = {
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


def _check_coreml_available() -> bool:
    """Check if Core ML framework is available."""
    try:
        import coremltools  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_ane_info() -> ANECapabilities:
    """Get Neural Engine capabilities for the current device.

    Returns:
        ANECapabilities with availability and feature information.

    Example:
        >>> info = get_ane_info()
        >>> if info.available:
        ...     print(f"ANE available with {info.tops} TOPS")
    """
    chip_info = get_chip_info()

    # Check Core ML availability
    coreml_available = _check_coreml_available()

    if not coreml_available:
        return ANECapabilities(
            available=False,
            tops=0.0,
            supported_dtypes=(),
            max_tensor_size_mb=0,
        )

    # Get TOPS based on chip
    tops = _ANE_TOPS.get(
        (chip_info.family, chip_info.tier),
        # Default to M1 BASE if unknown
        _ANE_TOPS.get((ChipFamily.M1, ChipTier.BASE), 11.0),
    )

    # ANE supports fp16 primarily, with some int8 support
    supported_dtypes = ("float16", "float32", "int8")

    # Max tensor size varies by chip, but ~512MB is a safe estimate
    max_tensor_size_mb = 512

    return ANECapabilities(
        available=True,
        tops=tops,
        supported_dtypes=supported_dtypes,
        max_tensor_size_mb=max_tensor_size_mb,
        supports_dynamic_shapes=False,
        supports_training=False,
        supports_matmul=True,
        supports_conv2d=True,
        supports_depthwise_conv=True,
        supports_batch_norm=True,
        supports_layer_norm=True,
        supports_activations=True,
    )


def is_ane_available() -> bool:
    """Quick check if ANE offload is available.

    Returns:
        True if ANE can be used for inference.
    """
    return get_ane_info().available


def get_ane_tops() -> float:
    """Get ANE throughput in TOPS.

    Returns:
        Trillion operations per second (0 if unavailable).
    """
    info = get_ane_info()
    return info.tops if info.available else 0.0


def supports_operation(operation: str) -> bool:
    """Check if ANE supports a specific operation.

    Args:
        operation: Operation name (e.g., "matmul", "conv2d").

    Returns:
        True if the operation is supported on ANE.
    """
    info = get_ane_info()
    if not info.available:
        return False

    op_support = {
        "matmul": info.supports_matmul,
        "conv2d": info.supports_conv2d,
        "depthwise_conv": info.supports_depthwise_conv,
        "batch_norm": info.supports_batch_norm,
        "layer_norm": info.supports_layer_norm,
        "gelu": info.supports_activations,
        "silu": info.supports_activations,
        "relu": info.supports_activations,
        "softmax": info.supports_activations,
    }

    return op_support.get(operation, False)
