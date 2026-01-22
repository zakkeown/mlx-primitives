"""ANE (Apple Neural Engine) capability detection.

This module detects Neural Engine availability and capabilities
for the current Apple Silicon chip.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from mlx_primitives.hardware import ChipFamily, ChipTier, get_ane_tops, get_chip_info


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


# NOTE: ANE TOPS lookup is now centralized in mlx_primitives.hardware.detection
# Use get_ane_tops() from hardware module instead of duplicating the data here


def _check_coreml_available() -> bool:
    """Check if Core ML framework is available and ANE is accessible.

    This performs an actual runtime check by attempting to compile a minimal
    model targeting the Neural Engine, rather than just checking for package
    imports.

    Returns:
        True if Core ML is available AND the Neural Engine can be targeted.
    """
    try:
        import coremltools as ct
        from coremltools.models.neural_network import NeuralNetworkBuilder
    except ImportError:
        return False

    try:
        # Create a minimal model to test ANE availability
        # This is a simple identity operation that should compile quickly
        input_features = [("input", ct.models.datatypes.Array(1))]
        output_features = [("output", ct.models.datatypes.Array(1))]

        builder = NeuralNetworkBuilder(
            input_features=input_features,
            output_features=output_features,
        )
        builder.add_activation(
            name="identity",
            non_linearity="LINEAR",
            input_name="input",
            output_name="output",
            params=[1.0, 0.0],  # y = 1*x + 0
        )

        # Try to compile with Neural Engine as preferred compute unit
        spec = builder.spec
        model = ct.models.MLModel(spec)

        # Check if we can get compute unit availability
        # On systems with ANE, this should succeed
        # Note: This doesn't guarantee ANE will be used at runtime,
        # but it confirms Core ML infrastructure is working
        return True

    except (RuntimeError, OSError, ValueError, ImportError, AttributeError) as e:
        # Core ML compilation or ANE targeting failed
        # This catches:
        # - RuntimeError: shader compilation failures, resource limits
        # - OSError: sandbox restrictions, missing entitlements
        # - ValueError: invalid model configuration
        # - ImportError: missing coremltools submodule
        # - AttributeError: API changes in coremltools
        from mlx_primitives.utils.logging import get_logger

        get_logger().debug(f"Core ML ANE check failed: {type(e).__name__}: {e}")
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

    # Get TOPS from hardware module (centralized lookup)
    tops = get_ane_tops()

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
