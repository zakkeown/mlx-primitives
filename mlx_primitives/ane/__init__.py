"""Apple Neural Engine (ANE) offload primitives.

This module provides automatic dispatch to the Neural Engine for suitable
operations during inference. ANE can provide significant throughput improvements
for specific operations like matrix multiplication and convolutions.

Note:
    ANE access requires Core ML (coremltools package).
    ANE is inference-only - training operations always use GPU.

Example:
    >>> from mlx_primitives.ane import ane_matmul, is_ane_available
    >>> if is_ane_available():
    ...     result = ane_matmul(a, b, use_ane="auto")
    ... else:
    ...     result = a @ b
"""

from mlx_primitives.ane.detection import (
    ANECapabilities,
    get_ane_info,
    get_ane_tops,
    is_ane_available,
    supports_operation,
)
from mlx_primitives.ane.dispatch import (
    ComputeTarget,
    DispatchDecision,
    estimate_transfer_overhead_ms,
    get_recommended_target,
    should_use_ane,
)
from mlx_primitives.ane.model_cache import (
    CoreMLModelCache,
    ModelSpec,
    get_model_cache,
)
from mlx_primitives.ane.primitives import ane_linear, ane_matmul

__all__ = [
    # Detection
    "ANECapabilities",
    "get_ane_info",
    "get_ane_tops",
    "is_ane_available",
    "supports_operation",
    # Dispatch
    "ComputeTarget",
    "DispatchDecision",
    "should_use_ane",
    "get_recommended_target",
    "estimate_transfer_overhead_ms",
    # Model cache
    "CoreMLModelCache",
    "ModelSpec",
    "get_model_cache",
    # Primitives
    "ane_matmul",
    "ane_linear",
]
