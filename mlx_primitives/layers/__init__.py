"""Custom neural network layers for MLX.

This module provides layers not included in mlx.nn:
- Normalization: RMSNorm, GroupNorm, InstanceNorm, AdaLayerNorm
- Activations: SwiGLU, GeGLU, ReGLU, Mish
- Pooling: AdaptiveAvgPool, AdaptiveMaxPool, GeM
"""

# Normalization layers
from mlx_primitives.layers.normalization import (
    RMSNorm,
    GroupNorm,
    InstanceNorm,
    AdaLayerNorm,
    QKNorm,
    rms_norm,
    group_norm,
)

# Activation functions
from mlx_primitives.layers.activations import (
    SwiGLU,
    GeGLU,
    ReGLU,
    FusedSwiGLU,
    Mish,
    mish,
    GELUTanh,
    gelu_tanh,
    SquaredReLU,
    squared_relu,
    QuickGELU,
    quick_gelu,
    Swish,
    swish,
    HardSwish,
    hard_swish,
    HardSigmoid,
    hard_sigmoid,
    get_activation,
    ACTIVATIONS,
)

# Pooling layers
from mlx_primitives.layers.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    GlobalAttentionPooling,
    GeM,
    SpatialPyramidPooling,
    AvgPool1d,
    MaxPool1d,
)

__all__ = [
    # Normalization
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    "AdaLayerNorm",
    "QKNorm",
    "rms_norm",
    "group_norm",
    # Activations
    "SwiGLU",
    "GeGLU",
    "ReGLU",
    "FusedSwiGLU",
    "Mish",
    "mish",
    "GELUTanh",
    "gelu_tanh",
    "SquaredReLU",
    "squared_relu",
    "QuickGELU",
    "quick_gelu",
    "Swish",
    "swish",
    "HardSwish",
    "hard_swish",
    "HardSigmoid",
    "hard_sigmoid",
    "get_activation",
    "ACTIVATIONS",
    # Pooling
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "GlobalAttentionPooling",
    "GeM",
    "SpatialPyramidPooling",
    "AvgPool1d",
    "MaxPool1d",
]
