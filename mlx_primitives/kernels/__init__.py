"""Fused operation kernels for MLX.

This module provides fused operations that reduce memory bandwidth:
- Fused RMSNorm + Linear
- Fused Softmax + Dropout + Matmul
- Fused Bias + Activation (SwiGLU, GeGLU)
- Quantized linear operations
"""

from mlx_primitives.kernels.fused_activations import (
    GeGLU,
    SwiGLU,
    fused_geglu,
    fused_swiglu,
    gelu,
    silu,
)
from mlx_primitives.kernels.fused_norm_linear import (
    FusedRMSNormLinear,
    fused_rmsnorm_linear,
    rmsnorm,
)
from mlx_primitives.kernels.quantized_linear import (
    QuantizedLinear,
    dequantize_int4,
    dequantize_int8,
    int4_linear,
    int8_linear,
    quantize_int4,
    quantize_int8,
)

__all__ = [
    # Fused RMSNorm + Linear
    "fused_rmsnorm_linear",
    "rmsnorm",
    "FusedRMSNormLinear",
    # Fused Activations
    "fused_swiglu",
    "fused_geglu",
    "silu",
    "gelu",
    "SwiGLU",
    "GeGLU",
    # Quantization
    "quantize_int8",
    "quantize_int4",
    "int8_linear",
    "int4_linear",
    "dequantize_int8",
    "dequantize_int4",
    "QuantizedLinear",
]
