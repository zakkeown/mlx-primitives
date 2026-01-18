"""Example kernels demonstrating Metal-Triton DSL.

These examples show how to write various kernels using the DSL.
"""

from mlx_primitives.dsl.examples.vector_ops import vector_add, vector_mul, relu
from mlx_primitives.dsl.examples.reductions import sum_reduce, max_reduce
from mlx_primitives.dsl.examples.activations import (
    silu, gelu_tanh, gelu_exact, quick_gelu,
    fused_silu_mul, fused_gelu_mul, softplus, mish,
)
from mlx_primitives.dsl.examples.normalization import (
    layer_norm, rms_norm, fused_add_layer_norm, fused_add_rms_norm,
)
from mlx_primitives.dsl.examples.rope import (
    rope_forward, rope_inline, rope_qk_fused, rope_neox,
    precompute_rope_cache,
)

__all__ = [
    # Vector ops
    "vector_add",
    "vector_mul",
    "relu",
    # Reductions
    "sum_reduce",
    "max_reduce",
    # Activations
    "silu",
    "gelu_tanh",
    "gelu_exact",
    "quick_gelu",
    "fused_silu_mul",
    "fused_gelu_mul",
    "softplus",
    "mish",
    # Normalization
    "layer_norm",
    "rms_norm",
    "fused_add_layer_norm",
    "fused_add_rms_norm",
    # RoPE
    "rope_forward",
    "rope_inline",
    "rope_qk_fused",
    "rope_neox",
    "precompute_rope_cache",
]
