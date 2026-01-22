"""Advanced primitives for MLX.

This module provides advanced building blocks:
- MoE: Mixture of Experts layers
- SSM: State Space Models (Mamba)
- Paged Attention: vLLM-style memory management for serving
- Quantization: Weight and activation quantization

Note:
    KV Cache implementations have moved to ``mlx_primitives.cache``.
    Use the ``create_kv_cache()`` factory function for the recommended API:

    >>> from mlx_primitives.cache import create_kv_cache, CacheType
    >>> cache = create_kv_cache(CacheType.SIMPLE, batch_size=1, ...)
"""

# Mixture of Experts
from mlx_primitives.advanced.moe import (
    TopKRouter,
    ExpertChoiceRouter,
    Expert,
    MoELayer,
    SwitchMoE,
    load_balancing_loss,
    router_z_loss,
)

# State Space Models
from mlx_primitives.advanced.ssm import (
    selective_scan,
    MambaBlock,
    S4Layer,
    Mamba,
    H3Layer,
    H3Block,
    H3,
)

# KV Cache - DEPRECATED: These classes are still importable for backwards
# compatibility, but emit deprecation warnings when instantiated.
# Users should use mlx_primitives.cache instead.
from mlx_primitives.advanced.kv_cache import (
    KVCache,
    SlidingWindowCache,
    PagedKVCache,
    RotatingKVCache,
    CompressedKVCache,
)

# Paged Attention (vLLM-style)
from mlx_primitives.advanced.paged_attention import (
    PagedKVCache as PagedKVCacheV2,
    BlockManager,
    BlockConfig,
    SequenceState,
    create_paged_attention_mask,
)

# Quantization
from mlx_primitives.advanced.quantization import (
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    DynamicQuantizer,
    CalibrationCollector,
    quantize_model_weights,
    Int4Linear,
    QLoRALinear,
    GPTQLinear,
    AWQLinear,
)

__all__ = [
    # MoE
    "TopKRouter",
    "ExpertChoiceRouter",
    "Expert",
    "MoELayer",
    "SwitchMoE",
    "load_balancing_loss",
    "router_z_loss",
    # SSM
    "selective_scan",
    "MambaBlock",
    "S4Layer",
    "Mamba",
    "H3Layer",
    "H3Block",
    "H3",
    # KV Cache (DEPRECATED - use mlx_primitives.cache instead)
    "KVCache",  # Deprecated: use mlx_primitives.cache.SimpleKVCache
    "SlidingWindowCache",  # Deprecated: use mlx_primitives.cache.SlidingWindowCache
    "PagedKVCache",  # Deprecated: use mlx_primitives.advanced.paged_attention.PagedKVCache
    "RotatingKVCache",  # Deprecated: use mlx_primitives.cache.RotatingKVCache
    "CompressedKVCache",  # Deprecated: use mlx_primitives.cache.CompressedKVCache
    # Paged Attention (vLLM-style)
    "PagedKVCacheV2",
    "BlockManager",
    "BlockConfig",
    "SequenceState",
    "create_paged_attention_mask",
    # Quantization
    "quantize_tensor",
    "dequantize_tensor",
    "QuantizedLinear",
    "DynamicQuantizer",
    "CalibrationCollector",
    "quantize_model_weights",
    "Int4Linear",
    "QLoRALinear",
    "GPTQLinear",
    "AWQLinear",
]
