"""Advanced primitives for MLX.

This module provides advanced building blocks:
- MoE: Mixture of Experts layers
- SSM: State Space Models (Mamba)
- KV Cache: Efficient cache implementations
- Paged Attention: vLLM-style memory management for serving
- Quantization: Weight and activation quantization
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

# KV Cache
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
    # KV Cache
    "KVCache",
    "SlidingWindowCache",
    "PagedKVCache",
    "RotatingKVCache",
    "CompressedKVCache",
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
