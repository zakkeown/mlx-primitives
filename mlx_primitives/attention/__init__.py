"""Attention mechanisms for MLX.

This module provides high-performance attention implementations including:
- FlashAttention: Memory-efficient attention with tiling
- GroupedQueryAttention: GQA with configurable head groups
- MultiQueryAttention: Single KV head shared across Q heads
- SlidingWindowAttention: Fixed window context attention
- RoPE: Rotary position embeddings
- ALiBi: Attention with Linear Biases
- BlockSparseAttention: Block-sparse patterns
- LongformerAttention: Sliding window + global attention
- BigBirdAttention: Random + window + global attention
- LinearAttention: O(n) attention approximation
- PerformerAttention: FAVOR+ random feature attention
- CosFormerAttention: Cosine-based linear attention
- QuantizedKVCache: INT8 quantized KV cache (~4x memory reduction)

Supports automatic precision selection (AUTO, FLOAT32, FLOAT16, MIXED) for
~2x memory bandwidth improvement when using float16 on supported inputs.
"""

from mlx_primitives.attention.rope import RoPE, apply_rope
from mlx_primitives.attention.flash import (
    FlashAttention,
    flash_attention_forward,
    scaled_dot_product_attention,
)
from mlx_primitives.config.precision import (
    PrecisionMode,
    PrecisionConfig,
    precision_context,
)
from mlx_primitives.attention.grouped_query import GroupedQueryAttention
from mlx_primitives.attention.multi_query import MultiQueryAttention
from mlx_primitives.attention.sliding_window import SlidingWindowAttention
from mlx_primitives.attention.alibi import ALiBi, alibi_bias
from mlx_primitives.attention.sparse import (
    BlockSparseAttention,
    LongformerAttention,
    BigBirdAttention,
    create_block_sparse_mask,
    create_sliding_window_mask,
    create_bigbird_mask,
)
from mlx_primitives.attention.linear import (
    LinearAttention,
    PerformerAttention,
    CosFormerAttention,
    elu_feature_map,
    relu_feature_map,
)
from mlx_primitives.kernels.fused_rope_attention import FusedRoPEFlashAttention
from mlx_primitives.attention.quantized_kv_cache import (
    QuantizedKVCache,
    QuantizedKVCacheAttention,
    quantize_kv_for_cache,
    dequantize_kv_from_cache,
)

__all__ = [
    # Core attention
    "FlashAttention",
    "FusedRoPEFlashAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "SlidingWindowAttention",
    # Functional API
    "flash_attention_forward",
    "scaled_dot_product_attention",
    # Precision configuration
    "PrecisionMode",
    "PrecisionConfig",
    "precision_context",
    # Position encodings
    "RoPE",
    "apply_rope",
    "ALiBi",
    "alibi_bias",
    # Sparse attention
    "BlockSparseAttention",
    "LongformerAttention",
    "BigBirdAttention",
    "create_block_sparse_mask",
    "create_sliding_window_mask",
    "create_bigbird_mask",
    # Linear attention
    "LinearAttention",
    "PerformerAttention",
    "CosFormerAttention",
    "elu_feature_map",
    "relu_feature_map",
    # Quantized KV cache
    "QuantizedKVCache",
    "QuantizedKVCacheAttention",
    "quantize_kv_for_cache",
    "dequantize_kv_from_cache",
]
