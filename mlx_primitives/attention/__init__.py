"""Attention mechanisms for MLX.

This module provides high-performance attention implementations including:
- FlashAttention: Memory-efficient attention with tiling
- GroupedQueryAttention: GQA with configurable head groups
- MultiQueryAttention: Single KV head shared across Q heads
- SlidingWindowAttention: Fixed window context attention
- RoPE: Rotary position embeddings
- ALiBi: Attention with Linear Biases
"""

from mlx_primitives.attention.rope import RoPE, apply_rope
from mlx_primitives.attention.flash import FlashAttention
from mlx_primitives.attention.grouped_query import GroupedQueryAttention
from mlx_primitives.attention.multi_query import MultiQueryAttention
from mlx_primitives.attention.sliding_window import SlidingWindowAttention
from mlx_primitives.attention.alibi import ALiBi, alibi_bias

__all__ = [
    "FlashAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "SlidingWindowAttention",
    "RoPE",
    "apply_rope",
    "ALiBi",
    "alibi_bias",
]
