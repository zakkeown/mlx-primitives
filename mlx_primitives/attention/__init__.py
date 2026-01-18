"""Attention mechanisms for MLX.

This module provides optimized attention variants including:
- Sliding window attention (fused Metal kernel)
- Flash Attention (O(n) memory via tiled online softmax)
- Chunked cross-attention for very long KV sequences
"""

from mlx_primitives.attention.sliding_window import (
    SlidingWindowAttention,
    create_sliding_window_mask,
    sliding_window_attention,
)
from mlx_primitives.attention.flash import (
    FlashAttention,
    flash_attention,
    get_optimal_flash_config,
)
from mlx_primitives.attention.chunked import (
    ChunkedCrossAttention,
    chunked_cross_attention,
    estimate_memory_savings,
)

__all__ = [
    # Sliding window
    "sliding_window_attention",
    "create_sliding_window_mask",
    "SlidingWindowAttention",
    # Flash attention
    "flash_attention",
    "FlashAttention",
    "get_optimal_flash_config",
    # Chunked cross-attention
    "chunked_cross_attention",
    "ChunkedCrossAttention",
    "estimate_memory_savings",
]
