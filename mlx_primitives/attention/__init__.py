"""Attention mechanisms for MLX.

This module provides optimized attention variants including:
- Sliding window attention (fused Metal kernel)
- Sparse attention patterns (BigBird, Longformer)
- Chunked cross-attention for long sequences
"""

from mlx_primitives.attention.sliding_window import (
    SlidingWindowAttention,
    create_sliding_window_mask,
    sliding_window_attention,
)

__all__ = [
    "sliding_window_attention",
    "create_sliding_window_mask",
    "SlidingWindowAttention",
]
