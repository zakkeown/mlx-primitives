"""Multi-Query Attention (MQA) for MLX.

MQA uses a single key-value head shared across all query heads,
dramatically reducing KV cache memory during inference.

Reference:
"Fast Transformer Decoding: One Write-Head is All You Need"
https://arxiv.org/abs/1911.02150
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention module.

    MQA is a special case of GQA where num_kv_heads == 1. All query heads
    share a single key-value head, which significantly reduces memory
    bandwidth during autoregressive decoding.

    Args:
        dims: Model dimension.
        num_heads: Number of query heads.
        head_dim: Dimension of each head (default: dims // num_heads).
        causal: Whether to apply causal masking.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = MultiQueryAttention(dims=768, num_heads=12, causal=True)
        >>> x = mx.random.normal((2, 1024, 768))
        >>> output, _ = attn(x)  # (2, 1024, 768)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = head_dim or dims // num_heads
        self.causal = causal
        self.dropout = dropout

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query projection: full num_heads
        q_dim = num_heads * self.head_dim
        # Key/Value projections: single head
        kv_dim = self.head_dim

        self.q_proj = nn.Linear(dims, q_dim, bias=bias)
        self.k_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.v_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.out_proj = nn.Linear(q_dim, dims, bias=bias)

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Compute multi-query attention.

        Args:
            queries: Query tensor of shape (batch, seq_q, dims).
            keys: Key tensor of shape (batch, seq_kv, dims).
                  If None, uses queries (self-attention).
            values: Value tensor of shape (batch, seq_kv, dims).
                    If None, uses keys.
            mask: Optional attention mask.
            cache: Optional KV cache tuple for incremental decoding.

        Returns:
            Tuple of:
            - Output tensor of shape (batch, seq_q, dims)
            - Updated cache tuple (or None if cache was None)
        """
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        batch_size, seq_q, _ = queries.shape
        _, seq_kv, _ = keys.shape

        # Project queries, keys, values
        q = self.q_proj(queries)  # (batch, seq_q, num_heads * head_dim)
        k = self.k_proj(keys)      # (batch, seq_kv, head_dim)
        v = self.v_proj(values)    # (batch, seq_kv, head_dim)

        # Reshape queries: (batch, seq_q, num_heads, head_dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)

        # Keys and values stay as (batch, seq_kv, head_dim)
        # but we'll add a head dimension: (batch, seq_kv, 1, head_dim)
        k = k[:, :, None, :]
        v = v[:, :, None, :]

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        # Always return cache for potential incremental decoding
        new_cache = (k, v)
        seq_kv = k.shape[1]

        # Broadcast KV to all heads
        # k, v: (batch, seq_kv, 1, head_dim) -> (batch, seq_kv, num_heads, head_dim)
        k = mx.broadcast_to(k, (batch_size, seq_kv, self.num_heads, self.head_dim))
        v = mx.broadcast_to(v, (batch_size, seq_kv, self.num_heads, self.head_dim))

        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        if self.causal:
            causal_mask = mx.triu(
                mx.full((seq_q, seq_kv), float("-inf")),
                k=seq_kv - seq_q + 1,
            )
            scores = scores + causal_mask

        # Apply custom mask
        if mask is not None:
            scores = scores + mask

        # Softmax and weighted sum
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape output
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_q, -1)

        return self.out_proj(output), new_cache


def mqa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    causal: bool = False,
) -> mx.array:
    """Functional interface for Multi-Query Attention.

    This function operates on pre-projected tensors where K and V have
    a single head dimension.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, head_dim) or (batch, seq_kv, 1, head_dim).
        v: Value tensor with same shape as k.
        scale: Scale factor (default: 1/sqrt(head_dim)).
        mask: Optional attention mask.
        causal: Whether to apply causal masking.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).
    """
    batch_size, seq_q, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Handle K/V shape: add head dim if missing
    if k.ndim == 3:
        k = k[:, :, None, :]
        v = v[:, :, None, :]

    seq_kv = k.shape[1]

    # Broadcast KV to all heads
    k = mx.broadcast_to(k, (batch_size, seq_kv, num_heads, head_dim))
    v = mx.broadcast_to(v, (batch_size, seq_kv, num_heads, head_dim))

    # Transpose for matmul
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Compute scores
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    if causal:
        causal_mask = mx.triu(
            mx.full((seq_q, seq_kv), float("-inf")),
            k=seq_kv - seq_q + 1,
        )
        scores = scores + causal_mask

    # Custom mask
    if mask is not None:
        scores = scores + mask

    # Softmax and output
    weights = mx.softmax(scores, axis=-1)
    output = weights @ v

    return output.transpose(0, 2, 1, 3)
