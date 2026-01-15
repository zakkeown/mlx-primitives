"""Grouped Query Attention (GQA) for MLX.

GQA uses fewer key-value heads than query heads, with each group of
query heads sharing the same key-value head. This reduces memory bandwidth
during inference while maintaining model quality.

Reference:
"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention module.

    In GQA, query heads are divided into groups, and each group shares
    a single key-value head. This is a generalization between:
    - Multi-Head Attention (MHA): num_kv_heads == num_heads
    - Multi-Query Attention (MQA): num_kv_heads == 1

    Args:
        dims: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key-value heads (must divide num_heads evenly).
        head_dim: Dimension of each head (default: dims // num_heads).
        causal: Whether to apply causal masking.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> # Llama 2 70B style: 64 query heads, 8 KV heads
        >>> attn = GroupedQueryAttention(
        ...     dims=8192,
        ...     num_heads=64,
        ...     num_kv_heads=8,
        ...     causal=True,
        ... )
        >>> x = mx.random.normal((2, 1024, 8192))
        >>> output = attn(x)  # (2, 1024, 8192)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim or dims // num_heads
        self.causal = causal
        self.dropout = dropout

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query projection: full num_heads
        q_dim = num_heads * self.head_dim
        # Key/Value projections: reduced num_kv_heads
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dims, q_dim, bias=bias)
        self.k_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.v_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.out_proj = nn.Linear(q_dim, dims, bias=bias)

    def _repeat_kv(self, x: mx.array) -> mx.array:
        """Repeat key/value heads to match number of query heads.

        Args:
            x: Tensor of shape (batch, seq, num_kv_heads, head_dim).

        Returns:
            Tensor of shape (batch, seq, num_heads, head_dim).
        """
        batch_size, seq_len, num_kv_heads, head_dim = x.shape

        # Expand to (batch, seq, num_kv_heads, num_groups, head_dim)
        x = x[:, :, :, None, :]
        # Broadcast to repeat along groups dimension
        x = mx.broadcast_to(
            x, (batch_size, seq_len, num_kv_heads, self.num_groups, head_dim)
        )
        # Reshape to (batch, seq, num_heads, head_dim)
        return x.reshape(batch_size, seq_len, self.num_heads, head_dim)

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Compute grouped query attention.

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
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        # Reshape: (batch, seq, num_heads/num_kv_heads, head_dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)

        # Handle KV cache for incremental decoding
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        # Always return cache for potential incremental decoding
        new_cache = (k, v)

        # Repeat KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        if self.causal:
            seq_k = k.shape[2]
            causal_mask = mx.triu(
                mx.full((seq_q, seq_k), float("-inf")),
                k=seq_k - seq_q + 1,
            )
            scores = scores + causal_mask

        # Apply custom mask
        if mask is not None:
            scores = scores + mask

        # Softmax and weighted sum
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape output: (batch, seq_q, dims)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_q, -1)

        return self.out_proj(output), new_cache


def gqa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    num_kv_groups: int,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    causal: bool = False,
) -> mx.array:
    """Functional interface for Grouped Query Attention.

    This is a lower-level function that operates on pre-projected Q, K, V
    tensors, useful for custom architectures.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        num_kv_groups: Number of query heads per KV head.
        scale: Scale factor (default: 1/sqrt(head_dim)).
        mask: Optional attention mask.
        causal: Whether to apply causal masking.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Repeat KV heads
    k = k[:, :, :, None, :]
    k = mx.broadcast_to(k, (batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim))
    k = k.reshape(batch_size, seq_kv, num_heads, head_dim)

    v = v[:, :, :, None, :]
    v = mx.broadcast_to(v, (batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim))
    v = v.reshape(batch_size, seq_kv, num_heads, head_dim)

    # Transpose for matmul
    q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, dim)
    k = k.transpose(0, 2, 1, 3)  # (batch, heads, seq_kv, dim)
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

    # Transpose back
    return output.transpose(0, 2, 1, 3)
