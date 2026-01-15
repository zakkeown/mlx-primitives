"""Sliding Window Attention for MLX.

Sliding window attention limits each position to attend only to a fixed
window of previous positions, enabling efficient processing of long sequences.
Used in models like Mistral and Longformer.

Reference:
"Longformer: The Long-Document Transformer"
https://arxiv.org/abs/2004.05150

"Mistral 7B"
https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> mx.array:
    """Create a sliding window attention mask.

    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window.
        causal: If True, each position can only attend to previous positions
                within the window. If False, positions attend to both
                directions within the window.

    Returns:
        Attention mask of shape (seq_len, seq_len) with -inf for masked
        positions and 0 for unmasked positions.
    """
    # Create position indices
    rows = mx.arange(seq_len)[:, None]
    cols = mx.arange(seq_len)[None, :]

    # Compute distance between positions
    distance = rows - cols

    if causal:
        # Can attend to positions within [pos - window_size + 1, pos]
        # distance >= 0 means past positions
        # distance < window_size means within window
        valid = (distance >= 0) & (distance < window_size)
    else:
        # Can attend to positions within [pos - window_size//2, pos + window_size//2]
        half_window = window_size // 2
        valid = mx.abs(distance) <= half_window

    # Convert to attention mask (0 for valid, -inf for invalid)
    mask = mx.where(valid, mx.array(0.0), mx.array(float("-inf")))
    return mask


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention module.

    Each position can only attend to a fixed window of previous positions,
    which limits memory usage and enables processing of very long sequences.
    Information propagates across the full sequence through multiple layers.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        window_size: Size of the attention window.
        head_dim: Dimension of each head (default: dims // num_heads).
        causal: Whether to use causal (unidirectional) windowing.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> # Mistral-style attention with 4096 window
        >>> attn = SlidingWindowAttention(
        ...     dims=4096,
        ...     num_heads=32,
        ...     window_size=4096,
        ...     causal=True,
        ... )
        >>> x = mx.random.normal((2, 8192, 4096))  # Long sequence
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        window_size: int,
        head_dim: Optional[int] = None,
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = head_dim or dims // num_heads
        self.causal = causal
        self.dropout = dropout

        self.scale = 1.0 / math.sqrt(self.head_dim)

        inner_dim = num_heads * self.head_dim
        self.q_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.k_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.v_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, dims, bias=bias)

        # Cache for the sliding window mask
        self._cached_mask: Optional[mx.array] = None
        self._cached_seq_len: int = 0

    def _get_mask(self, seq_len: int) -> mx.array:
        """Get or create sliding window mask."""
        if self._cached_mask is None or seq_len > self._cached_seq_len:
            self._cached_mask = create_sliding_window_mask(
                seq_len, self.window_size, self.causal
            )
            self._cached_seq_len = seq_len
        return self._cached_mask[:seq_len, :seq_len]

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Compute sliding window attention.

        Args:
            queries: Query tensor of shape (batch, seq_q, dims).
            keys: Key tensor of shape (batch, seq_kv, dims).
            values: Value tensor of shape (batch, seq_kv, dims).
            mask: Additional attention mask to combine with window mask.
            cache: KV cache for incremental decoding.

        Returns:
            Tuple of output tensor and optional updated cache.
        """
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        batch_size, seq_q, _ = queries.shape
        _, seq_kv, _ = keys.shape

        # Project Q, K, V
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        # Reshape to (batch, seq, num_heads, head_dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

            # For sliding window, we only need to keep the last window_size KV pairs
            if k.shape[1] > self.window_size:
                k = k[:, -self.window_size:]
                v = v[:, -self.window_size:]

        new_cache = (k, v) if cache is not None else None
        seq_kv = k.shape[1]

        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Get sliding window mask
        window_mask = self._get_mask(max(seq_q, seq_kv))

        # Handle different seq lengths for cross-attention
        if seq_q != seq_kv:
            # Create appropriate mask for cross-attention
            rows = mx.arange(seq_q)[:, None]
            cols = mx.arange(seq_kv)[None, :]
            distance = rows - cols + (seq_kv - seq_q)  # Align to end

            if self.causal:
                valid = (distance >= 0) & (distance < self.window_size)
            else:
                half_window = self.window_size // 2
                valid = mx.abs(distance) <= half_window

            window_mask = mx.where(valid, mx.array(0.0), mx.array(float("-inf")))
        else:
            window_mask = window_mask[:seq_q, :seq_kv]

        scores = scores + window_mask

        # Apply additional mask
        if mask is not None:
            scores = scores + mask

        # Softmax and weighted sum
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape output
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_q, -1)

        return self.out_proj(output), new_cache


class SlidingWindowCache:
    """KV Cache manager for sliding window attention.

    This class manages a fixed-size rolling buffer for KV pairs,
    automatically evicting old entries when the window fills up.

    Args:
        window_size: Maximum number of KV pairs to store.
        num_heads: Number of attention heads.
        head_dim: Dimension of each head.

    Example:
        >>> cache = SlidingWindowCache(window_size=4096, num_heads=32, head_dim=128)
        >>> for step in range(10000):
        ...     k_new = mx.random.normal((1, 1, 32, 128))
        ...     v_new = mx.random.normal((1, 1, 32, 128))
        ...     k_full, v_full = cache.update(k_new, v_new)
        ...     # k_full, v_full have at most window_size entries
    """

    def __init__(
        self,
        window_size: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self._k_buffer: Optional[mx.array] = None
        self._v_buffer: Optional[mx.array] = None
        self._length: int = 0

    def reset(self) -> None:
        """Clear the cache."""
        self._k_buffer = None
        self._v_buffer = None
        self._length = 0

    def update(
        self,
        k_new: mx.array,
        v_new: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Add new KV pairs to the cache.

        Args:
            k_new: New keys (batch, seq, num_heads, head_dim).
            v_new: New values (batch, seq, num_heads, head_dim).

        Returns:
            Tuple of (keys, values) including the full cache contents.
        """
        batch_size = k_new.shape[0]
        new_seq_len = k_new.shape[1]

        if self._k_buffer is None:
            # Initialize buffer
            self._k_buffer = k_new
            self._v_buffer = v_new
            self._length = new_seq_len
        else:
            # Append new entries
            self._k_buffer = mx.concatenate([self._k_buffer, k_new], axis=1)
            self._v_buffer = mx.concatenate([self._v_buffer, v_new], axis=1)
            self._length += new_seq_len

            # Trim to window size
            if self._length > self.window_size:
                excess = self._length - self.window_size
                self._k_buffer = self._k_buffer[:, excess:]
                self._v_buffer = self._v_buffer[:, excess:]
                self._length = self.window_size

        assert self._k_buffer is not None and self._v_buffer is not None
        return self._k_buffer, self._v_buffer

    @property
    def length(self) -> int:
        """Current number of cached KV pairs."""
        return self._length

    def get(self) -> Optional[Tuple[mx.array, mx.array]]:
        """Get the current cache contents."""
        if self._k_buffer is None:
            return None
        return self._k_buffer, self._v_buffer
