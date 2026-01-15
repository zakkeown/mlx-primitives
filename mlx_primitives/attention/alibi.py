"""ALiBi (Attention with Linear Biases) for MLX.

ALiBi adds a linear bias to attention scores based on the distance between
query and key positions. This provides positional information without
learned positional embeddings and extrapolates well to longer sequences.

Reference:
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def get_alibi_slopes(num_heads: int) -> mx.array:
    """Compute ALiBi slopes for each attention head.

    The slopes follow a geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    where n is the number of heads.

    For heads that aren't a power of 2, we use the closest power of 2 and
    interpolate the slopes.

    Args:
        num_heads: Number of attention heads.

    Returns:
        Array of slopes with shape (num_heads,).
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        # Power of 2: use exact formula
        slopes = _get_slopes_power_of_2(num_heads)
    else:
        # Not power of 2: interpolate between closest powers of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = _get_slopes_power_of_2(closest_power_of_2)

        # Get additional slopes from the next power of 2
        next_power_of_2 = closest_power_of_2 * 2
        extra_slopes = _get_slopes_power_of_2(next_power_of_2)

        # Take every other slope from the extra slopes
        extra_slopes = extra_slopes[0::2][: num_heads - closest_power_of_2]
        slopes = slopes + extra_slopes

    return mx.array(slopes, dtype=mx.float32)


def alibi_bias(
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    slopes: Optional[mx.array] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Compute ALiBi position bias matrix.

    Args:
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        num_heads: Number of attention heads.
        slopes: Pre-computed slopes (optional).
        dtype: Output data type.

    Returns:
        Bias matrix of shape (1, num_heads, seq_len_q, seq_len_k).
    """
    if slopes is None:
        slopes = get_alibi_slopes(num_heads)

    # Compute relative positions
    # For causal attention, we want negative values for past positions
    # position[i, j] = j - i (positive when key is after query)
    q_pos = mx.arange(seq_len_q)[:, None]
    k_pos = mx.arange(seq_len_k)[None, :]

    # Relative distance: negative for past tokens, positive for future
    # For causal models, we typically want to penalize further distances
    relative_pos = k_pos - q_pos  # (seq_q, seq_k)

    # Convert to float for multiplication
    relative_pos = relative_pos.astype(dtype)

    # Apply slopes: bias = -slope * |distance|
    # We use the signed distance, which naturally handles causal masking
    # (future tokens have positive distance, past have negative)
    # The bias should be 0 for the current position and decrease for further positions
    slopes = slopes[:, None, None]  # (num_heads, 1, 1)
    bias = slopes * relative_pos[None, :, :]  # (num_heads, seq_q, seq_k)

    return bias[None, :, :, :].astype(dtype)  # (1, num_heads, seq_q, seq_k)


class ALiBi(nn.Module):
    """Attention with Linear Biases module.

    ALiBi replaces positional embeddings with a linear bias added to
    attention scores. The bias decreases linearly with distance, with
    different slopes for different heads.

    Benefits:
    - No learned positional embeddings needed
    - Better extrapolation to longer sequences than trained on
    - Simple and efficient

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension of each head (default: dims // num_heads).
        causal: Whether to apply causal masking.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = ALiBi(dims=768, num_heads=12, causal=True)
        >>> x = mx.random.normal((2, 1024, 768))
        >>> output = attn(x)  # (2, 1024, 768)

        >>> # Extrapolate to longer sequences
        >>> x_long = mx.random.normal((2, 4096, 768))
        >>> output_long = attn(x_long)  # Works without retraining!
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

        # Pre-compute slopes (they don't change)
        self.slopes = get_alibi_slopes(num_heads)

        # Projections
        inner_dim = num_heads * self.head_dim
        self.q_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.k_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.v_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, dims, bias=bias)

        # Cache for ALiBi bias (indexed by seq length)
        self._bias_cache: dict[Tuple[int, int], mx.array] = {}

    def _get_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        """Get or compute ALiBi bias for given sequence lengths."""
        cache_key = (seq_len_q, seq_len_k)

        if cache_key not in self._bias_cache:
            bias = alibi_bias(
                seq_len_q, seq_len_k, self.num_heads, self.slopes, dtype
            )
            self._bias_cache[cache_key] = bias

        return self._bias_cache[cache_key]

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Compute attention with ALiBi positional bias.

        Args:
            queries: Query tensor of shape (batch, seq_q, dims).
            keys: Key tensor of shape (batch, seq_kv, dims).
            values: Value tensor of shape (batch, seq_kv, dims).
            mask: Additional attention mask.
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

        new_cache = (k, v) if cache is not None else None
        seq_kv = k.shape[1]

        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Add ALiBi bias
        alibi = self._get_alibi_bias(seq_q, seq_kv, scores.dtype)
        scores = scores + alibi

        # Apply causal mask
        if self.causal:
            causal_mask = mx.triu(
                mx.full((seq_q, seq_kv), float("-inf")),
                k=seq_kv - seq_q + 1,
            )
            scores = scores + causal_mask

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


class ALiBiGQA(nn.Module):
    """ALiBi combined with Grouped Query Attention.

    This combines ALiBi positional encoding with GQA for efficient
    long-context attention.

    Args:
        dims: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension of each head.
        causal: Whether to apply causal masking.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = ALiBiGQA(dims=4096, num_heads=32, num_kv_heads=8, causal=True)
        >>> x = mx.random.normal((2, 8192, 4096))
        >>> output, _ = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        causal: bool = False,
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

        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.slopes = get_alibi_slopes(num_heads)

        q_dim = num_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dims, q_dim, bias=bias)
        self.k_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.v_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.out_proj = nn.Linear(q_dim, dims, bias=bias)

        self._bias_cache: dict[Tuple[int, int], mx.array] = {}

    def _get_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        cache_key = (seq_len_q, seq_len_k)
        if cache_key not in self._bias_cache:
            self._bias_cache[cache_key] = alibi_bias(
                seq_len_q, seq_len_k, self.num_heads, self.slopes, dtype
            )
        return self._bias_cache[cache_key]

    def _repeat_kv(self, x: mx.array) -> mx.array:
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :]
        x = mx.broadcast_to(
            x, (batch_size, seq_len, num_kv_heads, self.num_groups, head_dim)
        )
        return x.reshape(batch_size, seq_len, self.num_heads, head_dim)

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        batch_size, seq_q, _ = queries.shape
        _, seq_kv, _ = keys.shape

        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)

        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        # Always return cache for potential incremental decoding
        new_cache = (k, v)
        seq_kv = k.shape[1]

        # Repeat KV heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Add ALiBi bias
        alibi = self._get_alibi_bias(seq_q, seq_kv, scores.dtype)
        scores = scores + alibi

        if self.causal:
            causal_mask = mx.triu(
                mx.full((seq_q, seq_kv), float("-inf")),
                k=seq_kv - seq_q + 1,
            )
            scores = scores + causal_mask

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_q, -1)

        return self.out_proj(output), new_cache
