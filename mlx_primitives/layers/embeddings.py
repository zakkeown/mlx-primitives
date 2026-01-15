"""Positional embedding layers for MLX.

This module provides various positional embedding implementations:
- SinusoidalEmbedding: Classic sinusoidal positional encoding
- LearnedPositionalEmbedding: Learnable absolute positions
- RotaryEmbedding: Standalone rotary embeddings module
- AlibiEmbedding: ALiBi as standalone module
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embeddings.

    Classic transformer positional encoding using sine and cosine functions
    of different frequencies.

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        dims: Embedding dimension.
        max_seq_len: Maximum sequence length.
        base: Base for the geometric progression (default: 10000).

    Reference:
        "Attention Is All You Need"
        https://arxiv.org/abs/1706.03762

    Example:
        >>> embed = SinusoidalEmbedding(dims=512, max_seq_len=2048)
        >>> positions = mx.arange(100)
        >>> pe = embed(positions)  # (100, 512)
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()

        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute embeddings
        self._embeddings = self._create_embeddings(max_seq_len)

    def _create_embeddings(self, seq_len: int) -> mx.array:
        """Create sinusoidal embeddings."""
        positions = mx.arange(seq_len)[:, None]  # (seq_len, 1)
        dims_range = mx.arange(0, self.dims, 2)  # (dims/2,)

        # Compute frequencies
        freqs = self.base ** (-dims_range / self.dims)  # (dims/2,)

        # Compute angles
        angles = positions * freqs  # (seq_len, dims/2)

        # Interleave sin and cos
        sin_embeddings = mx.sin(angles)
        cos_embeddings = mx.cos(angles)

        # Combine: [sin, cos, sin, cos, ...]
        embeddings = mx.zeros((seq_len, self.dims))
        embeddings = mx.concatenate([
            sin_embeddings[:, :, None],
            cos_embeddings[:, :, None]
        ], axis=2).reshape(seq_len, self.dims)

        return embeddings

    def __call__(
        self,
        positions: Optional[mx.array] = None,
        seq_len: Optional[int] = None,
    ) -> mx.array:
        """Get positional embeddings.

        Args:
            positions: Position indices (optional).
            seq_len: Sequence length (if positions not provided).

        Returns:
            Positional embeddings.
        """
        if positions is not None:
            # Index into precomputed embeddings
            return mx.take(self._embeddings, positions, axis=0)
        elif seq_len is not None:
            return self._embeddings[:seq_len]
        else:
            raise ValueError("Either positions or seq_len must be provided")

    def forward_with_offset(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> mx.array:
        """Get embeddings with position offset (for KV cache).

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Positional embeddings for positions [offset, offset + seq_len).
        """
        positions = mx.arange(offset, offset + seq_len)
        return self(positions)


class LearnedPositionalEmbedding(nn.Module):
    """Learnable absolute positional embeddings.

    Unlike sinusoidal embeddings, these are learned during training.

    Args:
        dims: Embedding dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        padding_idx: Padding index (optional).

    Example:
        >>> embed = LearnedPositionalEmbedding(dims=512, max_seq_len=2048)
        >>> x = mx.random.normal((2, 100, 512))
        >>> x_with_pos = x + embed(seq_len=100)
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.dims = dims
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # Learnable embedding table
        self.embedding = nn.Embedding(max_seq_len, dims)

        # Initialize with small values
        scale = dims ** -0.5
        self.embedding.weight = mx.random.normal((max_seq_len, dims)) * scale

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        positions: Optional[mx.array] = None,
        seq_len: Optional[int] = None,
    ) -> mx.array:
        """Get positional embeddings.

        Args:
            positions: Position indices (optional).
            seq_len: Sequence length (if positions not provided).

        Returns:
            Positional embeddings.
        """
        if positions is None:
            if seq_len is None:
                raise ValueError("Either positions or seq_len must be provided")
            positions = mx.arange(seq_len)

        embeddings = self.embedding(positions)

        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        return embeddings

    def forward_with_offset(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> mx.array:
        """Get embeddings with position offset."""
        positions = mx.arange(offset, offset + seq_len)
        return self(positions)


class RotaryEmbedding(nn.Module):
    """Standalone Rotary Position Embedding module.

    Can be used independently with any attention mechanism.

    Args:
        dims: Dimension per head.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Example:
        >>> rope = RotaryEmbedding(dims=64, max_seq_len=8192)
        >>> q = mx.random.normal((2, 12, 100, 64))  # (batch, heads, seq, head_dim)
        >>> k = mx.random.normal((2, 12, 100, 64))
        >>> q_rot, k_rot = rope(q, k, offset=0)
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()

        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        self._freqs_cis = self._compute_freqs(max_seq_len)

    def _compute_freqs(self, seq_len: int) -> mx.array:
        """Compute complex frequencies for RoPE."""
        freqs = 1.0 / (self.base ** (mx.arange(0, self.dims, 2) / self.dims))
        positions = mx.arange(seq_len)

        freqs_outer = positions[:, None] * freqs[None, :]  # (seq_len, dims/2)

        # Return cos and sin stacked
        cos_freqs = mx.cos(freqs_outer)
        sin_freqs = mx.sin(freqs_outer)

        return mx.stack([cos_freqs, sin_freqs], axis=-1)  # (seq_len, dims/2, 2)

    def _apply_rotary(
        self,
        x: mx.array,
        freqs: mx.array,
    ) -> mx.array:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor (..., seq_len, dims).
            freqs: Frequency tensor (seq_len, dims/2, 2).

        Returns:
            Rotated tensor.
        """
        # Split x into pairs
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], -1, 2)  # (..., seq_len, dims/2, 2)

        # Apply rotation
        cos = freqs[..., 0]  # (seq_len, dims/2)
        sin = freqs[..., 1]  # (seq_len, dims/2)

        # Broadcast to match x shape
        cos = cos[None, None, :, :, None] if len(x_shape) == 4 else cos[:, :, None]
        sin = sin[None, None, :, :, None] if len(x_shape) == 4 else sin[:, :, None]

        # Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x0, x1 = x[..., 0], x[..., 1]

        # Handle broadcasting properly
        if len(x_shape) == 4:
            cos = freqs[None, None, :, :, 0]
            sin = freqs[None, None, :, :, 1]
        else:
            cos = freqs[:, :, 0]
            sin = freqs[:, :, 1]

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        out = mx.stack([out0, out1], axis=-1)
        return out.reshape(x_shape)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> tuple:
        """Apply rotary embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq_len, head_dim).
            k: Key tensor (batch, heads, seq_len, head_dim).
            offset: Position offset for KV cache.

        Returns:
            Tuple of (rotated_q, rotated_k).
        """
        seq_len = q.shape[2]
        freqs = self._freqs_cis[offset:offset + seq_len]

        q_rot = self._apply_rotary(q, freqs)
        k_rot = self._apply_rotary(k, freqs)

        return q_rot, k_rot


class AlibiEmbedding(nn.Module):
    """Standalone ALiBi (Attention with Linear Biases) module.

    Adds linear biases to attention scores based on position differences.
    Does not require learned parameters.

    Args:
        num_heads: Number of attention heads.

    Reference:
        "Train Short, Test Long: Attention with Linear Biases"
        https://arxiv.org/abs/2108.12409

    Example:
        >>> alibi = AlibiEmbedding(num_heads=12)
        >>> attn_scores = mx.random.normal((2, 12, 100, 100))
        >>> biased_scores = alibi(attn_scores)
    """

    def __init__(self, num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self._slopes = self._compute_slopes()

    def _compute_slopes(self) -> mx.array:
        """Compute per-head slopes."""
        # Get slopes in geometric sequence
        ratio = 2 ** (-8 / self.num_heads)
        slopes = []
        for i in range(self.num_heads):
            slopes.append(ratio ** (i + 1))
        return mx.array(slopes)

    def get_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
    ) -> mx.array:
        """Get ALiBi bias matrix.

        Args:
            seq_len_q: Query sequence length.
            seq_len_k: Key sequence length.

        Returns:
            Bias matrix of shape (num_heads, seq_len_q, seq_len_k).
        """
        # Position differences: q_pos - k_pos
        q_pos = mx.arange(seq_len_q)[:, None]
        k_pos = mx.arange(seq_len_k)[None, :]
        rel_pos = q_pos - k_pos  # (seq_len_q, seq_len_k)

        # Apply slopes
        # slopes: (num_heads,) -> (num_heads, 1, 1)
        slopes = self._slopes[:, None, None]
        bias = slopes * rel_pos[None, :, :]  # (num_heads, seq_len_q, seq_len_k)

        return bias

    def __call__(
        self,
        attention_scores: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """Add ALiBi bias to attention scores.

        Args:
            attention_scores: Attention scores (batch, heads, seq_q, seq_k).
            offset: Position offset for KV cache.

        Returns:
            Biased attention scores.
        """
        batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape

        # Adjust for offset
        q_pos = mx.arange(offset, offset + seq_len_q)[:, None]
        k_pos = mx.arange(seq_len_k)[None, :]
        rel_pos = q_pos - k_pos

        slopes = self._slopes[:, None, None]
        bias = slopes * rel_pos[None, :, :]

        return attention_scores + bias[None, :, :, :]


class RelativePositionalEmbedding(nn.Module):
    """Relative positional embeddings (T5-style).

    Adds learnable bias based on relative positions between query and key.

    Args:
        num_heads: Number of attention heads.
        num_buckets: Number of buckets for relative positions.
        max_distance: Maximum distance for bucketing.
        bidirectional: Whether to use bidirectional positions.

    Reference:
        "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
        https://arxiv.org/abs/1910.10683
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Learnable relative position embeddings
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(
        self,
        relative_position: mx.array,
    ) -> mx.array:
        """Convert relative positions to bucket indices."""
        ret = mx.zeros_like(relative_position)

        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            # Positive positions go in second half
            ret = ret + mx.where(relative_position > 0, num_buckets, 0)
            relative_position = mx.abs(relative_position)
        else:
            # Clamp to non-negative
            relative_position = mx.maximum(-relative_position, 0)
            num_buckets = self.num_buckets

        # Half of buckets are for exact positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half are for log-spaced positions
        relative_position_if_large = max_exact + (
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)

        ret = ret + mx.where(is_small, relative_position, relative_position_if_large)

        return ret.astype(mx.int32)

    def get_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
    ) -> mx.array:
        """Get relative position bias.

        Args:
            seq_len_q: Query sequence length.
            seq_len_k: Key sequence length.

        Returns:
            Bias of shape (1, num_heads, seq_len_q, seq_len_k).
        """
        q_pos = mx.arange(seq_len_q)[:, None]
        k_pos = mx.arange(seq_len_k)[None, :]
        relative_position = k_pos - q_pos

        buckets = self._relative_position_bucket(relative_position)
        values = self.embedding(buckets.reshape(-1))  # (seq_q * seq_k, num_heads)
        values = values.reshape(seq_len_q, seq_len_k, self.num_heads)
        values = values.transpose(2, 0, 1)  # (num_heads, seq_q, seq_k)

        return values[None, :, :, :]  # (1, num_heads, seq_q, seq_k)

    def __call__(
        self,
        attention_scores: mx.array,
    ) -> mx.array:
        """Add relative position bias to attention scores.

        Args:
            attention_scores: (batch, heads, seq_q, seq_k).

        Returns:
            Biased attention scores.
        """
        seq_len_q = attention_scores.shape[2]
        seq_len_k = attention_scores.shape[3]

        bias = self.get_bias(seq_len_q, seq_len_k)

        return attention_scores + bias
