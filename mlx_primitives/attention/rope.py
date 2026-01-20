"""Rotary Position Embeddings (RoPE).

Implementation based on the RoFormer paper:
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input.

    This is a helper function for RoPE that splits the input along the last
    dimension and swaps/negates the halves to implement the rotation.

    Args:
        x: Input tensor of shape (..., dim) where dim is even.

    Returns:
        Rotated tensor of same shape as input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Precompute the frequency tensor for rotary embeddings.

    Args:
        dim: Dimension of the embeddings (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base for the geometric progression of frequencies.
        dtype: Data type for the output tensors.

    Returns:
        Tuple of (cos, sin, cos_doubled, sin_doubled) where:
        - cos, sin: Shape (max_seq_len, dim // 2)
        - cos_doubled, sin_doubled: Shape (max_seq_len, dim) - precomputed for efficiency
    """
    # Compute inverse frequencies: theta_i = base^(-2i/dim) for i = 0, 1, ..., dim/2-1
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    # Compute position indices
    t = mx.arange(max_seq_len, dtype=mx.float32)

    # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
    freqs = mx.outer(t, inv_freq)

    # Compute cos and sin
    cos = mx.cos(freqs).astype(dtype)
    sin = mx.sin(freqs).astype(dtype)

    # Precompute doubled versions to avoid per-call concatenation
    cos_doubled = mx.concatenate([cos, cos], axis=-1)
    sin_doubled = mx.concatenate([sin, sin], axis=-1)

    return cos, sin, cos_doubled, sin_doubled


def apply_rope(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
    cos_doubled: mx.array | None = None,
    sin_doubled: mx.array | None = None,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (..., seq_len, num_heads, head_dim) or
           (..., seq_len, head_dim).
        k: Key tensor of shape (..., seq_len, num_kv_heads, head_dim) or
           (..., seq_len, head_dim).
        cos: Precomputed cosine tensor of shape (max_seq_len, head_dim // 2).
        sin: Precomputed sine tensor of shape (max_seq_len, head_dim // 2).
        offset: Position offset for incremental decoding.
        cos_doubled: Optional precomputed doubled cos (max_seq_len, head_dim).
            If provided, avoids per-call concatenation.
        sin_doubled: Optional precomputed doubled sin (max_seq_len, head_dim).
            If provided, avoids per-call concatenation.

    Returns:
        Tuple of rotated (q, k) tensors with same shapes as input.
    """
    # Get sequence length from queries
    # Handle both (batch, seq, heads, dim) and (batch, seq, dim) formats
    if q.ndim == 4:
        seq_len = q.shape[1]
    else:
        seq_len = q.shape[-2]

    # Use precomputed doubled versions if available, otherwise compute
    if cos_doubled is not None and sin_doubled is not None:
        cos_d = cos_doubled[offset : offset + seq_len]
        sin_d = sin_doubled[offset : offset + seq_len]
    else:
        # Slice frequencies to match sequence length
        cos_slice = cos[offset : offset + seq_len]
        sin_slice = sin[offset : offset + seq_len]
        # Compute doubled versions (fallback for backward compatibility)
        cos_d = mx.concatenate([cos_slice, cos_slice], axis=-1)
        sin_d = mx.concatenate([sin_slice, sin_slice], axis=-1)

    # Reshape for broadcasting
    if q.ndim == 4:
        # (seq_len, dim) -> (1, seq_len, 1, dim)
        cos_d = cos_d[None, :, None, :]
        sin_d = sin_d[None, :, None, :]
    else:
        # (seq_len, dim) -> (1, seq_len, dim)
        cos_d = cos_d[None, :, :]
        sin_d = sin_d[None, :, :]

    # Apply rotation using complex multiplication formula:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos_d) + (_rotate_half(q) * sin_d)
    k_embed = (k * cos_d) + (_rotate_half(k) * sin_d)

    return q_embed, k_embed


class RoPE(nn.Module):
    """Rotary Position Embedding module.

    This module precomputes and caches sin/cos frequencies and provides
    a convenient interface for applying rotary embeddings to Q/K tensors.

    Args:
        dims: Dimension of the embeddings (head_dim, must be even).
        max_seq_len: Maximum sequence length to support.
        base: Base for frequency computation (default: 10000.0).
        scale: Optional scaling factor for extended context (default: 1.0).
            Note: The scale is squared when applied to the base frequency.

    Example:
        >>> rope = RoPE(dims=64, max_seq_len=8192)
        >>> q = mx.random.normal((2, 1024, 12, 64))  # (batch, seq, heads, head_dim)
        >>> k = mx.random.normal((2, 1024, 12, 64))
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scale: float = 1.0,
    ):
        super().__init__()

        if dims % 2 != 0:
            raise ValueError(f"dims must be even, got {dims}")

        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale = scale

        # Precomputed frequencies (lazily initialized)
        self._cos: mx.array | None = None
        self._sin: mx.array | None = None
        self._cos_doubled: mx.array | None = None
        self._sin_doubled: mx.array | None = None

    def _ensure_freqs(self, dtype: mx.Dtype = mx.float32) -> None:
        """Lazily compute frequencies on first use."""
        if self._cos is None or self._sin is None:
            # Apply scaling to base if using extended context
            # scale^2 is used per the original scaling paper convention
            effective_base = self.base * (self.scale**2) if self.scale != 1.0 else self.base
            self._cos, self._sin, self._cos_doubled, self._sin_doubled = precompute_freqs_cis(
                self.dims,
                self.max_seq_len,
                base=effective_base,
                dtype=dtype,
            )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
            k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim).
            offset: Position offset for incremental decoding.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        self._ensure_freqs(q.dtype)
        return apply_rope(
            q, k, self._cos, self._sin, offset,
            cos_doubled=self._cos_doubled,
            sin_doubled=self._sin_doubled,
        )

    def forward_one(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """Apply rotary embeddings to a single tensor.

        Useful when you need to rotate only queries or only keys.

        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
            offset: Position offset for incremental decoding.

        Returns:
            Rotated tensor with same shape as input.
        """
        self._ensure_freqs(x.dtype)

        seq_len = x.shape[1]

        # Use precomputed doubled cos/sin for efficiency
        cos_d = self._cos_doubled[offset : offset + seq_len]
        sin_d = self._sin_doubled[offset : offset + seq_len]

        # Reshape for broadcasting: (seq_len, dim) -> (1, seq_len, 1, dim)
        cos_d = cos_d[None, :, None, :]
        sin_d = sin_d[None, :, None, :]

        return (x * cos_d) + (_rotate_half(x) * sin_d)


class NTKAwareRoPE(RoPE):
    """RoPE with NTK-aware interpolation for extended context.

    This variant adjusts the base frequency to better handle sequences
    longer than the original training length, following the "NTK-aware"
    scaling approach.

    Args:
        dims: Dimension of the embeddings (head_dim, must be even).
        max_seq_len: Maximum sequence length to support.
        base: Base for frequency computation (default: 10000.0).
        original_max_seq_len: Original training sequence length.
        alpha: Scaling factor (computed automatically if not provided).

    Reference:
        https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        original_max_seq_len: int = 8192,
        alpha: float | None = None,
    ):
        # Compute alpha based on context extension ratio
        if alpha is None:
            alpha = (max_seq_len / original_max_seq_len) ** (dims / (dims - 2))

        # NTK scaling modifies the base
        scaled_base = base * alpha

        super().__init__(
            dims=dims,
            max_seq_len=max_seq_len,
            base=scaled_base,
            scale=1.0,  # Already applied via base
        )

        self.original_max_seq_len = original_max_seq_len
        self.alpha = alpha


class YaRNRoPE(nn.Module):
    """YaRN (Yet another RoPE extensioN) for extended context.

    Implements the YaRN method which combines NTK-aware scaling with
    attention scaling for better extrapolation to longer sequences.

    Args:
        dims: Dimension of the embeddings (head_dim, must be even).
        max_seq_len: Maximum sequence length to support.
        base: Base for frequency computation (default: 10000.0).
        original_max_seq_len: Original training sequence length.
        beta_fast: Fast beta for interpolation (default: 32).
        beta_slow: Slow beta for interpolation (default: 1).
        scale: Scale factor for attention (default: computed).

    Reference:
        "YaRN: Efficient Context Window Extension of Large Language Models"
        https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dims: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        original_max_seq_len: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        scale: float | None = None,
    ):
        super().__init__()

        if dims % 2 != 0:
            raise ValueError(f"dims must be even, got {dims}")

        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Compute scale factor
        self.extension_ratio = max_seq_len / original_max_seq_len
        if scale is None:
            self.scale = 0.1 * math.log(self.extension_ratio) + 1.0
        else:
            self.scale = scale

        # Precomputed frequencies with YaRN interpolation (lazily initialized)
        self._cos: mx.array | None = None
        self._sin: mx.array | None = None
        self._cos_doubled: mx.array | None = None
        self._sin_doubled: mx.array | None = None

    def _yarn_freqs(self, dtype: mx.Dtype = mx.float32) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute YaRN-interpolated frequencies.

        Returns:
            Tuple of (cos, sin, cos_doubled, sin_doubled).
        """
        # Compute wavelengths for each frequency
        two_pi = 2 * math.pi  # Named constant for clarity
        dim_indices = mx.arange(0, self.dims, 2, dtype=mx.float32)
        freqs = 1.0 / (self.base ** (dim_indices / self.dims))

        # Compute interpolation ratios
        low = max(
            math.floor(
                self.dims
                * math.log(self.original_max_seq_len / (self.beta_fast * two_pi))
                / (2 * math.log(self.base))
            ),
            0,
        )
        high = min(
            math.ceil(
                self.dims
                * math.log(self.original_max_seq_len / (self.beta_slow * two_pi))
                / (2 * math.log(self.base))
            ),
            self.dims // 2 - 1,
        )

        # Create interpolation mask
        ramp = mx.clip(
            (dim_indices / 2 - low) / max(high - low, 1), 0, 1
        )

        # Interpolate between original and scaled frequencies
        freqs_scaled = freqs / self.extension_ratio
        freqs_interpolated = (1 - ramp) * freqs_scaled + ramp * freqs

        # Compute position indices and frequencies
        t = mx.arange(self.max_seq_len, dtype=mx.float32)
        freqs_full = mx.outer(t, freqs_interpolated)

        cos = mx.cos(freqs_full).astype(dtype)
        sin = mx.sin(freqs_full).astype(dtype)

        # Precompute doubled versions for efficiency
        cos_doubled = mx.concatenate([cos, cos], axis=-1)
        sin_doubled = mx.concatenate([sin, sin], axis=-1)

        return cos, sin, cos_doubled, sin_doubled

    def _ensure_freqs(self, dtype: mx.Dtype = mx.float32) -> None:
        """Lazily compute frequencies on first use."""
        if self._cos is None or self._sin is None:
            self._cos, self._sin, self._cos_doubled, self._sin_doubled = self._yarn_freqs(dtype)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Apply YaRN rotary embeddings.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
            k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim).
            offset: Position offset for incremental decoding.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        self._ensure_freqs(q.dtype)
        return apply_rope(
            q, k, self._cos, self._sin, offset,
            cos_doubled=self._cos_doubled,
            sin_doubled=self._sin_doubled,
        )
