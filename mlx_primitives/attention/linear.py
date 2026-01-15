"""Linear Attention implementations for MLX.

This module provides O(n) attention approximations:
- LinearAttention: Basic linear attention with feature maps
- PerformerAttention: FAVOR+ random feature attention
- CosFormerAttention: Cosine-based linear attention
"""

from __future__ import annotations

import math
from typing import Optional, Callable

import mlx.core as mx
import mlx.nn as nn


def elu_feature_map(x: mx.array) -> mx.array:
    """ELU-based feature map for linear attention.

    Maps: x -> elu(x) + 1

    This ensures non-negative features for valid attention.
    """
    return nn.elu(x) + 1


def relu_feature_map(x: mx.array) -> mx.array:
    """ReLU-based feature map."""
    return nn.relu(x)


def softmax_feature_map(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Softmax-based feature map (Taylor approximation)."""
    return mx.exp(x - mx.max(x, axis=-1, keepdims=True)) + eps


class LinearAttention(nn.Module):
    """Linear attention with O(n) complexity.

    Instead of computing softmax(QK^T)V directly, uses:
    attn = phi(Q) @ (phi(K)^T @ V)

    where phi is a feature map that approximates the softmax kernel.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        feature_map: Feature map function (default: elu+1).
        eps: Numerical stability epsilon.
        dropout: Dropout rate.

    Reference:
        "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
        https://arxiv.org/abs/2006.16236

    Example:
        >>> attn = LinearAttention(dims=768, num_heads=12)
        >>> x = mx.random.normal((2, 8192, 768))  # Long sequence
        >>> y = attn(x)  # O(n) instead of O(n^2)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        feature_map: Callable = elu_feature_map,
        eps: float = 1e-6,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.feature_map = feature_map
        self.eps = eps
        self.causal = causal

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        causal: Optional[bool] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).
            causal: Whether to use causal (autoregressive) attention.

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        batch_size, seq_len, _ = x.shape

        # Use instance default if not provided
        use_causal = causal if causal is not None else self.causal

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply feature maps
        q = self.feature_map(q)
        k = self.feature_map(k)

        if use_causal:
            out = self._causal_linear_attention(q, k, v)
        else:
            out = self._linear_attention(q, k, v)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        if self.dropout is not None:
            out = self.dropout(out)

        return self.out_proj(out)

    def _linear_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Non-causal linear attention.

        Computes: phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
        """
        # k^T @ v: (batch, heads, head_dim, head_dim)
        kv = k.transpose(0, 1, 3, 2) @ v

        # q @ kv: (batch, heads, seq_len, head_dim)
        qkv = q @ kv

        # Normalization: q @ k^T @ 1 = q @ sum(k, axis=seq)
        k_sum = mx.sum(k, axis=2, keepdims=True)  # (batch, heads, 1, head_dim)
        normalizer = mx.sum(q * k_sum, axis=-1, keepdims=True)  # (batch, heads, seq_len, 1)

        return qkv / (normalizer + self.eps)

    def _causal_linear_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Causal linear attention using cumulative sums.

        For causal attention, we need:
        out_t = sum_{i<=t} phi(q_t) @ phi(k_i)^T @ v_i

        This can be computed as:
        S_t = sum_{i<=t} phi(k_i)^T @ v_i (cumsum)
        out_t = phi(q_t) @ S_t
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute k^T @ v for each position: (batch, heads, seq_len, head_dim, head_dim)
        # k: (batch, heads, seq_len, head_dim)
        # v: (batch, heads, seq_len, head_dim)
        kv = k[:, :, :, :, None] * v[:, :, :, None, :]  # (batch, heads, seq_len, head_dim, head_dim)

        # Cumulative sum along sequence dimension
        kv_cumsum = mx.cumsum(kv, axis=2)  # (batch, heads, seq_len, head_dim, head_dim)

        # q @ cumsum(kv): for each position, contract head_dim
        # q: (batch, heads, seq_len, head_dim)
        # kv_cumsum: (batch, heads, seq_len, head_dim, head_dim)
        out = mx.sum(q[:, :, :, :, None] * kv_cumsum, axis=3)  # (batch, heads, seq_len, head_dim)

        # Normalization: cumsum of k, then dot with q
        k_cumsum = mx.cumsum(k, axis=2)  # (batch, heads, seq_len, head_dim)
        normalizer = mx.sum(q * k_cumsum, axis=-1, keepdims=True)  # (batch, heads, seq_len, 1)

        return out / (normalizer + self.eps)


class PerformerAttention(nn.Module):
    """Performer attention using FAVOR+ random features.

    Approximates softmax attention using random feature maps:
    softmax(QK^T/sqrt(d)) ≈ phi(Q) @ phi(K)^T

    where phi uses random Fourier features.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        num_features: Number of random features (default: head_dim).
        redraw_interval: Steps between redrawing features (0 = never).
        dropout: Dropout rate.

    Reference:
        "Rethinking Attention with Performers"
        https://arxiv.org/abs/2009.14794

    Example:
        >>> attn = PerformerAttention(dims=768, num_heads=12)
        >>> x = mx.random.normal((2, 8192, 768))
        >>> y = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_features: Optional[int] = None,
        redraw_interval: int = 0,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.num_features = num_features or self.head_dim
        self.redraw_interval = redraw_interval
        self.scale = self.head_dim ** -0.25  # Different scaling for Performer
        self.causal = causal

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Random projection matrix (orthogonal for better approximation)
        self._init_random_features()
        self._step = 0

    def _init_random_features(self):
        """Initialize orthogonal random features."""
        # Create orthogonal random matrix
        random_matrix = mx.random.normal((self.num_features, self.head_dim))

        # Orthogonalize using QR decomposition approximation
        # For simplicity, we just normalize rows
        norms = mx.sqrt(mx.sum(random_matrix ** 2, axis=-1, keepdims=True))
        self.random_features = random_matrix / norms * math.sqrt(self.head_dim)

    def _redraw_if_needed(self):
        """Redraw random features if interval reached."""
        if self.redraw_interval > 0:
            self._step += 1
            if self._step >= self.redraw_interval:
                self._init_random_features()
                self._step = 0

    def _favor_features(self, x: mx.array) -> mx.array:
        """Apply FAVOR+ random feature map.

        Maps x to: exp(x @ W - ||x||^2/2) / sqrt(m)

        where W is the random feature matrix.
        """
        # x: (batch, heads, seq_len, head_dim)
        # random_features: (num_features, head_dim)

        # Scale input
        x_scaled = x * self.scale

        # Project through random features
        # (batch, heads, seq_len, num_features)
        x_proj = x_scaled @ self.random_features.T

        # Compute ||x||^2 / 2 for normalization
        x_norm_sq = mx.sum(x_scaled ** 2, axis=-1, keepdims=True) / 2

        # FAVOR+ positive features: exp(proj - norm) for stability
        features = mx.exp(x_proj - x_norm_sq) / math.sqrt(self.num_features)

        return features

    def __call__(
        self,
        x: mx.array,
        causal: Optional[bool] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).
            causal: Whether to use causal attention (overrides instance default).

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        self._redraw_if_needed()

        batch_size, seq_len, _ = x.shape
        use_causal = causal if causal is not None else self.causal

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply FAVOR+ features
        q_prime = self._favor_features(q)  # (batch, heads, seq_len, num_features)
        k_prime = self._favor_features(k)

        if use_causal:
            out = self._causal_attention(q_prime, k_prime, v)
        else:
            out = self._noncausal_attention(q_prime, k_prime, v)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        if self.dropout is not None:
            out = self.dropout(out)

        return self.out_proj(out)

    def _noncausal_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Non-causal FAVOR+ attention."""
        # k^T @ v
        kv = k.transpose(0, 1, 3, 2) @ v  # (batch, heads, num_features, head_dim)

        # q @ kv
        qkv = q @ kv  # (batch, heads, seq_len, head_dim)

        # Normalize
        k_sum = mx.sum(k, axis=2)  # (batch, heads, num_features)
        normalizer = q @ k_sum[:, :, :, None]  # (batch, heads, seq_len, 1)

        return qkv / (normalizer + 1e-6)

    def _causal_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Causal FAVOR+ attention using prefix sums."""
        # k: (batch, heads, seq_len, num_features)
        # v: (batch, heads, seq_len, head_dim)

        # Outer product then cumsum
        kv = k[:, :, :, :, None] * v[:, :, :, None, :]  # (batch, heads, seq, features, head_dim)
        kv_cumsum = mx.cumsum(kv, axis=2)

        # q @ cumsum
        out = mx.sum(q[:, :, :, :, None] * kv_cumsum, axis=3)  # (batch, heads, seq, head_dim)

        # Normalize with cumsum of k
        k_cumsum = mx.cumsum(k, axis=2)
        normalizer = mx.sum(q * k_cumsum, axis=-1, keepdims=True)

        return out / (normalizer + 1e-6)


class CosFormerAttention(nn.Module):
    """CosFormer: Cosine-based linear attention.

    Uses cos-based reweighting to achieve linear complexity while
    maintaining good performance on language modeling.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.

    Reference:
        "cosFormer: Rethinking Softmax in Attention"
        https://arxiv.org/abs/2202.08791

    Example:
        >>> attn = CosFormerAttention(dims=768, num_heads=12)
        >>> x = mx.random.normal((2, 4096, 768))
        >>> y = attn(x, causal=True)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        causal: Optional[bool] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).
            causal: Whether to use causal attention (overrides instance default).

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        batch_size, seq_len, _ = x.shape
        use_causal = causal if causal is not None else self.causal

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply ReLU feature map
        q = nn.relu(q)
        k = nn.relu(k)

        # Create position indices for cos reweighting
        positions = mx.arange(seq_len)
        pi_over_2M = math.pi / (2 * seq_len)

        # Cos weights: cos(pi * i / 2M) for position i
        cos_weights = mx.cos(positions * pi_over_2M)  # (seq_len,)

        if use_causal:
            out = self._causal_cosformer(q, k, v, cos_weights)
        else:
            out = self._noncausal_cosformer(q, k, v, cos_weights)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        if self.dropout is not None:
            out = self.dropout(out)

        return self.out_proj(out)

    def _noncausal_cosformer(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        cos_weights: mx.array,
    ) -> mx.array:
        """Non-causal cosFormer attention."""
        # Apply cos weights to q and k
        cos_w = cos_weights[None, None, :, None]  # (1, 1, seq_len, 1)
        q = q * cos_w
        k = k * cos_w

        # Standard linear attention
        kv = k.transpose(0, 1, 3, 2) @ v
        qkv = q @ kv

        k_sum = mx.sum(k, axis=2, keepdims=True)
        normalizer = mx.sum(q * k_sum, axis=-1, keepdims=True)

        return qkv / (normalizer + 1e-6)

    def _causal_cosformer(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        cos_weights: mx.array,
    ) -> mx.array:
        """Causal cosFormer with cos-sin decomposition."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        positions = mx.arange(seq_len)
        pi_over_2M = math.pi / (2 * seq_len)

        cos_pos = mx.cos(positions * pi_over_2M)[None, None, :, None]
        sin_pos = mx.sin(positions * pi_over_2M)[None, None, :, None]

        # Decompose: cos(i-j) = cos(i)cos(j) + sin(i)sin(j)
        q_cos = q * cos_pos
        q_sin = q * sin_pos
        k_cos = k * cos_pos
        k_sin = k * sin_pos

        # Causal linear attention for cos-cos and sin-sin terms
        # cos-cos term
        kv_cos = k_cos[:, :, :, :, None] * v[:, :, :, None, :]
        kv_cos_cumsum = mx.cumsum(kv_cos, axis=2)
        out_cos = mx.sum(q_cos[:, :, :, :, None] * kv_cos_cumsum, axis=3)

        # sin-sin term
        kv_sin = k_sin[:, :, :, :, None] * v[:, :, :, None, :]
        kv_sin_cumsum = mx.cumsum(kv_sin, axis=2)
        out_sin = mx.sum(q_sin[:, :, :, :, None] * kv_sin_cumsum, axis=3)

        out = out_cos + out_sin

        # Normalization
        k_cos_cumsum = mx.cumsum(k_cos, axis=2)
        k_sin_cumsum = mx.cumsum(k_sin, axis=2)
        norm = mx.sum(q_cos * k_cos_cumsum + q_sin * k_sin_cumsum, axis=-1, keepdims=True)

        return out / (norm + 1e-6)
