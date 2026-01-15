"""Quantized KV Cache for memory-efficient long-context inference.

This module provides INT8 quantized KV cache implementations that reduce
memory usage by 4x while maintaining accuracy through per-head, per-token
scale factors.

Key Benefits:
    - 4x memory reduction for KV cache (float32 -> int8)
    - 2x memory reduction for float16 models (float16 -> int8)
    - Minimal accuracy loss with per-head quantization
    - Compatible with MLX's fast SDPA after dequantization

Usage:
    >>> cache = QuantizedKVCache(num_heads=32, head_dim=128)
    >>> for step in range(seq_len):
    ...     k_step, v_step = get_kv(step)  # fp16/fp32
    ...     cache.update(k_step, v_step)
    ...     k_full, v_full = cache.get_dequantized()
    ...     # Use k_full, v_full with attention
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class QuantizedKVCache:
    """INT8 quantized KV cache for memory-efficient inference.

    Stores keys and values in INT8 format with per-head, per-token scale
    factors. Dequantizes on-the-fly for attention computation.

    Memory comparison for (batch=1, seq=8192, heads=32, dim=128):
        - float32: 8192 * 32 * 128 * 4 * 2 = 256 MB per layer
        - float16: 8192 * 32 * 128 * 2 * 2 = 128 MB per layer
        - int8: 8192 * 32 * 128 * 1 * 2 + scales = ~64 MB per layer

    Args:
        num_heads: Number of KV heads (for GQA, this is num_kv_heads).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length to pre-allocate.
        symmetric: Use symmetric quantization (default: True).

    Example:
        >>> cache = QuantizedKVCache(num_heads=8, head_dim=128)
        >>> k = mx.random.normal((1, 1, 8, 128))  # New K token
        >>> v = mx.random.normal((1, 1, 8, 128))  # New V token
        >>> cache.update(k, v)
        >>> k_full, v_full = cache.get_dequantized()
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        symmetric: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.symmetric = symmetric

        # Current sequence length
        self._seq_len = 0

        # Quantized storage (will be allocated on first update)
        self._k_quantized: Optional[mx.array] = None
        self._v_quantized: Optional[mx.array] = None

        # Scale factors: (batch, seq, heads) for per-token, per-head quantization
        self._k_scales: Optional[mx.array] = None
        self._v_scales: Optional[mx.array] = None

        # Zero points for asymmetric quantization
        self._k_zeros: Optional[mx.array] = None
        self._v_zeros: Optional[mx.array] = None

    @property
    def seq_len(self) -> int:
        """Current sequence length in cache."""
        return self._seq_len

    def reset(self):
        """Reset the cache, clearing all stored values."""
        self._seq_len = 0
        self._k_quantized = None
        self._v_quantized = None
        self._k_scales = None
        self._v_scales = None
        self._k_zeros = None
        self._v_zeros = None

    def _quantize_tensor(
        self, x: mx.array
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Quantize a tensor to INT8 with per-head, per-token scales.

        Args:
            x: Input tensor (batch, seq, heads, dim).

        Returns:
            Tuple of (quantized, scales, zeros).
            - quantized: int8 tensor (batch, seq, heads, dim)
            - scales: float32 tensor (batch, seq, heads, 1)
            - zeros: float32 tensor or None (for symmetric)
        """
        # Compute per-head, per-token statistics
        # Shape: (batch, seq, heads, 1)
        x_min = mx.min(x, axis=-1, keepdims=True)
        x_max = mx.max(x, axis=-1, keepdims=True)

        if self.symmetric:
            # Symmetric: scale = max(|min|, |max|) / 127
            abs_max = mx.maximum(mx.abs(x_min), mx.abs(x_max))
            scale = abs_max / 127.0
            # Avoid division by zero
            scale = mx.maximum(scale, 1e-10)
            # Quantize
            x_quant = mx.round(x / scale)
            x_quant = mx.clip(x_quant, -127, 127)
            return x_quant.astype(mx.int8), scale.astype(mx.float32), None
        else:
            # Asymmetric: maps [min, max] to [0, 255]
            scale = (x_max - x_min) / 255.0
            scale = mx.maximum(scale, 1e-10)
            zero = x_min
            x_quant = mx.round((x - zero) / scale)
            x_quant = mx.clip(x_quant, 0, 255)
            return (
                x_quant.astype(mx.uint8),
                scale.astype(mx.float32),
                zero.astype(mx.float32),
            )

    def _dequantize_tensor(
        self,
        x_quant: mx.array,
        scales: mx.array,
        zeros: Optional[mx.array],
    ) -> mx.array:
        """Dequantize INT8 tensor back to float.

        Args:
            x_quant: Quantized tensor (batch, seq, heads, dim).
            scales: Scale factors (batch, seq, heads, 1).
            zeros: Zero points or None (batch, seq, heads, 1).

        Returns:
            Dequantized float tensor (batch, seq, heads, dim).
        """
        x_float = x_quant.astype(mx.float32)
        if self.symmetric:
            return x_float * scales
        else:
            return x_float * scales + zeros

    def update(
        self,
        k_new: mx.array,
        v_new: mx.array,
    ) -> int:
        """Add new K/V tokens to the cache.

        Args:
            k_new: New key tensor (batch, new_seq, heads, dim).
            v_new: New value tensor (batch, new_seq, heads, dim).

        Returns:
            New total sequence length.
        """
        batch_size, new_seq, num_heads, head_dim = k_new.shape

        assert num_heads == self.num_heads, \
            f"num_heads mismatch: expected {self.num_heads}, got {num_heads}"
        assert head_dim == self.head_dim, \
            f"head_dim mismatch: expected {self.head_dim}, got {head_dim}"

        # Quantize new tokens
        k_quant, k_scale, k_zero = self._quantize_tensor(k_new)
        v_quant, v_scale, v_zero = self._quantize_tensor(v_new)

        if self._k_quantized is None:
            # First update - initialize storage
            self._k_quantized = k_quant
            self._v_quantized = v_quant
            self._k_scales = k_scale
            self._v_scales = v_scale
            self._k_zeros = k_zero
            self._v_zeros = v_zero
        else:
            # Append to existing cache
            self._k_quantized = mx.concatenate([self._k_quantized, k_quant], axis=1)
            self._v_quantized = mx.concatenate([self._v_quantized, v_quant], axis=1)
            self._k_scales = mx.concatenate([self._k_scales, k_scale], axis=1)
            self._v_scales = mx.concatenate([self._v_scales, v_scale], axis=1)
            if not self.symmetric:
                self._k_zeros = mx.concatenate([self._k_zeros, k_zero], axis=1)
                self._v_zeros = mx.concatenate([self._v_zeros, v_zero], axis=1)

        self._seq_len += new_seq
        return self._seq_len

    def get_dequantized(self) -> Tuple[mx.array, mx.array]:
        """Get dequantized K/V tensors for attention computation.

        Returns:
            Tuple of (K, V) in float32.
        """
        if self._k_quantized is None:
            raise ValueError("Cache is empty")

        k = self._dequantize_tensor(self._k_quantized, self._k_scales, self._k_zeros)
        v = self._dequantize_tensor(self._v_quantized, self._v_scales, self._v_zeros)
        return k, v

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics.

        Returns:
            Dict with memory usage in bytes.
        """
        if self._k_quantized is None:
            return {"quantized_bytes": 0, "scales_bytes": 0, "total_bytes": 0}

        quant_bytes = (
            self._k_quantized.nbytes + self._v_quantized.nbytes
        )
        scale_bytes = (
            self._k_scales.nbytes + self._v_scales.nbytes
        )
        if not self.symmetric:
            scale_bytes += self._k_zeros.nbytes + self._v_zeros.nbytes

        # Compare to float32 equivalent
        float32_bytes = self._seq_len * self.num_heads * self.head_dim * 4 * 2

        return {
            "quantized_bytes": quant_bytes,
            "scales_bytes": scale_bytes,
            "total_bytes": quant_bytes + scale_bytes,
            "float32_equivalent_bytes": float32_bytes,
            "compression_ratio": float32_bytes / (quant_bytes + scale_bytes),
        }


class QuantizedKVCacheAttention(nn.Module):
    """Attention module with integrated quantized KV cache.

    Uses INT8 quantized KV cache for memory efficiency while leveraging
    MLX's fast SDPA for computation.

    Args:
        dims: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head (default: dims // num_heads).
        max_seq_len: Maximum sequence length for cache.
        causal: Use causal masking (default: True).
        bias: Use bias in projections (default: False).

    Example:
        >>> attn = QuantizedKVCacheAttention(dims=2048, num_heads=32, num_kv_heads=8)
        >>> x = mx.random.normal((1, 1, 2048))  # Single token
        >>> output = attn(x)  # Uses quantized cache
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        causal: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dims // num_heads
        self.max_seq_len = max_seq_len
        self.causal = causal

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        q_dim = num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dims, q_dim, bias=bias)
        self.k_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.v_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.out_proj = nn.Linear(q_dim, dims, bias=bias)

        # Quantized KV cache
        self._cache = QuantizedKVCache(
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

    def reset_cache(self):
        """Reset the KV cache."""
        self._cache.reset()

    def get_cache_stats(self) -> dict:
        """Get cache memory statistics."""
        return self._cache.get_memory_stats()

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass with quantized KV cache.

        Args:
            x: Input tensor (batch, seq, dims).
            mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq, dims).
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq, heads, dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Update quantized cache
        self._cache.update(k, v)

        # Get full dequantized K, V
        k_full, v_full = self._cache.get_dequantized()

        # Use MLX fast SDPA
        # Transpose to (batch, heads, seq, dim) for SDPA
        q_t = q.transpose(0, 2, 1, 3)
        k_t = k_full.transpose(0, 2, 1, 3)
        v_t = v_full.transpose(0, 2, 1, 3)

        # Apply attention with optional causal mask
        sdpa_mask = "causal" if self.causal and mask is None else mask
        output = mx.fast.scaled_dot_product_attention(
            q_t, k_t, v_t,
            scale=self.scale,
            mask=sdpa_mask,
        )

        # Transpose back and reshape
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, -1)

        return self.out_proj(output)


def quantize_kv_for_cache(
    k: mx.array,
    v: mx.array,
    symmetric: bool = True,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array], Optional[mx.array]]:
    """Quantize K/V tensors for cache storage.

    Standalone function for integrating with existing attention implementations.

    Args:
        k: Key tensor (batch, seq, heads, dim).
        v: Value tensor (batch, seq, heads, dim).
        symmetric: Use symmetric quantization.

    Returns:
        Tuple of (k_quant, v_quant, k_scale, v_scale, k_zero, v_zero).
    """
    cache = QuantizedKVCache(
        num_heads=k.shape[2],
        head_dim=k.shape[3],
        symmetric=symmetric,
    )

    k_quant, k_scale, k_zero = cache._quantize_tensor(k)
    v_quant, v_scale, v_zero = cache._quantize_tensor(v)

    return k_quant, v_quant, k_scale, v_scale, k_zero, v_zero


def dequantize_kv_from_cache(
    k_quant: mx.array,
    v_quant: mx.array,
    k_scale: mx.array,
    v_scale: mx.array,
    k_zero: Optional[mx.array] = None,
    v_zero: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Dequantize K/V tensors from cache.

    Args:
        k_quant: Quantized keys.
        v_quant: Quantized values.
        k_scale: Key scales.
        v_scale: Value scales.
        k_zero: Key zero points (for asymmetric).
        v_zero: Value zero points (for asymmetric).

    Returns:
        Tuple of (K, V) in float32.
    """
    symmetric = k_zero is None

    if symmetric:
        k = k_quant.astype(mx.float32) * k_scale
        v = v_quant.astype(mx.float32) * v_scale
    else:
        k = k_quant.astype(mx.float32) * k_scale + k_zero
        v = v_quant.astype(mx.float32) * v_scale + v_zero

    return k, v
