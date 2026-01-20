"""Native MLX baseline implementations for comparison.

These are straightforward implementations using standard MLX operations,
serving as baselines to measure speedups from optimized primitives.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def naive_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Naive O(n^2) attention implementation.

    This is the standard scaled dot-product attention without any
    memory optimizations like FlashAttention.

    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        scale: Attention scale (default: 1/sqrt(head_dim))
        mask: Optional attention mask

    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    head_dim = query.shape[-1]
    scale = scale or (1.0 / (head_dim ** 0.5))

    # Standard attention: QK^T / sqrt(d)
    scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax over last dimension
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of values
    return mx.matmul(weights, value)


def naive_cumsum(x: mx.array, axis: int = -1) -> mx.array:
    """Naive cumulative sum using MLX built-in.

    This serves as a baseline for associative_scan implementations.

    Args:
        x: Input tensor
        axis: Axis along which to compute cumsum

    Returns:
        Cumulative sum along axis
    """
    return mx.cumsum(x, axis=axis)


def naive_layer_norm(
    x: mx.array,
    weight: Optional[mx.array] = None,
    bias: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    """Naive layer normalization using separate operations.

    Args:
        x: Input tensor [..., features]
        weight: Optional scale parameter [features]
        bias: Optional bias parameter [features]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    # Compute mean and variance
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / mx.sqrt(var + eps)

    # Apply affine transform
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias

    return x_norm


def naive_rms_norm(
    x: mx.array,
    weight: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    """Naive RMS normalization using separate operations.

    Args:
        x: Input tensor [..., features]
        weight: Optional scale parameter [features]
        eps: Epsilon for numerical stability

    Returns:
        RMS normalized tensor
    """
    # Compute RMS
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)

    # Normalize
    x_norm = x / rms

    # Apply scale
    if weight is not None:
        x_norm = x_norm * weight

    return x_norm


def naive_silu(x: mx.array) -> mx.array:
    """Naive SiLU activation using separate operations.

    SiLU(x) = x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        SiLU activated tensor
    """
    return x * mx.sigmoid(x)


def naive_gelu(x: mx.array, approximate: bool = True) -> mx.array:
    """Naive GELU activation using separate operations.

    Args:
        x: Input tensor
        approximate: Use tanh approximation if True

    Returns:
        GELU activated tensor
    """
    if approximate:
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        return 0.5 * x * (1.0 + mx.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))
    else:
        # Exact GELU using erf
        return 0.5 * x * (1.0 + mx.erf(x / 1.4142135623730951))


def naive_swiglu(x: mx.array, gate: mx.array) -> mx.array:
    """Naive SwiGLU activation using separate operations.

    SwiGLU(x, gate) = SiLU(gate) * x

    Args:
        x: Input tensor
        gate: Gate tensor

    Returns:
        SwiGLU activated tensor
    """
    return naive_silu(gate) * x


def naive_rotary_embedding(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Naive rotary position embedding using separate operations.

    Args:
        x: Input tensor [batch, seq, heads, head_dim]
        cos: Cosine cache [seq, head_dim//2]
        sin: Sine cache [seq, head_dim//2]

    Returns:
        Rotated tensor
    """
    # Split into pairs
    x0 = x[..., 0::2]  # Even indices
    x1 = x[..., 1::2]  # Odd indices

    # Reshape cos/sin for broadcasting
    cos = cos[None, :, None, :]  # [1, seq, 1, head_dim//2]
    sin = sin[None, :, None, :]

    # Apply rotation
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # Interleave back
    batch, seq, heads, half_dim = out0.shape
    out = mx.zeros((batch, seq, heads, half_dim * 2), dtype=x.dtype)

    # This is inefficient but serves as baseline
    out_even = out0
    out_odd = out1

    # Stack and reshape to interleave
    stacked = mx.stack([out_even, out_odd], axis=-1)  # [..., half_dim, 2]
    return stacked.reshape(batch, seq, heads, half_dim * 2)
