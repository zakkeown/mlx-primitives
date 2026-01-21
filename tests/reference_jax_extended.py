"""Extended JAX reference implementations for parity testing.

This module provides 50+ reference implementations covering all MLXPrimitives operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    jnn = None
    lax = None


def _check_jax():
    if not HAS_JAX:
        raise ImportError("JAX not available")


# =============================================================================
# Attention Operations (11 variants)
# =============================================================================

def jax_sliding_window_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    window_size: int, causal: bool = True
) -> np.ndarray:
    """Sliding window attention reference.

    Args:
        q: Query of shape (batch, seq, heads, dim).
        k: Key of shape (batch, seq, heads, dim).
        v: Value of shape (batch, seq, heads, dim).
        window_size: One-sided window size.
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_len, heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Create sliding window mask
    positions = jnp.arange(seq_len)
    row_pos = positions[:, None]
    col_pos = positions[None, :]
    mask = jnp.abs(row_pos - col_pos) <= window_size

    if causal:
        causal_mask = col_pos <= row_pos
        mask = mask & causal_mask

    # Apply mask
    scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax
    weights = jnn.softmax(scores, axis=-1)

    # Weighted sum
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_gqa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    num_kv_heads: int, causal: bool = False
) -> np.ndarray:
    """Grouped Query Attention reference.

    Args:
        q: Query of shape (batch, seq, num_heads, dim).
        k: Key of shape (batch, seq, num_kv_heads, dim).
        v: Value of shape (batch, seq, num_kv_heads, dim).
        num_kv_heads: Number of KV heads (must divide num_heads).
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq, num_heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_len, num_heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Expand K/V heads to match Q
    num_groups = num_heads // num_kv_heads
    k_j = jnp.repeat(k_j, num_groups, axis=2)
    v_j = jnp.repeat(v_j, num_groups, axis=2)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Apply causal mask if needed
    if causal:
        positions = jnp.arange(seq_len)
        mask = positions[None, :] <= positions[:, None]
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_mqa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    causal: bool = False
) -> np.ndarray:
    """Multi-Query Attention reference.

    Args:
        q: Query of shape (batch, seq, num_heads, dim).
        k: Key of shape (batch, seq, 1, dim) - single KV head.
        v: Value of shape (batch, seq, 1, dim) - single KV head.
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq, num_heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_len, num_heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Expand K/V to all heads
    k_j = jnp.broadcast_to(k_j, (batch, seq_len, num_heads, dim))
    v_j = jnp.broadcast_to(v_j, (batch, seq_len, num_heads, dim))

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Apply causal mask if needed
    if causal:
        positions = jnp.arange(seq_len)
        mask = positions[None, :] <= positions[:, None]
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_linear_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    feature_map: str = "elu"
) -> np.ndarray:
    """Linear attention reference.

    Linear attention computes attention in O(n) by avoiding explicit score matrix:
    y = phi(Q) @ (phi(K).T @ V) / (phi(Q) @ sum(phi(K)))

    Args:
        q: Query of shape (batch, seq, heads, dim).
        k: Key of shape (batch, seq, heads, dim).
        v: Value of shape (batch, seq, heads, dim).
        feature_map: Feature map to use ("elu" for ELU+1, "softmax").

    Returns:
        Output of shape (batch, seq, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    # Apply feature map
    if feature_map == "elu":
        q_j = jnn.elu(q_j) + 1
        k_j = jnn.elu(k_j) + 1
    elif feature_map == "softmax":
        q_j = jnn.softmax(q_j, axis=-1)
        k_j = jnn.softmax(k_j, axis=-1)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute KV: (batch, heads, dim_k, dim_v)
    kv = jnp.einsum("bhsd,bhsv->bhdv", k_j, v_j)

    # Compute output: Q @ KV
    out = jnp.einsum("bhqd,bhdv->bhqv", q_j, kv)

    # Normalize by sum of keys
    k_sum = k_j.sum(axis=2, keepdims=True)  # (batch, heads, 1, dim)
    normalizer = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_sum).squeeze(-1)  # (batch, heads, seq)
    normalizer = jnp.clip(normalizer[..., None], a_min=1e-6)

    out = out / normalizer

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_alibi_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    alibi_slopes: np.ndarray, causal: bool = True
) -> np.ndarray:
    """ALiBi attention reference.

    Args:
        q: Query of shape (batch, seq, heads, dim).
        k: Key of shape (batch, seq, heads, dim).
        v: Value of shape (batch, seq, heads, dim).
        alibi_slopes: Slopes per head of shape (heads,).
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)
    slopes = jnp.array(alibi_slopes, dtype=jnp.float32)

    batch, seq_len, heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Compute ALiBi bias
    positions = jnp.arange(seq_len)
    rel_pos = positions[:, None] - positions[None, :]  # (seq, seq)
    alibi_bias = slopes[:, None, None] * rel_pos[None, :, :]  # (heads, seq, seq)

    # Add ALiBi bias
    scores = scores + alibi_bias[None, :, :, :]

    # Apply causal mask if needed
    if causal:
        mask = positions[None, :] <= positions[:, None]
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_sparse_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    mask_pattern: str = "local", window_size: int = 128, stride: int = 64
) -> np.ndarray:
    """Sparse attention reference.

    Args:
        q: Query of shape (batch, seq, heads, dim).
        k: Key of shape (batch, seq, heads, dim).
        v: Value of shape (batch, seq, heads, dim).
        mask_pattern: Pattern type - "local", "strided", or "fixed".
        window_size: Window size for local attention.
        stride: Stride for strided attention.

    Returns:
        Output of shape (batch, seq, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_len, heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Create sparse mask based on pattern
    positions = jnp.arange(seq_len)
    row_pos = positions[:, None]
    col_pos = positions[None, :]

    if mask_pattern == "local":
        mask = jnp.abs(row_pos - col_pos) <= window_size
    elif mask_pattern == "strided":
        local_mask = jnp.abs(row_pos - col_pos) <= window_size
        strided_mask = (col_pos % stride) == 0
        mask = local_mask | strided_mask
    else:  # "fixed"
        local_mask = jnp.abs(row_pos - col_pos) <= window_size
        fixed_mask = col_pos < window_size
        mask = local_mask | fixed_mask

    # Apply mask
    scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_chunked_cross_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    chunk_size: int
) -> np.ndarray:
    """Chunked cross-attention reference.

    Processes attention in chunks to reduce memory usage.

    Args:
        q: Query of shape (batch, seq_q, heads, dim).
        k: Key of shape (batch, seq_kv, heads, dim).
        v: Value of shape (batch, seq_kv, heads, dim).
        chunk_size: Number of query positions to process at once.

    Returns:
        Output of shape (batch, seq_q, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_q, heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    outputs = []
    for start in range(0, seq_q, chunk_size):
        end = min(start + chunk_size, seq_q)
        q_chunk = q_j[:, :, start:end, :]

        # Compute attention for this chunk
        scores = jnp.einsum("bhqd,bhkd->bhqk", q_chunk, k_j) * scale
        weights = jnn.softmax(scores, axis=-1)
        out_chunk = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)
        outputs.append(out_chunk)

    out = jnp.concatenate(outputs, axis=2)
    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_rope_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, causal: bool = False
) -> np.ndarray:
    """RoPE + attention reference.

    Applies rotary position embeddings to Q and K, then computes attention.

    Args:
        q: Query of shape (batch, seq, heads, dim).
        k: Key of shape (batch, seq, heads, dim).
        v: Value of shape (batch, seq, heads, dim).
        cos: Cosine values for RoPE.
        sin: Sine values for RoPE.
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq, heads, dim).
    """
    _check_jax()
    # Apply RoPE to Q and K using existing function
    q_rotated = jax_rotary_embedding(q, cos, sin)
    k_rotated = jax_rotary_embedding(k, cos, sin)

    q_j = jnp.array(q_rotated, dtype=jnp.float32)
    k_j = jnp.array(k_rotated, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_len, heads, dim = q_j.shape
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Apply causal mask if needed
    if causal:
        positions = jnp.arange(seq_len)
        mask = positions[None, :] <= positions[:, None]
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


def jax_quantized_kv_attention(
    q: np.ndarray, k_quant: np.ndarray, v_quant: np.ndarray,
    k_scale: np.ndarray, v_scale: np.ndarray,
    k_zero_point: Optional[np.ndarray] = None,
    v_zero_point: Optional[np.ndarray] = None,
    causal: bool = False
) -> np.ndarray:
    """Quantized KV cache attention reference.

    Dequantizes K and V on-the-fly and computes attention.

    Args:
        q: Query of shape (batch, seq_q, heads, dim).
        k_quant: Quantized keys of shape (batch, seq_kv, heads, dim).
        v_quant: Quantized values of shape (batch, seq_kv, heads, dim).
        k_scale: Scale for K dequantization.
        v_scale: Scale for V dequantization.
        k_zero_point: Zero point for K (optional, for asymmetric).
        v_zero_point: Zero point for V (optional, for asymmetric).
        causal: If True, apply causal masking.

    Returns:
        Output of shape (batch, seq_q, heads, dim).
    """
    _check_jax()
    q_j = jnp.array(q, dtype=jnp.float32)
    k_q = jnp.array(k_quant, dtype=jnp.float32)
    v_q = jnp.array(v_quant, dtype=jnp.float32)
    k_s = jnp.array(k_scale, dtype=jnp.float32)
    v_s = jnp.array(v_scale, dtype=jnp.float32)

    # Dequantize K and V
    if k_zero_point is not None:
        k_zp = jnp.array(k_zero_point, dtype=jnp.float32)
        k_j = (k_q - k_zp) * k_s
    else:
        k_j = k_q * k_s

    if v_zero_point is not None:
        v_zp = jnp.array(v_zero_point, dtype=jnp.float32)
        v_j = (v_q - v_zp) * v_s
    else:
        v_j = v_q * v_s

    batch, seq_q, heads, dim = q_j.shape
    seq_kv = k_j.shape[1]
    scale = 1.0 / jnp.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q_j = jnp.transpose(q_j, (0, 2, 1, 3))
    k_j = jnp.transpose(k_j, (0, 2, 1, 3))
    v_j = jnp.transpose(v_j, (0, 2, 1, 3))

    # Compute attention scores
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_j, k_j) * scale

    # Apply causal mask if needed
    if causal:
        q_pos = jnp.arange(seq_q)
        k_pos = jnp.arange(seq_kv)
        mask = k_pos[None, :] <= q_pos[:, None]
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)

    # Softmax and weighted sum
    weights = jnn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_j)

    return np.array(jnp.transpose(out, (0, 2, 1, 3)))


# =============================================================================
# Activation Functions (12+)
# =============================================================================

def jax_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """SwiGLU activation reference.

    Computes: SiLU(x @ W_gate.T) * (x @ W_up.T)
    Weights are in (out_dim, in_dim) format, matching PyTorch convention.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    W_gate_j = jnp.array(W_gate, dtype=jnp.float32)
    W_up_j = jnp.array(W_up, dtype=jnp.float32)

    gate = jnn.silu(x_j @ W_gate_j.T)
    up = x_j @ W_up_j.T
    return np.array(gate * up)


def jax_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """GeGLU activation reference.

    Computes: GELU(x @ W_gate.T) * (x @ W_up.T)
    Weights are in (out_dim, in_dim) format, matching PyTorch convention.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    W_gate_j = jnp.array(W_gate, dtype=jnp.float32)
    W_up_j = jnp.array(W_up, dtype=jnp.float32)

    gate = jnn.gelu(x_j @ W_gate_j.T)
    up = x_j @ W_up_j.T
    return np.array(gate * up)


def jax_reglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """ReGLU activation reference.

    Computes: ReLU(x @ W_gate.T) * (x @ W_up.T)
    Weights are in (out_dim, in_dim) format, matching PyTorch convention.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    W_gate_j = jnp.array(W_gate, dtype=jnp.float32)
    W_up_j = jnp.array(W_up, dtype=jnp.float32)

    gate = jnn.relu(x_j @ W_gate_j.T)
    up = x_j @ W_up_j.T
    return np.array(gate * up)


def jax_quick_gelu(x: np.ndarray) -> np.ndarray:
    """QuickGELU activation reference.

    Computes: x * sigmoid(1.702 * x)
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(x_j * jnn.sigmoid(1.702 * x_j))


def jax_gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU with tanh approximation reference.

    Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(jnn.gelu(x_j, approximate=True))


def jax_mish(x: np.ndarray) -> np.ndarray:
    """Mish activation reference.

    Computes: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    softplus = jnp.log1p(jnp.exp(x_j))
    return np.array(x_j * jnp.tanh(softplus))


def jax_squared_relu(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation reference.

    Computes: ReLU(x)^2
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(jnn.relu(x_j) ** 2)


def jax_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish activation reference.

    Computes: x * sigmoid(beta * x)
    When beta=1, this equals SiLU.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(x_j * jnn.sigmoid(beta * x_j))


def jax_hard_swish(x: np.ndarray) -> np.ndarray:
    """Hard Swish activation reference.

    Computes: x * clip(x + 3, 0, 6) / 6
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(x_j * jnp.clip(x_j + 3, 0, 6) / 6)


def jax_hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """Hard Sigmoid activation reference.

    Computes: clip(x + 3, 0, 6) / 6
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    return np.array(jnp.clip(x_j + 3, 0, 6) / 6)


# =============================================================================
# Normalization Operations (5)
# =============================================================================

def jax_groupnorm(
    x: np.ndarray, num_groups: int,
    weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """GroupNorm reference.

    Args:
        x: Input tensor of shape (N, C, *) - channels first format.
        num_groups: Number of groups to divide channels into.
        weight: Scale parameter of shape (C,).
        bias: Shift parameter of shape (C,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized output as numpy array.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    w_j = jnp.array(weight, dtype=jnp.float32)
    b_j = jnp.array(bias, dtype=jnp.float32)

    original_shape = x_j.shape
    batch_size = x_j.shape[0]
    num_channels = x_j.shape[1]
    group_size = num_channels // num_groups

    # Reshape to (N, G, C//G, *)
    x_j = x_j.reshape(batch_size, num_groups, group_size, -1)

    # Compute mean/var per group
    mean = jnp.mean(x_j, axis=(2, 3), keepdims=True)
    var = jnp.var(x_j, axis=(2, 3), keepdims=True)
    x_norm = (x_j - mean) / jnp.sqrt(var + eps)

    # Reshape back
    x_norm = x_norm.reshape(original_shape)

    # Apply affine transform - reshape weight/bias for broadcasting
    ndim = len(original_shape)
    shape = [1, -1] + [1] * (ndim - 2)
    w_j = w_j.reshape(shape)
    b_j = b_j.reshape(shape)

    result = x_norm * w_j + b_j
    return np.array(result)


def jax_instancenorm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """InstanceNorm reference.

    Args:
        x: Input tensor of shape (N, C, *) - channels first format.
        weight: Scale parameter of shape (C,).
        bias: Shift parameter of shape (C,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized output as numpy array.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    w_j = jnp.array(weight, dtype=jnp.float32)
    b_j = jnp.array(bias, dtype=jnp.float32)

    original_shape = x_j.shape
    batch_size = x_j.shape[0]
    num_channels = x_j.shape[1]

    # Reshape to (N, C, -1) to flatten spatial dims
    x_j = x_j.reshape(batch_size, num_channels, -1)

    # Compute mean/var per instance (per batch, per channel)
    mean = jnp.mean(x_j, axis=2, keepdims=True)
    var = jnp.var(x_j, axis=2, keepdims=True)
    x_norm = (x_j - mean) / jnp.sqrt(var + eps)

    # Reshape back
    x_norm = x_norm.reshape(original_shape)

    # Apply affine transform - reshape weight/bias for broadcasting
    ndim = len(original_shape)
    shape = [1, -1] + [1] * (ndim - 2)
    w_j = w_j.reshape(shape)
    b_j = b_j.reshape(shape)

    result = x_norm * w_j + b_j
    return np.array(result)


def jax_adalayernorm(
    x: np.ndarray, scale: np.ndarray, shift: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Adaptive LayerNorm reference.

    AdaLayerNorm applies standard layer normalization then uses external
    conditioning to modulate the output: y = norm(x) * (1 + scale) + shift

    Args:
        x: Input tensor of shape (batch, seq, dims) or (batch, dims).
        scale: Scale conditioning of shape matching x's last dim broadcast shape.
        shift: Shift conditioning of shape matching x's last dim broadcast shape.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and modulated output as numpy array.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    scale_j = jnp.array(scale, dtype=jnp.float32)
    shift_j = jnp.array(shift, dtype=jnp.float32)

    # Standard layer normalization along last axis
    mean = jnp.mean(x_j, axis=-1, keepdims=True)
    var = jnp.var(x_j, axis=-1, keepdims=True)
    x_norm = (x_j - mean) / jnp.sqrt(var + eps)

    # Apply adaptive modulation: (1 + scale) * x_norm + shift
    result = x_norm * (1 + scale_j) + shift_j
    return np.array(result)


# =============================================================================
# Fused Operations (4)
# =============================================================================

def jax_fused_rmsnorm_linear(
    x: np.ndarray, norm_weight: np.ndarray,
    linear_weight: np.ndarray, linear_bias: Optional[np.ndarray], eps: float = 1e-5
) -> np.ndarray:
    """Fused RMSNorm + Linear reference.

    Computes: Linear(RMSNorm(x))

    Args:
        x: Input tensor.
        norm_weight: RMSNorm weight.
        linear_weight: Linear layer weight.
        linear_bias: Optional linear layer bias.
        eps: Epsilon for numerical stability.

    Returns:
        Output after RMSNorm + Linear.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    norm_w = jnp.array(norm_weight, dtype=jnp.float32)
    lin_w = jnp.array(linear_weight, dtype=jnp.float32)

    # RMSNorm
    rms = jnp.sqrt(jnp.mean(x_j * x_j, axis=-1, keepdims=True) + eps)
    x_norm = x_j / rms * norm_w

    # Linear
    out = jnp.einsum("...d,od->...o", x_norm, lin_w)
    if linear_bias is not None:
        lin_b = jnp.array(linear_bias, dtype=jnp.float32)
        out = out + lin_b

    return np.array(out)


def jax_fused_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused SwiGLU reference (same as jax_swiglu)."""
    return jax_swiglu(x, W_gate, W_up)


def jax_fused_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused GeGLU reference (same as jax_geglu)."""
    return jax_geglu(x, W_gate, W_up)


def jax_fused_rope_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, causal: bool = False
) -> np.ndarray:
    """Fused RoPE + Attention reference (same as jax_rope_attention)."""
    return jax_rope_attention(q, k, v, cos, sin, causal)


# =============================================================================
# Quantization Operations (6)
# =============================================================================

def jax_quantize_int8(
    weights: np.ndarray,
    per_channel: bool = False,
    symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """INT8 quantization reference.

    Args:
        weights: Float weights to quantize.
        per_channel: If True, compute scale per output channel (axis 0).
        symmetric: If True, use symmetric quantization (no zero point).

    Returns:
        Tuple of (quantized_weights, scale, zero_point).
        zero_point is None for symmetric quantization.
    """
    _check_jax()
    w = jnp.array(weights, dtype=jnp.float32)

    qmin, qmax = -128, 127
    eps = 1e-8

    if per_channel:
        # Quantize per output channel (axis 0)
        axis = tuple(range(1, w.ndim))
        w_min = jnp.min(w, axis=axis, keepdims=True)
        w_max = jnp.max(w, axis=axis, keepdims=True)
    else:
        w_min = jnp.min(w)
        w_max = jnp.max(w)

    if symmetric:
        # Symmetric: scale = max(|min|, |max|) / qmax
        w_absmax = jnp.maximum(jnp.abs(w_min), jnp.abs(w_max))
        scale = w_absmax / qmax
        scale = jnp.maximum(scale, eps)

        w_q = jnp.round(w / scale)
        w_q = jnp.clip(w_q, qmin, qmax).astype(jnp.int8)

        return np.array(w_q), np.array(scale), None
    else:
        # Asymmetric: compute zero point
        scale = (w_max - w_min) / (qmax - qmin)
        scale = jnp.maximum(scale, eps)

        zero_point = qmin - jnp.round(w_min / scale)
        zero_point = jnp.clip(zero_point, qmin, qmax)

        w_q = jnp.round(w / scale + zero_point)
        w_q = jnp.clip(w_q, qmin, qmax).astype(jnp.int8)

        return np.array(w_q), np.array(scale), np.array(zero_point)


def jax_dequantize_int8(
    W_quant: np.ndarray,
    scale: np.ndarray,
    zero_point: Optional[np.ndarray] = None,
) -> np.ndarray:
    """INT8 dequantization reference.

    Args:
        W_quant: Quantized INT8 weights.
        scale: Quantization scale.
        zero_point: Zero point (None for symmetric quantization).

    Returns:
        Dequantized float weights.
    """
    _check_jax()
    w_q = jnp.array(W_quant, dtype=jnp.float32)
    s = jnp.array(scale, dtype=jnp.float32)

    if zero_point is not None:
        zp = jnp.array(zero_point, dtype=jnp.float32)
        w = (w_q - zp) * s
    else:
        w = w_q * s

    return np.array(w)


def jax_quantize_int4(
    weights: np.ndarray,
    group_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT4 quantization reference with group-wise quantization.

    Args:
        weights: Float weights (out_features, in_features).
        group_size: Group size for quantization.

    Returns:
        Tuple of (quantized_weights, scales, num_groups).
        scales has shape (out_features, num_groups).
    """
    _check_jax()
    w = jnp.array(weights, dtype=jnp.float32)
    out_features, in_features = w.shape

    # Number of groups
    num_groups = (in_features + group_size - 1) // group_size

    # Pad to be divisible by group_size
    padded_in = num_groups * group_size
    if in_features < padded_in:
        padding = jnp.zeros((out_features, padded_in - in_features))
        w_padded = jnp.concatenate([w, padding], axis=1)
    else:
        w_padded = w

    # Reshape to (out_features, num_groups, group_size)
    grouped = w_padded.reshape(out_features, num_groups, group_size)

    # Compute absmax per group
    eps = 1e-8
    absmax = jnp.max(jnp.abs(grouped), axis=-1)
    scales = jnp.maximum(absmax / 7.0, eps)  # INT4 range: -8 to 7

    # Quantize all groups at once
    q_grouped = jnp.round(grouped / scales[:, :, None])
    q_grouped = jnp.clip(q_grouped, -8, 7).astype(jnp.int8)

    # Flatten back to (out_features, padded_in) and trim
    q_flat = q_grouped.reshape(out_features, padded_in)[:, :in_features]

    return np.array(q_flat), np.array(scales), num_groups


def jax_dequantize_int4(
    W_quant: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """INT4 dequantization reference with group-wise scales.

    Args:
        W_quant: Quantized INT4 weights (out_features, in_features).
        scales: Group-wise scales (out_features, num_groups).
        group_size: Group size used for quantization.

    Returns:
        Dequantized float weights.
    """
    _check_jax()
    w_q = jnp.array(W_quant, dtype=jnp.float32)
    s = jnp.array(scales, dtype=jnp.float32)

    out_features, in_features = w_q.shape

    # Expand scales to match weight dimensions
    # scales: (out_features, num_groups) -> (out_features, in_features)
    scale_expanded = jnp.repeat(s, group_size, axis=1)[:, :in_features]

    # Dequantize
    w = w_q * scale_expanded

    return np.array(w)


def jax_int8_linear(
    x: np.ndarray,
    W_quant: np.ndarray,
    scale: np.ndarray,
    zero_point: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """INT8 linear layer reference.

    Dequantizes weights and performs linear transformation.

    Args:
        x: Input tensor (*, in_features).
        W_quant: Quantized weights (out_features, in_features).
        scale: Quantization scale.
        zero_point: Zero point (None for symmetric).
        bias: Optional bias (out_features,).

    Returns:
        Output tensor (*, out_features).
    """
    _check_jax()
    # Dequantize weights
    W = jax_dequantize_int8(W_quant, scale, zero_point)
    W = jnp.array(W, dtype=jnp.float32)

    # Linear transformation: y = x @ W.T
    x_j = jnp.array(x, dtype=jnp.float32)
    y = x_j @ W.T

    if bias is not None:
        y = y + jnp.array(bias, dtype=jnp.float32)

    return np.array(y)


def jax_int4_linear(
    x: np.ndarray,
    W_quant: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """INT4 linear layer reference.

    Dequantizes weights and performs linear transformation.

    Args:
        x: Input tensor (*, in_features).
        W_quant: Quantized weights (out_features, in_features).
        scales: Group-wise scales (out_features, num_groups).
        group_size: Group size used for quantization.
        bias: Optional bias (out_features,).

    Returns:
        Output tensor (*, out_features).
    """
    _check_jax()
    # Dequantize weights
    W = jax_dequantize_int4(W_quant, scales, group_size)
    W = jnp.array(W, dtype=jnp.float32)

    # Linear transformation: y = x @ W.T
    x_j = jnp.array(x, dtype=jnp.float32)
    y = x_j @ W.T

    if bias is not None:
        y = y + jnp.array(bias, dtype=jnp.float32)

    return np.array(y)


# =============================================================================
# Primitive Operations (6)
# =============================================================================

def jax_associative_scan_add(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with add (cumsum) reference.

    Args:
        x: Input array.
        axis: Axis along which to scan.

    Returns:
        Cumulative sum as numpy array.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    result = lax.associative_scan(jnp.add, x_j, axis=axis)
    return np.array(result)


def jax_associative_scan_mul(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with multiply (cumprod) reference.

    Args:
        x: Input array.
        axis: Axis along which to scan.

    Returns:
        Cumulative product as numpy array.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    result = lax.associative_scan(jnp.multiply, x_j, axis=axis)
    return np.array(result)


def jax_ssm_scan_simple(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Simple SSM scan: h[t] = A[t] * h[t-1] + x[t].

    Uses JAX lax.associative_scan with the SSM binary operator.

    Args:
        A: Decay coefficients (batch, seq, state).
        x: Input values (batch, seq, state).

    Returns:
        Hidden states h of shape (batch, seq, state).
    """
    _check_jax()
    A_j = jnp.array(A, dtype=jnp.float32)
    x_j = jnp.array(x, dtype=jnp.float32)

    # Define the SSM associative binary operation
    # (A1, h1) ⊕ (A2, h2) = (A1*A2, A2*h1 + h2)
    def ssm_combine(left, right):
        A_left, h_left = left
        A_right, h_right = right
        return (A_left * A_right, A_right * h_left + h_right)

    # Apply associative scan along sequence axis (axis=1)
    init_elements = (A_j, x_j)
    _, h_result = lax.associative_scan(ssm_combine, init_elements, axis=1)

    return np.array(h_result)


def jax_ssm_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray,
    compute_dtype: Optional[str] = None,
    use_sequential: bool = False
) -> np.ndarray:
    """Full SSM-style selective scan reference (Mamba-style).

    Implements:
    1. Discretization: A_bar = exp(delta * A), B_bar = delta * B
    2. SSM recurrence: h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
    3. Output projection: y[t] = sum(C[t] * h[t]) + D * x[t]

    Args:
        A: State matrix diagonal (d_inner, d_state).
        B: Input projection (batch, seq, d_state).
        C: Output projection (batch, seq, d_state).
        D: Skip connection (d_inner,).
        x: Input tensor (batch, seq, d_inner).
        delta: Time step tensor (batch, seq, d_inner).
        compute_dtype: Optional dtype for computation ('fp16', 'bf16', or None for fp32).
        use_sequential: If True, use sequential scan instead of associative scan.

    Returns:
        Output tensor y of shape (batch, seq, d_inner).
    """
    _check_jax()

    # Select dtype based on compute_dtype parameter
    dtype_map = {
        'fp16': jnp.float16,
        'bf16': jnp.bfloat16,
        None: jnp.float32,
        'fp32': jnp.float32,
    }
    jax_dtype = dtype_map.get(compute_dtype, jnp.float32)

    A_j = jnp.array(A, dtype=jax_dtype)
    B_j = jnp.array(B, dtype=jax_dtype)
    C_j = jnp.array(C, dtype=jax_dtype)
    D_j = jnp.array(D, dtype=jax_dtype) if D is not None else None
    x_j = jnp.array(x, dtype=jax_dtype)
    delta_j = jnp.array(delta, dtype=jax_dtype)

    batch_size, seq_len, d_inner = x_j.shape
    d_state = A_j.shape[1]

    # Discretization
    delta_A = delta_j[..., None] * A_j[None, None, :, :]  # (batch, seq, d_inner, d_state)
    A_bar = jnp.exp(delta_A)

    # B_bar_x = delta * B * x
    B_x = B_j[:, :, None, :] * x_j[..., None]  # (batch, seq, d_inner, d_state)
    B_bar_x = delta_j[..., None] * B_x

    # Reshape for scan: (batch * d_inner, seq, d_state)
    A_bar_flat = A_bar.transpose(0, 2, 1, 3).reshape(batch_size * d_inner, seq_len, d_state)
    B_bar_x_flat = B_bar_x.transpose(0, 2, 1, 3).reshape(batch_size * d_inner, seq_len, d_state)

    if use_sequential:
        # Sequential scan (matches logical recurrence exactly)
        # Use Python loop with JAX arrays for deterministic order of operations
        h_list = []
        h_prev = jnp.zeros((batch_size * d_inner, d_state), dtype=jax_dtype)
        for t in range(seq_len):
            h_t = A_bar_flat[:, t, :] * h_prev + B_bar_x_flat[:, t, :]
            h_list.append(h_t)
            h_prev = h_t
        h_flat = jnp.stack(h_list, axis=1)
    else:
        # SSM scan using associative scan (parallel, may differ in reduced precision)
        def ssm_combine(left, right):
            A_left, h_left = left
            A_right, h_right = right
            return (A_left * A_right, A_right * h_left + h_right)

        _, h_flat = lax.associative_scan(ssm_combine, (A_bar_flat, B_bar_x_flat), axis=1)

    # Reshape back: (batch, seq, d_inner, d_state)
    h = h_flat.reshape(batch_size, d_inner, seq_len, d_state).transpose(0, 2, 1, 3)

    # Output projection: y = sum(C * h, axis=-1)
    y = jnp.sum(C_j[:, :, None, :] * h, axis=-1)

    # Skip connection
    if D_j is not None:
        y = y + x_j * D_j[None, None, :]

    # Always return float32 for comparison
    return np.array(y.astype(jnp.float32))


def jax_selective_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray,
    compute_dtype: Optional[str] = None,
    use_sequential: bool = False
) -> np.ndarray:
    """Mamba-style selective scan reference (alias for jax_ssm_scan)."""
    return jax_ssm_scan(A, B, C, D, x, delta, compute_dtype=compute_dtype, use_sequential=use_sequential)


def jax_selective_gather(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Selective gather reference.

    Args:
        x: Source array to gather from.
        indices: Indices to gather.

    Returns:
        Gathered values.
    """
    _check_jax()
    x_j = jnp.array(x)
    indices_j = jnp.array(indices)
    result = x_j[indices_j]
    return np.array(result)


def jax_selective_scatter_add(
    output: np.ndarray, values: np.ndarray,
    indices: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Selective scatter-add reference.

    Args:
        output: Target array to scatter into.
        values: Values to scatter.
        indices: Indices where to scatter.
        weights: Optional weights to multiply values.

    Returns:
        Output array with scattered values added.
    """
    _check_jax()
    output_j = jnp.array(output, dtype=jnp.float32)
    values_j = jnp.array(values, dtype=jnp.float32)
    indices_j = jnp.array(indices)

    if weights is not None:
        weights_j = jnp.array(weights, dtype=jnp.float32)
        values_j = values_j * weights_j

    # Use at[].add for scatter-add operation
    result = output_j.at[indices_j].add(values_j)
    return np.array(result)


# =============================================================================
# MoE Operations (3)
# =============================================================================

def jax_topk_routing(
    logits: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TopK routing reference.

    Routes each token to the top-k experts based on gating scores.

    Args:
        logits: Router logits (batch, seq_len, num_experts) or (n_tokens, num_experts).
        k: Number of experts to route to per token.

    Returns:
        Tuple of (gate_weights, expert_indices, router_logits):
        - gate_weights: Normalized weights for selected experts
        - expert_indices: Indices of selected experts
        - router_logits: Original logits (for aux loss computation)
    """
    _check_jax()
    logits_j = jnp.array(logits, dtype=jnp.float32)

    # Get top-k experts (descending sort)
    sorted_indices = jnp.argsort(-logits_j, axis=-1)
    expert_indices = sorted_indices[..., :k]

    # Gather logits for selected experts
    selected_logits = jnp.take_along_axis(logits_j, expert_indices, axis=-1)

    # Softmax over selected experts to get normalized weights
    gate_weights = jnn.softmax(selected_logits, axis=-1)

    return np.array(gate_weights), np.array(expert_indices), np.array(logits_j)


def jax_expert_dispatch(
    x: np.ndarray,
    expert_indices: np.ndarray,
    expert_weights: np.ndarray,
    expert_weights_list: List[np.ndarray],
    num_experts: int,
    capacity: Optional[int] = None,
) -> np.ndarray:
    """Expert dispatch reference.

    Dispatches tokens to experts and combines weighted outputs.
    Uses capacity-based dispatch to match MLX behavior.

    Args:
        x: Input tensor (n_tokens, dims).
        expert_indices: Expert indices per token (n_tokens, top_k).
        expert_weights: Routing weights per token (n_tokens, top_k).
        expert_weights_list: List of (w1, w2) weight tuples for each expert.
        num_experts: Number of experts.
        capacity: Max tokens per expert. If None, computes as in MLX.

    Returns:
        Combined output tensor (n_tokens, dims).
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    indices_j = jnp.array(expert_indices)
    weights_j = jnp.array(expert_weights, dtype=jnp.float32)

    n_tokens, dims = x_j.shape
    top_k = indices_j.shape[-1]

    # Calculate capacity as in MLX if not provided
    if capacity is None:
        capacity = (n_tokens * top_k + num_experts - 1) // num_experts
        capacity = max(1, capacity)

    output = jnp.zeros_like(x_j)

    # Process each expert with capacity-based dispatch (matching MLX)
    for expert_idx in range(num_experts):
        # Find tokens routed to this expert and their weights
        expert_mask = indices_j == expert_idx  # (n_tokens, top_k)
        expert_weights_per_topk = jnp.where(expert_mask, weights_j, 0.0)
        token_weights = jnp.sum(expert_weights_per_topk, axis=-1)  # (n_tokens,)

        # Sort by weight descending to get routing order (matches MLX argsort trick)
        sorted_order = jnp.argsort(-token_weights)

        # Get expert weights
        w1, w2 = expert_weights_list[expert_idx]
        w1_j = jnp.array(w1, dtype=jnp.float32)
        w2_j = jnp.array(w2, dtype=jnp.float32)

        # Take first 'capacity' tokens (includes routed + padding with weight=0)
        routed_indices = sorted_order[:capacity]
        x_expert = x_j[routed_indices]  # (capacity, dims)
        weights_expert = token_weights[routed_indices]  # (capacity,)

        # Process through expert
        hidden = jnn.silu(x_expert @ w1_j.T)  # (capacity, hidden_dims)
        expert_out = hidden @ w2_j.T  # (capacity, dims)

        # Weighted output
        weighted_out = expert_out * weights_expert[:, None]

        # Scatter-add back to output at original positions
        output = output.at[routed_indices].add(weighted_out)

    return np.array(output)


def jax_load_balancing_loss(
    router_logits: np.ndarray,
    expert_indices: np.ndarray,
    num_experts: int,
) -> float:
    """Load balancing auxiliary loss reference.

    Encourages balanced expert usage across tokens.

    Args:
        router_logits: Router logits (batch, seq_len, num_experts).
        expert_indices: Selected expert indices (batch, seq_len, top_k).
        num_experts: Total number of experts.

    Returns:
        Auxiliary loss value for load balancing.
    """
    _check_jax()
    logits_j = jnp.array(router_logits, dtype=jnp.float32)
    indices_j = jnp.array(expert_indices)

    batch_size, seq_len, _ = logits_j.shape
    top_k = indices_j.shape[-1]

    # Routing probabilities
    router_probs = jnn.softmax(logits_j, axis=-1)

    # Expert usage counts
    total_tokens = batch_size * seq_len * top_k

    # Count tokens for each expert
    expert_counts = []
    for e in range(num_experts):
        count = jnp.sum(indices_j == e)
        expert_counts.append(count)

    expert_fraction = jnp.stack(expert_counts).astype(jnp.float32) / total_tokens

    # Mean routing probability
    mean_prob = jnp.mean(router_probs, axis=(0, 1))

    # Loss: num_experts * sum(fraction * mean_prob)
    loss = num_experts * jnp.sum(expert_fraction * mean_prob)

    return float(loss)


# =============================================================================
# Pooling Operations (7)
# =============================================================================

def jax_adaptive_avg_pool1d(x: np.ndarray, output_size: int) -> np.ndarray:
    """AdaptiveAvgPool1d reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    input_size = x_j.shape[-1]
    if input_size == output_size:
        return np.array(x_j)
    outputs = []
    for i in range(output_size):
        start = (i * input_size) // output_size
        end = ((i + 1) * input_size + output_size - 1) // output_size
        pooled = jnp.mean(x_j[..., start:end], axis=-1, keepdims=True)
        outputs.append(pooled)
    return np.array(jnp.concatenate(outputs, axis=-1))


def jax_adaptive_avg_pool2d(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """AdaptiveAvgPool2d reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    _, _, in_h, in_w = x_j.shape
    out_h, out_w = output_size
    if in_h == out_h and in_w == out_w:
        return np.array(x_j)
    if out_h == 1 and out_w == 1:
        return np.array(jnp.mean(x_j, axis=(2, 3), keepdims=True))
    outputs = []
    for i in range(out_h):
        row_outputs = []
        start_h = (i * in_h) // out_h
        end_h = ((i + 1) * in_h + out_h - 1) // out_h
        for j in range(out_w):
            start_w = (j * in_w) // out_w
            end_w = ((j + 1) * in_w + out_w - 1) // out_w
            pooled = jnp.mean(x_j[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
            row_outputs.append(pooled)
        outputs.append(jnp.concatenate(row_outputs, axis=3))
    return np.array(jnp.concatenate(outputs, axis=2))


def jax_adaptive_max_pool1d(x: np.ndarray, output_size: int) -> np.ndarray:
    """AdaptiveMaxPool1d reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    input_size = x_j.shape[-1]
    if input_size == output_size:
        return np.array(x_j)
    outputs = []
    for i in range(output_size):
        start = (i * input_size) // output_size
        end = ((i + 1) * input_size + output_size - 1) // output_size
        pooled = jnp.max(x_j[..., start:end], axis=-1, keepdims=True)
        outputs.append(pooled)
    return np.array(jnp.concatenate(outputs, axis=-1))


def jax_adaptive_max_pool2d(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """AdaptiveMaxPool2d reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    _, _, in_h, in_w = x_j.shape
    out_h, out_w = output_size
    if in_h == out_h and in_w == out_w:
        return np.array(x_j)
    if out_h == 1 and out_w == 1:
        return np.array(jnp.max(x_j, axis=(2, 3), keepdims=True))
    outputs = []
    for i in range(out_h):
        row_outputs = []
        start_h = (i * in_h) // out_h
        end_h = ((i + 1) * in_h + out_h - 1) // out_h
        for j in range(out_w):
            start_w = (j * in_w) // out_w
            end_w = ((j + 1) * in_w + out_w - 1) // out_w
            pooled = jnp.max(x_j[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
            row_outputs.append(pooled)
        outputs.append(jnp.concatenate(row_outputs, axis=3))
    return np.array(jnp.concatenate(outputs, axis=2))


def jax_global_attention_pooling(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Global attention pooling reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    W1_j = jnp.array(W1, dtype=jnp.float32)
    W2_j = jnp.array(W2, dtype=jnp.float32)
    scores = jnp.tanh(x_j @ W1_j) @ W2_j
    weights = jnn.softmax(scores, axis=1)
    return np.array(jnp.sum(x_j * weights, axis=1))


def jax_gem(x: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """Generalized Mean (GeM) pooling reference."""
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    x_j = jnp.clip(x_j, a_min=eps)
    x_pow = jnp.power(x_j, p)
    mean_pow = jnp.mean(x_pow, axis=(2, 3), keepdims=True)
    return np.array(jnp.power(mean_pow, 1.0 / p))


def jax_spatial_pyramid_pooling(x: np.ndarray, levels: List[int]) -> np.ndarray:
    """Spatial Pyramid Pooling reference.

    Matches MLX SpatialPyramidPooling which outputs (batch, channels * sum(level^2)).
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    batch, channels, _, _ = x_j.shape
    outputs = []
    for level in levels:
        pooled = jnp.array(jax_adaptive_avg_pool2d(np.array(x_j), (level, level)))
        # Flatten each level's output to (batch, channels * level * level)
        pooled = pooled.reshape(batch, -1)
        outputs.append(pooled)
    # Concatenate all levels: (batch, channels * (1^2 + 2^2 + 4^2 + ...))
    return np.array(jnp.concatenate(outputs, axis=1))


# =============================================================================
# Embedding Operations (5)
# =============================================================================

def jax_sinusoidal_embedding(
    positions: np.ndarray, dim: int, base: float = 10000.0
) -> np.ndarray:
    """Sinusoidal positional embedding reference.

    PE(pos, 2i) = sin(pos / base^(2i/d))
    PE(pos, 2i+1) = cos(pos / base^(2i/d))

    Args:
        positions: Position indices.
        dim: Embedding dimension.
        base: Base for frequency computation.

    Returns:
        Sinusoidal embeddings of shape (len(positions), dim).
    """
    _check_jax()
    positions_j = jnp.array(positions, dtype=jnp.float32)

    # Ensure positions is 1D
    if positions_j.ndim == 0:
        positions_j = positions_j[None]
    positions_j = positions_j.flatten()[:, None]  # (seq_len, 1)

    dims_range = jnp.arange(0, dim, 2, dtype=jnp.float32)  # (dim/2,)
    freqs = base ** (-dims_range / dim)  # (dim/2,)

    angles = positions_j * freqs  # (seq_len, dim/2)

    sin_embeddings = jnp.sin(angles)
    cos_embeddings = jnp.cos(angles)

    # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
    embeddings = jnp.stack([sin_embeddings, cos_embeddings], axis=2)
    embeddings = embeddings.reshape(positions_j.shape[0], dim)

    return np.array(embeddings)


def jax_learned_positional_embedding(positions: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Learned positional embedding reference.

    Simple embedding lookup.

    Args:
        positions: Position indices.
        weight: Embedding weight matrix (max_seq_len, dim).

    Returns:
        Positional embeddings.
    """
    _check_jax()
    positions_j = jnp.array(positions)
    weight_j = jnp.array(weight, dtype=jnp.float32)

    result = weight_j[positions_j]
    return np.array(result)


def jax_rotary_embedding(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotary positional embedding reference.

    Applies rotary embedding to input tensor.
    RoPE rotates pairs of dimensions: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]

    Args:
        x: Input tensor (..., head_dim) where head_dim is even.
        cos: Cosine values for rotation.
        sin: Sine values for rotation.

    Returns:
        Rotated tensor of same shape as x.
    """
    _check_jax()
    x_j = jnp.array(x, dtype=jnp.float32)
    cos_j = jnp.array(cos, dtype=jnp.float32)
    sin_j = jnp.array(sin, dtype=jnp.float32)

    # Split x into pairs
    x1 = x_j[..., ::2]   # Even indices: (..., head_dim//2)
    x2 = x_j[..., 1::2]  # Odd indices: (..., head_dim//2)

    # Expand cos/sin to match x1/x2 shape if needed
    # Typical input: cos/sin are (seq_len, head_dim//2)
    # x1/x2 are (batch, seq_len, num_heads, head_dim//2)
    # Need cos/sin to be (1, seq_len, 1, head_dim//2) for broadcasting
    while cos_j.ndim < x1.ndim:
        if cos_j.ndim == 2:  # (seq, dim//2) -> (1, seq, 1, dim//2)
            cos_j = cos_j[None, :, None, :]
            sin_j = sin_j[None, :, None, :]
        else:
            cos_j = cos_j[None, ...]
            sin_j = sin_j[None, ...]

    # Apply rotation
    rotated_x1 = x1 * cos_j - x2 * sin_j
    rotated_x2 = x1 * sin_j + x2 * cos_j

    # Interleave back
    result = jnp.stack([rotated_x1, rotated_x2], axis=-1)
    result = result.reshape(x_j.shape)

    return np.array(result)


def jax_alibi_embedding(seq_len: int, num_heads: int) -> np.ndarray:
    """ALiBi position bias reference.

    Computes Attention with Linear Biases position bias matrix.
    slope_h = 2^(-8h/H) for head h in [1, H]

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.

    Returns:
        ALiBi bias matrix of shape (num_heads, seq_len, seq_len).
    """
    _check_jax()
    # Compute slopes: 2^(-8/num_heads) raised to powers 1, 2, ..., num_heads
    base = 2.0 ** (-8.0 / num_heads)
    slopes = jnp.array([base ** (i + 1) for i in range(num_heads)])  # (num_heads,)

    # Create relative position matrix
    positions = jnp.arange(seq_len)
    # Distance: query_pos - key_pos (negative for past positions)
    rel_pos = positions[:, None] - positions[None, :]  # (seq_len, seq_len)

    # Bias = slope * distance (negative distances give negative bias)
    # Shape: (num_heads, seq_len, seq_len)
    bias = slopes[:, None, None] * rel_pos[None, :, :]

    return np.array(bias)


def jax_relative_positional_embedding(
    q_len: int, k_len: int, num_buckets: int = 32, max_distance: int = 128
) -> np.ndarray:
    """Relative positional embedding (T5-style) reference.

    T5-style bucketed relative position bias.

    Args:
        q_len: Query length.
        k_len: Key length.
        num_buckets: Number of relative position buckets.
        max_distance: Maximum distance for bucket computation.

    Returns:
        Relative position bucket indices of shape (q_len, k_len).
    """
    _check_jax()
    # Compute relative positions
    q_pos = jnp.arange(q_len)[:, None]
    k_pos = jnp.arange(k_len)[None, :]
    relative_position = k_pos - q_pos  # (q_len, k_len)

    # Bucketing logic (simplified T5 style)
    num_buckets_half = num_buckets // 2
    relative_buckets = jnp.zeros((q_len, k_len), dtype=jnp.int32)

    # Positive positions (future)
    is_positive = relative_position > 0
    relative_position_abs = jnp.abs(relative_position)

    # Small distances get individual buckets
    max_exact = num_buckets_half // 2
    is_small = relative_position_abs < max_exact

    # Large distances use log-spaced buckets
    relative_position_if_large = max_exact + (
        jnp.log(relative_position_abs.astype(jnp.float32) / max_exact + 1e-6)
        / jnp.log(max_distance / max_exact)
        * (num_buckets_half - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.minimum(
        relative_position_if_large, num_buckets_half - 1
    )

    relative_buckets = jnp.where(is_small, relative_position_abs, relative_position_if_large)
    relative_buckets = jnp.where(is_positive, relative_buckets + num_buckets_half, relative_buckets)

    return np.array(relative_buckets)


# =============================================================================
# Cache Operations (4)
# =============================================================================

def jax_paged_attention(
    q: np.ndarray,
    k_pool: np.ndarray,
    v_pool: np.ndarray,
    block_tables: np.ndarray,
    context_lens: np.ndarray,
    block_size: int = 16,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Paged attention reference.

    Computes attention with paged KV cache using block tables for indexing.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k_pool: Key block pool (num_blocks, block_size, num_heads, head_dim).
        v_pool: Value block pool (num_blocks, block_size, num_heads, head_dim).
        block_tables: Block indices per sequence (batch, max_blocks).
        context_lens: Number of cached tokens per sequence (batch,).
        block_size: Tokens per block.
        scale: Attention scale. Defaults to 1/sqrt(head_dim).

    Returns:
        Output tensor same shape as q.
    """
    _check_jax()
    import math

    q_j = jnp.array(q, dtype=jnp.float32)
    k_pool_j = jnp.array(k_pool, dtype=jnp.float32)
    v_pool_j = jnp.array(v_pool, dtype=jnp.float32)
    block_tables_j = jnp.array(block_tables)
    context_lens_j = jnp.array(context_lens)

    batch_size, seq_q, num_heads, head_dim = q_j.shape
    max_blocks = block_tables_j.shape[1]
    max_context = max_blocks * block_size

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Replace invalid block IDs (-1) with 0 for safe gathering
    safe_block_tables = jnp.maximum(block_tables_j, 0)

    # Gather all blocks at once
    k_gathered = k_pool_j[safe_block_tables]  # (batch, max_blocks, block_size, num_heads, head_dim)
    v_gathered = v_pool_j[safe_block_tables]

    # Reshape to (batch, max_context, num_heads, head_dim)
    k = k_gathered.reshape(batch_size, max_context, num_heads, head_dim)
    v = v_gathered.reshape(batch_size, max_context, num_heads, head_dim)

    # Transpose for attention: (batch, num_heads, seq, head_dim)
    q_t = jnp.transpose(q_j, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    # Scores: (batch, num_heads, seq_q, max_context)
    scores = (q_t @ jnp.transpose(k_t, (0, 1, 3, 2))) * scale

    # Create mask for valid positions
    positions = jnp.arange(max_context)[None, None, None, :]
    padding_mask = positions < context_lens_j[:, None, None, None]

    # Block validity mask
    block_indices_1d = jnp.arange(max_context) // block_size
    block_validity = jnp.take(block_tables_j >= 0, block_indices_1d, axis=1)
    block_validity = block_validity[:, None, None, :]

    combined_mask = padding_mask & block_validity

    # Apply mask
    scores = jnp.where(combined_mask, scores, float("-inf"))

    # Causal mask
    if seq_q > 1:
        q_pos = jnp.arange(seq_q)[:, None]
        kv_pos = jnp.arange(max_context)[None, :]
        causal_mask = kv_pos <= q_pos
        causal_mask = causal_mask[None, None, :, :]
        scores = jnp.where(causal_mask, scores, float("-inf"))

    # Softmax and output
    attn_weights = jnn.softmax(scores, axis=-1)
    output = attn_weights @ v_t

    # Transpose back: (batch, seq_q, num_heads, head_dim)
    result = jnp.transpose(output, (0, 2, 1, 3))

    return np.array(result)


def jax_block_allocation(
    sequence_lengths: np.ndarray,
    block_size: int,
    max_blocks: Optional[int] = None,
) -> np.ndarray:
    """Block allocation reference.

    Creates block tables from sequence lengths by assigning sequential block IDs.

    Args:
        sequence_lengths: Lengths of each sequence (batch,).
        block_size: Tokens per block.
        max_blocks: Maximum blocks per sequence. If None, computed from lengths.

    Returns:
        Block tables (batch, max_blocks) with -1 padding for invalid blocks.
    """
    _check_jax()
    seq_lens = jnp.array(sequence_lengths)
    batch_size = seq_lens.shape[0]

    # Calculate blocks per sequence: ceil(length / block_size)
    blocks_per_seq = (seq_lens + block_size - 1) // block_size

    if max_blocks is None:
        max_blocks = int(jnp.max(blocks_per_seq))

    # Cumulative sum gives starting block ID for each sequence
    cum_blocks = jnp.concatenate([jnp.array([0]), jnp.cumsum(blocks_per_seq)[:-1]])

    # Create position indices
    positions = jnp.arange(max_blocks)[None, :]

    # Create mask: True where position < blocks_per_seq
    valid_mask = positions < blocks_per_seq[:, None]

    # Create block IDs
    block_ids = cum_blocks[:, None] + positions

    # Apply mask: -1 for invalid positions
    block_tables = jnp.where(valid_mask, block_ids, -1)

    return np.array(block_tables).astype(np.int32)


def jax_eviction_lru(access_times: np.ndarray, num_to_evict: int) -> np.ndarray:
    """LRU eviction reference.

    Returns indices of blocks to evict based on least recent access.

    Args:
        access_times: Last access time for each block of shape (num_blocks,).
        num_to_evict: Number of blocks to evict.

    Returns:
        Indices of blocks to evict (sorted by oldest access time).
    """
    _check_jax()
    times_j = jnp.array(access_times, dtype=jnp.float32)

    # Get indices sorted by access time (ascending = oldest first)
    sorted_indices = jnp.argsort(times_j)

    # Return the num_to_evict oldest
    return np.array(sorted_indices[:num_to_evict])


def jax_speculative_verification(
    draft_tokens: np.ndarray, target_probs: np.ndarray, draft_probs: np.ndarray,
    key: Optional[Any] = None
) -> Tuple[np.ndarray, int]:
    """Speculative decoding verification reference.

    Verifies draft tokens against target model probabilities.

    Args:
        draft_tokens: Draft tokens of shape (num_draft,).
        target_probs: Target model probabilities of shape (num_draft, vocab_size).
        draft_probs: Draft model probabilities of shape (num_draft, vocab_size).
        key: Optional JAX random key for rejection sampling.

    Returns:
        Tuple of (accepted_tokens, num_accepted):
        - accepted_tokens: Array of accepted token ids
        - num_accepted: Number of tokens accepted
    """
    _check_jax()
    import jax.random as random

    if key is None:
        key = random.PRNGKey(0)

    draft_t = jnp.array(draft_tokens, dtype=jnp.int32)
    target_p = jnp.array(target_probs, dtype=jnp.float32)
    draft_p = jnp.array(draft_probs, dtype=jnp.float32)

    num_draft = len(draft_tokens)
    accepted = []

    for i in range(num_draft):
        token = int(draft_t[i])

        # Get probabilities for this token
        p_target = float(target_p[i, token])
        p_draft = float(draft_p[i, token])

        # Accept if target prob >= draft prob, or with probability p_target/p_draft
        if p_target >= p_draft:
            accepted.append(token)
        else:
            # Rejection sampling
            key, subkey = random.split(key)
            r = float(random.uniform(subkey))
            if r < p_target / max(p_draft, 1e-10):
                accepted.append(token)
            else:
                # Reject this and all subsequent tokens
                break

    return np.array(accepted, dtype=np.int64), len(accepted)


# =============================================================================
# Generation/Sampling Operations (3)
# =============================================================================

def jax_temperature_sampling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled sampling reference.

    Applies temperature scaling to logits.

    Args:
        logits: Input logits (batch, vocab_size).
        temperature: Temperature value. Higher = more random.

    Returns:
        Temperature-scaled logits.
    """
    _check_jax()
    logits_j = jnp.array(logits, dtype=jnp.float32)

    if temperature == 1.0:
        return np.array(logits_j)
    if temperature == 0.0:
        return np.array(logits_j)  # Will use argmax anyway

    return np.array(logits_j / temperature)


def jax_top_k_sampling(logits: np.ndarray, k: int) -> np.ndarray:
    """Top-K sampling reference.

    Sets logits outside top-k to -inf.

    Args:
        logits: Input logits (batch, vocab_size).
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with values outside top-k set to -inf.
    """
    _check_jax()
    logits_j = jnp.array(logits, dtype=jnp.float32)

    if k <= 0 or k >= logits_j.shape[-1]:
        return np.array(logits_j)

    # Get k-th largest value for each batch
    sorted_logits = jnp.sort(logits_j, axis=-1)  # Ascending
    threshold = sorted_logits[:, -k : sorted_logits.shape[-1] - k + 1]

    # Mask values below threshold
    result = jnp.where(logits_j >= threshold, logits_j, float("-inf"))

    return np.array(result)


def jax_top_p_sampling(logits: np.ndarray, p: float) -> np.ndarray:
    """Top-P (nucleus) sampling reference.

    Keeps smallest set of tokens with cumulative probability >= p.

    Args:
        logits: Input logits (batch, vocab_size).
        p: Cumulative probability threshold.

    Returns:
        Filtered logits with values outside nucleus set to -inf.
    """
    _check_jax()
    logits_j = jnp.array(logits, dtype=jnp.float32)

    if p >= 1.0:
        return np.array(logits_j)

    # Sort by descending probability
    sorted_indices = jnp.argsort(logits_j, axis=-1)[:, ::-1]  # Descending
    sorted_logits = jnp.take_along_axis(logits_j, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    sorted_probs = jnn.softmax(sorted_logits, axis=-1)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

    # Find cutoff: keep tokens until cumulative prob exceeds p
    # Shift cumulative probs to include first token always
    shifted_cumulative = jnp.concatenate(
        [jnp.zeros((logits_j.shape[0], 1)), cumulative_probs[:, :-1]], axis=-1
    )
    sorted_mask = shifted_cumulative < p

    # Ensure at least the top token is always kept
    first_token_mask = jnp.zeros_like(sorted_mask).at[:, 0].set(True)
    sorted_mask = jnp.logical_or(sorted_mask, first_token_mask)

    # Set filtered positions to -inf
    sorted_logits = jnp.where(sorted_mask, sorted_logits, float("-inf"))

    # Unsort back to original order
    inverse_indices = jnp.argsort(sorted_indices, axis=-1)
    result = jnp.take_along_axis(sorted_logits, inverse_indices, axis=-1)

    return np.array(result)
