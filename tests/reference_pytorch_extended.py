"""Extended PyTorch reference implementations for parity testing.

This module provides 50+ reference implementations covering all MLXPrimitives operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")


# =============================================================================
# Attention Operations (11 variants)
# =============================================================================

def torch_sliding_window_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_len, heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Create sliding window mask
        positions = torch.arange(seq_len)
        row_pos = positions[:, None]
        col_pos = positions[None, :]
        mask = torch.abs(row_pos - col_pos) <= window_size

        if causal:
            causal_mask = col_pos <= row_pos
            mask = mask & causal_mask

        # Apply mask
        scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        # Transpose back
        return out.permute(0, 2, 1, 3).numpy()


def torch_gqa(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_len, num_heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Expand K/V heads to match Q
        num_groups = num_heads // num_kv_heads
        k_t = k_t.repeat_interleave(num_groups, dim=2)
        v_t = v_t.repeat_interleave(num_groups, dim=2)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Apply causal mask if needed
        if causal:
            positions = torch.arange(seq_len)
            mask = positions[None, :] <= positions[:, None]
            scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


def torch_mqa(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_len, num_heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Expand K/V to all heads
        k_t = k_t.expand(-1, -1, num_heads, -1)
        v_t = v_t.expand(-1, -1, num_heads, -1)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Apply causal mask if needed
        if causal:
            positions = torch.arange(seq_len)
            mask = positions[None, :] <= positions[:, None]
            scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


def torch_linear_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        # Apply feature map
        if feature_map == "elu":
            q_t = F.elu(q_t) + 1
            k_t = F.elu(k_t) + 1
        elif feature_map == "softmax":
            q_t = F.softmax(q_t, dim=-1)
            k_t = F.softmax(k_t, dim=-1)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute KV: (batch, heads, dim_k, dim_v)
        kv = torch.einsum("bhsd,bhsv->bhdv", k_t, v_t)

        # Compute output: Q @ KV
        out = torch.einsum("bhqd,bhdv->bhqv", q_t, kv)

        # Normalize by sum of keys
        k_sum = k_t.sum(dim=2, keepdim=True)  # (batch, heads, 1, dim)
        normalizer = torch.einsum("bhqd,bhkd->bhqk", q_t, k_sum).squeeze(-1)  # (batch, heads, seq)
        normalizer = normalizer.unsqueeze(-1).clamp(min=1e-6)

        out = out / normalizer

        return out.permute(0, 2, 1, 3).numpy()


def torch_alibi_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))
        slopes = torch.from_numpy(alibi_slopes.astype(np.float32))

        batch, seq_len, heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Compute ALiBi bias
        positions = torch.arange(seq_len)
        rel_pos = positions[:, None] - positions[None, :]  # (seq, seq)
        alibi_bias = slopes[:, None, None] * rel_pos[None, :, :]  # (heads, seq, seq)

        # Add ALiBi bias
        scores = scores + alibi_bias.unsqueeze(0)

        # Apply causal mask if needed
        if causal:
            mask = positions[None, :] <= positions[:, None]
            scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


def torch_sparse_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_len, heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Create sparse mask based on pattern
        positions = torch.arange(seq_len)
        row_pos = positions[:, None]
        col_pos = positions[None, :]

        if mask_pattern == "local":
            # Local window attention
            mask = torch.abs(row_pos - col_pos) <= window_size
        elif mask_pattern == "strided":
            # Every stride-th position globally + local window
            local_mask = torch.abs(row_pos - col_pos) <= window_size
            strided_mask = (col_pos % stride) == 0
            mask = local_mask | strided_mask
        else:  # "fixed"
            # Fixed positions at beginning + local window
            local_mask = torch.abs(row_pos - col_pos) <= window_size
            fixed_mask = col_pos < window_size
            mask = local_mask | fixed_mask

        # Apply mask
        scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


def torch_chunked_cross_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_t = torch.from_numpy(k.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_q, heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        outputs = []
        for start in range(0, seq_q, chunk_size):
            end = min(start + chunk_size, seq_q)
            q_chunk = q_t[:, :, start:end, :]

            # Compute attention for this chunk
            scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k_t) * scale
            weights = F.softmax(scores, dim=-1)
            out_chunk = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)
        return out.permute(0, 2, 1, 3).numpy()


def torch_rope_attention(
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
    _check_torch()
    with torch.no_grad():
        # Apply RoPE to Q and K
        q_rotated = torch_rotary_embedding(q, cos, sin)
        k_rotated = torch_rotary_embedding(k, cos, sin)

        q_t = torch.from_numpy(q_rotated.astype(np.float32))
        k_t = torch.from_numpy(k_rotated.astype(np.float32))
        v_t = torch.from_numpy(v.astype(np.float32))

        batch, seq_len, heads, dim = q_t.shape
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Apply causal mask if needed
        if causal:
            positions = torch.arange(seq_len)
            mask = positions[None, :] <= positions[:, None]
            scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


def torch_quantized_kv_attention(
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
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_q = torch.from_numpy(k_quant.astype(np.float32))
        v_q = torch.from_numpy(v_quant.astype(np.float32))
        k_s = torch.from_numpy(k_scale.astype(np.float32))
        v_s = torch.from_numpy(v_scale.astype(np.float32))

        # Dequantize K and V
        if k_zero_point is not None:
            k_zp = torch.from_numpy(k_zero_point.astype(np.float32))
            k_t = (k_q - k_zp) * k_s
        else:
            k_t = k_q * k_s

        if v_zero_point is not None:
            v_zp = torch.from_numpy(v_zero_point.astype(np.float32))
            v_t = (v_q - v_zp) * v_s
        else:
            v_t = v_q * v_s

        batch, seq_q, heads, dim = q_t.shape
        seq_kv = k_t.shape[1]
        scale = 1.0 / np.sqrt(dim)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

        # Apply causal mask if needed
        if causal:
            q_pos = torch.arange(seq_q)
            k_pos = torch.arange(seq_kv)
            mask = k_pos[None, :] <= q_pos[:, None]
            scores = torch.where(mask[None, None, :, :], scores, torch.tensor(float("-inf")))

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_t)

        return out.permute(0, 2, 1, 3).numpy()


# =============================================================================
# Activation Functions (12+)
# =============================================================================

def torch_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """SwiGLU activation reference.

    Computes: SiLU(x @ W_gate) * (x @ W_up)
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        W_gate_t = torch.from_numpy(W_gate.astype(np.float32))
        W_up_t = torch.from_numpy(W_up.astype(np.float32))

        gate = F.silu(x_t @ W_gate_t.T)
        up = x_t @ W_up_t.T
        return (gate * up).numpy()


def torch_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """GeGLU activation reference.

    Computes: GELU(x @ W_gate) * (x @ W_up)
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        W_gate_t = torch.from_numpy(W_gate.astype(np.float32))
        W_up_t = torch.from_numpy(W_up.astype(np.float32))

        gate = F.gelu(x_t @ W_gate_t.T)
        up = x_t @ W_up_t.T
        return (gate * up).numpy()


def torch_reglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """ReGLU activation reference.

    Computes: ReLU(x @ W_gate) * (x @ W_up)
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        W_gate_t = torch.from_numpy(W_gate.astype(np.float32))
        W_up_t = torch.from_numpy(W_up.astype(np.float32))

        gate = F.relu(x_t @ W_gate_t.T)
        up = x_t @ W_up_t.T
        return (gate * up).numpy()


def torch_quick_gelu(x: np.ndarray) -> np.ndarray:
    """QuickGELU activation reference.

    Computes: x * sigmoid(1.702 * x)
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return (x_t * torch.sigmoid(1.702 * x_t)).numpy()


def torch_gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU with tanh approximation reference.

    Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return F.gelu(x_t, approximate='tanh').numpy()


def torch_mish(x: np.ndarray) -> np.ndarray:
    """Mish activation reference.

    Computes: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return F.mish(x_t).numpy()


def torch_squared_relu(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation reference.

    Computes: ReLU(x)^2
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return (F.relu(x_t) ** 2).numpy()


def torch_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish activation reference.

    Computes: x * sigmoid(beta * x)
    When beta=1, this equals SiLU.
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return (x_t * torch.sigmoid(beta * x_t)).numpy()


def torch_hard_swish(x: np.ndarray) -> np.ndarray:
    """Hard Swish activation reference.

    Computes: x * clip(x + 3, 0, 6) / 6
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return F.hardswish(x_t).numpy()


def torch_hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """Hard Sigmoid activation reference.

    Computes: clip(x + 3, 0, 6) / 6
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return F.hardsigmoid(x_t).numpy()


# =============================================================================
# Normalization Operations (5)
# =============================================================================

def torch_groupnorm(
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
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        w_t = torch.from_numpy(weight.astype(np.float32))
        b_t = torch.from_numpy(bias.astype(np.float32))
        return F.group_norm(x_t, num_groups, w_t, b_t, eps).numpy()


def torch_instancenorm(
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
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        w_t = torch.from_numpy(weight.astype(np.float32))
        b_t = torch.from_numpy(bias.astype(np.float32))
        return F.instance_norm(x_t, weight=w_t, bias=b_t, eps=eps).numpy()


def torch_adalayernorm(
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
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        scale_t = torch.from_numpy(scale.astype(np.float32))
        shift_t = torch.from_numpy(shift.astype(np.float32))

        # Standard layer normalization along last axis
        normalized_shape = [x_t.shape[-1]]
        x_norm = F.layer_norm(x_t, normalized_shape, eps=eps)

        # Apply adaptive modulation: (1 + scale) * x_norm + shift
        result = x_norm * (1 + scale_t) + shift_t
        return result.numpy()


# =============================================================================
# Fused Operations (4)
# =============================================================================

def torch_fused_rmsnorm_linear(
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
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        norm_w = torch.from_numpy(norm_weight.astype(np.float32))
        lin_w = torch.from_numpy(linear_weight.astype(np.float32))

        # RMSNorm
        rms = torch.sqrt(torch.mean(x_t * x_t, dim=-1, keepdim=True) + eps)
        x_norm = x_t / rms * norm_w

        # Linear
        out = torch.einsum("...d,od->...o", x_norm, lin_w)
        if linear_bias is not None:
            lin_b = torch.from_numpy(linear_bias.astype(np.float32))
            out = out + lin_b

        return out.numpy()


def torch_fused_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused SwiGLU reference (same as torch_swiglu)."""
    return torch_swiglu(x, W_gate, W_up)


def torch_fused_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused GeGLU reference (same as torch_geglu)."""
    return torch_geglu(x, W_gate, W_up)


def torch_fused_rope_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, causal: bool = False
) -> np.ndarray:
    """Fused RoPE + Attention reference (same as torch_rope_attention)."""
    return torch_rope_attention(q, k, v, cos, sin, causal)


# =============================================================================
# Quantization Operations (6)
# =============================================================================

def torch_quantize_int8(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT8 quantization reference with per-channel (row) scale/zero_point.

    Returns:
        (quantized_weights, scales, zero_points)
    """
    _check_torch()
    with torch.no_grad():
        w_t = torch.from_numpy(weights.astype(np.float32))

        w_min = w_t.min(dim=1).values
        w_max = w_t.max(dim=1).values

        scale = (w_max - w_min) / 255.0
        scale = torch.maximum(scale, torch.tensor(1e-8))

        zero_point = -torch.round(w_min / scale)
        zero_point = torch.clamp(zero_point, 0, 255)

        # Quantize
        quantized = torch.round(w_t / scale[:, None]) + zero_point[:, None]
        quantized = torch.clamp(quantized, 0, 255)

        # Convert to signed int8
        quantized = (quantized - 128).to(torch.int8)
        zero_point = zero_point - 128

        return quantized.numpy(), scale.numpy(), zero_point.numpy()


def torch_dequantize_int8(
    W_quant: np.ndarray, scale: np.ndarray, zero_point: np.ndarray
) -> np.ndarray:
    """INT8 dequantization reference."""
    _check_torch()
    with torch.no_grad():
        W_float = torch.from_numpy(W_quant.astype(np.float32))
        scale_t = torch.from_numpy(scale.astype(np.float32))
        zp_t = torch.from_numpy(zero_point.astype(np.float32))

        if scale_t.numel() > 1:
            return (scale_t[:, None] * (W_float - zp_t[:, None])).numpy()
        else:
            return (scale_t * (W_float - zp_t)).numpy()


def torch_quantize_int4(
    weights: np.ndarray, group_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT4 quantization reference with group-wise quantization.

    Args:
        weights: Weight matrix of shape (out_features, in_features).
        group_size: Number of elements per quantization group.

    Returns:
        (quantized_weights, scales, zero_points) where quantized is packed INT4.
    """
    _check_torch()
    with torch.no_grad():
        w_t = torch.from_numpy(weights.astype(np.float32))
        out_features, in_features = w_t.shape

        # Reshape into groups
        num_groups = (in_features + group_size - 1) // group_size
        padded_in = num_groups * group_size

        if padded_in > in_features:
            w_t = F.pad(w_t, (0, padded_in - in_features))

        w_grouped = w_t.reshape(out_features, num_groups, group_size)

        # Compute per-group min/max
        w_min = w_grouped.min(dim=-1).values
        w_max = w_grouped.max(dim=-1).values

        # INT4 range: 0-15
        scale = (w_max - w_min) / 15.0
        scale = torch.maximum(scale, torch.tensor(1e-8))

        zero_point = -torch.round(w_min / scale)
        zero_point = torch.clamp(zero_point, 0, 15)

        # Quantize
        quantized = torch.round(w_grouped / scale.unsqueeze(-1)) + zero_point.unsqueeze(-1)
        quantized = torch.clamp(quantized, 0, 15).to(torch.uint8)

        # Pack two INT4 values into one INT8
        quantized_flat = quantized.reshape(out_features, -1)
        packed = quantized_flat[:, ::2] | (quantized_flat[:, 1::2] << 4)

        return packed.numpy(), scale.numpy(), zero_point.numpy()


def torch_dequantize_int4(
    W_quant: np.ndarray, scale: np.ndarray, zero_point: np.ndarray,
    group_size: int = 128
) -> np.ndarray:
    """INT4 dequantization reference.

    Args:
        W_quant: Packed INT4 weights.
        scale: Per-group scales of shape (out_features, num_groups).
        zero_point: Per-group zero points.
        group_size: Number of elements per group.

    Returns:
        Dequantized weights.
    """
    _check_torch()
    with torch.no_grad():
        W_packed = torch.from_numpy(W_quant.astype(np.uint8))
        scale_t = torch.from_numpy(scale.astype(np.float32))
        zp_t = torch.from_numpy(zero_point.astype(np.float32))

        out_features = W_packed.shape[0]
        num_groups = scale_t.shape[1]

        # Unpack INT4 values
        low_nibble = W_packed & 0x0F
        high_nibble = (W_packed >> 4) & 0x0F

        # Interleave
        unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(out_features, -1)

        # Reshape into groups
        unpacked = unpacked[:, :num_groups * group_size].reshape(out_features, num_groups, group_size)

        # Dequantize
        W_float = scale_t.unsqueeze(-1) * (unpacked.float() - zp_t.unsqueeze(-1))

        return W_float.reshape(out_features, -1).numpy()


def torch_int8_linear(
    x: np.ndarray, W_quant: np.ndarray,
    scale: np.ndarray, zero_point: np.ndarray, bias: Optional[np.ndarray] = None
) -> np.ndarray:
    """INT8 linear layer reference with dequantization."""
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))

        # Dequantize weights
        W_dequant = torch.from_numpy(
            torch_dequantize_int8(W_quant, scale, zero_point)
        )

        # Linear: x @ W.T
        out = torch.einsum("...d,od->...o", x_t, W_dequant)

        if bias is not None:
            bias_t = torch.from_numpy(bias.astype(np.float32))
            out = out + bias_t

        return out.numpy()


def torch_int4_linear(
    x: np.ndarray, W_quant: np.ndarray,
    scale: np.ndarray, zero_point: np.ndarray, bias: Optional[np.ndarray] = None,
    group_size: int = 128
) -> np.ndarray:
    """INT4 linear layer reference with dequantization."""
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))

        # Dequantize weights
        W_dequant = torch.from_numpy(
            torch_dequantize_int4(W_quant, scale, zero_point, group_size)
        )

        # Adjust for potential padding
        in_features = x_t.shape[-1]
        W_dequant = W_dequant[:, :in_features]

        # Linear: x @ W.T
        out = torch.einsum("...d,od->...o", x_t, W_dequant)

        if bias is not None:
            bias_t = torch.from_numpy(bias.astype(np.float32))
            out = out + bias_t

        return out.numpy()


# =============================================================================
# Primitive Operations (6)
# =============================================================================

def torch_associative_scan_add(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with add (cumsum) reference.

    Args:
        x: Input array.
        axis: Axis along which to scan.

    Returns:
        Cumulative sum as numpy array.
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return torch.cumsum(x_t, dim=axis).numpy()


def torch_associative_scan_mul(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with multiply (cumprod) reference.

    Args:
        x: Input array.
        axis: Axis along which to scan.

    Returns:
        Cumulative product as numpy array.
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        return torch.cumprod(x_t, dim=axis).numpy()


def torch_ssm_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray
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

    Returns:
        Output tensor y of shape (batch, seq, d_inner).
    """
    _check_torch()
    with torch.no_grad():
        A_t = torch.from_numpy(A.astype(np.float32))
        B_t = torch.from_numpy(B.astype(np.float32))
        C_t = torch.from_numpy(C.astype(np.float32))
        D_t = torch.from_numpy(D.astype(np.float32)) if D is not None else None
        x_t = torch.from_numpy(x.astype(np.float32))
        delta_t = torch.from_numpy(delta.astype(np.float32))

        batch_size, seq_len, d_inner = x_t.shape
        d_state = A_t.shape[1]

        # Discretization
        delta_A = delta_t.unsqueeze(-1) * A_t.unsqueeze(0).unsqueeze(0)  # (batch, seq, d_inner, d_state)
        A_bar = torch.exp(delta_A)

        # B_bar_x = delta * B * x
        B_x = B_t.unsqueeze(2) * x_t.unsqueeze(-1)  # (batch, seq, d_inner, d_state)
        B_bar_x = delta_t.unsqueeze(-1) * B_x

        # Sequential scan (PyTorch doesn't have native associative scan)
        h = torch.zeros(batch_size, d_inner, d_state, dtype=x_t.dtype)
        outputs = []

        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar_x[:, t]
            # Output: y = sum(C * h, axis=-1), where C is (batch, d_state)
            # h is (batch, d_inner, d_state)
            y_t = torch.sum(C_t[:, t].unsqueeze(1) * h, dim=-1)  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq, d_inner)

        # Skip connection: D is (d_inner,)
        if D_t is not None:
            y = y + x_t * D_t

        return y.numpy()


def torch_selective_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Mamba-style selective scan reference (alias for torch_ssm_scan)."""
    return torch_ssm_scan(A, B, C, D, x, delta)


def torch_selective_gather(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Selective gather reference.

    Args:
        x: Source array to gather from.
        indices: Indices to gather.

    Returns:
        Gathered values.
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        indices_t = torch.from_numpy(indices.astype(np.int64))
        return x_t[indices_t].numpy()


def torch_selective_scatter_add(
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
    _check_torch()
    with torch.no_grad():
        output_t = torch.from_numpy(output.astype(np.float32)).clone()
        values_t = torch.from_numpy(values.astype(np.float32))
        indices_t = torch.from_numpy(indices.astype(np.int64))

        if weights is not None:
            weights_t = torch.from_numpy(weights.astype(np.float32))
            values_t = values_t * weights_t.unsqueeze(-1)

        # Scatter add
        output_t.index_add_(0, indices_t, values_t)
        return output_t.numpy()


# =============================================================================
# MoE Operations (3)
# =============================================================================

def torch_topk_routing(
    logits: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TopK routing reference.

    Args:
        logits: Router logits of shape (batch, num_experts).
        k: Number of experts to select.

    Returns:
        Tuple of (indices, weights, softmax_probs):
        - indices: Top-k expert indices of shape (batch, k)
        - weights: Normalized weights for top-k experts of shape (batch, k)
        - softmax_probs: Full softmax probabilities of shape (batch, num_experts)
    """
    _check_torch()
    with torch.no_grad():
        logits_t = torch.from_numpy(logits.astype(np.float32))

        # Compute softmax probabilities over all experts
        softmax_probs = F.softmax(logits_t, dim=-1)

        # Get top-k experts
        topk_values, topk_indices = torch.topk(softmax_probs, k, dim=-1)

        # Renormalize weights to sum to 1
        weights = topk_values / topk_values.sum(dim=-1, keepdim=True)

        return topk_indices.numpy(), weights.numpy(), softmax_probs.numpy()


def torch_expert_dispatch(
    x: np.ndarray, expert_indices: np.ndarray, expert_weights: np.ndarray,
    num_experts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Expert dispatch reference.

    Organizes inputs for expert processing by gathering tokens per expert.

    Args:
        x: Input tensor of shape (batch, seq, dim).
        expert_indices: Expert indices of shape (batch, seq, k).
        expert_weights: Expert weights of shape (batch, seq, k).
        num_experts: Total number of experts.

    Returns:
        Tuple of (expert_inputs, dispatch_info):
        - expert_inputs: List of inputs per expert
        - dispatch_info: Information for combining outputs
    """
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        indices_t = torch.from_numpy(expert_indices.astype(np.int64))
        weights_t = torch.from_numpy(expert_weights.astype(np.float32))

        batch, seq, dim = x_t.shape
        k = indices_t.shape[-1]

        # Flatten for easier processing
        x_flat = x_t.reshape(-1, dim)  # (batch*seq, dim)
        indices_flat = indices_t.reshape(-1, k)  # (batch*seq, k)
        weights_flat = weights_t.reshape(-1, k)  # (batch*seq, k)

        # For each expert, gather the tokens assigned to it
        expert_inputs = []
        for e in range(num_experts):
            # Find tokens assigned to this expert (in any of the k slots)
            mask = (indices_flat == e).any(dim=-1)
            expert_tokens = x_flat[mask]
            expert_inputs.append(expert_tokens.numpy())

        return expert_inputs, (batch, seq, k)


def torch_load_balancing_loss(
    router_logits: np.ndarray, expert_mask: np.ndarray
) -> float:
    """Load balancing auxiliary loss reference.

    Computes the auxiliary loss to encourage balanced expert usage.
    Loss = num_experts * sum(f_i * P_i) where:
    - f_i = fraction of tokens routed to expert i
    - P_i = mean probability assigned to expert i

    Args:
        router_logits: Router logits of shape (batch*seq, num_experts).
        expert_mask: Binary mask of shape (batch*seq, num_experts) indicating
                     which experts each token was routed to.

    Returns:
        Scalar load balancing loss.
    """
    _check_torch()
    with torch.no_grad():
        logits_t = torch.from_numpy(router_logits.astype(np.float32))
        mask_t = torch.from_numpy(expert_mask.astype(np.float32))

        num_experts = logits_t.shape[-1]
        num_tokens = logits_t.shape[0]

        # Compute router probabilities
        router_probs = F.softmax(logits_t, dim=-1)

        # f_i: fraction of tokens routed to expert i
        tokens_per_expert = mask_t.sum(dim=0)
        f = tokens_per_expert / num_tokens

        # P_i: mean probability assigned to expert i
        P = router_probs.mean(dim=0)

        # Load balancing loss
        loss = num_experts * (f * P).sum()

        return float(loss.item())


# =============================================================================
# Pooling Operations (7)
# =============================================================================

def torch_adaptive_avg_pool1d(x: np.ndarray, output_size: int) -> np.ndarray:
    """AdaptiveAvgPool1d reference.

    Args:
        x: Input array of shape (batch, channels, length).
        output_size: Target output length.

    Returns:
        Pooled output of shape (batch, channels, output_size).
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    result = F.adaptive_avg_pool1d(x_torch, output_size)
    return result.numpy()


def torch_adaptive_avg_pool2d(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """AdaptiveAvgPool2d reference.

    Args:
        x: Input array of shape (batch, channels, height, width).
        output_size: Target output size as (H, W).

    Returns:
        Pooled output of shape (batch, channels, output_size[0], output_size[1]).
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    result = F.adaptive_avg_pool2d(x_torch, output_size)
    return result.numpy()


def torch_adaptive_max_pool1d(
    x: np.ndarray, output_size: int, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """AdaptiveMaxPool1d reference.

    Args:
        x: Input array of shape (batch, channels, length).
        output_size: Target output length.
        return_indices: If True, return indices of max values.

    Returns:
        Pooled output, optionally with indices.
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    if return_indices:
        result, indices = F.adaptive_max_pool1d(x_torch, output_size, return_indices=True)
        return result.numpy(), indices.numpy()
    result = F.adaptive_max_pool1d(x_torch, output_size)
    return result.numpy()


def torch_adaptive_max_pool2d(
    x: np.ndarray, output_size: Tuple[int, int], return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """AdaptiveMaxPool2d reference.

    Args:
        x: Input array of shape (batch, channels, height, width).
        output_size: Target output size as (H, W).
        return_indices: If True, return indices of max values.

    Returns:
        Pooled output, optionally with indices.
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    if return_indices:
        result, indices = F.adaptive_max_pool2d(x_torch, output_size, return_indices=True)
        return result.numpy(), indices.numpy()
    result = F.adaptive_max_pool2d(x_torch, output_size)
    return result.numpy()


def torch_global_attention_pooling(
    x: np.ndarray,
    attention_weights: Tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Global attention pooling reference.

    Implements attention-weighted pooling: softmax(MLP(x)) @ x

    Args:
        x: Input array of shape (batch, seq, dims).
        attention_weights: Tuple of (W1, W2, bias) for the attention MLP.
            W1: (dims, hidden_dims), W2: (hidden_dims, 1)
        mask: Optional mask of shape (batch, seq).

    Returns:
        Pooled output of shape (batch, dims).
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    W1 = torch.from_numpy(attention_weights[0])
    W2 = torch.from_numpy(attention_weights[1])

    # Attention MLP: Linear -> Tanh -> Linear
    hidden = torch.tanh(x_torch @ W1)  # (batch, seq, hidden_dims)
    scores = hidden @ W2  # (batch, seq, 1)
    scores = scores.squeeze(-1)  # (batch, seq)

    # Apply mask if provided
    if mask is not None:
        mask_torch = torch.from_numpy(mask)
        scores = torch.where(mask_torch, scores, torch.tensor(float("-inf")))

    # Softmax to get weights
    weights = torch.softmax(scores, dim=-1)  # (batch, seq)

    # Weighted average
    result = torch.sum(x_torch * weights.unsqueeze(-1), dim=1)  # (batch, dims)
    return result.numpy()


def torch_gem(x: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """Generalized Mean (GeM) pooling reference.

    Computes: (mean(x^p))^(1/p)

    Args:
        x: Input array of shape (batch, channels, height, width).
        p: Power parameter. p=1 is avg pooling, p->inf is max pooling.
        eps: Small constant for numerical stability.

    Returns:
        Pooled output of shape (batch, channels, 1, 1).
    """
    _check_torch()
    x_torch = torch.from_numpy(x)

    # Clamp to avoid numerical issues with negative values
    x_clamped = x_torch.clamp(min=eps)

    # Compute generalized mean
    x_pow = x_clamped.pow(p)
    mean_pow = x_pow.mean(dim=(2, 3), keepdim=True)
    result = mean_pow.pow(1.0 / p)

    return result.numpy()


def torch_spatial_pyramid_pooling(x: np.ndarray, levels: List[int]) -> np.ndarray:
    """Spatial Pyramid Pooling reference.

    Pools at multiple scales and concatenates results.

    Args:
        x: Input array of shape (batch, channels, height, width).
        levels: List of output sizes for each pyramid level.

    Returns:
        Flattened concatenated output of shape (batch, channels * sum(level^2)).
    """
    _check_torch()
    x_torch = torch.from_numpy(x)
    batch_size, channels = x_torch.shape[:2]

    pooled = []
    for level in levels:
        # Adaptive average pool to (level, level)
        p = F.adaptive_avg_pool2d(x_torch, (level, level))
        # Flatten spatial dimensions
        p = p.reshape(batch_size, -1)
        pooled.append(p)

    # Concatenate all levels
    result = torch.cat(pooled, dim=1)
    return result.numpy()


# =============================================================================
# Embedding Operations (5)
# =============================================================================

def torch_sinusoidal_embedding(positions: np.ndarray, dim: int, base: float = 10000.0) -> np.ndarray:
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
    _check_torch()
    with torch.no_grad():
        pos_t = torch.from_numpy(positions.astype(np.float32))

        # Ensure positions is 1D
        if pos_t.ndim == 0:
            pos_t = pos_t.unsqueeze(0)
        pos_t = pos_t.flatten().unsqueeze(1)  # (seq_len, 1)

        dims_range = torch.arange(0, dim, 2, dtype=torch.float32)  # (dim/2,)
        freqs = base ** (-dims_range / dim)  # (dim/2,)

        angles = pos_t * freqs  # (seq_len, dim/2)

        sin_embeddings = torch.sin(angles)
        cos_embeddings = torch.cos(angles)

        # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
        embeddings = torch.stack([sin_embeddings, cos_embeddings], dim=2)
        embeddings = embeddings.reshape(pos_t.shape[0], dim)

        return embeddings.numpy()


def torch_learned_positional_embedding(positions: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Learned positional embedding reference.

    Simple embedding lookup.

    Args:
        positions: Position indices.
        weight: Embedding weight matrix (max_seq_len, dim).

    Returns:
        Positional embeddings.
    """
    _check_torch()
    with torch.no_grad():
        pos_t = torch.from_numpy(positions.astype(np.int64))
        weight_t = torch.from_numpy(weight.astype(np.float32))
        return weight_t[pos_t].numpy()


def torch_rotary_embedding(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
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
    _check_torch()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        cos_t = torch.from_numpy(cos.astype(np.float32))
        sin_t = torch.from_numpy(sin.astype(np.float32))

        # Split x into pairs
        x1 = x_t[..., ::2]   # Even indices: (..., head_dim//2)
        x2 = x_t[..., 1::2]  # Odd indices: (..., head_dim//2)

        # Expand cos/sin to match x1/x2 shape if needed
        # Typical input: cos/sin are (seq_len, head_dim//2)
        # x1/x2 are (batch, seq_len, num_heads, head_dim//2)
        # Need cos/sin to be (1, seq_len, 1, head_dim//2) for broadcasting
        while cos_t.dim() < x1.dim():
            if cos_t.dim() == 2:  # (seq, dim//2) -> (1, seq, 1, dim//2)
                cos_t = cos_t.unsqueeze(0).unsqueeze(2)
                sin_t = sin_t.unsqueeze(0).unsqueeze(2)
            else:
                cos_t = cos_t.unsqueeze(0)
                sin_t = sin_t.unsqueeze(0)

        # Apply rotation
        rotated_x1 = x1 * cos_t - x2 * sin_t
        rotated_x2 = x1 * sin_t + x2 * cos_t

        # Interleave back
        result = torch.stack([rotated_x1, rotated_x2], dim=-1)
        result = result.reshape(x_t.shape)

        return result.numpy()


def torch_alibi_embedding(seq_len: int, num_heads: int) -> np.ndarray:
    """ALiBi position bias reference.

    Computes Attention with Linear Biases position bias matrix.
    slope_h = 2^(-8h/H) for head h in [1, H]

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.

    Returns:
        ALiBi bias matrix of shape (num_heads, seq_len, seq_len).
    """
    _check_torch()
    with torch.no_grad():
        # Compute slopes: 2^(-8/num_heads) raised to powers 1, 2, ..., num_heads
        base = 2.0 ** (-8.0 / num_heads)
        slopes = torch.tensor([base ** (i + 1) for i in range(num_heads)])  # (num_heads,)

        # Create relative position matrix
        positions = torch.arange(seq_len)
        # Distance: query_pos - key_pos (negative for past positions)
        rel_pos = positions[:, None] - positions[None, :]  # (seq_len, seq_len)

        # Bias = slope * distance (negative distances give negative bias)
        # Shape: (num_heads, seq_len, seq_len)
        bias = slopes[:, None, None] * rel_pos[None, :, :].float()

        return bias.numpy()


def torch_relative_positional_embedding(
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
    _check_torch()
    with torch.no_grad():
        # Compute relative positions
        q_pos = torch.arange(q_len)[:, None]
        k_pos = torch.arange(k_len)[None, :]
        relative_position = k_pos - q_pos  # (q_len, k_len)

        # Bucketing logic (simplified T5 style)
        num_buckets_half = num_buckets // 2

        # Positive positions (future)
        is_positive = relative_position > 0
        relative_position_abs = torch.abs(relative_position)

        # Small distances get individual buckets
        max_exact = num_buckets_half // 2
        is_small = relative_position_abs < max_exact

        # Large distances use log-spaced buckets
        relative_position_if_large = max_exact + (
            torch.log(relative_position_abs.float() / max_exact + 1e-6)
            / np.log(max_distance / max_exact)
            * (num_buckets_half - max_exact)
        ).long()
        relative_position_if_large = torch.minimum(
            relative_position_if_large, torch.tensor(num_buckets_half - 1)
        )

        relative_buckets = torch.where(is_small, relative_position_abs, relative_position_if_large)
        relative_buckets = torch.where(is_positive, relative_buckets + num_buckets_half, relative_buckets)

        return relative_buckets.numpy()


# =============================================================================
# Cache Operations (4)
# =============================================================================

def torch_paged_attention(
    q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
    block_tables: np.ndarray, seq_lens: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """Paged attention reference.

    Implements vLLM-style paged attention by gathering KV from block tables.

    Args:
        q: Query tensor of shape (batch, num_heads, head_dim).
        k_cache: Key cache of shape (num_blocks, block_size, num_heads, head_dim).
        v_cache: Value cache of shape (num_blocks, block_size, num_heads, head_dim).
        block_tables: Block table of shape (batch, max_num_blocks).
        seq_lens: Sequence lengths of shape (batch,).
        block_size: Size of each cache block.

    Returns:
        Attention output of shape (batch, num_heads, head_dim).
    """
    _check_torch()
    with torch.no_grad():
        q_t = torch.from_numpy(q.astype(np.float32))
        k_cache_t = torch.from_numpy(k_cache.astype(np.float32))
        v_cache_t = torch.from_numpy(v_cache.astype(np.float32))
        block_tables_t = torch.from_numpy(block_tables.astype(np.int64))
        seq_lens_t = torch.from_numpy(seq_lens.astype(np.int64))

        batch_size, num_heads, head_dim = q_t.shape
        scale = 1.0 / np.sqrt(head_dim)

        outputs = []
        for b in range(batch_size):
            seq_len = seq_lens_t[b].item()
            num_blocks_needed = (seq_len + block_size - 1) // block_size

            # Gather K and V from cache using block table
            k_list = []
            v_list = []
            for block_idx in range(num_blocks_needed):
                physical_block = block_tables_t[b, block_idx].item()
                k_block = k_cache_t[physical_block]  # (block_size, num_heads, head_dim)
                v_block = v_cache_t[physical_block]
                k_list.append(k_block)
                v_list.append(v_block)

            k_seq = torch.cat(k_list, dim=0)[:seq_len]  # (seq_len, num_heads, head_dim)
            v_seq = torch.cat(v_list, dim=0)[:seq_len]

            # Transpose to (num_heads, seq_len, head_dim)
            k_seq = k_seq.permute(1, 0, 2)
            v_seq = v_seq.permute(1, 0, 2)

            # Compute attention
            q_b = q_t[b]  # (num_heads, head_dim)
            scores = torch.einsum("hd,hsd->hs", q_b, k_seq) * scale
            weights = F.softmax(scores, dim=-1)
            out = torch.einsum("hs,hsd->hd", weights, v_seq)
            outputs.append(out)

        return torch.stack(outputs, dim=0).numpy()


def torch_block_allocation(num_blocks: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Block allocation reference.

    Initializes block allocator state.

    Args:
        num_blocks: Total number of blocks in the cache.
        block_size: Size of each block.

    Returns:
        Tuple of (free_blocks, allocated_blocks):
        - free_blocks: List of free block indices
        - allocated_blocks: Empty list (none allocated initially)
    """
    _check_torch()
    # All blocks start as free
    free_blocks = np.arange(num_blocks)
    allocated_blocks = np.array([], dtype=np.int64)
    return free_blocks, allocated_blocks


def torch_eviction_lru(access_times: np.ndarray, num_to_evict: int) -> np.ndarray:
    """LRU eviction reference.

    Returns indices of blocks to evict based on least recent access.

    Args:
        access_times: Last access time for each block of shape (num_blocks,).
        num_to_evict: Number of blocks to evict.

    Returns:
        Indices of blocks to evict (sorted by oldest access time).
    """
    _check_torch()
    with torch.no_grad():
        times_t = torch.from_numpy(access_times.astype(np.float32))

        # Get indices sorted by access time (ascending = oldest first)
        sorted_indices = torch.argsort(times_t)

        # Return the num_to_evict oldest
        return sorted_indices[:num_to_evict].numpy()


def torch_speculative_verification(
    draft_tokens: np.ndarray, target_probs: np.ndarray, draft_probs: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Speculative decoding verification reference.

    Verifies draft tokens against target model probabilities.

    Args:
        draft_tokens: Draft tokens of shape (num_draft,).
        target_probs: Target model probabilities of shape (num_draft, vocab_size).
        draft_probs: Draft model probabilities of shape (num_draft, vocab_size).

    Returns:
        Tuple of (accepted_tokens, num_accepted):
        - accepted_tokens: Array of accepted token ids
        - num_accepted: Number of tokens accepted
    """
    _check_torch()
    with torch.no_grad():
        draft_t = torch.from_numpy(draft_tokens.astype(np.int64))
        target_p = torch.from_numpy(target_probs.astype(np.float32))
        draft_p = torch.from_numpy(draft_probs.astype(np.float32))

        num_draft = len(draft_tokens)
        accepted = []

        for i in range(num_draft):
            token = draft_t[i].item()

            # Get probabilities for this token
            p_target = target_p[i, token].item()
            p_draft = draft_p[i, token].item()

            # Accept if target prob >= draft prob, or with probability p_target/p_draft
            if p_target >= p_draft:
                accepted.append(token)
            else:
                # Rejection sampling
                r = torch.rand(1).item()
                if r < p_target / max(p_draft, 1e-10):
                    accepted.append(token)
                else:
                    # Reject this and all subsequent tokens
                    break

        return np.array(accepted, dtype=np.int64), len(accepted)


# =============================================================================
# Generation/Sampling Operations (3)
# =============================================================================

def torch_temperature_sampling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled sampling reference.

    Applies temperature scaling to logits before softmax.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Temperature for scaling. Higher = more random.

    Returns:
        Scaled probabilities of shape (..., vocab_size).
    """
    _check_torch()
    with torch.no_grad():
        logits_t = torch.from_numpy(logits.astype(np.float32))
        scaled = logits_t / max(temperature, 1e-10)
        probs = F.softmax(scaled, dim=-1)
        return probs.numpy()


def torch_top_k_sampling(logits: np.ndarray, k: int) -> np.ndarray:
    """Top-K sampling reference.

    Masks all but the top-k logits before softmax.

    Args:
        logits: Logits of shape (..., vocab_size).
        k: Number of top logits to keep.

    Returns:
        Masked probabilities of shape (..., vocab_size).
    """
    _check_torch()
    with torch.no_grad():
        logits_t = torch.from_numpy(logits.astype(np.float32))

        # Get top-k values and mask others
        topk_values, _ = torch.topk(logits_t, k, dim=-1)
        threshold = topk_values[..., -1:]
        mask = logits_t < threshold
        logits_masked = logits_t.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits_masked, dim=-1)
        return probs.numpy()


def torch_top_p_sampling(logits: np.ndarray, p: float) -> np.ndarray:
    """Top-P (nucleus) sampling reference.

    Keeps smallest set of tokens with cumulative probability >= p.

    Args:
        logits: Logits of shape (..., vocab_size).
        p: Cumulative probability threshold (0 < p <= 1).

    Returns:
        Masked probabilities of shape (..., vocab_size).
    """
    _check_torch()
    with torch.no_grad():
        logits_t = torch.from_numpy(logits.astype(np.float32))

        # Sort by descending probability
        sorted_logits, sorted_indices = torch.sort(logits_t, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        # Keep at least one token
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Create mask in original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits_masked = logits_t.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits_masked, dim=-1)
        return probs.numpy()
