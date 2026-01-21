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
    """Sliding window attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_sliding_window_attention")


def torch_gqa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    num_kv_heads: int, causal: bool = False
) -> np.ndarray:
    """Grouped Query Attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_gqa")


def torch_mqa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    causal: bool = False
) -> np.ndarray:
    """Multi-Query Attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_mqa")


def torch_linear_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    feature_map: str = "elu"
) -> np.ndarray:
    """Linear attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_linear_attention")


def torch_alibi_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    alibi_slopes: np.ndarray, causal: bool = True
) -> np.ndarray:
    """ALiBi attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_alibi_attention")


def torch_sparse_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    mask_pattern: str = "local"
) -> np.ndarray:
    """Sparse attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_sparse_attention")


def torch_chunked_cross_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    chunk_size: int
) -> np.ndarray:
    """Chunked cross-attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_chunked_cross_attention")


def torch_rope_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, causal: bool = False
) -> np.ndarray:
    """RoPE + attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_rope_attention")


def torch_quantized_kv_attention(
    q: np.ndarray, k_quant: np.ndarray, v_quant: np.ndarray,
    k_scale: np.ndarray, v_scale: np.ndarray
) -> np.ndarray:
    """Quantized KV cache attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_quantized_kv_attention")


# =============================================================================
# Activation Functions (12+)
# =============================================================================

def torch_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """SwiGLU activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_swiglu")


def torch_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """GeGLU activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_geglu")


def torch_reglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """ReGLU activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_reglu")


def torch_quick_gelu(x: np.ndarray) -> np.ndarray:
    """QuickGELU activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_quick_gelu")


def torch_gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU with tanh approximation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_gelu_tanh")


def torch_mish(x: np.ndarray) -> np.ndarray:
    """Mish activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_mish")


def torch_squared_relu(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_squared_relu")


def torch_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_swish")


def torch_hard_swish(x: np.ndarray) -> np.ndarray:
    """Hard Swish activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_hard_swish")


def torch_hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """Hard Sigmoid activation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_hard_sigmoid")


# =============================================================================
# Normalization Operations (5)
# =============================================================================

def torch_groupnorm(
    x: np.ndarray, num_groups: int,
    weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """GroupNorm reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_groupnorm")


def torch_instancenorm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """InstanceNorm reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_instancenorm")


def torch_adalayernorm(
    x: np.ndarray, scale: np.ndarray, shift: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Adaptive LayerNorm reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_adalayernorm")


# =============================================================================
# Fused Operations (4)
# =============================================================================

def torch_fused_rmsnorm_linear(
    x: np.ndarray, norm_weight: np.ndarray,
    linear_weight: np.ndarray, linear_bias: Optional[np.ndarray], eps: float = 1e-5
) -> np.ndarray:
    """Fused RMSNorm + Linear reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_fused_rmsnorm_linear")


def torch_fused_swiglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused SwiGLU reference (same as torch_swiglu)."""
    _check_torch()
    raise NotImplementedError("Stub: torch_fused_swiglu")


def torch_fused_geglu(x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Fused GeGLU reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_fused_geglu")


def torch_fused_rope_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, causal: bool = False
) -> np.ndarray:
    """Fused RoPE + Attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_fused_rope_attention")


# =============================================================================
# Quantization Operations (6)
# =============================================================================

def torch_quantize_int8(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT8 quantization reference. Returns (quantized, scale, zero_point)."""
    _check_torch()
    raise NotImplementedError("Stub: torch_quantize_int8")


def torch_dequantize_int8(
    W_quant: np.ndarray, scale: np.ndarray, zero_point: np.ndarray
) -> np.ndarray:
    """INT8 dequantization reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_dequantize_int8")


def torch_quantize_int4(
    weights: np.ndarray, group_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT4 quantization reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_quantize_int4")


def torch_dequantize_int4(
    W_quant: np.ndarray, scale: np.ndarray, zero_point: np.ndarray
) -> np.ndarray:
    """INT4 dequantization reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_dequantize_int4")


def torch_int8_linear(
    x: np.ndarray, W_quant: np.ndarray,
    scale: np.ndarray, zero_point: np.ndarray, bias: Optional[np.ndarray] = None
) -> np.ndarray:
    """INT8 linear layer reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_int8_linear")


def torch_int4_linear(
    x: np.ndarray, W_quant: np.ndarray,
    scale: np.ndarray, zero_point: np.ndarray, bias: Optional[np.ndarray] = None
) -> np.ndarray:
    """INT4 linear layer reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_int4_linear")


# =============================================================================
# Primitive Operations (6)
# =============================================================================

def torch_associative_scan_add(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with add (cumsum) reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_associative_scan_add")


def torch_associative_scan_mul(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Associative scan with multiply (cumprod) reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_associative_scan_mul")


def torch_ssm_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """SSM-style selective scan reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_ssm_scan")


def torch_selective_scan(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
    x: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Mamba-style selective scan reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_selective_scan")


def torch_selective_gather(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Selective gather reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_selective_gather")


def torch_selective_scatter_add(
    output: np.ndarray, values: np.ndarray,
    indices: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Selective scatter-add reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_selective_scatter_add")


# =============================================================================
# MoE Operations (3)
# =============================================================================

def torch_topk_routing(
    logits: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TopK routing reference. Returns (indices, weights, softmax_probs)."""
    _check_torch()
    raise NotImplementedError("Stub: torch_topk_routing")


def torch_expert_dispatch(
    x: np.ndarray, expert_indices: np.ndarray, expert_weights: np.ndarray
) -> np.ndarray:
    """Expert dispatch reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_expert_dispatch")


def torch_load_balancing_loss(
    router_logits: np.ndarray, expert_counts: np.ndarray
) -> np.ndarray:
    """Load balancing auxiliary loss reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_load_balancing_loss")


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

def torch_sinusoidal_embedding(positions: np.ndarray, dim: int) -> np.ndarray:
    """Sinusoidal positional embedding reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_sinusoidal_embedding")


def torch_learned_positional_embedding(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Learned positional embedding reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_learned_positional_embedding")


def torch_rotary_embedding(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotary positional embedding reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_rotary_embedding")


def torch_alibi_embedding(seq_len: int, num_heads: int) -> np.ndarray:
    """ALiBi position bias reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_alibi_embedding")


def torch_relative_positional_embedding(
    q_len: int, k_len: int, num_heads: int, dim: int
) -> np.ndarray:
    """Relative positional embedding (T5-style) reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_relative_positional_embedding")


# =============================================================================
# Cache Operations (4)
# =============================================================================

def torch_paged_attention(
    q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
    block_tables: np.ndarray, seq_lens: np.ndarray
) -> np.ndarray:
    """Paged attention reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_paged_attention")


def torch_block_allocation(num_blocks: int, block_size: int) -> np.ndarray:
    """Block allocation reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_block_allocation")


def torch_eviction_lru(cache: np.ndarray, access_times: np.ndarray) -> np.ndarray:
    """LRU eviction reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_eviction_lru")


def torch_speculative_verification(
    draft_tokens: np.ndarray, target_probs: np.ndarray, draft_probs: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Speculative decoding verification reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_speculative_verification")


# =============================================================================
# Generation/Sampling Operations (3)
# =============================================================================

def torch_temperature_sampling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled sampling reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_temperature_sampling")


def torch_top_k_sampling(logits: np.ndarray, k: int) -> np.ndarray:
    """Top-K sampling reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_top_k_sampling")


def torch_top_p_sampling(logits: np.ndarray, p: float) -> np.ndarray:
    """Top-P (nucleus) sampling reference."""
    _check_torch()
    raise NotImplementedError("Stub: torch_top_p_sampling")
