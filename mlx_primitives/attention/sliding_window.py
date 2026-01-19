"""Fused sliding window attention for MLX.

Efficient attention where each query only attends to a local window of keys.
Instead of computing the full O(n²) attention matrix and masking, this
implementation iterates only over the valid KV range for each query position.

Memory: O(n * window_size) instead of O(n²)
Compute: O(n * window_size * d) instead of O(n² * d)

This is useful for:
- Long context models (Mistral, LongFormer local attention)
- Efficient inference with bounded attention span
- Memory-constrained environments
"""

import math
import threading
from typing import Optional

import mlx.core as mx

from mlx_primitives.constants import (
    ATTENTION_MASK_VALUE,
    METAL_ATTENTION_MAX_HEAD_DIM,
    METAL_SOFTMAX_EPSILON,
)
from mlx_primitives.hardware import get_chip_info, get_optimal_attention_config

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache with thread safety
_sliding_window_kernel: Optional[mx.fast.metal_kernel] = None
_sliding_window_lock = threading.Lock()


def _get_sliding_window_kernel() -> mx.fast.metal_kernel:
    """Get or create the sliding window attention kernel (thread-safe)."""
    global _sliding_window_kernel
    if _sliding_window_kernel is None:
        with _sliding_window_lock:
            # Double-check after acquiring lock
            if _sliding_window_kernel is not None:
                return _sliding_window_kernel
            # Simple version - each thread handles one (batch, head, q_pos)
            source = """
        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _num_heads = num_heads[0];
        uint _head_dim = head_dim[0];
        uint _window_size = window_size[0];
        uint _causal = causal[0];
        float _scale = scale[0];

        uint batch_idx = thread_position_in_grid.z;
        uint head_idx = thread_position_in_grid.y;
        uint q_pos = thread_position_in_grid.x;

        if (batch_idx >= _batch_size || head_idx >= _num_heads || q_pos >= _seq_len) return;

        // Compute valid KV range
        uint kv_start = (q_pos >= _window_size) ? (q_pos - _window_size) : 0;
        uint kv_end;
        if (_causal) {
            kv_end = q_pos + 1;
        } else {
            kv_end = (q_pos + _window_size + 1 < _seq_len) ? (q_pos + _window_size + 1) : _seq_len;
        }

        // Compute offsets
        uint qkv_stride = _num_heads * _head_dim;
        uint q_offset = batch_idx * _seq_len * qkv_stride + q_pos * qkv_stride + head_idx * _head_dim;
        uint kv_base = batch_idx * _seq_len * qkv_stride + head_idx * _head_dim;

        // First pass: find max score
        float max_score = -1e38f;
        for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
            uint k_offset = kv_base + kv_pos * qkv_stride;
            float score = 0.0f;
            for (uint d = 0; d < _head_dim; d++) {
                score += Q[q_offset + d] * K[k_offset + d];
            }
            score *= _scale;
            max_score = (score > max_score) ? score : max_score;
        }

        // Second pass: compute softmax and accumulate
        float sum_exp = 0.0f;
        float acc[128];
        for (uint d = 0; d < _head_dim; d++) acc[d] = 0.0f;

        for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
            uint k_offset = kv_base + kv_pos * qkv_stride;
            uint v_offset = kv_base + kv_pos * qkv_stride;

            float score = 0.0f;
            for (uint d = 0; d < _head_dim; d++) {
                score += Q[q_offset + d] * K[k_offset + d];
            }
            score *= _scale;

            float weight = exp(score - max_score);
            sum_exp += weight;

            for (uint d = 0; d < _head_dim; d++) {
                acc[d] += weight * V[v_offset + d];
            }
        }

        // Write normalized output
        float inv_sum = 1.0f / (sum_exp + SOFTMAX_EPSf);
        for (uint d = 0; d < _head_dim; d++) {
            O[q_offset + d] = acc[d] * inv_sum;
        }
        """.replace("SOFTMAX_EPS", str(METAL_SOFTMAX_EPSILON))
        _sliding_window_kernel = mx.fast.metal_kernel(
            name="sliding_window_attention",
            input_names=[
                "Q", "K", "V",
                "batch_size", "seq_len", "num_heads", "head_dim",
                "window_size", "causal", "scale"
            ],
            output_names=["O"],
            source=source,
        )
    return _sliding_window_kernel


def sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: Optional[float] = None,
    causal: bool = True,
    use_metal: bool = True,
    dropout_p: float = 0.0,
    training: bool = True,
) -> mx.array:
    """Fused sliding window attention.

    Computes attention where each query only attends to keys within a
    local window, avoiding O(n²) memory for long sequences.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        v: Value tensor of shape (batch, seq_len, num_heads, head_dim).
        window_size: One-sided window size. Each query attends to
            [pos - window_size, pos + window_size] (or [pos - window_size, pos]
            if causal).
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        causal: If True, queries can only attend to earlier positions.
        use_metal: Use Metal kernel if available.
        dropout_p: Dropout probability on attention weights. Default 0.0 (no dropout).
            When dropout_p > 0, falls back to reference implementation.
        training: If False, dropout is not applied. Default True.

    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim).

    Example:
        >>> q = mx.random.normal((1, 1024, 8, 64))
        >>> k = mx.random.normal((1, 1024, 8, 64))
        >>> v = mx.random.normal((1, 1024, 8, 64))
        >>> out = sliding_window_attention(q, k, v, window_size=128)
        >>> # Each position attends to at most 128 previous positions

    Note:
        When dropout_p > 0 and training=True, falls back to reference
        implementation since dropout requires materializing attention weights.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors (batch, seq, heads, dim), got {q.ndim}D")

    batch_size, seq_len, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Fall back to reference implementation when dropout is enabled
    if dropout_p > 0.0 and training:
        return _reference_sliding_window_attention_with_dropout(
            q, k, v, window_size, scale, causal, dropout_p
        )

    # For short sequences or small windows, use Metal kernel
    if use_metal and _HAS_METAL and seq_len >= 32:
        try:
            return _metal_sliding_window_attention(
                q, k, v, window_size, scale, causal
            )
        except RuntimeError as e:
            # Catch Metal kernel errors, but let programming bugs propagate
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("sliding_window_attention", e)

    # Reference implementation using masked attention
    return _reference_sliding_window_attention(
        q, k, v, window_size, scale, causal
    )


def _metal_sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Metal kernel implementation of sliding window attention."""
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Validate head_dim against Metal kernel limits
    if head_dim > METAL_ATTENTION_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={head_dim} exceeds Metal kernel limit of "
            f"{METAL_ATTENTION_MAX_HEAD_DIM}. Use use_metal=False for Python fallback."
        )

    kernel = _get_sliding_window_kernel()

    # Ensure contiguous float32 tensors
    q = mx.contiguous(q.astype(mx.float32))
    k = mx.contiguous(k.astype(mx.float32))
    v = mx.contiguous(v.astype(mx.float32))

    # Prepare scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    heads_arr = mx.array([num_heads], dtype=mx.uint32)
    dim_arr = mx.array([head_dim], dtype=mx.uint32)
    window_arr = mx.array([window_size], dtype=mx.uint32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Grid: (seq_len, num_heads, batch_size)
    outputs = kernel(
        inputs=[q, k, v, batch_arr, seq_arr, heads_arr, dim_arr, window_arr, causal_arr, scale_arr],
        grid=(seq_len, num_heads, batch_size),
        threadgroup=(min(seq_len, 64), 1, 1),
        output_shapes=[(batch_size, seq_len, num_heads, head_dim)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Reference implementation using dense mask."""
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Create sliding window mask
    # mask[i, j] = True if j is within window of i
    positions = mx.arange(seq_len)
    row_positions = positions[:, None]  # (seq_len, 1)
    col_positions = positions[None, :]  # (1, seq_len)

    # Window mask: |i - j| <= window_size
    distance = mx.abs(row_positions - col_positions)
    mask = distance <= window_size

    if causal:
        # Also apply causal mask: j <= i
        causal_mask = col_positions <= row_positions
        mask = mask & causal_mask

    # Convert to attention mask (0 for valid, -inf for masked)
    attn_mask = mx.where(mask, 0.0, ATTENTION_MASK_VALUE)

    # Reshape for batched matmul: (batch, heads, seq, dim)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Compute attention scores
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (batch, heads, seq, seq)

    # Apply mask
    scores = scores + attn_mask[None, None, :, :]

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum of values
    out = weights @ v  # (batch, heads, seq, dim)

    # Reshape back: (batch, seq, heads, dim)
    return out.transpose(0, 2, 1, 3)


def _reference_sliding_window_attention_with_dropout(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> mx.array:
    """Reference sliding window attention with dropout support.

    Args:
        q: Query tensor (batch, seq, heads, dim).
        k: Key tensor (batch, seq, heads, dim).
        v: Value tensor (batch, seq, heads, dim).
        window_size: One-sided window size.
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability (0.0 to 1.0).

    Returns:
        Output tensor (batch, seq, heads, dim).
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Create sliding window mask
    positions = mx.arange(seq_len)
    row_positions = positions[:, None]
    col_positions = positions[None, :]

    # Window mask: |i - j| <= window_size
    distance = mx.abs(row_positions - col_positions)
    mask = distance <= window_size

    if causal:
        causal_mask = col_positions <= row_positions
        mask = mask & causal_mask

    # Convert to attention mask
    attn_mask = mx.where(mask, 0.0, ATTENTION_MASK_VALUE)

    # Reshape for batched matmul
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Compute attention scores
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    scores = scores + attn_mask[None, None, :, :]

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Apply dropout to attention weights
    if dropout_p > 0.0:
        dropout_mask = mx.random.bernoulli(1.0 - dropout_p, weights.shape)
        weights = weights * dropout_mask / (1.0 - dropout_p)

    # Weighted sum of values
    out = weights @ v

    return out.transpose(0, 2, 1, 3)


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> mx.array:
    """Create a sliding window attention mask.

    Args:
        seq_len: Sequence length.
        window_size: One-sided window size.
        causal: If True, apply causal masking.

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True indicates
        positions that can be attended to.
    """
    positions = mx.arange(seq_len)
    row_positions = positions[:, None]
    col_positions = positions[None, :]

    # Window constraint
    distance = mx.abs(row_positions - col_positions)
    mask = distance <= window_size

    if causal:
        causal_mask = col_positions <= row_positions
        mask = mask & causal_mask

    return mask


class SlidingWindowAttention:
    """Sliding window attention module.

    A reusable attention module with sliding window masking.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        window_size: One-sided window size.
        causal: Whether to use causal masking.

    Example:
        >>> attn = SlidingWindowAttention(num_heads=8, head_dim=64, window_size=128)
        >>> q = mx.random.normal((2, 1024, 8, 64))
        >>> k = mx.random.normal((2, 1024, 8, 64))
        >>> v = mx.random.normal((2, 1024, 8, 64))
        >>> out = attn(q, k, v)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        window_size: int,
        causal: bool = True,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Apply sliding window attention.

        Args:
            q: Query tensor (batch, seq, num_heads, head_dim).
            k: Key tensor (batch, seq, num_heads, head_dim).
            v: Value tensor (batch, seq, num_heads, head_dim).

        Returns:
            Output tensor of same shape as q.
        """
        return sliding_window_attention(
            q, k, v,
            window_size=self.window_size,
            scale=self.scale,
            causal=self.causal,
        )
