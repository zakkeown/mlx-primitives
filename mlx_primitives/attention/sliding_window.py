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
from typing import Optional

import mlx.core as mx

from mlx_primitives.hardware import get_chip_info, get_optimal_attention_config

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_sliding_window_kernel: Optional[mx.fast.metal_kernel] = None


def _get_sliding_window_kernel() -> mx.fast.metal_kernel:
    """Get or create the sliding window attention kernel."""
    global _sliding_window_kernel
    if _sliding_window_kernel is None:
        # Simple version - each thread handles one (batch, head, q_pos)
        source = """
        uint batch_idx = thread_position_in_grid.z;
        uint head_idx = thread_position_in_grid.y;
        uint q_pos = thread_position_in_grid.x;

        if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len) return;

        // Compute valid KV range
        uint kv_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;
        uint kv_end;
        if (causal) {
            kv_end = q_pos + 1;
        } else {
            kv_end = (q_pos + window_size + 1 < seq_len) ? (q_pos + window_size + 1) : seq_len;
        }

        // Compute offsets
        uint qkv_stride = num_heads * head_dim;
        uint q_offset = batch_idx * seq_len * qkv_stride + q_pos * qkv_stride + head_idx * head_dim;
        uint kv_base = batch_idx * seq_len * qkv_stride + head_idx * head_dim;

        // First pass: find max score
        float max_score = -1e38f;
        for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
            uint k_offset = kv_base + kv_pos * qkv_stride;
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += Q[q_offset + d] * K[k_offset + d];
            }
            score *= scale;
            max_score = (score > max_score) ? score : max_score;
        }

        // Second pass: compute softmax and accumulate
        float sum_exp = 0.0f;
        float acc[128];
        for (uint d = 0; d < head_dim; d++) acc[d] = 0.0f;

        for (uint kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
            uint k_offset = kv_base + kv_pos * qkv_stride;
            uint v_offset = kv_base + kv_pos * qkv_stride;

            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += Q[q_offset + d] * K[k_offset + d];
            }
            score *= scale;

            float weight = exp(score - max_score);
            sum_exp += weight;

            for (uint d = 0; d < head_dim; d++) {
                acc[d] += weight * V[v_offset + d];
            }
        }

        // Write normalized output
        float inv_sum = 1.0f / sum_exp;
        for (uint d = 0; d < head_dim; d++) {
            O[q_offset + d] = acc[d] * inv_sum;
        }
        """
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

    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim).

    Example:
        >>> q = mx.random.normal((1, 1024, 8, 64))
        >>> k = mx.random.normal((1, 1024, 8, 64))
        >>> v = mx.random.normal((1, 1024, 8, 64))
        >>> out = sliding_window_attention(q, k, v, window_size=128)
        >>> # Each position attends to at most 128 previous positions
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors (batch, seq, heads, dim), got {q.ndim}D")

    batch_size, seq_len, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # For short sequences or small windows, use Metal kernel
    if use_metal and _HAS_METAL and seq_len >= 32:
        try:
            return _metal_sliding_window_attention(
                q, k, v, window_size, scale, causal
            )
        except Exception as e:
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

    kernel = _get_sliding_window_kernel()

    # Ensure contiguous float32 tensors
    q = mx.ascontiguousarray(q.astype(mx.float32))
    k = mx.ascontiguousarray(k.astype(mx.float32))
    v = mx.ascontiguousarray(v.astype(mx.float32))

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
    attn_mask = mx.where(mask, 0.0, -1e9)

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
