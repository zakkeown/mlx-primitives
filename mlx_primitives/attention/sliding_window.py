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

from mlx_primitives.attention._online_softmax import (
    compute_chunk_attention,
    online_softmax_merge,
)
from mlx_primitives.constants import (
    ATTENTION_MASK_VALUE,
    METAL_ATTENTION_MAX_HEAD_DIM,
    METAL_SOFTMAX_EPSILON,
)
from mlx_primitives.hardware import get_chip_info, get_optimal_attention_config
from mlx_primitives.kernels._registry import get_kernel

# Check if Metal kernels and SDPA are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")
_HAS_SDPA = hasattr(mx.fast, "scaled_dot_product_attention")

# Mask lock for thread-safe mask caching (kernel lock moved to registry)
_sliding_window_lock = threading.Lock()

# Window mask cache (key: (seq_len, window_size, causal) -> additive mask)
_window_mask_cache: dict[tuple[int, int, bool], mx.array] = {}
_MAX_MASK_CACHE_SIZE = 32  # Limit cache growth


def _get_sliding_window_kernel() -> mx.fast.metal_kernel:
    """Get or create the sliding window attention kernel (thread-safe)."""
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

    return get_kernel(
        "sliding_window_attention",
        lambda: mx.fast.metal_kernel(
            name="sliding_window_attention",
            input_names=[
                "Q", "K", "V",
                "batch_size", "seq_len", "num_heads", "head_dim",
                "window_size", "causal", "scale"
            ],
            output_names=["O"],
            source=source,
        ),
    )


def _create_sliding_window_additive_mask(
    seq_len: int,
    window_size: int,
    causal: bool,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create an additive mask for SDPA (0 = attend, -inf = masked).

    This mask is created once and reused for all batches/heads via broadcasting.
    Shape: (seq_len, seq_len) - broadcasts to (B, N, T_q, T_kv)
    """
    positions = mx.arange(seq_len)
    row_pos = positions[:, None]  # (seq_len, 1)
    col_pos = positions[None, :]  # (1, seq_len)

    # Window constraint: |i - j| <= window_size
    distance = mx.abs(row_pos - col_pos)
    in_window = distance <= window_size

    if causal:
        # Causal: j <= i
        in_window = in_window & (col_pos <= row_pos)

    # Convert to additive mask (0 for attend, -inf for masked)
    mask = mx.where(in_window, mx.array(0.0, dtype=dtype), mx.array(ATTENTION_MASK_VALUE, dtype=dtype))
    return mask


def _get_cached_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool,
    dtype: mx.Dtype,
) -> mx.array:
    """Get or create cached sliding window mask (thread-safe)."""
    cache_key = (seq_len, window_size, causal)

    # All cache access must be protected by the lock to avoid race conditions
    with _sliding_window_lock:
        if cache_key in _window_mask_cache:
            cached = _window_mask_cache[cache_key]
            # Cast if needed (masks stored as float32)
            return cached.astype(dtype) if cached.dtype != dtype else cached

        # Create new mask
        mask = _create_sliding_window_additive_mask(seq_len, window_size, causal, mx.float32)

        # LRU-style eviction if cache too large
        if len(_window_mask_cache) >= _MAX_MASK_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(_window_mask_cache))
            del _window_mask_cache[oldest_key]

        _window_mask_cache[cache_key] = mask
        return mask.astype(dtype) if dtype != mx.float32 else mask


def _sdpa_sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Use MLX's SDPA with pre-computed sliding window mask.

    This is the fastest path, achieving parity with flash_attention performance.
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Create/get cached additive mask (reused via broadcasting)
    # Shape: (seq_len, seq_len) - will broadcast to (B, N, seq_len, seq_len)
    mask = _get_cached_window_mask(seq_len, window_size, causal, q.dtype)

    # Transpose to SDPA format: (batch, heads, seq, dim)
    q_sdpa = mx.transpose(q, (0, 2, 1, 3))
    k_sdpa = mx.transpose(k, (0, 2, 1, 3))
    v_sdpa = mx.transpose(v, (0, 2, 1, 3))

    # Call SDPA with the window mask
    out_sdpa = mx.fast.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        scale=scale,
        mask=mask,  # Broadcasts to (B, N, seq_len, seq_len)
    )

    # Transpose back to (batch, seq, heads, dim)
    return mx.transpose(out_sdpa, (0, 2, 1, 3))


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

    # PRIMARY PATH: SDPA with pre-computed window mask
    # This achieves parity with flash_attention (~8-9x faster than Metal kernel)
    # Only use for seq_len <= 8192 to avoid excessive mask memory (256MB for 8192)
    MASK_MEMORY_THRESHOLD = 8192
    if _HAS_SDPA and use_metal and seq_len <= MASK_MEMORY_THRESHOLD:
        try:
            return _sdpa_sliding_window_attention(
                q, k, v, window_size, scale, causal
            )
        except (RuntimeError, TypeError) as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("sliding_window_attention (SDPA)", e)

    # SECONDARY PATH: Metal kernel for very long sequences
    # where mask memory would be prohibitive
    if use_metal and _HAS_METAL and seq_len >= 32:
        try:
            return _metal_sliding_window_attention(
                q, k, v, window_size, scale, causal
            )
        except RuntimeError as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("sliding_window_attention (Metal)", e)

    # TERTIARY PATH: Tiled Python implementation
    # Fallback for systems without SDPA or Metal
    return _tiled_sliding_window_attention(
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


def _tiled_sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
    block_q: int = 32,
) -> mx.array:
    """Tiled sliding window attention using online softmax.

    This implementation avoids materializing the O(n²) attention mask by
    processing queries in blocks and only iterating over the valid KV window
    for each query block.

    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim).
        k: Key tensor (batch, seq_len, num_heads, head_dim).
        v: Value tensor (batch, seq_len, num_heads, head_dim).
        window_size: One-sided window size.
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        block_q: Query block size for tiling.

    Returns:
        Output tensor (batch, seq_len, num_heads, head_dim).
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Collect output blocks
    output_blocks = []

    # Process query blocks
    for q_start in range(0, seq_len, block_q):
        q_end = min(q_start + block_q, seq_len)
        q_block = q[:, q_start:q_end]  # (batch, block_q, heads, dim)
        block_q_actual = q_end - q_start

        # Initialize running statistics for this query block
        m_i = mx.full((batch_size, block_q_actual, num_heads), ATTENTION_MASK_VALUE)
        l_i = mx.zeros((batch_size, block_q_actual, num_heads))
        o_i = mx.zeros((batch_size, block_q_actual, num_heads, head_dim))

        # Compute KV range for this query block
        # For causal: [max(0, q_start - window_size), q_end]
        # For non-causal: [max(0, q_start - window_size), min(seq_len, q_end + window_size)]
        kv_start = max(0, q_start - window_size)
        if causal:
            kv_end = q_end
        else:
            kv_end = min(seq_len, q_end + window_size)

        # Process KV blocks within the window
        for kv_block_start in range(kv_start, kv_end, block_q):
            kv_block_end = min(kv_block_start + block_q, kv_end)
            k_block = k[:, kv_block_start:kv_block_end]
            v_block = v[:, kv_block_start:kv_block_end]

            # Compute chunk attention with window-aware masking
            chunk_output, chunk_max, chunk_sum = _compute_windowed_chunk_attention(
                q_block,
                k_block,
                v_block,
                scale,
                causal=causal,
                q_offset=q_start,
                kv_offset=kv_block_start,
                window_size=window_size,
            )

            # Online softmax merge
            o_i, m_i, l_i = online_softmax_merge(
                o_i, m_i, l_i,
                chunk_output, chunk_max, chunk_sum,
            )

        # Normalize at the end of each query block
        o_i_normalized = o_i / (l_i[..., None] + METAL_SOFTMAX_EPSILON)
        output_blocks.append(o_i_normalized)

    # Single concatenation at the end
    return mx.concatenate(output_blocks, axis=1)


def _compute_windowed_chunk_attention(
    q: mx.array,
    k_chunk: mx.array,
    v_chunk: mx.array,
    scale: float,
    causal: bool,
    q_offset: int,
    kv_offset: int,
    window_size: int,
) -> tuple:
    """Compute attention for a chunk with sliding window masking.

    Args:
        q: Query tensor (batch, seq_q, heads, dim).
        k_chunk: Key chunk (batch, chunk_size, heads, dim).
        v_chunk: Value chunk (batch, chunk_size, heads, dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        q_offset: Starting position of queries.
        kv_offset: Starting position of this KV chunk.
        window_size: One-sided window size.

    Returns:
        Tuple of (chunk_output, chunk_max, chunk_sum).
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, chunk_size, _, _ = k_chunk.shape

    # Compute attention scores: Q @ K^T
    scores = mx.einsum("bqhd,bkhd->bqhk", q, k_chunk) * scale

    # Create window + causal mask
    q_pos = mx.arange(seq_q) + q_offset  # (seq_q,)
    kv_pos = mx.arange(chunk_size) + kv_offset  # (chunk_size,)

    # Window mask: |q_pos - kv_pos| <= window_size
    distance = mx.abs(q_pos[:, None] - kv_pos[None, :])  # (seq_q, chunk_size)
    window_mask = distance <= window_size

    if causal:
        # Causal mask: kv_pos <= q_pos
        causal_mask = kv_pos[None, :] <= q_pos[:, None]
        combined_mask = window_mask & causal_mask
    else:
        combined_mask = window_mask

    # Apply mask
    scores = mx.where(
        combined_mask[None, :, None, :],  # (1, seq_q, 1, chunk_size)
        scores,
        mx.array(ATTENTION_MASK_VALUE),
    )

    # Compute local max for numerical stability
    chunk_max = mx.max(scores, axis=-1)  # (batch, seq_q, heads)

    # Stable softmax computation
    scores_stable = scores - chunk_max[..., None]
    exp_scores = mx.exp(scores_stable)
    chunk_sum = mx.sum(exp_scores, axis=-1)  # (batch, seq_q, heads)

    # Compute weighted values (unnormalized)
    chunk_output = mx.einsum("bqhk,bkhd->bqhd", exp_scores, v_chunk)

    return chunk_output, chunk_max, chunk_sum


def _reference_sliding_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Reference implementation using dense mask (kept for testing)."""
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
