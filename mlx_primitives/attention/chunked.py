"""Chunked cross-attention for very long KV sequences.

For scenarios where the key-value sequence is extremely long (e.g., attending
to a 100K token document), the standard attention mechanism would require
O(seq_q * seq_kv) memory which may not fit in GPU memory.

Chunked cross-attention processes the KV sequence in chunks, using online
softmax to accumulate results without materializing the full attention matrix.

Memory: O(seq_q * chunk_size) instead of O(seq_q * seq_kv)
Compute: Same O(seq_q * seq_kv * d) as standard attention

This is different from sliding window attention which limits the attention
span. Chunked attention allows full attention span but limits memory usage.

Example:
    >>> # Attending to 100K tokens document
    >>> q = mx.random.normal((1, 512, 8, 64))     # 512 queries
    >>> k = mx.random.normal((1, 100000, 8, 64))  # 100K keys
    >>> v = mx.random.normal((1, 100000, 8, 64))
    >>> out = chunked_cross_attention(q, k, v, chunk_size=4096)
    >>> # Memory: ~O(512 * 4096) instead of O(512 * 100000)
"""

import math
from typing import Optional, Tuple

import mlx.core as mx

from mlx_primitives.attention._online_softmax import (
    compute_chunk_attention,
    online_softmax_merge,
)
from mlx_primitives.constants import ATTENTION_MASK_VALUE, METAL_SOFTMAX_EPSILON

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_chunked_attention_kernel: Optional[mx.fast.metal_kernel] = None


def chunked_cross_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk_size: int = 1024,
    scale: Optional[float] = None,
    causal: bool = False,
    use_metal: bool = True,
    dropout_p: float = 0.0,
    training: bool = True,
) -> mx.array:
    """Cross-attention with chunked KV processing.

    For very long KV sequences that don't fit in memory, processes K/V
    in chunks and accumulates outputs using online softmax for numerical
    stability.

    Args:
        q: Query tensor of shape (batch, seq_q, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_kv, num_heads, head_dim).
            Can be very long (e.g., 100K tokens).
        v: Value tensor of shape (batch, seq_kv, num_heads, head_dim).
        chunk_size: KV chunk size. Larger chunks are faster but use more
            memory. Should fit in GPU memory: memory ~ O(seq_q * chunk_size * d).
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        causal: If True, apply causal masking. For cross-attention between
            different sequences, this is often False.
        use_metal: Use Metal kernel if available.
        dropout_p: Dropout probability on attention weights. Default 0.0 (no dropout).
            When dropout_p > 0, falls back to reference implementation.
        training: If False, dropout is not applied. Default True.

    Returns:
        Output tensor of shape (batch, seq_q, num_heads, head_dim).

    Example:
        >>> # Standard cross-attention to long context
        >>> q = mx.random.normal((1, 512, 8, 64))      # 512 query tokens
        >>> k = mx.random.normal((1, 100000, 8, 64))   # 100K context
        >>> v = mx.random.normal((1, 100000, 8, 64))
        >>> out = chunked_cross_attention(q, k, v, chunk_size=4096)
        >>>
        >>> # With causal masking (for self-attention on long sequences)
        >>> q = mx.random.normal((1, 10000, 8, 64))
        >>> k = q  # Self-attention
        >>> v = q
        >>> out = chunked_cross_attention(q, k, v, chunk_size=2048, causal=True)

    Note:
        - Different from sliding_window_attention which limits attention SPAN
        - This limits attention MEMORY while preserving full attention span
        - For very long sequences, Flash Attention may be more efficient if
          Q and KV have the same length
        - When dropout_p > 0 and training=True, falls back to reference
          implementation since dropout requires materializing attention weights.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors (batch, seq, heads, dim), got {q.ndim}D")

    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Fall back to reference implementation when dropout is enabled
    if dropout_p > 0.0 and training:
        return _reference_cross_attention_with_dropout(q, k, v, scale, causal, dropout_p)

    # For short KV sequences, just use standard attention
    if seq_kv <= chunk_size:
        return _reference_cross_attention(q, k, v, scale, causal)

    # Try Metal kernel for longer sequences
    if use_metal and _HAS_METAL and seq_kv >= 2048:
        try:
            return _metal_chunked_cross_attention(q, k, v, chunk_size, scale, causal)
        except RuntimeError as e:
            # Catch Metal kernel errors, but let programming bugs propagate
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("chunked_cross_attention", e)

    # Python implementation
    return _chunked_cross_attention_impl(q, k, v, chunk_size, scale, causal)


def _chunked_cross_attention_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Python implementation of chunked cross-attention."""
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Initialize accumulators for online softmax
    # output_acc stores unnormalized weighted values
    output_acc = mx.zeros((batch_size, seq_q, num_heads, head_dim))
    max_scores = mx.full((batch_size, seq_q, num_heads), ATTENTION_MASK_VALUE)
    sum_exp = mx.zeros((batch_size, seq_q, num_heads))

    # Process KV in chunks
    for chunk_start in range(0, seq_kv, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_kv)

        k_chunk = k[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end]

        # Compute attention for this chunk
        chunk_output, chunk_max, chunk_sum = compute_chunk_attention(
            q, k_chunk, v_chunk, scale,
            causal=causal,
            q_offset=0,
            kv_offset=chunk_start,
        )

        # Online softmax merge
        output_acc, max_scores, sum_exp = online_softmax_merge(
            output_acc, max_scores, sum_exp,
            chunk_output, chunk_max, chunk_sum,
        )

    # Normalize at the end
    return output_acc / (sum_exp[..., None] + METAL_SOFTMAX_EPSILON)


def _reference_cross_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
) -> mx.array:
    """Reference implementation using standard attention."""
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Attention scores: (batch, heads, seq_q, seq_kv)
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask if needed
    if causal:
        # For cross-attention, causal mask compares absolute positions
        q_pos = mx.arange(seq_q)[:, None]
        kv_pos = mx.arange(seq_kv)[None, :]
        mask = kv_pos <= q_pos
        scores = mx.where(mask[None, None, :, :], scores, ATTENTION_MASK_VALUE)

    # Softmax and output
    weights = mx.softmax(scores, axis=-1)
    output = weights @ v_t

    return output.transpose(0, 2, 1, 3)


def _reference_cross_attention_with_dropout(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> mx.array:
    """Reference cross-attention with dropout support.

    Args:
        q: Query tensor (batch, seq_q, heads, dim).
        k: Key tensor (batch, seq_kv, heads, dim).
        v: Value tensor (batch, seq_kv, heads, dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability (0.0 to 1.0).

    Returns:
        Output tensor (batch, seq_q, heads, dim).
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Attention scores: (batch, heads, seq_q, seq_kv)
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask if needed
    if causal:
        q_pos = mx.arange(seq_q)[:, None]
        kv_pos = mx.arange(seq_kv)[None, :]
        mask = kv_pos <= q_pos
        scores = mx.where(mask[None, None, :, :], scores, ATTENTION_MASK_VALUE)

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Apply dropout to attention weights
    if dropout_p > 0.0:
        dropout_mask = mx.random.bernoulli(1.0 - dropout_p, weights.shape)
        weights = weights * dropout_mask / (1.0 - dropout_p)

    output = weights @ v_t

    return output.transpose(0, 2, 1, 3)


def _get_chunked_attention_kernel() -> mx.fast.metal_kernel:
    """Get or create the chunked cross-attention Metal kernel."""
    global _chunked_attention_kernel
    if _chunked_attention_kernel is None:
        # Note: MLX metal_kernel only supports 1 thread per threadgroup effectively
        # (threads_per_threadgroup.x always returns 1), so we use one threadgroup
        # per query position and stream K/V from global memory.
        source = """
        // Chunked Cross-Attention Metal Kernel
        // Each threadgroup handles one query position
        // Q stays in registers, KV streams through global memory

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_q = seq_q[0];
        uint _seq_kv = seq_kv[0];
        uint _num_heads = num_heads[0];
        uint _head_dim = head_dim[0];
        uint _causal = causal[0];
        float _scale = scale[0];

        // Grid layout: (seq_q, num_heads, batch_size)
        uint batch_idx = threadgroup_position_in_grid.z;
        uint head_idx = threadgroup_position_in_grid.y;
        uint q_pos = threadgroup_position_in_grid.x;

        if (batch_idx >= _batch_size || head_idx >= _num_heads || q_pos >= _seq_q) return;

        // Compute base offsets
        uint qkv_stride = _num_heads * _head_dim;

        uint q_offset = batch_idx * _seq_q * qkv_stride + q_pos * qkv_stride + head_idx * _head_dim;
        uint kv_base = batch_idx * _seq_kv * qkv_stride + head_idx * _head_dim;

        // Initialize online softmax state
        float max_val = -1e38f;
        float sum_exp = 0.0f;
        float acc[132];  // Support up to head_dim=128 + padding
        for (uint d = 0; d < _head_dim; d++) {
            acc[d] = 0.0f;
        }

        // Load Q into registers
        float q_reg[132];
        for (uint d = 0; d < _head_dim; d++) {
            q_reg[d] = Q[q_offset + d];
        }

        // Determine KV range based on causal masking
        uint kv_end_pos = _causal ? (q_pos + 1) : _seq_kv;

        // Process all KV positions
        for (uint kv_pos = 0; kv_pos < kv_end_pos; kv_pos++) {
            uint kv_offset = kv_base + kv_pos * qkv_stride;

            // Compute score: Q @ K^T
            float score = 0.0f;
            for (uint d = 0; d < _head_dim; d++) {
                score += q_reg[d] * K[kv_offset + d];
            }
            score *= _scale;

            // Online softmax update
            if (score > max_val) {
                float ratio = exp(max_val - score);
                sum_exp = sum_exp * ratio + 1.0f;
                for (uint d = 0; d < _head_dim; d++) {
                    acc[d] = acc[d] * ratio + V[kv_offset + d];
                }
                max_val = score;
            } else {
                float weight = exp(score - max_val);
                sum_exp += weight;
                for (uint d = 0; d < _head_dim; d++) {
                    acc[d] += weight * V[kv_offset + d];
                }
            }
        }

        // Normalize and write output
        uint o_offset = batch_idx * _seq_q * qkv_stride + q_pos * qkv_stride + head_idx * _head_dim;
        float inv_sum = 1.0f / (sum_exp + SOFTMAX_EPSf);
        for (uint d = 0; d < _head_dim; d++) {
            O[o_offset + d] = acc[d] * inv_sum;
        }
        """.replace("SOFTMAX_EPS", str(METAL_SOFTMAX_EPSILON))

        _chunked_attention_kernel = mx.fast.metal_kernel(
            name="chunked_cross_attention",
            input_names=[
                "Q", "K", "V",
                "batch_size", "seq_q", "seq_kv", "num_heads", "head_dim",
                "causal", "scale"
            ],
            output_names=["O"],
            source=source,
        )
    return _chunked_attention_kernel


def _metal_chunked_cross_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk_size: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Metal kernel implementation of chunked cross-attention.

    Note: chunk_size parameter is kept for API compatibility but is not used
    by the current kernel implementation. The kernel processes all KV positions
    in a single pass due to MLX metal_kernel limitations (no shared memory).
    The Python fallback (_chunked_cross_attention_impl) does true chunking.
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    kernel = _get_chunked_attention_kernel()

    # Ensure contiguous float32 tensors
    q = mx.contiguous(q.astype(mx.float32))
    k = mx.contiguous(k.astype(mx.float32))
    v = mx.contiguous(v.astype(mx.float32))

    # Prepare scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_q_arr = mx.array([seq_q], dtype=mx.uint32)
    seq_kv_arr = mx.array([seq_kv], dtype=mx.uint32)
    heads_arr = mx.array([num_heads], dtype=mx.uint32)
    dim_arr = mx.array([head_dim], dtype=mx.uint32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Grid: (seq_q, num_heads, batch_size) - one threadgroup per query position
    # Threadgroup: (1, 1, 1) - MLX metal_kernel only supports 1 thread per threadgroup
    outputs = kernel(
        inputs=[
            q, k, v,
            batch_arr, seq_q_arr, seq_kv_arr, heads_arr, dim_arr,
            causal_arr, scale_arr
        ],
        grid=(seq_q, num_heads, batch_size),
        threadgroup=(1, 1, 1),
        output_shapes=[(batch_size, seq_q, num_heads, head_dim)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


class ChunkedCrossAttention:
    """Chunked cross-attention module for long KV sequences.

    A reusable attention module that processes KV in chunks, allowing
    attention to very long sequences without O(n²) memory.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        chunk_size: Size of KV chunks to process at a time.
        causal: Whether to apply causal masking.

    Example:
        >>> # Create module for attending to long documents
        >>> attn = ChunkedCrossAttention(
        ...     num_heads=8, head_dim=64, chunk_size=4096
        ... )
        >>>
        >>> # Query tokens attending to 100K document
        >>> q = mx.random.normal((1, 512, 8, 64))
        >>> doc_k = mx.random.normal((1, 100000, 8, 64))
        >>> doc_v = mx.random.normal((1, 100000, 8, 64))
        >>> out = attn(q, doc_k, doc_v)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        chunk_size: int = 1024,
        causal: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Apply chunked cross-attention.

        Args:
            q: Query tensor (batch, seq_q, num_heads, head_dim).
            k: Key tensor (batch, seq_kv, num_heads, head_dim).
            v: Value tensor (batch, seq_kv, num_heads, head_dim).

        Returns:
            Output tensor of shape (batch, seq_q, num_heads, head_dim).
        """
        return chunked_cross_attention(
            q, k, v,
            chunk_size=self.chunk_size,
            scale=self.scale,
            causal=self.causal,
        )


def estimate_memory_savings(
    seq_q: int,
    seq_kv: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    dtype_bytes: int = 4,
) -> Tuple[int, int, float]:
    """Estimate memory savings from using chunked attention.

    Args:
        seq_q: Query sequence length.
        seq_kv: Key-value sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        chunk_size: Chunk size for chunked attention.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).

    Returns:
        Tuple of (standard_memory_bytes, chunked_memory_bytes, savings_ratio).
    """
    # Standard attention: stores full attention matrix
    standard_attn_matrix = seq_q * seq_kv * num_heads * dtype_bytes
    standard_qkv = 3 * seq_q * num_heads * head_dim * dtype_bytes  # Assuming seq_q == seq_kv typically
    standard_total = standard_attn_matrix + standard_qkv

    # Chunked attention: only stores one chunk at a time
    chunked_attn_matrix = seq_q * chunk_size * num_heads * dtype_bytes
    chunked_qkv = (seq_q + 2 * seq_kv) * num_heads * head_dim * dtype_bytes
    # Plus accumulators
    chunked_accum = seq_q * num_heads * (head_dim + 2) * dtype_bytes
    chunked_total = chunked_attn_matrix + chunked_qkv + chunked_accum

    # For large seq_kv, the main savings come from attention matrix
    # But we can't avoid storing K and V entirely (just access in chunks)
    # The key insight is we don't materialize the full attention matrix

    # Simplified comparison: attention matrix memory
    savings_ratio = standard_attn_matrix / max(chunked_attn_matrix, 1)

    return standard_attn_matrix, chunked_attn_matrix, savings_ratio
