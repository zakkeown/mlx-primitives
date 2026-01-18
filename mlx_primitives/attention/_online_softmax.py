"""Online softmax utilities for memory-efficient attention.

The online softmax algorithm allows computing softmax incrementally without
materializing the full attention matrix. This is the key insight behind
Flash Attention and chunked attention implementations.

Key insight: softmax can be computed in a single pass by maintaining:
- m: running maximum of scores (for numerical stability)
- l: running sum of exp(scores - m)
- o: running weighted sum of values (unnormalized)

When processing a new chunk, we can merge with the running statistics:
- m_new = max(m_old, m_chunk)
- l_new = l_old * exp(m_old - m_new) + l_chunk * exp(m_chunk - m_new)
- o_new = (o_old * l_old * exp(m_old - m_new) + o_chunk * exp(m_chunk - m_new)) / l_new
"""

from typing import Tuple

import mlx.core as mx


def online_softmax_merge(
    acc_output: mx.array,
    acc_max: mx.array,
    acc_sum: mx.array,
    new_output: mx.array,
    new_max: mx.array,
    new_sum: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Merge two online softmax accumulators.

    This function combines running statistics from two chunks of attention
    computation, maintaining numerical stability through the online softmax
    algorithm.

    Args:
        acc_output: Accumulated output from previous chunks.
            Shape: (batch, seq_q, heads, dim) - unnormalized weighted values.
        acc_max: Running maximum scores from previous chunks.
            Shape: (batch, seq_q, heads).
        acc_sum: Running sum of exponentials from previous chunks.
            Shape: (batch, seq_q, heads).
        new_output: Output from the new chunk (unnormalized).
            Shape: (batch, seq_q, heads, dim).
        new_max: Maximum scores in the new chunk.
            Shape: (batch, seq_q, heads).
        new_sum: Sum of exponentials in the new chunk.
            Shape: (batch, seq_q, heads).

    Returns:
        Tuple of (merged_output, merged_max, merged_sum) with same shapes.

    Example:
        >>> # After processing chunk 1:
        >>> acc_out, acc_max, acc_sum = process_chunk(q, k1, v1)
        >>>
        >>> # Process chunk 2 and merge:
        >>> new_out, new_max, new_sum = process_chunk(q, k2, v2)
        >>> acc_out, acc_max, acc_sum = online_softmax_merge(
        ...     acc_out, acc_max, acc_sum,
        ...     new_out, new_max, new_sum
        ... )
        >>>
        >>> # Final normalized output:
        >>> output = acc_out / acc_sum[..., None]
    """
    # Compute new global maximum
    global_max = mx.maximum(acc_max, new_max)

    # Correction factors for rescaling
    acc_correction = mx.exp(acc_max - global_max)
    new_correction = mx.exp(new_max - global_max)

    # Update sum of exponentials
    new_total_sum = acc_sum * acc_correction + new_sum * new_correction

    # Update output accumulator
    # Note: outputs are stored as sum(exp(s - local_max) * v), need to rescale
    merged_output = (
        acc_output * (acc_correction * acc_sum)[..., None]
        + new_output * new_correction[..., None]
    ) / (new_total_sum[..., None] + 1e-6)  # FP16-safe epsilon

    return merged_output, global_max, new_total_sum


def compute_chunk_attention(
    q: mx.array,
    k_chunk: mx.array,
    v_chunk: mx.array,
    scale: float,
    causal: bool = False,
    q_offset: int = 0,
    kv_offset: int = 0,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Compute attention for a single KV chunk.

    Returns the unnormalized weighted output along with statistics needed
    for online softmax merge.

    Args:
        q: Query tensor. Shape: (batch, seq_q, heads, dim).
        k_chunk: Key chunk. Shape: (batch, chunk_size, heads, dim).
        v_chunk: Value chunk. Shape: (batch, chunk_size, heads, dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        q_offset: Starting position of queries (for causal masking).
        kv_offset: Starting position of this KV chunk (for causal masking).

    Returns:
        Tuple of:
        - chunk_output: Unnormalized weighted values (batch, seq_q, heads, dim).
        - chunk_max: Maximum scores per query (batch, seq_q, heads).
        - chunk_sum: Sum of exponentials per query (batch, seq_q, heads).
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, chunk_size, _, _ = k_chunk.shape

    # Compute attention scores: Q @ K^T
    # (batch, seq_q, heads, dim) @ (batch, chunk_size, heads, dim)^T
    # -> (batch, seq_q, heads, chunk_size)
    scores = mx.einsum("bqhd,bkhd->bqhk", q, k_chunk) * scale

    # Apply causal mask if needed
    if causal:
        # Create position indices
        q_pos = mx.arange(seq_q) + q_offset  # (seq_q,)
        kv_pos = mx.arange(chunk_size) + kv_offset  # (chunk_size,)

        # Causal mask: kv_pos <= q_pos
        # Shape: (seq_q, chunk_size)
        causal_mask = kv_pos[None, :] <= q_pos[:, None]

        # Apply mask: set future positions to -inf
        scores = mx.where(
            causal_mask[None, :, None, :],  # (1, seq_q, 1, chunk_size)
            scores,
            mx.array(-1e9),
        )

    # Compute local max for numerical stability
    chunk_max = mx.max(scores, axis=-1)  # (batch, seq_q, heads)

    # Stable softmax computation
    scores_stable = scores - chunk_max[..., None]
    exp_scores = mx.exp(scores_stable)
    chunk_sum = mx.sum(exp_scores, axis=-1)  # (batch, seq_q, heads)

    # Compute weighted values (unnormalized)
    # (batch, seq_q, heads, chunk_size) @ (batch, chunk_size, heads, dim)
    # -> (batch, seq_q, heads, dim)
    chunk_output = mx.einsum("bqhk,bkhd->bqhd", exp_scores, v_chunk)

    return chunk_output, chunk_max, chunk_sum
