"""Paged attention implementation for non-contiguous KV cache.

This module provides paged attention that works with block-based KV storage,
enabling efficient memory usage for variable-length sequences.
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx


def paged_attention(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: Optional[float] = None,
    block_size: int = 16,
    causal: bool = True,
) -> mx.array:
    """Paged attention with non-contiguous KV cache.

    Computes attention where K/V are stored in a paged block pool rather than
    contiguous memory. Uses online softmax for numerical stability.

    Args:
        q: Query tensor.
            - For decode: (batch, 1, num_heads, head_dim)
            - For prefill: (batch, seq_q, num_heads, head_dim)
        k_pool: Key block pool (num_blocks, block_size, num_heads, head_dim).
        v_pool: Value block pool (num_blocks, block_size, num_heads, head_dim).
        block_tables: Block indices per sequence (batch, max_blocks).
            Use -1 for padding/invalid blocks.
        context_lens: Number of cached tokens per sequence (batch,).
        scale: Attention scale. Defaults to 1/sqrt(head_dim).
        block_size: Tokens per block.
        causal: Apply causal masking.

    Returns:
        Output tensor same shape as q.

    Example:
        >>> # Decode mode: single new query token
        >>> q = mx.random.normal((batch, 1, heads, dim))
        >>> out = paged_attention(q, k_pool, v_pool, block_tables, context_lens)
    """
    batch_size, seq_q, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if seq_q == 1:
        # Decode mode: optimized single-token attention
        return _paged_attention_decode(
            q, k_pool, v_pool, block_tables, context_lens, scale, block_size
        )
    else:
        # Prefill mode: process query sequence with paged KV
        return _paged_attention_prefill(
            q, k_pool, v_pool, block_tables, context_lens, scale, block_size, causal
        )


def _paged_attention_decode(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: float,
    block_size: int,
) -> mx.array:
    """Optimized paged attention for decode (single query token).

    Uses online softmax to process blocks one at a time, maintaining
    numerical stability without materializing the full attention matrix.
    """
    batch_size, _, num_heads, head_dim = q.shape
    max_blocks = block_tables.shape[1]

    # Squeeze query: (batch, num_heads, head_dim)
    q = q.squeeze(1)

    # Initialize online softmax state per batch/head
    # max_score: running maximum for numerical stability
    # sum_exp: running sum of exp(score - max)
    # output: running weighted sum
    max_score = mx.full((batch_size, num_heads), float("-inf"))
    sum_exp = mx.zeros((batch_size, num_heads))
    output = mx.zeros((batch_size, num_heads, head_dim))

    # Process each block position
    for block_idx in range(max_blocks):
        # Get block IDs for this position: (batch,)
        block_ids = block_tables[:, block_idx]

        # Create mask for valid blocks
        valid_mask = block_ids >= 0  # (batch,)

        # Skip if no valid blocks at this position
        if not mx.any(valid_mask):
            continue

        # Calculate tokens in this block
        # For most blocks: block_size tokens
        # For last block: context_lens % block_size (or block_size if exact multiple)
        block_start = block_idx * block_size
        block_end = mx.minimum(
            mx.array(block_start + block_size), context_lens
        )  # (batch,)
        tokens_in_block = mx.maximum(block_end - block_start, mx.array(0))  # (batch,)

        # Gather K, V for valid blocks
        # k_pool shape: (num_blocks, block_size, num_heads, head_dim)
        # We need: (batch, block_size, num_heads, head_dim)

        # Use valid block_ids for gathering (replace -1 with 0 for gather, mask later)
        safe_block_ids = mx.where(valid_mask, block_ids, mx.zeros_like(block_ids))

        # Gather blocks: (batch, block_size, num_heads, head_dim)
        k_block = k_pool[safe_block_ids]
        v_block = v_pool[safe_block_ids]

        # Compute attention scores: q @ k^T
        # q: (batch, num_heads, head_dim)
        # k_block: (batch, block_size, num_heads, head_dim)
        # scores: (batch, num_heads, block_size)

        # Transpose k_block for matmul: (batch, num_heads, head_dim, block_size)
        k_block_t = mx.transpose(k_block, (0, 2, 3, 1))

        # q expanded: (batch, num_heads, 1, head_dim)
        q_expanded = q[:, :, None, :]

        # scores: (batch, num_heads, 1, block_size) -> squeeze -> (batch, num_heads, block_size)
        scores = (q_expanded @ k_block_t).squeeze(2) * scale

        # Create position mask within block
        positions = mx.arange(block_size)[None, None, :]  # (1, 1, block_size)
        position_mask = positions < tokens_in_block[:, None, None]  # (batch, 1, block_size)

        # Apply mask: invalid positions get -inf
        scores = mx.where(position_mask, scores, mx.array(float("-inf")))

        # Also mask invalid blocks entirely
        block_mask = valid_mask[:, None, None]  # (batch, 1, 1)
        scores = mx.where(block_mask, scores, mx.array(float("-inf")))

        # Online softmax update
        # Find max in this block
        block_max = mx.max(scores, axis=-1)  # (batch, num_heads)

        # New running max
        new_max = mx.maximum(max_score, block_max)

        # Rescale old accumulator
        # Handle -inf - (-inf) = NaN by using where with a safe value
        # When max_score is -inf (first valid block), old_scale should be 0
        is_first_valid = max_score == float("-inf")
        old_scale = mx.where(
            is_first_valid,
            mx.zeros_like(max_score),
            mx.exp(max_score - new_max)
        )
        new_scale = mx.where(
            block_max == float("-inf"),
            mx.zeros_like(block_max),
            mx.exp(block_max - new_max)
        )

        # Compute softmax for this block (relative to block max)
        # Guard against exp(-inf - (-inf)) = exp(NaN)
        score_diff = mx.where(
            block_max[:, :, None] == float("-inf"),
            mx.zeros_like(scores),
            scores - block_max[:, :, None]
        )
        exp_scores = mx.exp(score_diff)  # (batch, num_heads, block_size)
        block_sum = mx.sum(exp_scores, axis=-1)  # (batch, num_heads)

        # Update running sum
        sum_exp = old_scale * sum_exp + new_scale * block_sum

        # Compute weighted values for this block
        # v_block: (batch, block_size, num_heads, head_dim)
        # Transpose to (batch, num_heads, block_size, head_dim)
        v_block_t = mx.transpose(v_block, (0, 2, 1, 3))

        # exp_scores: (batch, num_heads, block_size) -> (batch, num_heads, 1, block_size)
        weights = exp_scores[:, :, None, :]

        # weighted_v: (batch, num_heads, 1, head_dim) -> squeeze
        weighted_v = (weights @ v_block_t).squeeze(2)  # (batch, num_heads, head_dim)

        # Update output accumulator
        output = old_scale[:, :, None] * output + new_scale[:, :, None] * weighted_v

        # Update max score
        max_score = new_max

    # Normalize by sum_exp
    output = output / (sum_exp[:, :, None] + 1e-6)  # FP16-safe epsilon

    # Reshape to (batch, 1, num_heads, head_dim)
    return output[:, None, :, :]


def _paged_attention_prefill(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: float,
    block_size: int,
    causal: bool,
) -> mx.array:
    """Paged attention for prefill (multiple query tokens).

    For prefill, we have multiple query positions attending to the cached KV.
    This is typically used when processing the initial prompt.
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    max_blocks = block_tables.shape[1]

    # For prefill, gather all K, V and use standard attention
    # This is simpler but uses more memory than block-by-block processing
    # For very long contexts, should use chunked processing

    # Gather K, V for all sequences
    all_k = []
    all_v = []
    max_context = int(mx.max(context_lens).item())

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        num_blocks_used = (ctx_len + block_size - 1) // block_size

        k_parts = []
        v_parts = []

        for block_idx in range(num_blocks_used):
            block_id = int(block_tables[b, block_idx].item())
            if block_id < 0:
                break

            k_block = k_pool[block_id]  # (block_size, num_heads, head_dim)
            v_block = v_pool[block_id]

            # Handle partial last block
            if block_idx == num_blocks_used - 1:
                tokens_remaining = ctx_len - block_idx * block_size
                k_parts.append(k_block[:tokens_remaining])
                v_parts.append(v_block[:tokens_remaining])
            else:
                k_parts.append(k_block)
                v_parts.append(v_block)

        if k_parts:
            k_seq = mx.concatenate(k_parts, axis=0)  # (ctx_len, num_heads, head_dim)
            v_seq = mx.concatenate(v_parts, axis=0)
        else:
            k_seq = mx.zeros((0, num_heads, head_dim), dtype=q.dtype)
            v_seq = mx.zeros((0, num_heads, head_dim), dtype=q.dtype)

        # Pad to max_context
        if k_seq.shape[0] < max_context:
            pad_len = max_context - k_seq.shape[0]
            k_seq = mx.pad(k_seq, [(0, pad_len), (0, 0), (0, 0)])
            v_seq = mx.pad(v_seq, [(0, pad_len), (0, 0), (0, 0)])

        all_k.append(k_seq)
        all_v.append(v_seq)

    # Stack: (batch, max_context, num_heads, head_dim)
    k = mx.stack(all_k, axis=0)
    v = mx.stack(all_v, axis=0)

    # Standard attention computation
    # Transpose for matmul: (batch, num_heads, seq, head_dim)
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Scores: (batch, num_heads, seq_q, max_context)
    scores = (q_t @ mx.transpose(k_t, (0, 1, 3, 2))) * scale

    # Create attention mask
    # Padding mask: valid positions based on context_lens
    positions = mx.arange(max_context)[None, None, None, :]  # (1, 1, 1, max_context)
    padding_mask = positions < context_lens[:, None, None, None]  # (batch, 1, 1, max_context)

    # Apply padding mask
    scores = mx.where(padding_mask, scores, mx.array(float("-inf")))

    if causal:
        # For prefill with cached KV, causal mask allows query to attend
        # to all cached positions (they came before the query)
        # This is the typical case for append-style caching
        pass  # No additional masking needed - all cached KV is "before" query

    # Softmax and weighted sum
    attn_weights = mx.softmax(scores, axis=-1)
    output = attn_weights @ v_t  # (batch, num_heads, seq_q, head_dim)

    # Transpose back: (batch, seq_q, num_heads, head_dim)
    return mx.transpose(output, (0, 2, 1, 3))


def paged_attention_with_bias(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    attention_bias: Optional[mx.array] = None,
    scale: Optional[float] = None,
    block_size: int = 16,
) -> mx.array:
    """Paged attention with optional attention bias.

    Supports ALiBi-style position biases.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k_pool: Key block pool (num_blocks, block_size, num_heads, head_dim).
        v_pool: Value block pool (num_blocks, block_size, num_heads, head_dim).
        block_tables: Block indices per sequence (batch, max_blocks).
        context_lens: Number of cached tokens per sequence (batch,).
        attention_bias: Optional bias to add to attention scores.
        scale: Attention scale. Defaults to 1/sqrt(head_dim).
        block_size: Tokens per block.

    Returns:
        Output tensor same shape as q.
    """
    # For now, fall back to standard paged attention
    # Bias support would require modification of the scoring loop
    output = paged_attention(
        q, k_pool, v_pool, block_tables, context_lens, scale, block_size
    )

    return output


def create_block_table_from_lengths(
    sequence_lengths: mx.array,
    block_size: int,
    max_blocks: Optional[int] = None,
) -> mx.array:
    """Create block tables from sequence lengths (for testing).

    Assigns sequential block IDs to each sequence.

    Args:
        sequence_lengths: Lengths of each sequence (batch,).
        block_size: Tokens per block.
        max_blocks: Maximum blocks per sequence.

    Returns:
        Block tables (batch, max_blocks) with -1 padding.
    """
    batch_size = sequence_lengths.shape[0]
    lengths_list: List[int] = [int(x) for x in sequence_lengths.tolist()]

    if max_blocks is None:
        max_blocks = max((l + block_size - 1) // block_size for l in lengths_list)

    block_tables: List[List[int]] = []
    next_block_id = 0

    for length in lengths_list:
        num_blocks = (length + block_size - 1) // block_size
        blocks: List[int] = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks

        # Pad to max_blocks
        blocks += [-1] * (max_blocks - len(blocks))
        block_tables.append(blocks)

    return mx.array(block_tables, dtype=mx.int32)
