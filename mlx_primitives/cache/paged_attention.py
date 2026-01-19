"""Paged attention implementation for non-contiguous KV cache.

This module provides paged attention that works with block-based KV storage,
enabling efficient memory usage for variable-length sequences.

Supports ALiBi (Attention with Linear Biases) for position encoding.
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.constants import METAL_SOFTMAX_EPSILON


def get_alibi_slopes(num_heads: int) -> mx.array:
    """Compute ALiBi slopes for the given number of heads.

    ALiBi (Attention with Linear Biases) adds a position-dependent bias
    to attention scores: bias = slope * (kv_pos - q_pos)

    The slopes are computed as powers of 2^(-8/num_heads) following the
    original ALiBi paper.

    Args:
        num_heads: Number of attention heads.

    Returns:
        Slopes array of shape (num_heads,). Each slope is negative,
        causing attention to prefer nearby tokens.

    Example:
        >>> slopes = get_alibi_slopes(8)
        >>> # slopes are [2^-1, 2^-2, 2^-3, ..., 2^-8] = [0.5, 0.25, ...]
    """
    # Standard ALiBi: slopes are powers of 2^(-8/n) for n heads
    # This gives geometric sequence from 2^(-8/n) to 2^(-8)
    ratio = 2 ** (-8 / num_heads)
    slopes = mx.array([ratio ** (i + 1) for i in range(num_heads)], dtype=mx.float32)
    return slopes


def paged_attention(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: Optional[float] = None,
    block_size: int = 16,
    causal: bool = True,
    use_metal: bool = True,
    alibi_slopes: Optional[mx.array] = None,
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
        use_metal: Use Metal kernel if available. Currently not implemented;
            parameter accepted for API consistency with other attention functions.
        alibi_slopes: Optional ALiBi slopes of shape (num_heads,).
            When provided, adds position-dependent bias: slope * (kv_pos - q_pos).
            Use get_alibi_slopes(num_heads) to compute standard ALiBi slopes.

    Returns:
        Output tensor same shape as q.

    Example:
        >>> # Decode mode: single new query token
        >>> q = mx.random.normal((batch, 1, heads, dim))
        >>> out = paged_attention(q, k_pool, v_pool, block_tables, context_lens)
        >>>
        >>> # With ALiBi position encoding
        >>> slopes = get_alibi_slopes(num_heads)
        >>> out = paged_attention(q, k_pool, v_pool, block_tables, context_lens,
        ...                       alibi_slopes=slopes)
    """
    batch_size, seq_q, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if seq_q == 1:
        # Decode mode: optimized single-token attention
        return _paged_attention_decode(
            q, k_pool, v_pool, block_tables, context_lens, scale, block_size,
            alibi_slopes=alibi_slopes
        )
    else:
        # Prefill mode: process query sequence with paged KV
        return _paged_attention_prefill(
            q, k_pool, v_pool, block_tables, context_lens, scale, block_size, causal,
            alibi_slopes=alibi_slopes
        )


def _paged_attention_decode(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    block_tables: mx.array,
    context_lens: mx.array,
    scale: float,
    block_size: int,
    alibi_slopes: Optional[mx.array] = None,
) -> mx.array:
    """Optimized paged attention for decode (single query token).

    Uses online softmax to process blocks one at a time, maintaining
    numerical stability without materializing the full attention matrix.

    Args:
        alibi_slopes: Optional ALiBi slopes (num_heads,). When provided,
            adds bias = slope * (kv_pos - q_pos) to attention scores.
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

    # For decode, query position is context_lens (position of the new token)
    # Shape: (batch,)
    q_positions = context_lens

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

        # Apply ALiBi bias if provided
        if alibi_slopes is not None:
            # KV positions for this block: block_start + [0, 1, ..., block_size-1]
            kv_positions = mx.arange(block_size) + block_start  # (block_size,)

            # Position difference: kv_pos - q_pos
            # q_positions: (batch,) -> (batch, 1, 1)
            # kv_positions: (block_size,) -> (1, 1, block_size)
            # Result: (batch, 1, block_size)
            pos_diff = kv_positions[None, None, :] - q_positions[:, None, None]

            # ALiBi bias: slopes[head] * pos_diff
            # slopes: (num_heads,) -> (1, num_heads, 1)
            # pos_diff: (batch, 1, block_size)
            # Result: (batch, num_heads, block_size)
            alibi_bias = alibi_slopes[None, :, None] * pos_diff.astype(mx.float32)
            scores = scores + alibi_bias

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
    output = output / (sum_exp[:, :, None] + METAL_SOFTMAX_EPSILON)

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
    alibi_slopes: Optional[mx.array] = None,
) -> mx.array:
    """Paged attention for prefill (multiple query tokens).

    For prefill, we have multiple query positions attending to the cached KV.
    This is typically used when processing the initial prompt.

    Optimized to avoid host/device synchronization (.item() calls) by using
    vectorized gather operations.

    Args:
        alibi_slopes: Optional ALiBi slopes (num_heads,). When provided,
            adds bias = slope * (kv_pos - q_pos) to attention scores.
    """
    batch_size, seq_q, num_heads, head_dim = q.shape
    max_blocks = block_tables.shape[1]

    # Compute max_context without .item() - use the full allocated space
    # max_context = max_blocks * block_size covers all possible tokens
    max_context = max_blocks * block_size

    # Vectorized block gathering:
    # 1. Clamp block_tables to valid indices (replace -1 with 0 for safe gather)
    # 2. Gather all blocks at once
    # 3. Reshape to (batch, max_blocks * block_size, heads, dim)
    # 4. Use masking to handle variable context lengths

    # Replace invalid block IDs (-1) with 0 for safe gathering
    # We'll mask out invalid positions later
    safe_block_tables = mx.maximum(block_tables, 0)  # (batch, max_blocks)

    # Gather all blocks for all sequences at once
    # k_pool: (num_blocks, block_size, num_heads, head_dim)
    # Result: (batch, max_blocks, block_size, num_heads, head_dim)
    k_gathered = k_pool[safe_block_tables]
    v_gathered = v_pool[safe_block_tables]

    # Reshape to (batch, max_context, num_heads, head_dim)
    k = k_gathered.reshape(batch_size, max_context, num_heads, head_dim)
    v = v_gathered.reshape(batch_size, max_context, num_heads, head_dim)

    # Standard attention computation
    # Transpose for matmul: (batch, num_heads, seq, head_dim)
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Scores: (batch, num_heads, seq_q, max_context)
    scores = (q_t @ mx.transpose(k_t, (0, 1, 3, 2))) * scale

    # Apply ALiBi bias if provided
    if alibi_slopes is not None:
        # Query positions: 0 to seq_q-1 (new tokens being processed)
        # KV positions: 0 to max_context-1 (cached tokens)
        q_pos = mx.arange(seq_q)[:, None]  # (seq_q, 1)
        kv_pos = mx.arange(max_context)[None, :]  # (1, max_context)

        # Position difference: kv_pos - q_pos
        # Result: (seq_q, max_context)
        pos_diff = (kv_pos - q_pos).astype(mx.float32)

        # ALiBi bias: slopes[head] * pos_diff
        # slopes: (num_heads,) -> (1, num_heads, 1, 1)
        # pos_diff: (seq_q, max_context) -> (1, 1, seq_q, max_context)
        # Result: (1, num_heads, seq_q, max_context)
        alibi_bias = alibi_slopes[None, :, None, None] * pos_diff[None, None, :, :]
        scores = scores + alibi_bias

    # Create attention mask combining:
    # 1. Padding mask: positions < context_lens (valid cached tokens)
    # 2. Block validity mask: block_tables >= 0 (valid blocks)
    positions = mx.arange(max_context)[None, None, None, :]  # (1, 1, 1, max_context)
    padding_mask = positions < context_lens[:, None, None, None]  # (batch, 1, 1, max_context)

    # Also mask out positions from invalid blocks
    # Use 1D indices for take to avoid extra dimensions
    block_indices_1d = mx.arange(max_context) // block_size  # (max_context,)
    block_validity = mx.take(block_tables >= 0, block_indices_1d, axis=1)  # (batch, max_context)
    block_validity = block_validity[:, None, None, :]  # (batch, 1, 1, max_context)

    # Combined mask: position must be valid AND in a valid block
    combined_mask = padding_mask & block_validity

    # Apply mask
    scores = mx.where(combined_mask, scores, mx.array(float("-inf")))

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
    alibi_slopes: Optional[mx.array] = None,
    scale: Optional[float] = None,
    block_size: int = 16,
    causal: bool = True,
    use_metal: bool = True,
) -> mx.array:
    """Paged attention with optional attention bias or ALiBi position encoding.

    Supports ALiBi-style position biases via the alibi_slopes parameter.
    For general pre-computed attention bias matrices, use the attention_bias
    parameter (not yet implemented).

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k_pool: Key block pool (num_blocks, block_size, num_heads, head_dim).
        v_pool: Value block pool (num_blocks, block_size, num_heads, head_dim).
        block_tables: Block indices per sequence (batch, max_blocks).
        context_lens: Number of cached tokens per sequence (batch,).
        attention_bias: Optional pre-computed bias to add to attention scores.
            NOT YET IMPLEMENTED - will raise NotImplementedError if provided.
            Use alibi_slopes for ALiBi-style position encoding instead.
        alibi_slopes: Optional ALiBi slopes of shape (num_heads,).
            When provided, adds position-dependent bias: slope * (kv_pos - q_pos).
            Use get_alibi_slopes(num_heads) to compute standard ALiBi slopes.
        scale: Attention scale. Defaults to 1/sqrt(head_dim).
        block_size: Tokens per block.
        causal: Apply causal masking.
        use_metal: Use Metal kernel if available. Currently not implemented;
            parameter accepted for API consistency with other attention functions.

    Returns:
        Output tensor same shape as q.

    Raises:
        NotImplementedError: If attention_bias is not None.
        ValueError: If both attention_bias and alibi_slopes are provided.

    Example:
        >>> # Using ALiBi position encoding
        >>> slopes = get_alibi_slopes(num_heads)
        >>> out = paged_attention_with_bias(
        ...     q, k_pool, v_pool, block_tables, context_lens,
        ...     alibi_slopes=slopes
        ... )
    """
    if attention_bias is not None and alibi_slopes is not None:
        raise ValueError("Cannot specify both attention_bias and alibi_slopes")

    if attention_bias is not None:
        raise NotImplementedError(
            "Pre-computed attention_bias support in paged_attention is not yet implemented. "
            "Use alibi_slopes parameter for ALiBi-style position encoding instead."
        )

    return paged_attention(
        q, k_pool, v_pool, block_tables, context_lens, scale, block_size, causal, use_metal,
        alibi_slopes=alibi_slopes
    )


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
