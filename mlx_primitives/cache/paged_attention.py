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
    """Vectorized paged attention for decode (single query token).

    Gathers all KV blocks at once and computes attention in a single pass,
    eliminating Python loop overhead. For decode with seq_q=1, the attention
    matrix is (batch, heads, context_len) which fits in memory.

    Args:
        alibi_slopes: Optional ALiBi slopes (num_heads,). When provided,
            adds bias = slope * (kv_pos - q_pos) to attention scores.
    """
    batch_size, _, num_heads, head_dim = q.shape
    max_blocks = block_tables.shape[1]
    max_context = max_blocks * block_size

    # Squeeze query: (batch, num_heads, head_dim)
    q = q.squeeze(1)

    # Vectorized block gathering (same approach as prefill):
    # Replace invalid block IDs (-1) with 0 for safe gathering, mask later
    safe_block_tables = mx.maximum(block_tables, 0)  # (batch, max_blocks)

    # Gather all blocks at once
    # k_pool: (num_blocks, block_size, num_heads, head_dim)
    # Result: (batch, max_blocks, block_size, num_heads, head_dim)
    k_gathered = k_pool[safe_block_tables]
    v_gathered = v_pool[safe_block_tables]

    # Reshape to (batch, max_context, num_heads, head_dim)
    k = k_gathered.reshape(batch_size, max_context, num_heads, head_dim)
    v = v_gathered.reshape(batch_size, max_context, num_heads, head_dim)

    # Transpose for attention computation
    # k: (batch, max_context, num_heads, head_dim) -> (batch, num_heads, head_dim, max_context)
    k_t = mx.transpose(k, (0, 2, 3, 1))
    # v: (batch, max_context, num_heads, head_dim) -> (batch, num_heads, max_context, head_dim)
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Compute attention scores: q @ k^T
    # q: (batch, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
    # k_t: (batch, num_heads, head_dim, max_context)
    # scores: (batch, num_heads, 1, max_context) -> squeeze -> (batch, num_heads, max_context)
    q_expanded = q[:, :, None, :]
    scores = (q_expanded @ k_t).squeeze(2) * scale  # (batch, num_heads, max_context)

    # Apply ALiBi bias if provided
    if alibi_slopes is not None:
        # For decode, query position is context_lens (position of the new token)
        q_positions = context_lens  # (batch,)
        kv_positions = mx.arange(max_context)  # (max_context,)

        # Position difference: kv_pos - q_pos
        # q_positions: (batch,) -> (batch, 1, 1)
        # kv_positions: (max_context,) -> (1, 1, max_context)
        # Result: (batch, 1, max_context)
        pos_diff = kv_positions[None, None, :] - q_positions[:, None, None]

        # ALiBi bias: slopes[head] * pos_diff
        # slopes: (num_heads,) -> (1, num_heads, 1)
        # Result: (batch, num_heads, max_context)
        alibi_bias = alibi_slopes[None, :, None] * pos_diff.astype(mx.float32)
        scores = scores + alibi_bias

    # Create combined mask:
    # 1. Padding mask: positions < context_lens (valid cached tokens)
    # 2. Block validity mask: block_tables >= 0 (valid blocks)

    positions = mx.arange(max_context)[None, None, :]  # (1, 1, max_context)
    padding_mask = positions < context_lens[:, None, None]  # (batch, 1, max_context)

    # Block validity: map each position to its block and check validity
    block_indices_1d = mx.arange(max_context) // block_size  # (max_context,)
    block_validity = mx.take(block_tables >= 0, block_indices_1d, axis=1)  # (batch, max_context)
    block_validity = block_validity[:, None, :]  # (batch, 1, max_context)

    # Combined mask
    combined_mask = padding_mask & block_validity

    # Apply mask: invalid positions get -inf
    scores = mx.where(combined_mask, scores, mx.array(float("-inf")))

    # Softmax and weighted sum
    attn_weights = mx.softmax(scores, axis=-1)  # (batch, num_heads, max_context)

    # Output: attn_weights @ v
    # attn_weights: (batch, num_heads, max_context) -> (batch, num_heads, 1, max_context)
    # v_t: (batch, num_heads, max_context, head_dim)
    # output: (batch, num_heads, 1, head_dim) -> squeeze -> (batch, num_heads, head_dim)
    output = (attn_weights[:, :, None, :] @ v_t).squeeze(2)

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
        # Apply causal mask: query position i can only attend to positions 0..i
        # This is standard for prompt prefill where q_pos and kv_pos are aligned
        q_positions = mx.arange(seq_q)[:, None]  # (seq_q, 1)
        kv_positions = mx.arange(max_context)[None, :]  # (1, max_context)
        causal_mask = kv_positions <= q_positions  # (seq_q, max_context)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_q, max_context)
        scores = mx.where(causal_mask, scores, mx.array(float("-inf")))

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
        q, k_pool, v_pool, block_tables, context_lens, scale, block_size, causal,
        alibi_slopes=alibi_slopes
    )


def create_block_table_from_lengths(
    sequence_lengths: mx.array,
    block_size: int,
    max_blocks: Optional[int] = None,
) -> mx.array:
    """Create block tables from sequence lengths (for testing).

    Assigns sequential block IDs to each sequence.

    Note: This function is intended for testing/setup, not for hot paths.
    It uses tensor operations to avoid GPU→CPU sync where possible.

    Args:
        sequence_lengths: Lengths of each sequence (batch,).
        block_size: Tokens per block.
        max_blocks: Maximum blocks per sequence.

    Returns:
        Block tables (batch, max_blocks) with -1 padding.
    """
    batch_size = sequence_lengths.shape[0]

    # Calculate blocks per sequence without Python iteration
    # blocks_per_seq = ceil(length / block_size)
    blocks_per_seq = (sequence_lengths + block_size - 1) // block_size

    # Evaluate to get max_blocks if not provided (requires sync)
    if max_blocks is None:
        max_blocks_val = int(mx.max(blocks_per_seq).item())
    else:
        max_blocks_val = max_blocks

    # Build block tables using vectorized operations where possible
    # cumsum gives us the starting block ID for each sequence
    cum_blocks = mx.concatenate([mx.array([0]), mx.cumsum(blocks_per_seq)[:-1]])

    # Create block indices for each position
    # For each sequence i and block position j, block_id = cum_blocks[i] + j
    # if j < blocks_per_seq[i], else -1

    # Create position indices: (1, max_blocks)
    positions = mx.arange(max_blocks_val)[None, :]

    # Create mask: (batch, max_blocks) - True where position < blocks_per_seq
    valid_mask = positions < blocks_per_seq[:, None]

    # Create block IDs: (batch, max_blocks)
    # block_id = cum_blocks[batch_idx] + position
    block_ids = cum_blocks[:, None] + positions

    # Apply mask: -1 for invalid positions
    block_tables = mx.where(valid_mask, block_ids, mx.array(-1))

    return block_tables.astype(mx.int32)
