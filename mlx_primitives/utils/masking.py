"""Centralized attention masking utilities.

This module provides consistent attention mask generation across the codebase,
ensuring identical behavior regardless of which attention implementation is used.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_primitives.constants import ATTENTION_MASK_VALUE


def create_causal_mask(
    seq_q: int,
    seq_kv: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a causal attention mask.

    Creates a mask where position i can only attend to positions j <= i
    (in the aligned coordinate system). This handles both square (seq_q == seq_kv)
    and non-square (e.g., autoregressive decoding) attention scenarios.

    The mask uses an additive formulation: positions that should be masked
    receive -inf, and unmasked positions receive 0. Apply the mask by adding
    it to attention scores before softmax.

    Args:
        seq_q: Query sequence length.
        seq_kv: Key/Value sequence length.
        dtype: Output dtype for the mask (default: float32).

    Returns:
        Mask of shape (seq_q, seq_kv) where:
        - mask[i, j] = 0.0 if query i can attend to key j
        - mask[i, j] = -inf if query i should NOT attend to key j

    Example:
        >>> # Square attention (self-attention)
        >>> mask = create_causal_mask(4, 4)
        >>> mask.shape
        (4, 4)
        >>> # Non-square (autoregressive decoding: 1 new token attending to 100 cached)
        >>> mask = create_causal_mask(1, 100)
        >>> mask.shape
        (1, 100)
        >>> # The single query can attend to all 100 KV positions
        >>> mx.sum(mask == 0.0).item()
        100

    Note:
        For non-square attention where seq_q < seq_kv, the diagonal offset is
        computed so that the last query position can attend to all KV positions.
        This is the standard autoregressive behavior where a new token can
        attend to all previous tokens.
    """
    # Use triu to create upper triangular mask of -inf values.
    # k parameter controls diagonal offset:
    # - k=1 for square: position i attends to positions [0, i]
    # - k=seq_kv-seq_q+1 for non-square: aligns the "causal diagonal"
    diagonal_offset = seq_kv - seq_q + 1
    return mx.triu(
        mx.full((seq_q, seq_kv), ATTENTION_MASK_VALUE, dtype=dtype),
        k=diagonal_offset,
    )


def create_causal_mask_broadcast(
    seq_q: int,
    seq_kv: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a causal mask with batch/head broadcast dimensions.

    Same as create_causal_mask but returns shape (1, 1, seq_q, seq_kv)
    for easy broadcasting with attention scores of shape
    (batch, heads, seq_q, seq_kv).

    Args:
        seq_q: Query sequence length.
        seq_kv: Key/Value sequence length.
        dtype: Output dtype for the mask.

    Returns:
        Mask of shape (1, 1, seq_q, seq_kv).
    """
    mask = create_causal_mask(seq_q, seq_kv, dtype)
    return mask[None, None, :, :]


def create_sliding_window_mask(
    seq_q: int,
    seq_kv: int,
    window_size: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a sliding window attention mask.

    Each query position can only attend to positions within the window.
    Combines with causal behavior: position i attends to
    [max(0, i - window_size + 1), i].

    Args:
        seq_q: Query sequence length.
        seq_kv: Key/Value sequence length.
        window_size: Size of the attention window.
        dtype: Output dtype for the mask.

    Returns:
        Mask of shape (seq_q, seq_kv).
    """
    # Start with causal mask
    mask = create_causal_mask(seq_q, seq_kv, dtype)

    # Add lower bound for sliding window (mask positions too far in the past)
    row_indices = mx.arange(seq_q)[:, None]
    col_indices = mx.arange(seq_kv)[None, :]

    # For non-square, align the window properly
    offset = seq_kv - seq_q
    aligned_col = col_indices - offset

    # Mask positions outside the window (too far in the past)
    too_far_mask = aligned_col < (row_indices - window_size + 1)
    mask = mx.where(too_far_mask, ATTENTION_MASK_VALUE, mask)

    return mask
