"""Sparse Attention implementations for MLX.

This module provides sparse attention patterns:
- BlockSparseAttention: Block-sparse attention patterns
- LongformerAttention: Sliding window + global attention
- BigBirdAttention: Random + window + global attention
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


def create_block_sparse_mask(
    seq_len: int,
    block_size: int,
    num_random_blocks: int = 0,
    num_global_blocks: int = 0,
) -> mx.array:
    """Create a block-sparse attention mask.

    Args:
        seq_len: Sequence length.
        block_size: Size of each block.
        num_random_blocks: Number of random blocks to attend to.
        num_global_blocks: Number of global blocks (attend to all).

    Returns:
        Boolean mask of shape (seq_len, seq_len).
    """
    num_blocks = (seq_len + block_size - 1) // block_size

    # Start with diagonal blocks (local attention)
    mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)

    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, seq_len)
        # Each block attends to itself
        for j in range(start, end):
            for k in range(start, end):
                mask = mx.where(
                    (mx.arange(seq_len) == j)[:, None] & (mx.arange(seq_len) == k)[None, :],
                    True,
                    mask
                )

    # Simpler approach: create block diagonal
    block_indices = mx.arange(seq_len) // block_size
    mask = block_indices[:, None] == block_indices[None, :]

    # Add global blocks (first num_global_blocks blocks attend to/from all)
    if num_global_blocks > 0:
        global_end = num_global_blocks * block_size
        # Global tokens can attend to all
        global_mask_rows = mx.arange(seq_len) < global_end
        global_mask_cols = mx.arange(seq_len) < global_end
        # All tokens can attend to global tokens
        mask = mask | global_mask_cols[None, :]
        # Global tokens can attend to all
        mask = mx.where(global_mask_rows[:, None], True, mask)

    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    global_indices: Optional[List[int]] = None,
) -> mx.array:
    """Create a sliding window attention mask with optional global tokens.

    Args:
        seq_len: Sequence length.
        window_size: Size of the sliding window (one side).
        global_indices: Indices of global tokens that attend to all.

    Returns:
        Boolean mask of shape (seq_len, seq_len).
    """
    # Create position indices
    positions = mx.arange(seq_len)

    # Sliding window: |i - j| <= window_size
    row_pos = positions[:, None]
    col_pos = positions[None, :]
    mask = mx.abs(row_pos - col_pos) <= window_size

    # Add global tokens
    if global_indices is not None and len(global_indices) > 0:
        global_idx = mx.array(global_indices)
        # Global tokens can attend to all positions
        is_global_row = mx.any(positions[:, None] == global_idx[None, :], axis=1)
        # All positions can attend to global tokens
        is_global_col = mx.any(positions[:, None] == global_idx[None, :], axis=1)

        mask = mask | is_global_row[:, None] | is_global_col[None, :]

    return mask


def create_bigbird_mask(
    seq_len: int,
    block_size: int,
    window_size: int,
    num_global_tokens: int,
    num_random_blocks: int,
    seed: int = 0,
) -> mx.array:
    """Create BigBird attention pattern.

    Combines:
    - Global attention (first/last tokens)
    - Sliding window attention
    - Random block attention

    Args:
        seq_len: Sequence length.
        block_size: Size of blocks for random attention.
        window_size: Sliding window size.
        num_global_tokens: Number of global tokens at start.
        num_random_blocks: Number of random blocks per query block.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask of shape (seq_len, seq_len).
    """
    mx.random.seed(seed)

    positions = mx.arange(seq_len)
    num_blocks = (seq_len + block_size - 1) // block_size

    # Start with sliding window
    row_pos = positions[:, None]
    col_pos = positions[None, :]
    mask = mx.abs(row_pos - col_pos) <= window_size

    # Add global tokens
    is_global = positions < num_global_tokens
    mask = mask | is_global[None, :] | is_global[:, None]

    # Add random blocks
    if num_random_blocks > 0:
        block_indices = positions // block_size
        for query_block in range(num_blocks):
            # Randomly select key blocks
            available_blocks = [b for b in range(num_blocks) if b != query_block]
            if len(available_blocks) > 0:
                num_to_select = min(num_random_blocks, len(available_blocks))
                # Simple random selection using shuffle
                perm = mx.random.permutation(len(available_blocks))
                selected = [available_blocks[int(perm[i])] for i in range(num_to_select)]

                query_mask = block_indices == query_block
                for key_block in selected:
                    key_mask = block_indices == key_block
                    mask = mask | (query_mask[:, None] & key_mask[None, :])

    return mask


class BlockSparseAttention(nn.Module):
    """Block-sparse attention with configurable sparsity patterns.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        block_size: Size of attention blocks.
        num_random_blocks: Random blocks per query block.
        num_global_blocks: Number of global blocks.
        dropout: Dropout rate.

    Example:
        >>> attn = BlockSparseAttention(dims=768, num_heads=12, block_size=64)
        >>> x = mx.random.normal((2, 512, 768))
        >>> y = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        block_size: int = 64,
        num_random_blocks: int = 1,
        num_global_blocks: int = 1,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.num_global_blocks = num_global_blocks
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).
            mask: Optional custom attention mask.

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Create sparse mask if not provided
        if mask is None:
            mask = create_block_sparse_mask(
                seq_len,
                self.block_size,
                self.num_random_blocks,
                self.num_global_blocks,
            )

        # Apply causal mask if needed
        if self.causal:
            causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
            mask = mask & causal_mask

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply sparse mask
        scores = mx.where(mask[None, None, :, :], scores, float('-inf'))

        # Softmax and dropout
        attn_weights = mx.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        return self.out_proj(out)


class LongformerAttention(nn.Module):
    """Longformer-style attention with sliding window and global tokens.

    Uses a combination of:
    - Local sliding window attention for most tokens
    - Global attention for special tokens (e.g., [CLS])

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        window_size: One-sided window size.
        global_token_indices: Indices of tokens with global attention.
        dropout: Dropout rate.

    Reference:
        "Longformer: The Long-Document Transformer"
        https://arxiv.org/abs/2004.05150

    Example:
        >>> attn = LongformerAttention(dims=768, num_heads=12, window_size=256)
        >>> x = mx.random.normal((2, 4096, 768))
        >>> y = attn(x, global_indices=[0])  # First token is global
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
        num_global_tokens: int = 1,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.num_global_tokens = num_global_tokens

        # Separate projections for local and global attention
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)

        # Global attention projections
        self.q_global_proj = nn.Linear(dims, dims, bias=False)
        self.k_global_proj = nn.Linear(dims, dims, bias=False)
        self.v_global_proj = nn.Linear(dims, dims, bias=False)

        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        global_indices: Optional[List[int]] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).
            global_indices: Indices of tokens with global attention (overrides num_global_tokens).

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        batch_size, seq_len, _ = x.shape

        if global_indices is None:
            # Use first num_global_tokens tokens as global
            global_indices = list(range(self.num_global_tokens))

        # Create combined mask
        mask = create_sliding_window_mask(seq_len, self.window_size, global_indices)

        # Project Q, K, V for local attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention with mask
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.where(mask[None, None, :, :], scores, float('-inf'))

        attn_weights = mx.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v

        # Handle global tokens with full attention
        if len(global_indices) > 0:
            global_idx = mx.array(global_indices)

            # Global Q, K, V
            q_global = self.q_global_proj(x)
            k_global = self.k_global_proj(x)
            v_global = self.v_global_proj(x)

            q_global = q_global.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_global = k_global.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_global = v_global.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            # For global tokens, compute full attention
            for idx in global_indices:
                # Query from global token to all keys
                q_g = q_global[:, :, idx:idx+1, :]  # (batch, heads, 1, head_dim)
                scores_g = (q_g @ k_global.transpose(0, 1, 3, 2)) * self.scale
                attn_g = mx.softmax(scores_g, axis=-1)
                out_g = attn_g @ v_global  # (batch, heads, 1, head_dim)

                # Update output at global position
                out = mx.concatenate([
                    out[:, :, :idx, :],
                    out_g,
                    out[:, :, idx+1:, :]
                ], axis=2)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        return self.out_proj(out)


class BigBirdAttention(nn.Module):
    """BigBird attention with random, window, and global patterns.

    Combines three attention patterns:
    - Random attention: Each query attends to random keys
    - Window attention: Local sliding window
    - Global attention: Special tokens attend to all

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        block_size: Block size for random attention.
        window_size: Sliding window size.
        num_global_tokens: Number of global tokens.
        num_random_blocks: Number of random blocks per query.
        dropout: Dropout rate.

    Reference:
        "Big Bird: Transformers for Longer Sequences"
        https://arxiv.org/abs/2007.14062

    Example:
        >>> attn = BigBirdAttention(dims=768, num_heads=12)
        >>> x = mx.random.normal((2, 4096, 768))
        >>> y = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        block_size: int = 64,
        window_size: int = 3,  # In blocks
        num_global_tokens: int = 2,
        num_random_blocks: int = 3,
        num_random_tokens: Optional[int] = None,  # Alias for num_random_blocks
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.block_size = block_size
        self.window_size = window_size * block_size
        self.num_global_tokens = num_global_tokens
        # Support both num_random_blocks and num_random_tokens
        self.num_random_blocks = num_random_tokens if num_random_tokens is not None else num_random_blocks
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._mask_cache = {}

    def _get_mask(self, seq_len: int) -> mx.array:
        """Get or create BigBird attention mask."""
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = create_bigbird_mask(
                seq_len,
                self.block_size,
                self.window_size,
                self.num_global_tokens,
                self.num_random_blocks,
            )
        return self._mask_cache[seq_len]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dims).

        Returns:
            Output tensor (batch, seq_len, dims).
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Get BigBird mask
        mask = self._get_mask(seq_len)

        # Compute attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.where(mask[None, None, :, :], scores, float('-inf'))

        attn_weights = mx.softmax(scores, axis=-1)
        # Handle NaN from all-masked rows
        attn_weights = mx.where(mx.isnan(attn_weights), 0.0, attn_weights)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)

        return self.out_proj(out)
