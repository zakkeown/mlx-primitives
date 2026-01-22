"""KV Cache utilities for MLX.

.. deprecated::
    This module is deprecated. Please use the canonical implementations:
    - ``mlx_primitives.cache.SimpleKVCache`` instead of ``KVCache``
    - ``mlx_primitives.cache.SlidingWindowCache`` instead of ``SlidingWindowCache``
    - ``mlx_primitives.cache.RotatingKVCache`` instead of ``RotatingKVCache``
    - ``mlx_primitives.cache.CompressedKVCache`` instead of ``CompressedKVCache``

    For paged attention, use ``mlx_primitives.cache.KVCache`` (the managed version)
    or ``mlx_primitives.advanced.paged_attention.PagedKVCache``.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import mlx.core as mx

# Re-export CompressedKVCache from canonical location
from mlx_primitives.cache.compressed_cache import CompressedKVCache as _CompressedKVCache


def _deprecation_warning(old_class: str, new_module: str, new_class: str) -> None:
    """Emit a deprecation warning for old cache classes."""
    warnings.warn(
        f"mlx_primitives.advanced.kv_cache.{old_class} is deprecated. "
        f"Use mlx_primitives.{new_module}.{new_class} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class CompressedKVCache(_CompressedKVCache):
    """Deprecated: Use mlx_primitives.cache.CompressedKVCache instead."""

    def __init__(self, *args, **kwargs):
        _deprecation_warning("CompressedKVCache", "cache", "CompressedKVCache")
        super().__init__(*args, **kwargs)


class KVCache:
    """Basic Key-Value cache for transformer inference.

    .. deprecated::
        Use ``mlx_primitives.cache.SimpleKVCache`` for single-layer cache,
        or ``mlx_primitives.cache.MultiLayerKVCache`` for multi-layer cache.

    Stores key and value tensors for all layers, enabling efficient
    autoregressive generation without recomputing past attention.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type for cache tensors.

    Example:
        >>> cache = KVCache(
        ...     num_layers=32,
        ...     max_batch_size=1,
        ...     max_seq_len=2048,
        ...     num_heads=32,
        ...     head_dim=128,
        ... )
        >>> k, v = cache.get(layer_idx=0)
        >>> cache.update(layer_idx=0, new_k, new_v)
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        _deprecation_warning("KVCache", "cache", "SimpleKVCache")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Cache shape: (batch, num_heads, seq_len, head_dim)
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)

        # Initialize caches for all layers
        self.k_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]

        # Track current sequence length per batch
        self.seq_len = 0

    def reset(self) -> None:
        """Reset the cache (clear all stored KV pairs)."""
        self.seq_len = 0
        # Note: We don't need to zero out tensors, just reset the length

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached K, V for a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Tuple of (K, V) tensors up to current sequence length.
        """
        k = self.k_cache[layer_idx][:, :, : self.seq_len, :]
        v = self.v_cache[layer_idx][:, :, : self.seq_len, :]
        return k, v

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new K, V and return full cache.

        Args:
            layer_idx: Layer index.
            new_k: New key tensor (batch, num_heads, new_seq_len, head_dim).
            new_v: New value tensor.

        Returns:
            Tuple of full (K, V) tensors including new tokens.
        """
        new_seq_len = new_k.shape[2]
        start_pos = self.seq_len

        # Update cache
        # Note: MLX doesn't support item assignment, so we use concatenation
        if start_pos == 0:
            self.k_cache[layer_idx] = mx.pad(
                new_k,
                [(0, 0), (0, 0), (0, self.max_seq_len - new_seq_len), (0, 0)],
            )
            self.v_cache[layer_idx] = mx.pad(
                new_v,
                [(0, 0), (0, 0), (0, self.max_seq_len - new_seq_len), (0, 0)],
            )
        else:
            # Get existing cache up to start_pos
            existing_k = self.k_cache[layer_idx][:, :, :start_pos, :]
            existing_v = self.v_cache[layer_idx][:, :, :start_pos, :]

            # Concatenate with new
            full_k = mx.concatenate([existing_k, new_k], axis=2)
            full_v = mx.concatenate([existing_v, new_v], axis=2)

            # Pad to max length
            current_len = full_k.shape[2]
            if current_len < self.max_seq_len:
                self.k_cache[layer_idx] = mx.pad(
                    full_k,
                    [(0, 0), (0, 0), (0, self.max_seq_len - current_len), (0, 0)],
                )
                self.v_cache[layer_idx] = mx.pad(
                    full_v,
                    [(0, 0), (0, 0), (0, self.max_seq_len - current_len), (0, 0)],
                )
            else:
                self.k_cache[layer_idx] = full_k
                self.v_cache[layer_idx] = full_v

        # Update sequence length (only on first layer to avoid double counting)
        if layer_idx == 0:
            self.seq_len = start_pos + new_seq_len

        # Return full cache
        return self.get(layer_idx)


class SlidingWindowCache:
    """Sliding window KV cache for long sequences.

    .. deprecated::
        Use ``mlx_primitives.cache.SlidingWindowCache`` instead.

    Maintains a fixed-size window of recent KV pairs, useful for
    models with sliding window attention (e.g., Mistral).

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        window_size: Size of the sliding window.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type for cache tensors.

    Example:
        >>> cache = SlidingWindowCache(
        ...     num_layers=32,
        ...     max_batch_size=1,
        ...     window_size=4096,
        ...     num_heads=32,
        ...     head_dim=128,
        ... )
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        window_size: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        _deprecation_warning("SlidingWindowCache", "cache", "SlidingWindowCache")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        cache_shape = (max_batch_size, num_heads, window_size, head_dim)

        self.k_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]

        self.current_len = 0
        self.total_len = 0  # Total tokens seen (for position encoding)

    def reset(self) -> None:
        """Reset the cache."""
        self.current_len = 0
        self.total_len = 0

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached K, V for a layer."""
        k = self.k_cache[layer_idx][:, :, : self.current_len, :]
        v = self.v_cache[layer_idx][:, :, : self.current_len, :]
        return k, v

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with sliding window semantics.

        Args:
            layer_idx: Layer index.
            new_k: New key tensor.
            new_v: New value tensor.

        Returns:
            Tuple of windowed (K, V) tensors.
        """
        new_seq_len = new_k.shape[2]

        # Get current cache
        current_k = self.k_cache[layer_idx][:, :, : self.current_len, :]
        current_v = self.v_cache[layer_idx][:, :, : self.current_len, :]

        # Concatenate with new tokens
        full_k = mx.concatenate([current_k, new_k], axis=2)
        full_v = mx.concatenate([current_v, new_v], axis=2)

        # Apply sliding window (keep only last window_size tokens)
        total_len = full_k.shape[2]
        if total_len > self.window_size:
            start = total_len - self.window_size
            full_k = full_k[:, :, start:, :]
            full_v = full_v[:, :, start:, :]
            new_len = self.window_size
        else:
            new_len = total_len

        # Pad to window size for storage
        if new_len < self.window_size:
            self.k_cache[layer_idx] = mx.pad(
                full_k,
                [(0, 0), (0, 0), (0, self.window_size - new_len), (0, 0)],
            )
            self.v_cache[layer_idx] = mx.pad(
                full_v,
                [(0, 0), (0, 0), (0, self.window_size - new_len), (0, 0)],
            )
        else:
            self.k_cache[layer_idx] = full_k
            self.v_cache[layer_idx] = full_v

        # Update lengths
        if layer_idx == 0:
            self.current_len = new_len
            self.total_len += new_seq_len

        return full_k, full_v

    @property
    def position_offset(self) -> int:
        """Get position offset for rotary embeddings."""
        return max(0, self.total_len - self.window_size)


class PagedKVCache:
    """Paged KV cache for efficient memory management.

    .. deprecated::
        Use ``mlx_primitives.advanced.paged_attention.PagedKVCache`` for
        the full-featured paged attention implementation with block management.

    Implements vLLM-style paged attention where KV cache is stored
    in fixed-size blocks that can be allocated/freed independently.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Number of tokens per block (default: 16).
        num_blocks: Maximum number of blocks to allocate.
        dtype: Data type for cache tensors.

    Reference:
        "Efficient Memory Management for Large Language Model Serving with PagedAttention"
        https://arxiv.org/abs/2309.06180

    Example:
        >>> cache = PagedKVCache(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     block_size=16,
        ...     num_blocks=1024,
        ... )
        >>> seq_id = cache.allocate_sequence(max_len=512)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        dtype: mx.Dtype = mx.float16,
    ):
        _deprecation_warning("PagedKVCache", "advanced.paged_attention", "PagedKVCache")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype

        # Block storage: (num_blocks, num_heads, block_size, head_dim)
        block_shape = (num_blocks, num_heads, block_size, head_dim)
        self.k_blocks = [
            mx.zeros(block_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_blocks = [
            mx.zeros(block_shape, dtype=dtype) for _ in range(num_layers)
        ]

        # Block allocation table: maps sequence_id -> list of block indices
        self.block_tables: dict[int, List[int]] = {}

        # Free block list
        self.free_blocks = list(range(num_blocks))

        # Sequence lengths
        self.seq_lengths: dict[int, int] = {}

        # Next sequence ID
        self._next_seq_id = 0

    def allocate_sequence(self, max_len: int) -> int:
        """Allocate blocks for a new sequence.

        Args:
            max_len: Maximum sequence length.

        Returns:
            Sequence ID.

        Raises:
            RuntimeError: If not enough blocks available.
        """
        num_blocks_needed = (max_len + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"Not enough free blocks. Need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)

        self.block_tables[seq_id] = allocated
        self.seq_lengths[seq_id] = 0

        return seq_id

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence.

        Args:
            seq_id: Sequence ID to free.
        """
        if seq_id not in self.block_tables:
            return

        # Return blocks to free list
        self.free_blocks.extend(self.block_tables[seq_id])
        del self.block_tables[seq_id]
        del self.seq_lengths[seq_id]

    def get(
        self, layer_idx: int, seq_id: int
    ) -> Tuple[mx.array, mx.array]:
        """Get cached K, V for a sequence at a layer.

        Args:
            layer_idx: Layer index.
            seq_id: Sequence ID.

        Returns:
            Tuple of (K, V) tensors.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Unknown sequence ID: {seq_id}")

        block_indices = self.block_tables[seq_id]
        seq_len = self.seq_lengths[seq_id]

        if seq_len == 0:
            return (
                mx.zeros((1, self.num_heads, 0, self.head_dim), dtype=self.dtype),
                mx.zeros((1, self.num_heads, 0, self.head_dim), dtype=self.dtype),
            )

        # Gather blocks
        k_parts = []
        v_parts = []

        for block_idx in block_indices:
            k_parts.append(self.k_blocks[layer_idx][block_idx])
            v_parts.append(self.v_blocks[layer_idx][block_idx])

        # Concatenate blocks
        k = mx.concatenate([p[None, ...] for p in k_parts], axis=2)
        v = mx.concatenate([p[None, ...] for p in v_parts], axis=2)

        # Trim to actual sequence length
        k = k[:, :, :seq_len, :]
        v = v[:, :, :seq_len, :]

        return k, v

    def update(
        self,
        layer_idx: int,
        seq_id: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache for a sequence.

        Args:
            layer_idx: Layer index.
            seq_id: Sequence ID.
            new_k: New key tensor (1, num_heads, new_len, head_dim).
            new_v: New value tensor.

        Returns:
            Tuple of full (K, V) including new tokens.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Unknown sequence ID: {seq_id}")

        new_len = new_k.shape[2]
        current_len = self.seq_lengths[seq_id]
        block_indices = self.block_tables[seq_id]

        # Compute positions, block indices, and offsets for all new tokens
        positions = current_len + mx.arange(new_len)
        block_idxs = positions // self.block_size
        offsets = positions % self.block_size

        # Check we have enough blocks
        max_block_idx = int(mx.max(block_idxs))
        if max_block_idx >= len(block_indices):
            raise RuntimeError("Sequence exceeded allocated blocks")

        # Group updates by block for efficiency
        # For each block that receives updates, compute a single vectorized update
        unique_block_idxs = set(int(b) for b in block_idxs)

        for block_idx in unique_block_idxs:
            actual_block = block_indices[block_idx]

            # Find which new tokens go to this block
            # Create mask: which positions in new_k map to this block
            block_mask = (block_idxs == block_idx)  # (new_len,)

            # Get the offsets for tokens going to this block
            token_offsets = mx.where(block_mask, offsets, -1)  # -1 for non-matching

            # For each position in the block, check if any token maps there
            block_positions = mx.arange(self.block_size)  # (block_size,)

            # Create update mask: (new_len, block_size)
            update_match = (offsets[:, None] == block_positions[None, :]) & block_mask[:, None]

            # any_update[j] = True if position j in block should be updated
            any_update = mx.any(update_match, axis=0)  # (block_size,)

            # which_token[j] = index of token that updates position j
            token_indices = mx.arange(new_len)[:, None]
            which_token = mx.sum(
                update_match.astype(mx.float32) * token_indices.astype(mx.float32),
                axis=0
            ).astype(mx.int32)  # (block_size,)

            # Gather new values - new_k is (1, heads, new_len, head_dim)
            # We need (heads, block_size, head_dim)
            new_k_gathered = mx.take(new_k[0], which_token, axis=1)  # (heads, block_size, head_dim)
            new_v_gathered = mx.take(new_v[0], which_token, axis=1)

            # Update block using mx.where
            # any_update shape: (block_size,) -> (1, block_size, 1) for broadcasting
            any_update_expanded = any_update[None, :, None]

            k_block = self.k_blocks[layer_idx][actual_block]  # (heads, block_size, head_dim)
            v_block = self.v_blocks[layer_idx][actual_block]

            updated_k = mx.where(any_update_expanded, new_k_gathered, k_block)
            updated_v = mx.where(any_update_expanded, new_v_gathered, v_block)

            # Update the block in the blocks array
            # Create mask for which block to update: (num_blocks,)
            block_update_mask = (mx.arange(self.num_blocks) == actual_block)
            block_update_mask = block_update_mask[:, None, None, None]  # (num_blocks, 1, 1, 1)

            # Broadcast updated block to full shape
            updated_k_full = mx.broadcast_to(
                updated_k[None, :, :, :],
                self.k_blocks[layer_idx].shape
            )
            updated_v_full = mx.broadcast_to(
                updated_v[None, :, :, :],
                self.v_blocks[layer_idx].shape
            )

            self.k_blocks[layer_idx] = mx.where(
                block_update_mask,
                updated_k_full,
                self.k_blocks[layer_idx]
            )
            self.v_blocks[layer_idx] = mx.where(
                block_update_mask,
                updated_v_full,
                self.v_blocks[layer_idx]
            )

        # Update length
        if layer_idx == 0:
            self.seq_lengths[seq_id] = current_len + new_len

        return self.get(layer_idx, seq_id)

    @property
    def num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self.free_blocks)

    @property
    def memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        block_size_bytes = (
            self.num_heads * self.block_size * self.head_dim * 2  # float16
        )
        used_blocks = self.num_blocks - len(self.free_blocks)
        return (used_blocks * block_size_bytes * self.num_layers * 2) / (1024 * 1024)


class RotatingKVCache:
    """Rotating buffer KV cache.

    .. deprecated::
        Use ``mlx_primitives.cache.RotatingKVCache`` instead.

    Uses a circular buffer for efficient sliding window operations.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        buffer_size: Size of rotating buffer.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        buffer_size: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        _deprecation_warning("RotatingKVCache", "cache", "RotatingKVCache")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.buffer_size = buffer_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        cache_shape = (max_batch_size, num_heads, buffer_size, head_dim)

        self.k_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_cache = [
            mx.zeros(cache_shape, dtype=dtype) for _ in range(num_layers)
        ]

        # Write position (circular)
        self.write_pos = 0
        self.total_len = 0

    def reset(self) -> None:
        """Reset the cache."""
        self.write_pos = 0
        self.total_len = 0

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached K, V in correct order."""
        if self.total_len <= self.buffer_size:
            # Haven't wrapped yet
            k = self.k_cache[layer_idx][:, :, : self.total_len, :]
            v = self.v_cache[layer_idx][:, :, : self.total_len, :]
        else:
            # Need to reorder: [write_pos:] + [:write_pos]
            k1 = self.k_cache[layer_idx][:, :, self.write_pos :, :]
            k2 = self.k_cache[layer_idx][:, :, : self.write_pos, :]
            k = mx.concatenate([k1, k2], axis=2)

            v1 = self.v_cache[layer_idx][:, :, self.write_pos :, :]
            v2 = self.v_cache[layer_idx][:, :, : self.write_pos, :]
            v = mx.concatenate([v1, v2], axis=2)

        return k, v

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with circular write using vectorized operations."""
        new_len = new_k.shape[2]

        # Compute target positions for each new token (circular)
        positions = mx.arange(new_len)
        target_positions = (self.write_pos + positions) % self.buffer_size

        # Create position indices for the buffer: (buffer_size,)
        buffer_indices = mx.arange(self.buffer_size)

        # Create a mask: (new_len, buffer_size) where mask[i, j] = 1 if target_positions[i] == j
        # Reshape for broadcasting: target_positions (new_len, 1), buffer_indices (1, buffer_size)
        update_mask = (target_positions[:, None] == buffer_indices[None, :])  # (new_len, buffer_size)

        # For each buffer position, find which new token (if any) should go there
        # If multiple new tokens map to same position (shouldn't happen normally), take the last one
        # any_update[j] = True if position j should be updated
        any_update = mx.any(update_mask, axis=0)  # (buffer_size,)

        # Find which new token index updates each position (use argmax on reversed mask for "last" token)
        # We'll use sum of weighted indices: for position j, which i has update_mask[i,j]=1?
        token_indices = mx.arange(new_len)[:, None]  # (new_len, 1)
        # Weighted sum where only the matching token contributes
        which_token = mx.sum(
            update_mask.astype(mx.float32) * token_indices.astype(mx.float32),
            axis=0
        ).astype(mx.int32)  # (buffer_size,)

        # Gather new values for each position that gets updated
        # new_k shape: (batch, heads, new_len, head_dim)
        # We want to select new_k[:, :, which_token[j], :] for each j

        # Expand which_token for gathering: (1, 1, buffer_size, 1)
        which_token_expanded = which_token[None, None, :, None]

        # Broadcast new_k to (batch, heads, buffer_size, head_dim) by gathering
        # Use take to gather along the sequence dimension
        batch_size = new_k.shape[0]
        num_heads = new_k.shape[1]

        # Gather new values: for each buffer position, get the corresponding new token
        new_k_gathered = mx.take(new_k, which_token, axis=2)  # (batch, heads, buffer_size, head_dim)
        new_v_gathered = mx.take(new_v, which_token, axis=2)  # (batch, heads, buffer_size, head_dim)

        # Reshape mask for broadcasting: (1, 1, buffer_size, 1)
        any_update_expanded = any_update[None, None, :, None]

        # Update cache using mx.where
        self.k_cache[layer_idx] = mx.where(
            any_update_expanded,
            new_k_gathered,
            self.k_cache[layer_idx]
        )
        self.v_cache[layer_idx] = mx.where(
            any_update_expanded,
            new_v_gathered,
            self.v_cache[layer_idx]
        )

        if layer_idx == 0:
            self.write_pos = (self.write_pos + new_len) % self.buffer_size
            self.total_len += new_len

        return self.get(layer_idx)

    @property
    def position_offset(self) -> int:
        """Get position offset for position encoding."""
        return max(0, self.total_len - self.buffer_size)
