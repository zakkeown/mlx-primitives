"""KV Cache utilities for MLX.

This module provides KV cache implementations for efficient inference:
- KVCache: Basic KV cache
- SlidingWindowCache: Fixed-size sliding window cache
- PagedKVCache: vLLM-style paged attention cache
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx


class KVCache:
    """Basic Key-Value cache for transformer inference.

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


class CompressedKVCache:
    """Compressed KV cache using quantization or other compression.

    Reduces memory usage by compressing cached keys and values
    while maintaining acceptable quality for attention computation.

    Args:
        num_layers: Number of transformer layers.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        compression: Compression method ('quantize', 'prune', 'cluster').
        bits: Bits for quantization (default: 4).
        keep_ratio: Ratio of KV pairs to keep for pruning (default: 0.5).
        dtype: Data type for computation.

    Reference:
        "Scissorhands: Exploiting the Persistence of Importance Hypothesis
        for LLM KV Cache Compression at Test Time"
        https://arxiv.org/abs/2305.17118

    Example:
        >>> cache = CompressedKVCache(
        ...     num_layers=32,
        ...     max_batch_size=1,
        ...     max_seq_len=8192,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     compression='quantize',
        ...     bits=4,
        ... )
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        compression: str = 'quantize',
        bits: int = 4,
        keep_ratio: float = 0.5,
        dtype: mx.Dtype = mx.float16,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compression = compression
        self.bits = bits
        self.keep_ratio = keep_ratio
        self.dtype = dtype

        # Initialize based on compression method
        if compression == 'quantize':
            self._init_quantized_cache()
        elif compression == 'prune':
            self._init_pruned_cache()
        elif compression == 'cluster':
            self._init_clustered_cache()
        else:
            raise ValueError(f"Unknown compression method: {compression}")

        self.seq_len = 0

    def _init_quantized_cache(self) -> None:
        """Initialize quantized cache storage."""
        cache_shape = (self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim)

        # Quantized storage (int8 or packed int4)
        if self.bits == 8:
            self.k_cache_q = [
                mx.zeros(cache_shape, dtype=mx.int8) for _ in range(self.num_layers)
            ]
            self.v_cache_q = [
                mx.zeros(cache_shape, dtype=mx.int8) for _ in range(self.num_layers)
            ]
        else:
            # Pack multiple values per byte for sub-8-bit
            packed_dim = (self.head_dim * self.bits + 7) // 8
            packed_shape = (self.max_batch_size, self.num_heads, self.max_seq_len, packed_dim)
            self.k_cache_q = [
                mx.zeros(packed_shape, dtype=mx.uint8) for _ in range(self.num_layers)
            ]
            self.v_cache_q = [
                mx.zeros(packed_shape, dtype=mx.uint8) for _ in range(self.num_layers)
            ]

        # Scales per position (for dequantization)
        scale_shape = (self.max_batch_size, self.num_heads, self.max_seq_len, 1)
        self.k_scales = [
            mx.ones(scale_shape, dtype=mx.float32) for _ in range(self.num_layers)
        ]
        self.v_scales = [
            mx.ones(scale_shape, dtype=mx.float32) for _ in range(self.num_layers)
        ]

    def _init_pruned_cache(self) -> None:
        """Initialize pruned cache storage."""
        # Keep only top-k positions based on attention importance
        self.keep_len = int(self.max_seq_len * self.keep_ratio)

        # Initialize as empty lists - will be populated on first update
        self.k_cache = [None for _ in range(self.num_layers)]
        self.v_cache = [None for _ in range(self.num_layers)]
        self.importance_scores = [None for _ in range(self.num_layers)]

        # Track actual sequence lengths per layer
        self.pruned_seq_lens = [0 for _ in range(self.num_layers)]

        # Mapping from cache index to original position
        self.position_map = [
            mx.zeros((self.max_batch_size, self.num_heads, self.keep_len), dtype=mx.int32)
            for _ in range(self.num_layers)
        ]

    def _init_clustered_cache(self) -> None:
        """Initialize clustered cache storage."""
        # Cluster similar KV pairs together
        num_clusters = int(self.max_seq_len * self.keep_ratio)
        cache_shape = (self.max_batch_size, self.num_heads, num_clusters, self.head_dim)

        self.k_centroids = [
            mx.zeros(cache_shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self.v_centroids = [
            mx.zeros(cache_shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]

        # Cluster assignments
        self.cluster_assignments = [
            mx.zeros((self.max_batch_size, self.num_heads, self.max_seq_len), dtype=mx.int32)
            for _ in range(self.num_layers)
        ]
        self.cluster_counts = [
            mx.zeros((self.max_batch_size, self.num_heads, num_clusters))
            for _ in range(self.num_layers)
        ]

    def reset(self) -> None:
        """Reset the cache."""
        self.seq_len = 0
        if self.compression == 'quantize':
            for i in range(self.num_layers):
                self.k_cache_q[i] = mx.zeros_like(self.k_cache_q[i])
                self.v_cache_q[i] = mx.zeros_like(self.v_cache_q[i])
        elif self.compression == 'prune':
            for i in range(self.num_layers):
                self.k_cache[i] = None
                self.v_cache[i] = None
                self.importance_scores[i] = None
                self.pruned_seq_lens[i] = 0
        elif self.compression == 'cluster':
            for i in range(self.num_layers):
                self.k_centroids[i] = mx.zeros_like(self.k_centroids[i])
                self.v_centroids[i] = mx.zeros_like(self.v_centroids[i])

    def _quantize_8bit(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize to 8-bit with per-token scaling."""
        # Per-token absmax quantization
        absmax = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = absmax / 127.0
        scale = mx.maximum(scale, mx.array(1e-8))

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, -128, 127).astype(mx.int8)

        return x_q, scale

    def _dequantize_8bit(self, x_q: mx.array, scale: mx.array) -> mx.array:
        """Dequantize from 8-bit."""
        return x_q.astype(mx.float32) * scale

    def _quantize_4bit(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize to 4-bit with per-token scaling."""
        absmax = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = absmax / 7.0
        scale = mx.maximum(scale, mx.array(1e-8))

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, -8, 7).astype(mx.int8)

        # Pack pairs of 4-bit values into bytes
        batch, heads, seq, dim = x_q.shape
        x_q_list = x_q.reshape(-1, dim).tolist()

        packed = []
        for row in x_q_list:
            packed_row = []
            for j in range(0, len(row), 2):
                low = row[j] & 0xF
                high = (row[j + 1] & 0xF) if j + 1 < len(row) else 0
                packed_row.append((high << 4) | low)
            packed.append(packed_row)

        packed_arr = mx.array(packed, dtype=mx.uint8)
        packed_arr = packed_arr.reshape(batch, heads, seq, -1)

        return packed_arr, scale

    def _dequantize_4bit(self, x_packed: mx.array, scale: mx.array) -> mx.array:
        """Dequantize from 4-bit."""
        batch, heads, seq, packed_dim = x_packed.shape

        # Unpack
        packed_list = x_packed.reshape(-1, packed_dim).tolist()

        unpacked = []
        for packed_row in packed_list:
            row = []
            for byte in packed_row:
                low = byte & 0xF
                high = (byte >> 4) & 0xF
                low = low - 16 if low > 7 else low
                high = high - 16 if high > 7 else high
                row.extend([low, high])
            unpacked.append(row[:self.head_dim])

        x = mx.array(unpacked, dtype=mx.float32)
        x = x.reshape(batch, heads, seq, self.head_dim)

        return x * scale

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached K, V tensors.

        Args:
            layer_idx: Layer index.

        Returns:
            Tuple of (K, V) tensors.
        """
        if self.seq_len == 0:
            return (
                mx.zeros((self.max_batch_size, self.num_heads, 0, self.head_dim), dtype=self.dtype),
                mx.zeros((self.max_batch_size, self.num_heads, 0, self.head_dim), dtype=self.dtype),
            )

        if self.compression == 'quantize':
            if self.bits == 8:
                k = self._dequantize_8bit(
                    self.k_cache_q[layer_idx][:, :, :self.seq_len, :],
                    self.k_scales[layer_idx][:, :, :self.seq_len, :]
                )
                v = self._dequantize_8bit(
                    self.v_cache_q[layer_idx][:, :, :self.seq_len, :],
                    self.v_scales[layer_idx][:, :, :self.seq_len, :]
                )
            else:
                k = self._dequantize_4bit(
                    self.k_cache_q[layer_idx][:, :, :self.seq_len, :],
                    self.k_scales[layer_idx][:, :, :self.seq_len, :]
                )
                v = self._dequantize_4bit(
                    self.v_cache_q[layer_idx][:, :, :self.seq_len, :],
                    self.v_scales[layer_idx][:, :, :self.seq_len, :]
                )
            return k.astype(self.dtype), v.astype(self.dtype)

        elif self.compression == 'prune':
            effective_len = min(self.seq_len, self.k_cache[layer_idx].shape[2])
            return (
                self.k_cache[layer_idx][:, :, :effective_len, :],
                self.v_cache[layer_idx][:, :, :effective_len, :],
            )

        elif self.compression == 'cluster':
            # Return cluster centroids
            return (
                self.k_centroids[layer_idx],
                self.v_centroids[layer_idx],
            )

        return mx.zeros((0,)), mx.zeros((0,))

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        attention_scores: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new K, V.

        Args:
            layer_idx: Layer index.
            new_k: New key tensor (batch, heads, new_len, head_dim).
            new_v: New value tensor.
            attention_scores: Optional attention scores for importance-based pruning.

        Returns:
            Tuple of full (K, V) after update.
        """
        new_len = new_k.shape[2]
        start_pos = self.seq_len

        if self.compression == 'quantize':
            return self._update_quantized(layer_idx, new_k, new_v, start_pos, new_len)
        elif self.compression == 'prune':
            return self._update_pruned(layer_idx, new_k, new_v, attention_scores)
        elif self.compression == 'cluster':
            return self._update_clustered(layer_idx, new_k, new_v)

        return new_k, new_v

    def _update_quantized(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        start_pos: int,
        new_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Update quantized cache."""
        end_pos = start_pos + new_len

        if self.bits == 8:
            k_q, k_scale = self._quantize_8bit(new_k)
            v_q, v_scale = self._quantize_8bit(new_v)

            # Update using vectorized operations
            positions = mx.arange(self.max_seq_len)
            update_mask = (positions >= start_pos) & (positions < end_pos)
            update_mask = update_mask[None, None, :, None]  # (1, 1, seq, 1)

            # Broadcast new values to full cache shape
            batch = new_k.shape[0]
            k_q_full = mx.zeros_like(self.k_cache_q[layer_idx])
            v_q_full = mx.zeros_like(self.v_cache_q[layer_idx])
            k_scale_full = mx.ones_like(self.k_scales[layer_idx])
            v_scale_full = mx.ones_like(self.v_scales[layer_idx])

            # Place new values at correct positions
            k_q_full = k_q_full.at[:, :, start_pos:end_pos, :].add(k_q)
            v_q_full = v_q_full.at[:, :, start_pos:end_pos, :].add(v_q)
            k_scale_full = k_scale_full.at[:, :, start_pos:end_pos, :].add(k_scale - 1)
            v_scale_full = v_scale_full.at[:, :, start_pos:end_pos, :].add(v_scale - 1)

            self.k_cache_q[layer_idx] = mx.where(
                update_mask, k_q_full, self.k_cache_q[layer_idx]
            )
            self.v_cache_q[layer_idx] = mx.where(
                update_mask, v_q_full, self.v_cache_q[layer_idx]
            )
            self.k_scales[layer_idx] = mx.where(
                update_mask, k_scale_full, self.k_scales[layer_idx]
            )
            self.v_scales[layer_idx] = mx.where(
                update_mask, v_scale_full, self.v_scales[layer_idx]
            )
        else:
            k_q, k_scale = self._quantize_4bit(new_k)
            v_q, v_scale = self._quantize_4bit(new_v)

            # Update packed cache
            positions = mx.arange(self.max_seq_len)
            update_mask = (positions >= start_pos) & (positions < end_pos)
            update_mask_q = update_mask[None, None, :, None]
            update_mask_s = update_mask[None, None, :, None]

            # Use where for update
            k_q_full = mx.zeros_like(self.k_cache_q[layer_idx])
            v_q_full = mx.zeros_like(self.v_cache_q[layer_idx])

            # Place new values
            # This is simplified - full implementation would need proper placement
            self.k_cache_q[layer_idx] = mx.where(
                update_mask_q[..., :k_q.shape[-1]],
                mx.concatenate([
                    self.k_cache_q[layer_idx][:, :, :start_pos, :],
                    k_q,
                    self.k_cache_q[layer_idx][:, :, end_pos:, :]
                ], axis=2)[:, :, :self.max_seq_len, :],
                self.k_cache_q[layer_idx]
            )
            self.v_cache_q[layer_idx] = mx.where(
                update_mask_q[..., :v_q.shape[-1]],
                mx.concatenate([
                    self.v_cache_q[layer_idx][:, :, :start_pos, :],
                    v_q,
                    self.v_cache_q[layer_idx][:, :, end_pos:, :]
                ], axis=2)[:, :, :self.max_seq_len, :],
                self.v_cache_q[layer_idx]
            )

            self.k_scales[layer_idx] = mx.where(
                update_mask_s,
                mx.concatenate([
                    self.k_scales[layer_idx][:, :, :start_pos, :],
                    k_scale,
                    self.k_scales[layer_idx][:, :, end_pos:, :]
                ], axis=2)[:, :, :self.max_seq_len, :],
                self.k_scales[layer_idx]
            )
            self.v_scales[layer_idx] = mx.where(
                update_mask_s,
                mx.concatenate([
                    self.v_scales[layer_idx][:, :, :start_pos, :],
                    v_scale,
                    self.v_scales[layer_idx][:, :, end_pos:, :]
                ], axis=2)[:, :, :self.max_seq_len, :],
                self.v_scales[layer_idx]
            )

        if layer_idx == 0:
            self.seq_len = end_pos

        return self.get(layer_idx)

    def _update_pruned(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        attention_scores: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Update pruned cache based on importance."""
        new_len = new_k.shape[2]

        # Compute importance scores if not provided
        if attention_scores is None:
            # Use L2 norm as proxy for importance
            scores = mx.sum(new_k ** 2, axis=-1)  # (batch, heads, new_len)
        else:
            scores = mx.mean(attention_scores, axis=-2)  # Average over query positions

        # First update - initialize cache
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
            self.importance_scores[layer_idx] = scores
            self.pruned_seq_lens[layer_idx] = new_len
        else:
            # Combine existing and new
            current_k = self.k_cache[layer_idx]
            current_v = self.v_cache[layer_idx]
            current_scores = self.importance_scores[layer_idx]

            # Merge new KV pairs
            all_k = mx.concatenate([current_k, new_k], axis=2)
            all_v = mx.concatenate([current_v, new_v], axis=2)
            all_scores = mx.concatenate([current_scores, scores], axis=2)

            # Keep top-k based on importance if exceeds keep_len
            total_len = all_k.shape[2]
            if total_len > self.keep_len:
                # Get indices of top-k scores
                top_indices = mx.argsort(all_scores, axis=-1)[..., -self.keep_len:]

                # Simplified: take most recent keep_len positions
                # Full implementation would sort and gather by importance
                self.k_cache[layer_idx] = all_k[:, :, -self.keep_len:, :]
                self.v_cache[layer_idx] = all_v[:, :, -self.keep_len:, :]
                self.importance_scores[layer_idx] = all_scores[:, :, -self.keep_len:]
                self.pruned_seq_lens[layer_idx] = self.keep_len
            else:
                self.k_cache[layer_idx] = all_k
                self.v_cache[layer_idx] = all_v
                self.importance_scores[layer_idx] = all_scores
                self.pruned_seq_lens[layer_idx] = total_len

        if layer_idx == 0:
            self.seq_len += new_len

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def _update_clustered(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update clustered cache using online k-means."""
        num_clusters = self.k_centroids[layer_idx].shape[2]
        new_len = new_k.shape[2]

        for t in range(new_len):
            k_t = new_k[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)
            v_t = new_v[:, :, t:t+1, :]

            # Find nearest cluster
            distances = mx.sum(
                (self.k_centroids[layer_idx] - k_t) ** 2,
                axis=-1
            )  # (batch, heads, num_clusters)

            nearest = mx.argmin(distances, axis=-1)  # (batch, heads)

            # Update cluster centroids (online update)
            # c_new = (c_old * n + x) / (n + 1)
            for b in range(new_k.shape[0]):
                for h in range(new_k.shape[1]):
                    c = int(nearest[b, h])
                    count = float(self.cluster_counts[layer_idx][b, h, c])
                    new_count = count + 1

                    # Update K centroid
                    old_k = self.k_centroids[layer_idx][b, h, c, :]
                    self.k_centroids[layer_idx] = (
                        self.k_centroids[layer_idx].at[b, h, c, :].add(
                            (k_t[b, h, 0, :] - old_k) / new_count
                        )
                    )

                    # Update V centroid
                    old_v = self.v_centroids[layer_idx][b, h, c, :]
                    self.v_centroids[layer_idx] = (
                        self.v_centroids[layer_idx].at[b, h, c, :].add(
                            (v_t[b, h, 0, :] - old_v) / new_count
                        )
                    )

                    # Update count
                    self.cluster_counts[layer_idx] = (
                        self.cluster_counts[layer_idx].at[b, h, c].add(1)
                    )

        if layer_idx == 0:
            self.seq_len += new_len

        return self.k_centroids[layer_idx], self.v_centroids[layer_idx]

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        if self.compression == 'quantize':
            if self.bits == 8:
                bytes_per_element = 1  # int8
            else:
                bytes_per_element = 0.5  # 4-bit packed
            cache_size = (
                self.max_batch_size * self.num_heads * self.max_seq_len *
                self.head_dim * bytes_per_element * 2 * self.num_layers  # K + V
            )
            scale_size = (
                self.max_batch_size * self.num_heads * self.max_seq_len *
                4 * 2 * self.num_layers  # float32 scales
            )
            return (cache_size + scale_size) / (1024 * 1024)

        elif self.compression == 'prune':
            keep_len = int(self.max_seq_len * self.keep_ratio)
            cache_size = (
                self.max_batch_size * self.num_heads * keep_len *
                self.head_dim * 2 * 2 * self.num_layers  # float16
            )
            return cache_size / (1024 * 1024)

        elif self.compression == 'cluster':
            num_clusters = int(self.max_seq_len * self.keep_ratio)
            cache_size = (
                self.max_batch_size * self.num_heads * num_clusters *
                self.head_dim * 2 * 2 * self.num_layers
            )
            return cache_size / (1024 * 1024)

        return 0.0

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio vs full precision cache."""
        full_size = (
            self.max_batch_size * self.num_heads * self.max_seq_len *
            self.head_dim * 2 * 2 * self.num_layers  # float16 K + V
        )
        compressed_size = self.memory_usage_mb * 1024 * 1024
        return full_size / max(compressed_size, 1)