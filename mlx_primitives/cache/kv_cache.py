"""Core KV cache implementation for transformer inference.

This module provides the main KVCache class that integrates block allocation,
page tables, and eviction policies for efficient KV storage.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx

from mlx_primitives.cache.block_allocator import (
    BlockAllocator,
    BlockConfig,
    calculate_num_blocks,
    get_optimal_block_size,
)
from mlx_primitives.cache.eviction import (
    CacheMemoryStats,
    EvictionPolicy,
    LRUEvictionPolicy,
    MemoryBudgetManager,
)
from mlx_primitives.cache.page_table import PageTable


@dataclass
class KVCacheConfig:
    """Configuration for KV cache.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        num_layers: Number of transformer layers.
        max_memory_gb: Maximum memory for cache in GB.
        block_size: Tokens per block (auto if None).
        dtype: Data type for storage.
        enable_cow: Enable copy-on-write for prefix sharing.
    """

    num_heads: int
    head_dim: int
    num_layers: int
    max_memory_gb: float = 8.0
    block_size: Optional[int] = None
    dtype: mx.Dtype = mx.float16
    enable_cow: bool = True

    def __post_init__(self):
        """Set optimal block size if not specified."""
        if self.block_size is None:
            self.block_size = get_optimal_block_size(
                self.head_dim, self.num_heads, self.dtype
            )

    @property
    def block_config(self) -> BlockConfig:
        """Get BlockConfig for this cache configuration."""
        return BlockConfig(
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
        )

    def calculate_capacity(self) -> Tuple[int, int]:
        """Calculate cache capacity.

        Returns:
            Tuple of (num_blocks_per_layer, max_tokens).
        """
        blocks_per_layer = calculate_num_blocks(
            self.max_memory_gb / self.num_layers,
            self.block_config,
        )
        max_tokens = blocks_per_layer * self.block_size
        return blocks_per_layer, max_tokens


class KVCache:
    """Paged KV cache for transformer inference.

    Key Features:
    - Paged attention with block-based storage
    - Dynamic memory allocation without fragmentation
    - Prefix sharing for beam search and continuous batching
    - Configurable eviction policies

    Example:
        >>> config = KVCacheConfig(
        ...     num_heads=32, head_dim=128, num_layers=32,
        ...     max_memory_gb=8.0, block_size=16
        ... )
        >>> cache = KVCache(config)
        >>>
        >>> # Add tokens for a sequence
        >>> seq_id = cache.create_sequence()
        >>> cache.update(seq_id, k_new, v_new, layer_idx=0)
        >>>
        >>> # Get KV for attention
        >>> k, v = cache.get_kv(seq_id, layer_idx=0)
    """

    def __init__(
        self,
        config: KVCacheConfig,
        eviction_policy: Optional[EvictionPolicy] = None,
    ):
        """Initialize the KV cache.

        Args:
            config: Cache configuration.
            eviction_policy: Policy for eviction. Defaults to LRU.
        """
        self._config = config
        self._eviction_policy = eviction_policy or LRUEvictionPolicy()

        # Calculate blocks per layer
        blocks_per_layer, _ = config.calculate_capacity()

        # Create per-layer allocators and page tables
        self._allocators: List[BlockAllocator] = []
        self._page_tables: List[PageTable] = []

        for _ in range(config.num_layers):
            allocator = BlockAllocator(
                config.block_config,
                num_blocks=blocks_per_layer,
                enable_cow=config.enable_cow,
            )
            page_table = PageTable(allocator)

            self._allocators.append(allocator)
            self._page_tables.append(page_table)

        # Memory budget manager
        total_memory = sum(a.total_memory_bytes for a in self._allocators)
        self._budget_manager = MemoryBudgetManager(
            max_memory_bytes=total_memory,
            eviction_policy=self._eviction_policy,
        )

        # Mapping from user sequence IDs to internal layer sequence IDs
        self._sequence_ids: Dict[int, List[int]] = {}
        self._next_user_seq_id = 0

    @property
    def config(self) -> KVCacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        return self._config.num_layers

    def create_sequence(
        self,
        initial_k: Optional[mx.array] = None,
        initial_v: Optional[mx.array] = None,
    ) -> int:
        """Create a new sequence, optionally with initial KV.

        Args:
            initial_k: Initial keys, shape (seq_len, num_heads, head_dim).
            initial_v: Initial values, shape (seq_len, num_heads, head_dim).

        Returns:
            User-facing sequence ID.
        """
        user_seq_id = self._next_user_seq_id
        self._next_user_seq_id += 1

        # Create sequence in each layer's page table
        layer_seq_ids = []
        initial_tokens = initial_k.shape[0] if initial_k is not None else 0

        for layer_idx in range(self.num_layers):
            layer_seq_id = self._page_tables[layer_idx].create_sequence(
                initial_tokens=initial_tokens
            )
            layer_seq_ids.append(layer_seq_id)

        self._sequence_ids[user_seq_id] = layer_seq_ids
        self._eviction_policy.on_create(user_seq_id)

        # Store initial KV if provided
        if initial_k is not None and initial_v is not None:
            for layer_idx in range(self.num_layers):
                self._store_kv(user_seq_id, initial_k, initial_v, layer_idx)

        return user_seq_id

    def _store_kv(
        self,
        user_seq_id: int,
        k: mx.array,
        v: mx.array,
        layer_idx: int,
        start_position: Optional[int] = None,
    ) -> None:
        """Store KV data for a sequence at a specific layer.

        Args:
            user_seq_id: User sequence ID.
            k: Keys to store, shape (seq_len, num_heads, head_dim).
            v: Values to store, shape (seq_len, num_heads, head_dim).
            layer_idx: Layer index.
            start_position: Position to start writing at. If None, writes at
                position (num_tokens - seq_len), assuming extend_sequence was
                already called.
        """
        layer_seq_id = self._sequence_ids[user_seq_id][layer_idx]
        page_table = self._page_tables[layer_idx]
        allocator = self._allocators[layer_idx]

        metadata = page_table.get_sequence_metadata(layer_seq_id)
        block_size = self._config.block_size

        seq_len = k.shape[0]
        tokens_written = 0

        # Calculate starting write position
        # After extend_sequence, num_tokens already includes the new tokens,
        # so we need to write starting at (num_tokens - seq_len)
        if start_position is None:
            write_start = metadata.num_tokens - seq_len
        else:
            write_start = start_position

        while tokens_written < seq_len:
            # Calculate block and position
            total_pos = write_start + tokens_written
            block_idx = total_pos // block_size
            pos_in_block = total_pos % block_size

            # Ensure block exists
            if block_idx >= len(metadata.block_table):
                raise RuntimeError(
                    f"Block index {block_idx} out of range. "
                    f"Have {len(metadata.block_table)} blocks for {metadata.num_tokens} tokens."
                )

            block_id = metadata.block_table[block_idx]

            # Calculate how many tokens fit in this block
            tokens_in_block = min(
                seq_len - tokens_written,
                block_size - pos_in_block,
            )

            # Write to block
            k_chunk = k[tokens_written : tokens_written + tokens_in_block]
            v_chunk = v[tokens_written : tokens_written + tokens_in_block]
            allocator.set_block_data(block_id, k_chunk, v_chunk, pos_in_block)

            tokens_written += tokens_in_block

    def update(
        self,
        sequence_id: int,
        k: mx.array,
        v: mx.array,
        layer_idx: int,
        position: Optional[int] = None,
    ) -> None:
        """Append or update KV at position.

        Args:
            sequence_id: User sequence ID.
            k: Keys to add, shape (new_tokens, num_heads, head_dim).
            v: Values to add, shape (new_tokens, num_heads, head_dim).
            layer_idx: Transformer layer index.
            position: If specified, update at position. Otherwise append.
        """
        if sequence_id not in self._sequence_ids:
            raise KeyError(f"Sequence {sequence_id} not found")

        self._eviction_policy.on_access(sequence_id)

        layer_seq_id = self._sequence_ids[sequence_id][layer_idx]
        page_table = self._page_tables[layer_idx]

        if position is None:
            # Append mode
            num_new_tokens = k.shape[0]
            page_table.extend_sequence(layer_seq_id, num_new_tokens)
            self._store_kv(sequence_id, k, v, layer_idx)
        else:
            # Update at position (for speculative decoding rollback)
            # Truncate first, then store
            page_table.truncate_sequence(layer_seq_id, position)
            page_table.extend_sequence(layer_seq_id, k.shape[0])
            self._store_kv(sequence_id, k, v, layer_idx)

    def get_kv(
        self,
        sequence_id: int,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """Get all cached K, V for a sequence (materialized).

        Args:
            sequence_id: User sequence ID.
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (K, V), each shape (num_tokens, num_heads, head_dim).
        """
        if sequence_id not in self._sequence_ids:
            raise KeyError(f"Sequence {sequence_id} not found")

        self._eviction_policy.on_access(sequence_id)

        layer_seq_id = self._sequence_ids[sequence_id][layer_idx]
        page_table = self._page_tables[layer_idx]
        allocator = self._allocators[layer_idx]

        metadata = page_table.get_sequence_metadata(layer_seq_id)

        if metadata.num_tokens == 0:
            # Empty sequence
            shape = (0, self._config.num_heads, self._config.head_dim)
            return mx.zeros(shape, dtype=self._config.dtype), mx.zeros(
                shape, dtype=self._config.dtype
            )

        # Gather all KV from blocks
        block_size = self._config.block_size
        k_parts = []
        v_parts = []

        for block_idx, block_id in enumerate(metadata.block_table):
            k_block, v_block = allocator.get_block_data(block_id)

            # Handle partial last block
            if block_idx == len(metadata.block_table) - 1:
                tokens_in_block = metadata.num_tokens_in_last_block
                k_parts.append(k_block[:tokens_in_block])
                v_parts.append(v_block[:tokens_in_block])
            else:
                k_parts.append(k_block)
                v_parts.append(v_block)

        k = mx.concatenate(k_parts, axis=0)
        v = mx.concatenate(v_parts, axis=0)

        return k, v

    def get_kv_paged(
        self,
        sequence_ids: List[int],
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Get data for paged attention.

        Args:
            sequence_ids: List of user sequence IDs.
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (K_pool, V_pool, block_tables, context_lens):
            - K_pool: (num_blocks, block_size, num_heads, head_dim)
            - V_pool: (num_blocks, block_size, num_heads, head_dim)
            - block_tables: (batch, max_blocks) with -1 padding
            - context_lens: (batch,) number of tokens per sequence
        """
        for seq_id in sequence_ids:
            if seq_id not in self._sequence_ids:
                raise KeyError(f"Sequence {seq_id} not found")
            self._eviction_policy.on_access(seq_id)

        allocator = self._allocators[layer_idx]
        page_table = self._page_tables[layer_idx]

        k_pool, v_pool = allocator.get_pools()

        # Get layer-specific sequence IDs
        layer_seq_ids = [self._sequence_ids[sid][layer_idx] for sid in sequence_ids]

        block_tables = page_table.get_block_table_tensor(layer_seq_ids)
        context_lens = page_table.get_context_lengths(layer_seq_ids)

        return k_pool, v_pool, block_tables, context_lens

    def fork_sequence(self, sequence_id: int) -> int:
        """Fork a sequence for beam search (sharing prefix via COW).

        Args:
            sequence_id: Sequence to fork.

        Returns:
            New sequence ID for the fork.
        """
        if sequence_id not in self._sequence_ids:
            raise KeyError(f"Sequence {sequence_id} not found")

        new_user_seq_id = self._next_user_seq_id
        self._next_user_seq_id += 1

        # Fork in each layer
        layer_seq_ids = []
        for layer_idx in range(self.num_layers):
            old_layer_seq_id = self._sequence_ids[sequence_id][layer_idx]
            new_layer_seq_id = self._page_tables[layer_idx].fork_sequence(
                old_layer_seq_id
            )
            layer_seq_ids.append(new_layer_seq_id)

        self._sequence_ids[new_user_seq_id] = layer_seq_ids
        self._eviction_policy.on_create(new_user_seq_id)

        return new_user_seq_id

    def delete_sequence(self, sequence_id: int) -> None:
        """Remove a sequence and free its memory.

        Args:
            sequence_id: User sequence ID to delete.

        Raises:
            KeyError: If sequence does not exist.
        """
        if sequence_id not in self._sequence_ids:
            raise KeyError(f"Sequence {sequence_id} not found")

        # Delete from each layer
        for layer_idx in range(self.num_layers):
            layer_seq_id = self._sequence_ids[sequence_id][layer_idx]
            self._page_tables[layer_idx].delete_sequence(layer_seq_id)

        del self._sequence_ids[sequence_id]
        self._eviction_policy.on_delete(sequence_id)

    def get_sequence_length(self, sequence_id: int) -> int:
        """Get the number of cached tokens for a sequence.

        Args:
            sequence_id: User sequence ID.

        Returns:
            Number of cached tokens.
        """
        if sequence_id not in self._sequence_ids:
            raise KeyError(f"Sequence {sequence_id} not found")

        # All layers should have the same count, use layer 0
        layer_seq_id = self._sequence_ids[sequence_id][0]
        metadata = self._page_tables[0].get_sequence_metadata(layer_seq_id)
        return metadata.num_tokens

    def clear(self) -> None:
        """Clear all cached data."""
        for page_table in self._page_tables:
            page_table.clear()
        for allocator in self._allocators:
            allocator.clear()
        self._sequence_ids.clear()

    @property
    def memory_stats(self) -> CacheMemoryStats:
        """Get current memory usage statistics."""
        total_bytes = sum(a.total_memory_bytes for a in self._allocators)
        used_bytes = sum(a.memory_usage_bytes for a in self._allocators)
        num_blocks_allocated = sum(a.num_allocated_blocks for a in self._allocators)
        num_blocks_free = sum(a.num_free_blocks for a in self._allocators)

        return CacheMemoryStats(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            num_sequences=len(self._sequence_ids),
            num_blocks_allocated=num_blocks_allocated,
            num_blocks_free=num_blocks_free,
        )

    def get_all_sequence_ids(self) -> List[int]:
        """Get all active sequence IDs.

        Returns:
            List of user sequence IDs.
        """
        return list(self._sequence_ids.keys())

    def evict_if_needed(
        self,
        bytes_needed: int,
        protected_sequences: Optional[List[int]] = None,
    ) -> int:
        """Evict sequences if needed to free memory.

        Args:
            bytes_needed: Bytes that need to be freed.
            protected_sequences: Sequences that cannot be evicted.

        Returns:
            Number of sequences evicted.
        """
        # Check if eviction is needed
        stats = self.memory_stats
        if stats.used_bytes + bytes_needed <= self._budget_manager.max_memory_bytes:
            return 0

        # Evict from layer 0 page table (all layers have same sequence IDs)
        evicted = self._budget_manager.evict_to_free(
            bytes_needed,
            self._page_tables[0],
            protected_sequences,
        )

        return evicted


class MultiLayerKVCache:
    """Convenience wrapper for using KVCache with a model.

    Provides a simpler interface for common use cases.

    Example:
        >>> cache = MultiLayerKVCache.from_model_config(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_memory_gb=8.0,
        ... )
        >>>
        >>> # In attention layer
        >>> k, v = cache.get_or_update(
        ...     sequence_id=seq_id,
        ...     layer_idx=layer_idx,
        ...     new_k=k_proj,
        ...     new_v=v_proj,
        ... )
    """

    def __init__(self, cache: KVCache):
        """Initialize with a KVCache instance.

        Args:
            cache: The underlying KVCache.
        """
        self._cache = cache

    @classmethod
    def from_model_config(
        cls,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_memory_gb: float = 8.0,
        block_size: Optional[int] = None,
        dtype: mx.Dtype = mx.float16,
    ) -> "MultiLayerKVCache":
        """Create cache from model configuration.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            max_memory_gb: Maximum memory in GB.
            block_size: Tokens per block (auto if None).
            dtype: Data type for storage.

        Returns:
            Initialized MultiLayerKVCache.
        """
        config = KVCacheConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            max_memory_gb=max_memory_gb,
            block_size=block_size,
            dtype=dtype,
        )
        return cls(KVCache(config))

    def create_sequence(self) -> int:
        """Create a new sequence."""
        return self._cache.create_sequence()

    def get_or_update(
        self,
        sequence_id: int,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Get existing KV and append new KV.

        Args:
            sequence_id: Sequence ID.
            layer_idx: Layer index.
            new_k: New keys to append, shape (new_tokens, heads, dim).
            new_v: New values to append, shape (new_tokens, heads, dim).

        Returns:
            Full (K, V) including new tokens.
        """
        # Append new KV
        self._cache.update(sequence_id, new_k, new_v, layer_idx)

        # Return full KV
        return self._cache.get_kv(sequence_id, layer_idx)

    def delete_sequence(self, sequence_id: int) -> None:
        """Delete a sequence."""
        self._cache.delete_sequence(sequence_id)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    @property
    def inner(self) -> KVCache:
        """Access the underlying KVCache."""
        return self._cache
