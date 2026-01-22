"""Protocol definitions for KV cache implementations.

This module defines the interfaces that KV cache implementations should follow.
Use these protocols for type hints when writing generic cache code.

Example:
    >>> def process_cache(cache: KVCacheProtocol) -> None:
    ...     seq_id = cache.create_sequence()
    ...     cache.update(seq_id, k, v, layer_idx=0)
    ...     k, v = cache.get_kv(seq_id, layer_idx=0)
    ...     cache.delete_sequence(seq_id)
"""

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import mlx.core as mx


@runtime_checkable
class SimpleCacheProtocol(Protocol):
    """Protocol for simple KV caches (SimpleKVCache, SlidingWindowCache, etc.).

    Simple caches manage a single sequence or batch and provide a straightforward
    update/retrieve interface without explicit sequence IDs.

    Example:
        >>> cache: SimpleCacheProtocol = SimpleKVCache(...)
        >>> k_full, v_full = cache.update(k_new, v_new)
        >>> cache.reset()
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the cache (batch_size, max_seq_len, num_heads, head_dim)."""
        ...

    @property
    def dtype(self) -> mx.Dtype:
        """Data type of cached values."""
        ...

    @property
    def current_length(self) -> int:
        """Current number of cached tokens."""
        ...

    def update(
        self,
        k: mx.array,
        v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Add new K/V to the cache and return full cached K/V.

        Args:
            k: New keys, shape (batch, new_tokens, heads, dim).
            v: New values, shape (batch, new_tokens, heads, dim).

        Returns:
            Tuple of (full_k, full_v) including all cached tokens.
        """
        ...

    def reset(self) -> None:
        """Clear the cache and reset position to 0."""
        ...


@runtime_checkable
class KVCacheProtocol(Protocol):
    """Protocol for multi-sequence KV caches (KVCache, PagedKVCache, etc.).

    Multi-sequence caches manage multiple sequences with explicit IDs and
    support features like forking, layer-specific storage, and eviction.

    Example:
        >>> cache: KVCacheProtocol = KVCache(config)
        >>> seq_id = cache.create_sequence()
        >>> cache.update(seq_id, k, v, layer_idx=0)
        >>> k, v = cache.get_kv(seq_id, layer_idx=0)
        >>> forked_id = cache.fork_sequence(seq_id)
        >>> cache.delete_sequence(seq_id)
    """

    @property
    def num_layers(self) -> int:
        """Number of transformer layers this cache supports."""
        ...

    def create_sequence(
        self,
        initial_k: Optional[mx.array] = None,
        initial_v: Optional[mx.array] = None,
    ) -> int:
        """Create a new sequence, optionally with initial K/V.

        Args:
            initial_k: Optional initial keys.
            initial_v: Optional initial values.

        Returns:
            Sequence ID for the new sequence.
        """
        ...

    def update(
        self,
        sequence_id: int,
        k: mx.array,
        v: mx.array,
        layer_idx: int,
        position: Optional[int] = None,
    ) -> None:
        """Add or update K/V for a sequence at a specific layer.

        Args:
            sequence_id: ID of the sequence to update.
            k: Keys to add/update.
            v: Values to add/update.
            layer_idx: Transformer layer index.
            position: Optional position to update at. If None, appends.
        """
        ...

    def get_kv(
        self,
        sequence_id: int,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """Get cached K/V for a sequence at a specific layer.

        Args:
            sequence_id: ID of the sequence.
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (k, v) arrays.
        """
        ...

    def delete_sequence(self, sequence_id: int) -> None:
        """Delete a sequence and free its memory.

        Args:
            sequence_id: ID of the sequence to delete.
        """
        ...

    def fork_sequence(self, sequence_id: int) -> int:
        """Fork a sequence (copy-on-write for shared prefixes).

        Args:
            sequence_id: ID of the sequence to fork.

        Returns:
            New sequence ID for the forked sequence.
        """
        ...

    def get_sequence_length(self, sequence_id: int) -> int:
        """Get the number of cached tokens for a sequence.

        Args:
            sequence_id: ID of the sequence.

        Returns:
            Number of cached tokens.
        """
        ...

    def clear(self) -> None:
        """Clear all cached data."""
        ...

    def get_all_sequence_ids(self) -> List[int]:
        """Get all active sequence IDs.

        Returns:
            List of sequence IDs.
        """
        ...
