"""Simple KV cache implementations for transformer inference.

This module provides simpler KV cache implementations for common use cases:
- SimpleKVCache: Basic pre-allocated cache with batch support
- SlidingWindowCache: Maintains a sliding window of recent tokens
- RotatingKVCache: Circular buffer cache with configurable size

These are simpler alternatives to the paged KVCache for use cases
that don't require dynamic memory management.
"""

from typing import Optional, Tuple

import mlx.core as mx


class SimpleKVCache:
    """Simple pre-allocated KV cache for transformer inference.

    Pre-allocates memory for a fixed batch size and maximum sequence length.
    This is simpler than the paged KVCache but less memory efficient for
    variable-length sequences.

    Example:
        >>> cache = SimpleKVCache(
        ...     batch_size=4,
        ...     num_heads=8,
        ...     head_dim=64,
        ...     max_seq_len=2048,
        ... )
        >>> # Add new KV
        >>> cache.update(new_k, new_v)
        >>> # Get cached KV
        >>> k, v = cache.get_kv()
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize the cache.

        Args:
            batch_size: Number of sequences in batch.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            max_seq_len: Maximum sequence length to cache.
            dtype: Data type for storage (default: float16 for memory efficiency).
        """
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._dtype = dtype

        # Pre-allocate cache: (batch, max_seq_len, num_heads, head_dim)
        cache_shape = (batch_size, max_seq_len, num_heads, head_dim)
        self._k_cache = mx.zeros(cache_shape, dtype=dtype)
        self._v_cache = mx.zeros(cache_shape, dtype=dtype)

        # Track current length
        self._current_len = 0

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Dimension per head."""
        return self._head_dim

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length."""
        return self._max_seq_len

    @property
    def current_length(self) -> int:
        """Current cached sequence length."""
        return self._current_len

    @property
    def k_cache(self) -> mx.array:
        """Get current K cache."""
        return self._k_cache[:, :self._current_len]

    @property
    def v_cache(self) -> mx.array:
        """Get current V cache."""
        return self._v_cache[:, :self._current_len]

    def update(
        self,
        k: mx.array,
        v: mx.array,
        position: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new KV and return full cached KV.

        Args:
            k: New keys, shape (batch, seq_len, num_heads, head_dim).
            v: New values, shape (batch, seq_len, num_heads, head_dim).
            position: Position to write at (default: append).

        Returns:
            Tuple of (K, V) with all cached tokens up to new position.
        """
        new_len = k.shape[1]

        if position is None:
            position = self._current_len

        end_pos = position + new_len
        if end_pos > self._max_seq_len:
            raise ValueError(
                f"Cache overflow: {end_pos} > {self._max_seq_len}"
            )

        # Update cache using slice assignment
        self._k_cache = self._k_cache.at[:, position:end_pos].add(
            k - self._k_cache[:, position:end_pos]
        )
        self._v_cache = self._v_cache.at[:, position:end_pos].add(
            v - self._v_cache[:, position:end_pos]
        )

        self._current_len = max(self._current_len, end_pos)

        # Return cached KV up to current length
        return self._k_cache[:, :self._current_len], self._v_cache[:, :self._current_len]

    def get_kv(self) -> Tuple[mx.array, mx.array]:
        """Get all cached K, V.

        Returns:
            Tuple of (K, V), each shape (batch, seq_len, num_heads, head_dim).
        """
        return self._k_cache[:, :self._current_len], self._v_cache[:, :self._current_len]

    def clear(self) -> None:
        """Clear the cache."""
        self._k_cache = mx.zeros_like(self._k_cache)
        self._v_cache = mx.zeros_like(self._v_cache)
        self._current_len = 0

    def reset_to(self, length: int) -> None:
        """Reset cache to a specific length (for rollback).

        Args:
            length: Length to reset to.
        """
        if length < 0 or length > self._current_len:
            raise ValueError(f"Invalid reset length: {length}")
        self._current_len = length

    def __del__(self) -> None:
        """Clean up cache memory when garbage collected."""
        self._k_cache = None
        self._v_cache = None


class SlidingWindowCache:
    """Sliding window KV cache for long context attention.

    Maintains only the most recent `window_size` tokens, allowing
    efficient processing of arbitrarily long sequences with bounded memory.

    Example:
        >>> cache = SlidingWindowCache(
        ...     batch_size=4,
        ...     num_heads=8,
        ...     head_dim=64,
        ...     window_size=512,
        ... )
        >>> # Add tokens - old ones are automatically evicted
        >>> k, v = cache.update(new_k, new_v)
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize sliding window cache.

        Args:
            batch_size: Number of sequences in batch.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            window_size: Number of recent tokens to keep.
            dtype: Data type for storage (default: float16 for memory efficiency).
        """
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._window_size = window_size
        self._dtype = dtype

        # Pre-allocate cache
        cache_shape = (batch_size, window_size, num_heads, head_dim)
        self._k_cache = mx.zeros(cache_shape, dtype=dtype)
        self._v_cache = mx.zeros(cache_shape, dtype=dtype)

        # Track position
        self._current_len = 0
        self._total_seen = 0  # Total tokens seen (for position encoding)

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Dimension per head."""
        return self._head_dim

    @property
    def window_size(self) -> int:
        """Window size."""
        return self._window_size

    @property
    def current_length(self) -> int:
        """Current number of tokens in cache (up to window_size)."""
        return self._current_len

    @property
    def total_seen(self) -> int:
        """Total tokens seen since creation/clear."""
        return self._total_seen

    @property
    def k_cache(self) -> mx.array:
        """Get current K cache."""
        return self._k_cache[:, :self._current_len]

    @property
    def v_cache(self) -> mx.array:
        """Get current V cache."""
        return self._v_cache[:, :self._current_len]

    def update(
        self,
        k: mx.array,
        v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new KV.

        If adding new tokens would exceed window_size, oldest tokens
        are discarded.

        Args:
            k: New keys, shape (batch, seq_len, num_heads, head_dim).
            v: New values, shape (batch, seq_len, num_heads, head_dim).

        Returns:
            Tuple of (K, V) for the sliding window.
        """
        new_len = k.shape[1]
        self._total_seen += new_len

        if new_len >= self._window_size:
            # New tokens fill entire window
            self._k_cache = k[:, -self._window_size:]
            self._v_cache = v[:, -self._window_size:]
            self._current_len = self._window_size
        elif self._current_len + new_len <= self._window_size:
            # Fits in window without eviction
            start = self._current_len
            end = start + new_len
            self._k_cache = self._k_cache.at[:, start:end].add(
                k - self._k_cache[:, start:end]
            )
            self._v_cache = self._v_cache.at[:, start:end].add(
                v - self._v_cache[:, start:end]
            )
            self._current_len = end
        else:
            # Need to shift window
            shift = self._current_len + new_len - self._window_size
            keep = self._window_size - new_len

            # Shift old data
            old_k = self._k_cache[:, shift:self._current_len]
            old_v = self._v_cache[:, shift:self._current_len]

            # Create new cache with shifted old + new
            new_k_cache = mx.concatenate([old_k, k], axis=1)
            new_v_cache = mx.concatenate([old_v, v], axis=1)

            self._k_cache = self._k_cache.at[:, :self._window_size].add(
                new_k_cache - self._k_cache[:, :self._window_size]
            )
            self._v_cache = self._v_cache.at[:, :self._window_size].add(
                new_v_cache - self._v_cache[:, :self._window_size]
            )
            self._current_len = self._window_size

        return self.get_kv()

    def get_kv(self) -> Tuple[mx.array, mx.array]:
        """Get cached K, V.

        Returns:
            Tuple of (K, V) for the current window.
        """
        return self._k_cache[:, :self._current_len], self._v_cache[:, :self._current_len]

    def clear(self) -> None:
        """Clear the cache."""
        self._k_cache = mx.zeros_like(self._k_cache)
        self._v_cache = mx.zeros_like(self._v_cache)
        self._current_len = 0
        self._total_seen = 0

    def __del__(self) -> None:
        """Clean up cache memory when garbage collected."""
        self._k_cache = None
        self._v_cache = None


class RotatingKVCache:
    """Rotating (circular buffer) KV cache.

    Uses a circular buffer to store KV pairs, automatically overwriting
    the oldest entries when the buffer is full. This is useful for
    streaming scenarios where you want to maintain a fixed context window.

    Example:
        >>> cache = RotatingKVCache(
        ...     batch_size=4,
        ...     num_heads=8,
        ...     head_dim=64,
        ...     max_size=1024,
        ... )
        >>> # Tokens automatically rotate
        >>> k, v = cache.update(new_k, new_v)
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_size: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize rotating cache.

        Args:
            batch_size: Number of sequences in batch.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            max_size: Maximum number of tokens to store.
            dtype: Data type for storage (default: float16 for memory efficiency).
        """
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_size = max_size
        self._dtype = dtype

        # Pre-allocate circular buffer
        cache_shape = (batch_size, max_size, num_heads, head_dim)
        self._k_cache = mx.zeros(cache_shape, dtype=dtype)
        self._v_cache = mx.zeros(cache_shape, dtype=dtype)

        # Ring buffer state
        self._write_pos = 0  # Next write position
        self._filled = False  # Whether buffer has been filled at least once

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Dimension per head."""
        return self._head_dim

    @property
    def max_size(self) -> int:
        """Maximum buffer size."""
        return self._max_size

    @property
    def current_length(self) -> int:
        """Current number of valid tokens."""
        if self._filled:
            return self._max_size
        return self._write_pos

    @property
    def k_cache(self) -> mx.array:
        """Get K cache in order (oldest to newest)."""
        return self._get_ordered_cache(self._k_cache)

    @property
    def v_cache(self) -> mx.array:
        """Get V cache in order (oldest to newest)."""
        return self._get_ordered_cache(self._v_cache)

    def _get_ordered_cache(self, cache: mx.array) -> mx.array:
        """Get cache contents in chronological order."""
        if not self._filled:
            return cache[:, :self._write_pos]

        # Reorder: [write_pos:] + [:write_pos]
        first = cache[:, self._write_pos:]
        second = cache[:, :self._write_pos]
        return mx.concatenate([first, second], axis=1)

    def update(
        self,
        k: mx.array,
        v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new KV.

        New tokens are written at the current position, wrapping around
        if necessary.

        Args:
            k: New keys, shape (batch, seq_len, num_heads, head_dim).
            v: New values, shape (batch, seq_len, num_heads, head_dim).

        Returns:
            Tuple of (K, V) in chronological order.
        """
        new_len = k.shape[1]

        if new_len >= self._max_size:
            # New data fills entire buffer
            self._k_cache = k[:, -self._max_size:]
            self._v_cache = v[:, -self._max_size:]
            self._write_pos = 0
            self._filled = True
        else:
            # Write tokens, potentially wrapping
            for i in range(new_len):
                pos = (self._write_pos + i) % self._max_size

                # Update single position
                k_slice = k[:, i:i+1]
                v_slice = v[:, i:i+1]

                self._k_cache = self._k_cache.at[:, pos:pos+1].add(
                    k_slice - self._k_cache[:, pos:pos+1]
                )
                self._v_cache = self._v_cache.at[:, pos:pos+1].add(
                    v_slice - self._v_cache[:, pos:pos+1]
                )

            new_write_pos = (self._write_pos + new_len) % self._max_size
            if new_write_pos < self._write_pos or self._write_pos + new_len >= self._max_size:
                self._filled = True
            self._write_pos = new_write_pos

        return self.get_kv()

    def get_kv(self) -> Tuple[mx.array, mx.array]:
        """Get cached K, V in chronological order.

        Returns:
            Tuple of (K, V) from oldest to newest.
        """
        return self.k_cache, self.v_cache

    def clear(self) -> None:
        """Clear the cache."""
        self._k_cache = mx.zeros_like(self._k_cache)
        self._v_cache = mx.zeros_like(self._v_cache)
        self._write_pos = 0
        self._filled = False

    def __del__(self) -> None:
        """Clean up cache memory when garbage collected."""
        self._k_cache = None
        self._v_cache = None
