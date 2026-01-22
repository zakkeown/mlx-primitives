"""KV cache management for MLX.

This module provides the canonical KV cache implementations for MLX Primitives.

Protocol Classes:
    - **KVCacheProtocol**: Defines the interface that all KV cache implementations
      should follow. Use this for type hints when writing generic cache code.

Cache Selection Guide:
    - **SimpleKVCache**: Basic pre-allocated cache. Best for single sequences with
      known maximum length. Simple API, no dynamic memory management.

    - **SlidingWindowCache**: Maintains a sliding window of recent tokens. Best for
      long sequences with models that use sliding window attention (e.g., Mistral).

    - **RotatingKVCache**: Circular buffer cache. Best for streaming scenarios where
      you want to maintain a fixed context window.

    - **KVCache** (with KVCacheConfig): Full-featured paged cache with block management,
      eviction policies, and multi-sequence support. Best for serving scenarios with
      multiple concurrent sequences.

    - **CompressedKVCache**: Quantized cache for memory efficiency. Best when memory
      is constrained and some precision loss is acceptable.

Quick Start:
    >>> from mlx_primitives.cache import create_kv_cache, CacheType
    >>>
    >>> # Simple cache for single sequence
    >>> cache = create_kv_cache(
    ...     cache_type=CacheType.SIMPLE,
    ...     batch_size=1,
    ...     num_heads=32,
    ...     head_dim=128,
    ...     max_seq_len=2048,
    ... )
    >>> k, v = cache.update(new_k, new_v)
    >>>
    >>> # Paged cache for serving
    >>> cache = create_kv_cache(
    ...     cache_type=CacheType.PAGED,
    ...     num_heads=32,
    ...     head_dim=128,
    ...     num_layers=32,
    ...     max_memory_gb=8.0,
    ... )
    >>> seq_id = cache.create_sequence()
    >>> cache.update(seq_id, k, v, layer_idx=0)

Advanced Usage:
    >>> from mlx_primitives.cache import KVCache, KVCacheConfig, paged_attention
    >>>
    >>> config = KVCacheConfig(
    ...     num_heads=32,
    ...     head_dim=128,
    ...     num_layers=32,
    ...     max_memory_gb=8.0,
    ... )
    >>> cache = KVCache(config)
    >>>
    >>> # Create sequence and add tokens
    >>> seq_id = cache.create_sequence()
    >>> cache.update(seq_id, k, v, layer_idx=0)
    >>>
    >>> # Use paged attention
    >>> k_pool, v_pool, tables, lens = cache.get_kv_paged([seq_id], layer_idx=0)
    >>> out = paged_attention(q, k_pool, v_pool, tables, lens)
"""

from mlx_primitives.cache.protocols import (
    KVCacheProtocol,
    SimpleCacheProtocol,
)
from mlx_primitives.cache.block_allocator import (
    BlockAllocator,
    BlockConfig,
    calculate_num_blocks,
    get_optimal_block_size,
)
from mlx_primitives.cache.page_table import (
    PageTable,
    SequenceMetadata,
)
from mlx_primitives.cache.eviction import (
    AttentionScoreEvictionPolicy,
    CacheMemoryStats,
    CompositeEvictionPolicy,
    EvictionPolicy,
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    MemoryBudgetManager,
)
from mlx_primitives.cache.kv_cache import (
    KVCache,
    KVCacheConfig,
    MultiLayerKVCache,
)
from mlx_primitives.cache.paged_attention import (
    create_block_table_from_lengths,
    paged_attention,
    paged_attention_with_bias,
)
from mlx_primitives.cache.speculative import (
    SpeculativeCache,
    SpeculativeToken,
    TreeSpeculation,
    speculative_verify,
)
from mlx_primitives.cache.simple_cache import (
    SimpleKVCache,
    SlidingWindowCache,
    RotatingKVCache,
)
from mlx_primitives.cache.compressed_cache import (
    CompressedKVCache,
)
from mlx_primitives.constants import CacheType

import mlx.core as mx
from typing import Optional, Union


def create_kv_cache(
    cache_type: Union[str, CacheType],
    *,
    # Common parameters
    num_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
    # SimpleKVCache, SlidingWindowCache, RotatingKVCache parameters
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    # SlidingWindowCache parameter
    window_size: Optional[int] = None,
    # RotatingKVCache parameter
    max_size: Optional[int] = None,
    # KVCache (paged) parameters
    num_layers: Optional[int] = None,
    max_memory_gb: Optional[float] = None,
    block_size: Optional[int] = None,
    eviction_policy: Optional[str] = None,
):
    """Create a KV cache of the specified type.

    Factory function for creating KV caches with appropriate parameters.
    This is the recommended way to create caches as it provides a unified
    interface and validates parameters for each cache type.

    Args:
        cache_type: Type of cache to create. Use CacheType enum or string:
            - "simple": SimpleKVCache - basic pre-allocated cache
            - "sliding": SlidingWindowCache - sliding window cache
            - "rotating": RotatingKVCache - circular buffer cache
            - "paged": KVCache - full-featured paged cache
            - "compressed": CompressedKVCache - quantized cache
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        dtype: Data type for cache storage (default: float16).
        batch_size: Batch size (required for simple, sliding, rotating).
        max_seq_len: Maximum sequence length (required for simple).
        window_size: Window size (required for sliding).
        max_size: Buffer size (required for rotating).
        num_layers: Number of transformer layers (required for paged).
        max_memory_gb: Maximum memory budget in GB (required for paged).
        block_size: Block size for paged cache (optional, auto-tuned if not set).
        eviction_policy: Eviction policy for paged cache ("lru" or "fifo").

    Returns:
        A KV cache instance of the requested type.

    Raises:
        ValueError: If required parameters for the cache type are missing.

    Example:
        >>> # Create a simple cache
        >>> cache = create_kv_cache(
        ...     CacheType.SIMPLE,
        ...     batch_size=1,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     max_seq_len=2048,
        ... )
        >>>
        >>> # Create a paged cache for serving
        >>> cache = create_kv_cache(
        ...     CacheType.PAGED,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     num_layers=32,
        ...     max_memory_gb=8.0,
        ... )
    """
    # Normalize cache_type to enum
    if isinstance(cache_type, str):
        try:
            cache_type = CacheType(cache_type.lower())
        except ValueError:
            valid_types = [ct.value for ct in CacheType]
            raise ValueError(
                f"Unknown cache type: {cache_type}. "
                f"Valid types are: {', '.join(valid_types)}"
            )

    if cache_type == CacheType.SIMPLE:
        if batch_size is None or max_seq_len is None:
            raise ValueError(
                "SimpleKVCache requires batch_size and max_seq_len"
            )
        return SimpleKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )

    elif cache_type == CacheType.SLIDING:
        if batch_size is None or window_size is None:
            raise ValueError(
                "SlidingWindowCache requires batch_size and window_size"
            )
        return SlidingWindowCache(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            dtype=dtype,
        )

    elif cache_type == CacheType.ROTATING:
        if batch_size is None or max_size is None:
            raise ValueError(
                "RotatingKVCache requires batch_size and max_size"
            )
        return RotatingKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_size=max_size,
            dtype=dtype,
        )

    elif cache_type == CacheType.PAGED:
        if num_layers is None or max_memory_gb is None:
            raise ValueError(
                "KVCache (paged) requires num_layers and max_memory_gb"
            )
        config_kwargs = {
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "max_memory_gb": max_memory_gb,
            "dtype": dtype,
        }
        if block_size is not None:
            config_kwargs["block_size"] = block_size
        if eviction_policy is not None:
            config_kwargs["eviction_policy"] = eviction_policy

        config = KVCacheConfig(**config_kwargs)
        return KVCache(config)

    elif cache_type == CacheType.COMPRESSED:
        if batch_size is None or max_seq_len is None:
            raise ValueError(
                "CompressedKVCache requires batch_size and max_seq_len"
            )
        return CompressedKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )

    else:
        valid_types = [ct.value for ct in CacheType]
        raise ValueError(
            f"Unknown cache type: {cache_type}. "
            f"Valid types are: {', '.join(valid_types)}"
        )


__all__ = [
    # Protocol classes (for type hints)
    "KVCacheProtocol",
    "SimpleCacheProtocol",
    # Factory function (recommended entry point)
    "create_kv_cache",
    "CacheType",
    # Block allocation
    "BlockAllocator",
    "BlockConfig",
    "calculate_num_blocks",
    "get_optimal_block_size",
    # Page table
    "PageTable",
    "SequenceMetadata",
    # Eviction policies
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "FIFOEvictionPolicy",
    "AttentionScoreEvictionPolicy",
    "CompositeEvictionPolicy",
    "MemoryBudgetManager",
    "CacheMemoryStats",
    # Core cache
    "KVCache",
    "KVCacheConfig",
    "MultiLayerKVCache",
    # Paged attention
    "paged_attention",
    "paged_attention_with_bias",
    "create_block_table_from_lengths",
    # Speculative decoding
    "SpeculativeCache",
    "SpeculativeToken",
    "TreeSpeculation",
    "speculative_verify",
    # Simple cache implementations
    "SimpleKVCache",
    "SlidingWindowCache",
    "RotatingKVCache",
    # Compressed cache
    "CompressedKVCache",
]
