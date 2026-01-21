"""KV cache management for MLX.

This module provides KV cache implementations:
- Paged attention with block management
- Cache eviction policies (LRU, FIFO, attention-based)
- Speculative decoding support

Example:
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

__all__ = [
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
]
