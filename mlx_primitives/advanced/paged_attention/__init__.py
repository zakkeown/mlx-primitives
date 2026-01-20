"""Paged Attention for efficient KV cache memory management.

This module implements vLLM-style paged attention for efficient serving
of multiple sequences with variable lengths.

Key Features:
    - Block-based KV cache allocation eliminates memory fragmentation
    - Efficient batching with different sequence lengths
    - Copy-on-write (COW) for beam search and speculative decoding
    - ~20-30% memory savings vs pre-allocated caches

Architecture:
    Physical Memory Pool          Page Tables (per sequence)
    +------------------+         +------------------+
    | Block 0 [KV]     | <------ | Seq 0: [0,3,7]   |
    | Block 1 [KV]     | <------ | Seq 1: [1,2,5]   |
    | Block 2 [KV]     | <-------+                  |
    | Block 3 [KV]     | <------ | (shares block 3) |
    | ...              |         | Seq 2: [4,3,6]   |
    +------------------+         +------------------+

Usage:
    >>> from mlx_primitives.advanced.paged_attention import PagedKVCache
    >>>
    >>> # Create cache
    >>> cache = PagedKVCache(num_kv_heads=8, head_dim=128, num_layers=32)
    >>>
    >>> # Create sequences
    >>> seq1 = cache.create_sequence()
    >>> seq2 = cache.create_sequence()
    >>>
    >>> # Append KV during generation
    >>> cache.append_kv(seq1, layer_idx=0, k=k_tensor, v=v_tensor)
    >>>
    >>> # Fork for beam search (copy-on-write)
    >>> seq1_beam = cache.fork_sequence(seq1)
    >>>
    >>> # Get KV for attention
    >>> k, v = cache.get_kv(seq1, layer_idx=0)
"""

from mlx_primitives.advanced.paged_attention.block_manager import (
    BlockManager,
    BlockConfig,
    PhysicalBlock,
)
from mlx_primitives.advanced.paged_attention.paged_kv_cache import (
    PagedKVCache,
    SequenceState,
    create_paged_attention_mask,
)

__all__ = [
    "PagedKVCache",
    "BlockManager",
    "BlockConfig",
    "PhysicalBlock",
    "SequenceState",
    "create_paged_attention_mask",
]
