"""Unified memory exploitation primitives for Apple Silicon.

This module provides primitives that explicitly leverage Apple Silicon's
unified memory architecture, enabling:
- Zero-copy tensor views between CPU and GPU
- Memory-efficient streaming for large datasets
- Efficient CPU/GPU hybrid workloads
- Memory residency hints and prefetching
"""

from mlx_primitives.memory.coordination import (
    PingPongBuffer,
    SyncPoint,
    WorkQueue,
    overlap_compute_io,
    parallel_cpu_gpu,
    ping_pong_buffer,
)
from mlx_primitives.memory.residency import (
    CacheEstimate,
    ResidencyHint,
    estimate_cache_usage,
    evict_from_cache,
    gpu_residency_context,
    prefetch_batch,
    prefetch_to_cpu,
    prefetch_to_gpu,
    recommend_chunk_size,
    set_residency_hint,
    streaming_context,
)
from mlx_primitives.memory.streaming import (
    StreamingDataLoader,
    StreamingTensor,
    streaming_reduce,
)
from mlx_primitives.memory.unified import (
    AccessMode,
    MemoryInfo,
    UnifiedView,
    create_unified_buffer,
    ensure_contiguous,
    get_memory_info,
    shares_memory,
    zero_copy_slice,
)

__all__ = [
    # Unified memory
    "UnifiedView",
    "MemoryInfo",
    "AccessMode",
    "create_unified_buffer",
    "zero_copy_slice",
    "get_memory_info",
    "ensure_contiguous",
    "shares_memory",
    # Streaming
    "StreamingTensor",
    "StreamingDataLoader",
    "streaming_reduce",
    # Coordination
    "WorkQueue",
    "SyncPoint",
    "PingPongBuffer",
    "parallel_cpu_gpu",
    "ping_pong_buffer",
    "overlap_compute_io",
    # Residency hints
    "ResidencyHint",
    "CacheEstimate",
    "set_residency_hint",
    "prefetch_to_gpu",
    "prefetch_to_cpu",
    "prefetch_batch",
    "evict_from_cache",
    "gpu_residency_context",
    "streaming_context",
    "estimate_cache_usage",
    "recommend_chunk_size",
]
