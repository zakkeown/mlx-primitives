"""MLX Primitives: High-performance building blocks for MLX.

Custom Metal kernels for missing MLX primitives, fused operations,
quantization, and memory-efficient implementations.
"""

from mlx_primitives.primitives import (
    ExpertDispatch,
    SparseMoELayer,
    associative_scan,
    build_expert_dispatch,
    compute_load_balancing_loss,
    selective_gather,
    selective_scan,
    selective_scatter_add,
)

from mlx_primitives.attention import (
    ChunkedCrossAttention,
    FlashAttention,
    SlidingWindowAttention,
    chunked_cross_attention,
    flash_attention,
    sliding_window_attention,
)

from mlx_primitives.training import (
    checkpoint,
    checkpoint_sequential,
)

# Cache management
from mlx_primitives.cache import (
    create_kv_cache,
    KVCache,
    KVCacheConfig,
    MultiLayerKVCache,
    SimpleKVCache,
    SlidingWindowCache,
    RotatingKVCache,
    CompressedKVCache,
    paged_attention,
)

# Hardware detection
from mlx_primitives.hardware import (
    get_chip_info,
    ChipFamily,
    ChipTier,
)

# Type-safe enums
from mlx_primitives.constants import (
    Layout,
    CacheType,
    EvictionPolicy,
)

__version__ = "0.2.0"

__all__ = [
    # Primitives
    "associative_scan",
    "selective_scan",
    "selective_gather",
    "selective_scatter_add",
    "build_expert_dispatch",
    "ExpertDispatch",
    "SparseMoELayer",
    "compute_load_balancing_loss",
    # Attention
    "sliding_window_attention",
    "SlidingWindowAttention",
    "flash_attention",
    "FlashAttention",
    "chunked_cross_attention",
    "ChunkedCrossAttention",
    # Training
    "checkpoint",
    "checkpoint_sequential",
    # Cache
    "create_kv_cache",
    "KVCache",
    "KVCacheConfig",
    "MultiLayerKVCache",
    "SimpleKVCache",
    "SlidingWindowCache",
    "RotatingKVCache",
    "CompressedKVCache",
    "paged_attention",
    # Hardware
    "get_chip_info",
    "ChipFamily",
    "ChipTier",
    # Enums
    "Layout",
    "CacheType",
    "EvictionPolicy",
]
