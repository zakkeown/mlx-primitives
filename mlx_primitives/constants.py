"""Centralized numerical constants and enums for MLX Primitives.

This module provides documented constants and type-safe enums used throughout
the codebase to ensure consistency and traceability.
"""

import math
from enum import Enum, auto


# =============================================================================
# ENUMS - Type-safe alternatives to string literals
# =============================================================================


class Layout(str, Enum):
    """Tensor layout for attention operations.

    BSHD: (Batch, Sequence, Heads, Dimension) - default for most operations
    BHSD: (Batch, Heads, Sequence, Dimension) - native SDPA format, faster for some ops

    Using str enum allows backward compatibility with string comparisons:
        layout == "BSHD"  # Still works
        layout == Layout.BSHD  # Type-safe alternative
    """

    BSHD = "BSHD"
    BHSD = "BHSD"


class CacheType(str, Enum):
    """Type of KV cache implementation.

    SIMPLE: Pre-allocated fixed-size cache (SimpleKVCache)
    PAGED: Dynamic paged attention cache (KVCache)
    SLIDING: Sliding window cache (SlidingWindowCache)
    ROTATING: Circular buffer cache (RotatingKVCache)
    """

    SIMPLE = "simple"
    PAGED = "paged"
    SLIDING = "sliding"
    ROTATING = "rotating"


class EvictionPolicy(str, Enum):
    """Cache eviction policy type.

    LRU: Least Recently Used - evicts oldest accessed sequence
    FIFO: First In First Out - evicts oldest created sequence
    """

    LRU = "lru"
    FIFO = "fifo"

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# FP16-safe epsilon for division and normalization operations
# FP16 machine epsilon is ~1e-3, so 1e-6 provides safety margin
EPSILON_FP16_SAFE = 1e-6

# Standard epsilon for FP32 normalization (RMSNorm, LayerNorm)
EPSILON_NORM = 1e-5


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

# GELU approximation constants
# From: "Gaussian Error Linear Units (GELUs)" - Hendrycks & Gimpel, 2016
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
GELU_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # sqrt(2/π) ≈ 0.7978845608
GELU_TANH_COEFF = 0.044715  # Empirical coefficient for tanh approximation

# Quick GELU constant (less accurate but faster)
# QuickGELU(x) = x * sigmoid(1.702 * x)
QUICK_GELU_COEFF = 1.702


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

# Default base frequency for RoPE
# From: "RoFormer" - Su et al., 2021
# Standard value used in LLaMA, Mistral, and most modern LLMs
ROPE_BASE_FREQUENCY = 10000.0


# =============================================================================
# ATTENTION
# =============================================================================

# Mask value for causal attention (large negative to become ~0 after softmax)
# Using -1e38 to align with Metal kernels which use -1e38f for consistency
# between Python and Metal code paths
ATTENTION_MASK_VALUE = -1e38

# Maximum head dimension supported by Metal attention kernels
# Kernels use fixed-size arrays: float acc[132] supports head_dim <= 128
METAL_ATTENTION_MAX_HEAD_DIM = 128

# Epsilon for softmax normalization in Metal kernels
# Prevents division by zero when all attention scores are masked
# Using 1e-6 as minimum safe value for FP32; FP16 would need 1e-4
# Industry standard (PyTorch, JAX) uses 1e-6 minimum
METAL_SOFTMAX_EPSILON = 1e-6


# =============================================================================
# QUANTIZATION
# =============================================================================

# INT8 quantization range
QUANT_INT8_MIN = -128
QUANT_INT8_MAX = 127
QUANT_INT8_RANGE = 255

# UINT4 quantization range (unsigned 4-bit, common in LLM quantization)
# Note: Using unsigned 0-15 range, not signed -8 to 7
QUANT_UINT4_MIN = 0
QUANT_UINT4_MAX = 15
QUANT_UINT4_RANGE = 15


# =============================================================================
# HARDWARE
# =============================================================================

# Default L2 cache size assumption (MB) - conservative default for base chips
# M1/M2/M3 base: 8MB, Pro: 24MB, Max: 48MB, Ultra: 96MB
# Hardware detection in hardware/detection.py provides accurate values;
# this is only used as fallback.
DEFAULT_L2_CACHE_MB = 8.0

# Minimum sequence length for Metal kernel dispatch
# Lowered from 32 to 8 to use Metal for more SSM workloads
MIN_SEQ_FOR_METAL = 8

# Maximum sequence length for Metal associative scan (single-block limit)
# Multi-block algorithm at seq_len > 1024 has significant overhead due to:
# - 3 kernel launches instead of 1 (phase1, phase2, phase3)
# - Memory round-trips between phases
# - Inefficient warp utilization for small block counts
# Benchmark shows: seq=2048 is 14-45x slower than native cumsum
# For seq_len > this threshold, fall back to native MLX ops
MAX_SEQ_FOR_METAL_SCAN = 1024

# Sequence length threshold for SSM sequential fallback warning
# When using sequential SSM scan, warn if sequence length exceeds this
SSM_SEQUENTIAL_WARNING_THRESHOLD = 256

# =============================================================================
# FLASH ATTENTION THRESHOLDS
# =============================================================================

# Maximum sequence length for using reference (O(n²) memory) attention
# Below this, the O(n²) memory overhead is acceptable and avoids tiling overhead
FLASH_MAX_SEQ_FOR_REFERENCE = 256

# Maximum sequence length for reference attention with small batches
# When batch_size is small, we can tolerate slightly longer sequences
FLASH_SMALL_BATCH_SEQ_THRESHOLD = 512

# Maximum batch size considered "small" for the relaxed sequence threshold
FLASH_SMALL_BATCH_THRESHOLD = 4

# Minimum sequence length for Metal kernel dispatch in flash attention
# Metal kernel launch overhead dominates for very short sequences
FLASH_MIN_SEQ_FOR_METAL = 64
