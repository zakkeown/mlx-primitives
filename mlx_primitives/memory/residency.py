"""Memory residency hints for Apple Silicon.

This module provides advisory hints for cache behavior and memory
placement on Apple Silicon's unified memory architecture.

IMPORTANT: Current Implementation Status
    Most functions in this module are currently NO-OPs because MLX does not
    expose low-level cache control APIs. These functions exist to:

    1. Document intended access patterns for code clarity
    2. Provide a stable API for when MLX adds cache control features
    3. Allow performance-sensitive code to express intent

    Functions that actually do something:
    - prefetch_to_gpu(): Calls mx.eval() to materialize tensors
    - prefetch_to_cpu(): Calls mx.eval() and touches numpy view
    - prefetch_batch(): Same as above for multiple tensors
    - gpu_residency_context(): Calls mx.eval() on context entry
    - estimate_cache_usage(): Returns size estimates (heuristic)
    - recommend_chunk_size(): Returns chunk size recommendation

    Functions that are pure NO-OPs:
    - set_residency_hint(): Returns tensor unchanged (except GPU_PREFERRED calls eval)
    - evict_from_cache(): Does nothing
    - streaming_context(): Just applies no-op hints

Note: These are hints that the system may or may not honor depending
on memory pressure, access patterns, and hardware generation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import mlx.core as mx
import numpy as np


class ResidencyHint(Enum):
    """Hints for memory placement optimization."""

    GPU_PREFERRED = "gpu_preferred"  # Keep in GPU cache when possible
    CPU_PREFERRED = "cpu_preferred"  # Keep in CPU cache when possible
    STREAMING = "streaming"  # Don't cache, stream through
    SHARED_FREQUENT = "shared_frequent"  # Frequent CPU/GPU access


@dataclass
class CacheEstimate:
    """Estimated cache usage for a set of tensors.

    Attributes:
        total_bytes: Total memory footprint.
        l2_estimate_bytes: Estimated L2 cache usage.
        shared_estimate_bytes: Estimated shared cache usage.
        fits_in_l2: Whether data likely fits in L2 cache.
    """

    total_bytes: int
    l2_estimate_bytes: int
    shared_estimate_bytes: int
    fits_in_l2: bool


def set_residency_hint(
    tensor: mx.array,
    hint: ResidencyHint,
) -> mx.array:
    """Apply residency hint to a tensor.

    This is advisory; the system may ignore hints based on
    memory pressure and access patterns.

    Args:
        tensor: Tensor to hint.
        hint: Residency hint to apply.

    Returns:
        The tensor (possibly with hint metadata).

    Note:
        On current MLX, this is a no-op but provides documentation
        of intended access patterns.
    """
    # Currently, MLX doesn't expose cache control APIs
    # This function serves as documentation of intent and
    # may be implemented when MLX adds such capabilities

    if hint == ResidencyHint.GPU_PREFERRED:
        # Ensure tensor is evaluated (materialized on GPU)
        mx.eval(tensor)
    elif hint == ResidencyHint.STREAMING:
        # For streaming, we don't want to cache
        pass

    return tensor


def prefetch_to_gpu(
    tensor: mx.array,
    priority: int = 0,
) -> None:
    """Hint to prefetch tensor data to GPU caches.

    Non-blocking; returns immediately. Useful before a compute-heavy
    operation to hide memory latency.

    Args:
        tensor: Tensor to prefetch.
        priority: Higher = more urgent (0-10).

    Note:
        This is advisory and may not have effect on all operations.
    """
    # The most reliable way to "prefetch" to GPU on MLX is to
    # trigger evaluation, which materializes the tensor
    mx.eval(tensor)


def prefetch_to_cpu(
    tensor: mx.array,
    priority: int = 0,
) -> None:
    """Hint to prefetch tensor data to CPU caches.

    Non-blocking when possible. Useful before CPU-side operations.

    Args:
        tensor: Tensor to prefetch.
        priority: Higher = more urgent (0-10).
    """
    # Ensure tensor is evaluated
    mx.eval(tensor)

    # Touch data through numpy to bring into CPU cache
    # This is a hint - may not have effect on all systems
    np_view = np.array(tensor, copy=False)
    _ = np_view.flat[0]  # Touch first element


def prefetch_batch(
    tensors: List[mx.array],
    target: str = "gpu",
) -> None:
    """Prefetch multiple tensors to target device cache.

    More efficient than individual prefetch calls when handling
    multiple tensors.

    Args:
        tensors: List of tensors to prefetch.
        target: Target cache - "gpu" or "cpu".
    """
    if not tensors:
        return

    if target == "gpu":
        mx.eval(*tensors)
    elif target == "cpu":
        mx.eval(*tensors)
        # Touch each tensor's data
        for tensor in tensors:
            np_view = np.array(tensor, copy=False)
            _ = np_view.flat[0]


def evict_from_cache(
    tensor: mx.array,
    device: str = "both",
) -> None:
    """Hint to evict tensor from caches.

    Useful after processing to free cache for other data.
    This is purely advisory.

    Args:
        tensor: Tensor to evict.
        device: "cpu", "gpu", or "both".

    Warning:
        THIS IS A NO-OP. MLX does not expose cache eviction APIs.
        This function exists for API completeness and future compatibility.
        On unified memory, actual eviction is managed by the system.
    """
    # NO-OP: On Apple Silicon unified memory, we can't directly evict
    # from caches. This is a placeholder for future MLX cache APIs.
    pass


@contextmanager
def gpu_residency_context(*tensors: mx.array):
    """Context manager ensuring tensors are GPU-resident.

    Useful for ensuring weights stay in GPU cache during
    a compute-intensive section.

    Example:
        >>> with gpu_residency_context(weights, biases):
        ...     for batch in data:
        ...         output = model(batch)

    Args:
        *tensors: Tensors to keep GPU-resident.
    """
    # Prefetch tensors to GPU at context entry
    if tensors:
        mx.eval(*tensors)

    try:
        yield
    finally:
        # On context exit, tensors may be evicted naturally
        # No explicit cleanup needed on unified memory
        pass


@contextmanager
def streaming_context(*tensors: mx.array):
    """Context manager for streaming access pattern.

    Hints that tensors should not be cached aggressively,
    as they will only be accessed once.

    Args:
        *tensors: Tensors to mark for streaming access.
    """
    # Apply streaming hints
    for tensor in tensors:
        set_residency_hint(tensor, ResidencyHint.STREAMING)

    try:
        yield
    finally:
        pass


def estimate_cache_usage(tensors: List[mx.array]) -> CacheEstimate:
    """Estimate cache usage for a set of tensors.

    Provides estimates based on tensor sizes and chip characteristics.

    Args:
        tensors: List of tensors to analyze.

    Returns:
        CacheEstimate with size breakdown and fit analysis.
    """
    from mlx_primitives.hardware import get_chip_info

    chip_info = get_chip_info()

    # Calculate total size
    total_bytes = 0
    for tensor in tensors:
        size = tensor.size
        dtype_size = _get_dtype_size(tensor.dtype)
        total_bytes += size * dtype_size

    # Estimate L2 usage (heuristic: ~50% of accessed data stays in L2)
    l2_estimate = int(total_bytes * 0.5)

    # L2 cache size
    l2_size_bytes = chip_info.l2_cache_mb * 1024 * 1024

    # Check if data fits
    fits_in_l2 = total_bytes <= l2_size_bytes

    return CacheEstimate(
        total_bytes=total_bytes,
        l2_estimate_bytes=min(l2_estimate, l2_size_bytes),
        shared_estimate_bytes=total_bytes,  # All memory is shared on unified
        fits_in_l2=fits_in_l2,
    )


def recommend_chunk_size(
    total_elements: int,
    dtype: mx.Dtype = mx.float32,
    target_cache_fraction: float = 0.5,
) -> int:
    """Recommend chunk size for processing large data.

    Suggests a chunk size that should fit comfortably in cache,
    optimizing for memory access efficiency.

    Args:
        total_elements: Total number of elements to process.
        dtype: Data type.
        target_cache_fraction: Target fraction of L2 cache to use (0.0-1.0).

    Returns:
        Recommended chunk size in elements.
    """
    from mlx_primitives.hardware import get_chip_info

    chip_info = get_chip_info()

    dtype_size = _get_dtype_size(dtype)
    l2_size_bytes = chip_info.l2_cache_mb * 1024 * 1024

    # Target chunk should use specified fraction of L2
    target_bytes = int(l2_size_bytes * target_cache_fraction)
    chunk_elements = target_bytes // dtype_size

    # Don't exceed total elements
    chunk_elements = min(chunk_elements, total_elements)

    # Round to power of 2 for efficiency (optional)
    if chunk_elements > 0:
        # Find nearest power of 2
        import math
        log2 = math.log2(chunk_elements)
        chunk_elements = 2 ** int(log2)

    return max(1, chunk_elements)


def _get_dtype_size(dtype: mx.Dtype) -> int:
    """Get size in bytes for an MLX dtype."""
    sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int16: 2,
        mx.int8: 1,
        mx.uint32: 4,
        mx.uint16: 2,
        mx.uint8: 1,
    }
    return sizes.get(dtype, 4)
