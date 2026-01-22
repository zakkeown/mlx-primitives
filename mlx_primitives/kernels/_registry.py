"""Thread-safe Metal kernel registry.

Provides centralized kernel caching with lazy initialization using
the double-checked locking pattern for thread safety.

This module consolidates the repeated kernel caching pattern found
throughout the codebase into a single reusable utility.

Example:
    from mlx_primitives.kernels._registry import get_kernel

    def _get_rmsnorm_kernel():
        def factory():
            return mx.fast.metal_kernel(
                name="rmsnorm",
                input_names=["x", "weight"],
                output_names=["y"],
                source="...",
            )
        return get_kernel("rmsnorm", factory)
"""

import threading
from typing import Callable, Dict, Optional, TypeVar

try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

T = TypeVar("T")


class KernelRegistry:
    """Thread-safe registry for Metal kernels with per-kernel locking.

    Uses double-checked locking pattern to minimize lock contention
    while ensuring thread-safe initialization of kernels.

    The registry is a singleton - use get_instance() to access it.
    """

    _instance: Optional["KernelRegistry"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the registry. Use get_instance() instead."""
        self._kernels: Dict[str, "mx.fast.metal_kernel"] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "KernelRegistry":
        """Get the singleton registry instance (thread-safe)."""
        if cls._instance is None:
            with cls._init_lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], "mx.fast.metal_kernel"],
    ) -> "mx.fast.metal_kernel":
        """Get kernel from cache or create using factory.

        Thread-safe with per-kernel locking for maximum concurrency.
        Multiple different kernels can be initialized in parallel,
        but each individual kernel is only created once.

        Args:
            name: Unique identifier for the kernel.
            factory: Callable that creates the kernel when needed.

        Returns:
            The cached or newly created kernel.
        """
        # Fast path - kernel already exists (no lock needed)
        if name in self._kernels:
            return self._kernels[name]

        # Get or create the per-kernel lock
        with self._global_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            kernel_lock = self._locks[name]

        # Acquire per-kernel lock for initialization
        with kernel_lock:
            # Double-check after acquiring lock
            if name in self._kernels:
                return self._kernels[name]

            # Create kernel using factory
            kernel = factory()
            self._kernels[name] = kernel
            return kernel

    def clear(self) -> None:
        """Clear all cached kernels.

        Useful for testing or when dynamic reconfiguration is needed.
        """
        with self._global_lock:
            self._kernels.clear()
            # Keep locks to prevent race conditions during clear

    def is_cached(self, name: str) -> bool:
        """Check if a kernel is already cached."""
        return name in self._kernels

    @property
    def cached_kernel_names(self) -> list:
        """Get list of currently cached kernel names."""
        return list(self._kernels.keys())


def get_kernel(
    name: str,
    factory: Callable[[], "mx.fast.metal_kernel"],
) -> "mx.fast.metal_kernel":
    """Get or create a Metal kernel with thread-safe caching.

    This is the main entry point for kernel caching. It provides a simple
    interface that handles all the complexity of thread-safe lazy
    initialization internally.

    Args:
        name: Unique identifier for the kernel. Should be descriptive
            and unique across the codebase (e.g., "rmsnorm", "rope_forward").
        factory: A callable that creates the kernel. This is only called
            once, on first access. Should return an mx.fast.metal_kernel.

    Returns:
        The Metal kernel (cached or newly created).

    Example:
        def _get_my_kernel():
            return get_kernel(
                "my_kernel",
                lambda: mx.fast.metal_kernel(
                    name="my_kernel",
                    input_names=["x"],
                    output_names=["y"],
                    source="...",
                ),
            )

        # Later uses return the cached kernel
        kernel = _get_my_kernel()
    """
    return KernelRegistry.get_instance().get_or_create(name, factory)


def clear_kernel_cache() -> None:
    """Clear all cached kernels.

    This is primarily useful for testing scenarios where you need
    to reset the kernel cache between tests.
    """
    KernelRegistry.get_instance().clear()
