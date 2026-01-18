"""Kernel cache for compiled Metal-Triton kernels.

Caches compiled kernels to avoid recompilation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import hashlib
import threading


@dataclass
class CachedKernel:
    """A cached compiled kernel."""
    name: str
    metal_source: str
    mlx_kernel: Any
    config_hash: str
    hit_count: int = 0


class KernelCache:
    """Thread-safe cache for compiled kernels.

    Keyed by (kernel_name, config_hash) to support multiple
    configurations of the same kernel.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: dict[tuple[str, str], CachedKernel] = {}
        self._lock = threading.RLock()

    def get(self, name: str, config_hash: str) -> Optional[CachedKernel]:
        """Get cached kernel if available."""
        with self._lock:
            key = (name, config_hash)
            cached = self._cache.get(key)
            if cached:
                cached.hit_count += 1
            return cached

    def put(
        self,
        name: str,
        config_hash: str,
        metal_source: str,
        mlx_kernel: Any,
    ) -> CachedKernel:
        """Cache a compiled kernel."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()

            cached = CachedKernel(
                name=name,
                metal_source=metal_source,
                mlx_kernel=mlx_kernel,
                config_hash=config_hash,
            )
            self._cache[(name, config_hash)] = cached
            return cached

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with lowest hit count
        min_hits = float('inf')
        min_key = None
        for key, cached in self._cache.items():
            if cached.hit_count < min_hits:
                min_hits = cached.hit_count
                min_key = key

        if min_key:
            del self._cache[min_key]

    def clear(self) -> None:
        """Clear all cached kernels."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Global cache instance
_global_cache: Optional[KernelCache] = None
_cache_lock = threading.Lock()


def get_kernel_cache() -> KernelCache:
    """Get the global kernel cache."""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = KernelCache()
    return _global_cache


def hash_config(config: Any) -> str:
    """Create hash from configuration."""
    items = []
    if hasattr(config, "num_warps"):
        items.append(f"warps={config.num_warps}")
    if hasattr(config, "num_stages"):
        items.append(f"stages={config.num_stages}")
    if hasattr(config, "kwargs"):
        for k, v in sorted(config.kwargs.items()):
            items.append(f"{k}={v}")

    config_str = ",".join(items)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]
