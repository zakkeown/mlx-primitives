"""Core ML model compilation and caching for ANE primitives.

This module manages the compilation and caching of Core ML models
used to dispatch operations to the Neural Engine.
"""

import hashlib
import json
import logging
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a cached Core ML model.

    Uniquely identifies a compiled model based on operation
    type, input shapes, and configuration.

    Attributes:
        operation: Operation type (e.g., "matmul", "conv2d").
        input_shapes: Tuple of input tensor shapes.
        dtype: Data type string (e.g., "float16").
        params: Frozen tuple of (key, value) parameter pairs.
    """

    operation: str
    input_shapes: Tuple[Tuple[int, ...], ...]
    dtype: str
    params: Tuple[Tuple[str, Any], ...]

    def cache_key(self) -> str:
        """Generate unique cache key for this spec.

        Returns:
            16-character hex string uniquely identifying this model.
        """
        key_str = f"{self.operation}_{self.input_shapes}_{self.dtype}_{self.params}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "input_shapes": self.input_shapes,
            "dtype": self.dtype,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        """Create from dictionary."""
        return cls(
            operation=data["operation"],
            input_shapes=tuple(tuple(s) for s in data["input_shapes"]),
            dtype=data["dtype"],
            params=tuple(sorted(data.get("params", {}).items())),
        )


class CoreMLModelCache:
    """Cache for compiled Core ML models targeting ANE.

    Provides both in-memory and disk caching to avoid expensive
    recompilation of Core ML models.

    Example:
        >>> cache = CoreMLModelCache()
        >>> spec = ModelSpec("matmul", ((1024, 512), (512, 1024)), "float16", ())
        >>> model = cache.get_or_compile(spec, compile_matmul_model)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_cache: int = 32,
    ):
        """Initialize the model cache.

        Args:
            cache_dir: Directory for disk cache.
                Defaults to ~/.mlx_primitives/ane_cache/
            max_memory_cache: Maximum models to keep in memory.
        """
        self._cache_dir = cache_dir or Path.home() / ".mlx_primitives" / "ane_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Use OrderedDict for O(1) LRU operations instead of list
        self._memory_cache: OrderedDict[str, Any] = OrderedDict()
        self._max_memory_cache = max_memory_cache

    def get_or_compile(
        self,
        spec: ModelSpec,
        compile_fn: Callable[[ModelSpec], Any],
    ) -> Any:
        """Get cached model or compile new one.

        Args:
            spec: Model specification.
            compile_fn: Function to compile the model if not cached.

        Returns:
            Compiled Core ML model.
        """
        cache_key = spec.cache_key()

        # Check memory cache first (fastest)
        if cache_key in self._memory_cache:
            # Move to end for LRU (O(1) with OrderedDict)
            self._memory_cache.move_to_end(cache_key)
            return self._memory_cache[cache_key]

        # Check disk cache
        disk_path = self._cache_dir / f"{cache_key}.mlmodelc"
        if disk_path.exists():
            model = self._load_from_disk(disk_path, cache_key)
            if model is not None:
                self._add_to_memory_cache(cache_key, model)
                return model

        # Compile new model
        logger.debug(f"Compiling new Core ML model for {spec.operation} with key {cache_key}")
        model = compile_fn(spec)

        # Save to caches
        self._save_to_disk(model, disk_path, spec)
        self._add_to_memory_cache(cache_key, model)

        return model

    def _add_to_memory_cache(self, key: str, model: Any) -> None:
        """Add model to memory cache with LRU eviction."""
        # Evict oldest if at capacity (O(1) with OrderedDict)
        while len(self._memory_cache) >= self._max_memory_cache:
            # popitem(last=False) removes the oldest entry
            evicted_key, _ = self._memory_cache.popitem(last=False)
            logger.debug(f"Evicted model {evicted_key} from memory cache (LRU)")

        self._memory_cache[key] = model

    def _load_from_disk(self, path: Path, cache_key: str = "") -> Optional[Any]:
        """Load compiled model from disk.

        Args:
            path: Path to .mlmodelc directory.
            cache_key: Cache key for logging purposes.

        Returns:
            Loaded model or None if loading fails.
        """
        try:
            import coremltools as ct
            model = ct.models.MLModel(str(path))
            logger.debug(f"Loaded cached Core ML model from {path}")
            return model
        except Exception as e:
            # If loading fails, remove corrupted cache and log the event
            logger.warning(
                f"Failed to load cached Core ML model from {path} (key: {cache_key}): {e}. "
                "Removing corrupted cache and will recompile."
            )
            if path.exists():
                import shutil
                try:
                    shutil.rmtree(path)
                    logger.info(f"Removed corrupted cache directory: {path}")
                except OSError as rm_err:
                    logger.error(f"Failed to remove corrupted cache {path}: {rm_err}")
            return None

    def _save_to_disk(
        self,
        model: Any,
        path: Path,
        spec: ModelSpec,
    ) -> None:
        """Save compiled model to disk.

        Args:
            model: Core ML model to save.
            path: Path to save to.
            spec: Model specification (for metadata).
        """
        try:
            model.save(str(path))
            logger.debug(f"Saved Core ML model to {path}")

            # Save metadata alongside
            metadata_path = path.parent / f"{path.stem}.json"
            with open(metadata_path, "w") as f:
                json.dump(spec.to_dict(), f, indent=2)
        except OSError as e:
            # Disk full, permissions, etc. - log but don't fail
            logger.warning(f"Failed to save Core ML model to {path}: {e}")
        except Exception as e:
            # Other unexpected errors - log at warning level since they are unexpected
            logger.warning(f"Unexpected error saving Core ML model to {path}: {e}")

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache."""
        self._memory_cache.clear()
        logger.debug("Cleared in-memory model cache")

    def clear_disk_cache(self) -> None:
        """Clear disk cache."""
        import shutil
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_all(self) -> None:
        """Clear both memory and disk caches."""
        self.clear_memory_cache()
        self.clear_disk_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        disk_models = list(self._cache_dir.glob("*.mlmodelc"))
        disk_size = sum(
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            for p in disk_models
        )

        return {
            "memory_cache_count": len(self._memory_cache),
            "memory_cache_max": self._max_memory_cache,
            "disk_cache_count": len(disk_models),
            "disk_cache_size_mb": disk_size / (1024 * 1024),
            "cache_dir": str(self._cache_dir),
        }


# Global cache instance
_model_cache: Optional[CoreMLModelCache] = None


@lru_cache(maxsize=1)
def get_model_cache() -> CoreMLModelCache:
    """Get the global model cache instance.

    Returns:
        Singleton CoreMLModelCache instance.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = CoreMLModelCache()
    return _model_cache


def warmup_cache(operations: list[str], shapes: list[Tuple[int, ...]]) -> None:
    """Pre-compile models for common operations.

    Useful for reducing first-call latency by triggering
    compilation ahead of time. This actually compiles models,
    not just generates cache keys.

    Args:
        operations: List of operation names (e.g., ["matmul"]).
        shapes: List of common tensor shapes to pre-compile for.

    Example:
        >>> warmup_cache(
        ...     operations=["matmul"],
        ...     shapes=[(1024, 512), (512, 1024), (2048, 768)]
        ... )
    """
    try:
        from mlx_primitives.ane.dispatch import is_ane_available
        if not is_ane_available():
            logger.debug("ANE not available, skipping warmup")
            return
    except ImportError:
        logger.debug("ANE dispatch module not available, skipping warmup")
        return

    try:
        from mlx_primitives.ane.primitives.matmul import _compile_matmul_model
    except ImportError:
        logger.debug("Matmul primitives not available, skipping warmup")
        return

    cache = get_model_cache()
    compiled_count = 0

    for operation in operations:
        if operation == "matmul":
            # Pre-compile matmul for common shape pairs
            for i in range(len(shapes)):
                for j in range(len(shapes)):
                    shape_a = shapes[i]
                    shape_b = shapes[j]

                    # Only compile if shapes are compatible for matmul
                    # Shape A: (M, K), Shape B: (K, N) -> inner dims must match
                    if len(shape_a) >= 2 and len(shape_b) >= 2:
                        if shape_a[-1] != shape_b[-2]:
                            continue  # Incompatible shapes

                        spec = ModelSpec(
                            operation="matmul",
                            input_shapes=(shape_a, shape_b),
                            dtype="float16",
                            params=(),
                        )
                        try:
                            # Actually trigger compilation via get_or_compile
                            cache.get_or_compile(spec, _compile_matmul_model)
                            compiled_count += 1
                            logger.debug(f"Warmed up matmul for shapes {shape_a} @ {shape_b}")
                        except Exception as e:
                            logger.debug(f"Warmup failed for {shape_a} @ {shape_b}: {e}")

    logger.info(f"Cache warmup complete: compiled {compiled_count} models")
