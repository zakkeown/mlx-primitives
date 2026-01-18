"""Auto-tuning infrastructure for Metal-Triton DSL kernels.

Provides benchmarking and configuration selection for kernels
decorated with @autotune.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_primitives.dsl.decorators import Config, CompiledKernel


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single Config.

    Attributes:
        config: The configuration that was tested.
        times_ms: All individual timing measurements in milliseconds.
        min_ms: Minimum time across all runs.
        median_ms: Median time across all runs.
        mean_ms: Mean time across all runs.
        std_ms: Standard deviation of times.
        valid: Whether the benchmark completed successfully.
        error: Error message if benchmark failed.
    """

    config: "Config"
    times_ms: list[float] = field(default_factory=list)
    min_ms: float = float("inf")
    median_ms: float = float("inf")
    mean_ms: float = float("inf")
    std_ms: float = 0.0
    valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "config": {
                "num_warps": self.config.num_warps,
                "num_stages": self.config.num_stages,
                **self.config.kwargs,
            },
            "times_ms": self.times_ms,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "valid": self.valid,
            "error": self.error,
        }


@dataclass
class AutoTuneResult:
    """Complete auto-tuning result for a kernel.

    Attributes:
        best_config: The configuration with the best performance.
        all_results: Benchmark results for all tested configurations.
        tuning_key: The key tuple used for caching (e.g., (seq_len, head_dim)).
        kernel_name: Name of the kernel that was tuned.
        timestamp: When the tuning was performed.
        best_time_ms: Best timing achieved.
    """

    best_config: "Config"
    all_results: list[BenchmarkResult]
    tuning_key: tuple
    kernel_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    best_time_ms: float = float("inf")

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "best_config": {
                "num_warps": self.best_config.num_warps,
                "num_stages": self.best_config.num_stages,
                **self.best_config.kwargs,
            },
            "all_results": [r.to_dict() for r in self.all_results],
            "tuning_key": list(self.tuning_key),
            "kernel_name": self.kernel_name,
            "timestamp": self.timestamp,
            "best_time_ms": self.best_time_ms,
        }


class DSLAutoTuner:
    """Auto-tuner for Metal-Triton DSL kernels.

    Benchmarks multiple configurations and selects the fastest one.
    Uses proper GPU synchronization for accurate timing.

    Example:
        >>> tuner = DSLAutoTuner(warmup=3, rep=10)
        >>> result = tuner.tune(kernel, configs, args, kwargs, grid, tuning_key)
        >>> print(f"Best config: {result.best_config}")
    """

    def __init__(
        self,
        warmup: int = 3,
        rep: int = 10,
        timeout_ms: float = 5000.0,
        use_median: bool = True,
    ):
        """Initialize the auto-tuner.

        Args:
            warmup: Number of warmup iterations before timing.
            rep: Number of timed repetitions.
            timeout_ms: Maximum total time for all reps of one config.
            use_median: Use median time for selection (vs min).
        """
        self.warmup = warmup
        self.rep = rep
        self.timeout_ms = timeout_ms
        self.use_median = use_median

    def benchmark_config(
        self,
        kernel: "CompiledKernel",
        config: "Config",
        args: tuple,
        kwargs: dict,
        grid: tuple,
    ) -> BenchmarkResult:
        """Benchmark a single configuration.

        Args:
            kernel: The compiled kernel to benchmark.
            config: Configuration to test.
            args: Positional arguments for the kernel.
            kwargs: Keyword arguments for the kernel.
            grid: Grid dimensions for execution.

        Returns:
            BenchmarkResult with timing statistics.
        """
        try:
            import mlx.core as mx
        except ImportError:
            return BenchmarkResult(
                config=config,
                valid=False,
                error="MLX not available",
            )

        times = []

        try:
            # Merge config kwargs into kwargs
            merged_kwargs = {**kwargs}
            merged_kwargs.update(config.kwargs)

            # Force compilation by doing one execution
            # Get the compiled kernel for this config
            from mlx_primitives.dsl.compiler import compile_kernel

            cache_key = kernel._make_cache_key(config)
            if cache_key not in kernel._compiled_cache:
                metal_source, kernel_info = compile_kernel(
                    kernel.kernel_def,
                    config,
                    debug=kernel.debug,
                )
                kernel._compiled_cache[cache_key] = kernel_info
                kernel._metal_source_cache[cache_key] = metal_source

            kernel_info = kernel._compiled_cache[cache_key]

            # Build args for the kernel
            from mlx_primitives.dsl.compiler import execute_kernel

            full_args = list(args)
            for name in kernel_info.input_names:
                if name in merged_kwargs:
                    full_args.append(merged_kwargs[name])

            # Warmup runs
            for _ in range(self.warmup):
                result = execute_kernel(kernel_info, tuple(full_args), grid, config)
                if isinstance(result, (list, tuple)):
                    mx.eval(*result)
                else:
                    mx.eval(result)

            # Timed runs
            for _ in range(self.rep):
                mx.synchronize()
                start = time.perf_counter()

                result = execute_kernel(kernel_info, tuple(full_args), grid, config)
                if isinstance(result, (list, tuple)):
                    mx.eval(*result)
                else:
                    mx.eval(result)
                mx.synchronize()

                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

                # Timeout check
                if sum(times) > self.timeout_ms:
                    break

            if not times:
                return BenchmarkResult(
                    config=config,
                    valid=False,
                    error="No timing data collected",
                )

            return BenchmarkResult(
                config=config,
                times_ms=times,
                min_ms=min(times),
                median_ms=statistics.median(times),
                mean_ms=statistics.mean(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                valid=True,
            )

        except Exception as e:
            return BenchmarkResult(
                config=config,
                valid=False,
                error=str(e),
            )

    def tune(
        self,
        kernel: "CompiledKernel",
        configs: list["Config"],
        args: tuple,
        kwargs: dict,
        grid: tuple,
        tuning_key: tuple,
    ) -> AutoTuneResult:
        """Run full auto-tuning over all configurations.

        Args:
            kernel: The compiled kernel to tune.
            configs: List of configurations to test.
            args: Positional arguments for the kernel.
            kwargs: Keyword arguments for the kernel.
            grid: Grid dimensions for execution.
            tuning_key: Key tuple for caching results.

        Returns:
            AutoTuneResult with the best configuration.
        """
        results = []

        for config in configs:
            result = self.benchmark_config(kernel, config, args, kwargs, grid)
            results.append(result)

            # Log progress (could add verbose flag)
            if result.valid:
                pass  # Could print timing info here

        # Filter to valid results
        valid_results = [r for r in results if r.valid]

        if not valid_results:
            # All failed, use first config as fallback
            return AutoTuneResult(
                best_config=configs[0],
                all_results=results,
                tuning_key=tuning_key,
                kernel_name=kernel.kernel_def.name,
                best_time_ms=float("inf"),
            )

        # Select best based on median or min time
        if self.use_median:
            best = min(valid_results, key=lambda r: r.median_ms)
            best_time = best.median_ms
        else:
            best = min(valid_results, key=lambda r: r.min_ms)
            best_time = best.min_ms

        return AutoTuneResult(
            best_config=best.config,
            all_results=results,
            tuning_key=tuning_key,
            kernel_name=kernel.kernel_def.name,
            best_time_ms=best_time,
        )


class AutoTuneCache:
    """Thread-safe cache for auto-tuned configurations.

    Provides both in-memory caching and persistent disk storage
    for tuning results across sessions.

    Cache key format: (kernel_name, tuning_key_tuple)
    Example: ("flash_attention_fwd", (1024, 64))
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_disk_cache: bool = True,
    ):
        """Initialize the cache.

        Args:
            cache_dir: Directory for persistent cache. Defaults to ~/.mlx_primitives/dsl_autotune/
            enable_disk_cache: Whether to persist results to disk.
        """
        self._memory_cache: dict[tuple, "Config"] = {}
        self._results_cache: dict[tuple, AutoTuneResult] = {}
        self._lock = threading.RLock()
        self._enable_disk = enable_disk_cache

        if cache_dir is None:
            self._cache_dir = Path.home() / ".mlx_primitives" / "dsl_autotune"
        else:
            self._cache_dir = cache_dir

        if self._enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, kernel_name: str, tuning_key: tuple) -> tuple:
        """Create cache key from kernel name and tuning key."""
        return (kernel_name, tuning_key)

    def _key_to_filename(self, kernel_name: str) -> Path:
        """Get cache filename for a kernel."""
        # Sanitize kernel name for filesystem
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in kernel_name)
        return self._cache_dir / f"{safe_name}.json"

    def _tuning_key_to_str(self, tuning_key: tuple) -> str:
        """Convert tuning key to string for JSON."""
        return str(tuning_key)

    def get(self, kernel_name: str, tuning_key: tuple) -> Optional["Config"]:
        """Get cached best config for kernel + key combination.

        Args:
            kernel_name: Name of the kernel.
            tuning_key: Tuning key tuple.

        Returns:
            Cached Config if found, None otherwise.
        """
        cache_key = self._make_key(kernel_name, tuning_key)

        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]

            # Check disk cache
            if self._enable_disk:
                config = self._load_from_disk(kernel_name, tuning_key)
                if config is not None:
                    self._memory_cache[cache_key] = config
                    return config

        return None

    def put(
        self,
        kernel_name: str,
        tuning_key: tuple,
        config: "Config",
        result: Optional[AutoTuneResult] = None,
    ) -> None:
        """Cache the best config.

        Args:
            kernel_name: Name of the kernel.
            tuning_key: Tuning key tuple.
            config: Best configuration to cache.
            result: Full tuning result (optional, for disk persistence).
        """
        cache_key = self._make_key(kernel_name, tuning_key)

        with self._lock:
            self._memory_cache[cache_key] = config

            if result is not None:
                self._results_cache[cache_key] = result

            # Persist to disk
            if self._enable_disk:
                if result is not None:
                    self._save_to_disk(kernel_name, tuning_key, result)
                else:
                    # Save minimal config data without full result
                    self._save_config_to_disk(kernel_name, tuning_key, config)

    def _load_from_disk(self, kernel_name: str, tuning_key: tuple) -> Optional["Config"]:
        """Load config from persistent cache."""
        from mlx_primitives.dsl.decorators import Config

        cache_file = self._key_to_filename(kernel_name)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            key_str = self._tuning_key_to_str(tuning_key)
            if key_str not in data:
                return None

            entry = data[key_str]
            config_data = entry.get("best_config", {})

            return Config(
                num_warps=config_data.get("num_warps", 4),
                num_stages=config_data.get("num_stages", 1),
                **{k: v for k, v in config_data.items() if k not in ("num_warps", "num_stages")},
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_to_disk(
        self,
        kernel_name: str,
        tuning_key: tuple,
        result: AutoTuneResult,
    ) -> None:
        """Persist tuning result to disk."""
        cache_file = self._key_to_filename(kernel_name)

        # Load existing data
        data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

        # Add new entry
        key_str = self._tuning_key_to_str(tuning_key)
        data[key_str] = result.to_dict()

        # Write back
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Silently fail on disk errors

    def _save_config_to_disk(
        self,
        kernel_name: str,
        tuning_key: tuple,
        config: "Config",
    ) -> None:
        """Persist just a config to disk (without full result)."""
        cache_file = self._key_to_filename(kernel_name)

        # Load existing data
        data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

        # Add new entry with minimal data
        key_str = self._tuning_key_to_str(tuning_key)
        data[key_str] = {
            "best_config": {
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                **config.kwargs,
            }
        }

        # Write back
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Silently fail on disk errors

    def clear(self, kernel_name: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            kernel_name: If provided, only clear entries for this kernel.
                        If None, clear all entries.
        """
        with self._lock:
            if kernel_name is None:
                self._memory_cache.clear()
                self._results_cache.clear()
                if self._enable_disk:
                    for f in self._cache_dir.glob("*.json"):
                        try:
                            f.unlink()
                        except OSError:
                            pass
            else:
                # Clear entries for specific kernel
                keys_to_remove = [
                    k for k in self._memory_cache if k[0] == kernel_name
                ]
                for k in keys_to_remove:
                    del self._memory_cache[k]
                    if k in self._results_cache:
                        del self._results_cache[k]

                if self._enable_disk:
                    cache_file = self._key_to_filename(kernel_name)
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                        except OSError:
                            pass


# Global cache instance
_autotune_cache: Optional[AutoTuneCache] = None
_cache_lock = threading.Lock()


def get_autotune_cache() -> AutoTuneCache:
    """Get the global auto-tune cache instance.

    Returns:
        Singleton AutoTuneCache instance.
    """
    global _autotune_cache
    if _autotune_cache is None:
        with _cache_lock:
            if _autotune_cache is None:
                _autotune_cache = AutoTuneCache()
    return _autotune_cache


def clear_autotune_cache(kernel_name: Optional[str] = None) -> None:
    """Clear the auto-tune cache.

    Args:
        kernel_name: If provided, only clear cache for this kernel.
    """
    cache = get_autotune_cache()
    cache.clear(kernel_name)
