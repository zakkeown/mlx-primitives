"""Benchmark utility functions."""

import time
import statistics
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import mlx.core as mx


@dataclass
class BenchmarkResult:
    """Result of a single benchmark.

    Attributes:
        name: Benchmark name.
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation in milliseconds.
        min_ms: Minimum execution time.
        max_ms: Maximum execution time.
        iterations: Number of iterations run.
        throughput_ops_per_sec: Operations per second (optional).
        memory_mb: Peak memory usage in MB (optional).
    """

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    throughput_ops_per_sec: Optional[float] = None
    memory_mb: Optional[float] = None


def warmup(fn: Callable, warmup_iterations: int = 3) -> None:
    """Run warmup iterations to stabilize GPU.

    Args:
        fn: Function to run.
        warmup_iterations: Number of warmup iterations.
    """
    for _ in range(warmup_iterations):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, (list, tuple)):
            arrays = [x for x in result if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)


def benchmark_fn(
    fn: Callable,
    iterations: int = 10,
    warmup_iterations: int = 3,
) -> List[float]:
    """Benchmark a function, returning list of times in milliseconds.

    Args:
        fn: Function to benchmark.
        iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.

    Returns:
        List of execution times in milliseconds.
    """
    warmup(fn, warmup_iterations)

    times = []
    for _ in range(iterations):
        # Sync before timing
        mx.synchronize()

        start = time.perf_counter()
        result = fn()

        # Force evaluation
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, (list, tuple)):
            arrays = [x for x in result if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)

        # Sync after to ensure complete
        mx.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return times


def compute_statistics(times: List[float], name: str) -> BenchmarkResult:
    """Compute statistics from timing results.

    Args:
        times: List of execution times in milliseconds.
        name: Benchmark name.

    Returns:
        BenchmarkResult with computed statistics.
    """
    return BenchmarkResult(
        name=name,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        iterations=len(times),
    )


def format_time(ms: float) -> str:
    """Format time in milliseconds to human-readable string.

    Args:
        ms: Time in milliseconds.

    Returns:
        Formatted string (e.g., "1.23 ms", "123 us").
    """
    if ms >= 1.0:
        return f"{ms:.3f} ms"
    else:
        return f"{ms * 1000:.1f} us"
