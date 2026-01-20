"""Benchmark utility functions."""

import time
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.config import BenchmarkConfig

import mlx.core as mx

try:
    import numpy as np
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    np = None
    scipy_stats = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark.

    Attributes:
        name: Benchmark name.
        mean_time: Mean execution time in seconds.
        std_time: Standard deviation in seconds.
        min_time: Minimum execution time in seconds.
        max_time: Maximum execution time in seconds.
        iterations: Number of iterations run.
        metadata: Optional metadata dict (batch_size, seq_len, type, etc.).
        throughput_ops_per_sec: Operations per second (optional).
        memory_mb: Peak memory usage in MB (optional).
        median_time: Median execution time in seconds (optional).
        percentile_95: 95th percentile execution time (optional).
        percentile_99: 99th percentile execution time (optional).
        coefficient_of_variation: Std/mean ratio for stability assessment (optional).
        confidence_interval_95: 95% confidence interval tuple (optional).
        raw_times: Raw timing data for advanced analysis (optional).
    """

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    metadata: Optional[dict] = None
    throughput_ops_per_sec: Optional[float] = None
    memory_mb: Optional[float] = None
    median_time: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    coefficient_of_variation: Optional[float] = None
    confidence_interval_95: Optional[Tuple[float, float]] = None
    raw_times: Optional[List[float]] = field(default=None, repr=False)


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
    name: str = "benchmark",
) -> BenchmarkResult:
    """Benchmark a function, returning a BenchmarkResult.

    Args:
        fn: Function to benchmark.
        iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        name: Name for the benchmark result.

    Returns:
        BenchmarkResult with timing statistics (times in seconds).
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

        times.append(end - start)  # Time in seconds

    return compute_statistics(times, name)


def compute_statistics(
    times: List[float],
    name: str,
    store_raw: bool = False,
) -> BenchmarkResult:
    """Compute comprehensive statistics from timing results.

    Args:
        times: List of execution times in seconds.
        name: Benchmark name.
        store_raw: Whether to store raw timing data.

    Returns:
        BenchmarkResult with computed statistics including percentiles and CI.
    """
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0

    # Compute additional statistics
    median_time = statistics.median(times)
    cv = std / mean if mean > 0 else 0.0

    # Compute percentiles and confidence intervals if scipy available
    percentile_95 = None
    percentile_99 = None
    ci_95 = None

    if HAS_SCIPY and n > 1:
        times_arr = np.array(times)
        percentile_95 = float(np.percentile(times_arr, 95))
        percentile_99 = float(np.percentile(times_arr, 99))

        # 95% confidence interval using t-distribution
        t_crit = scipy_stats.t.ppf(0.975, n - 1)
        margin = t_crit * std / (n ** 0.5)
        ci_95 = (mean - margin, mean + margin)
    else:
        # Fallback without scipy
        sorted_times = sorted(times)
        if n >= 20:
            percentile_95 = sorted_times[int(n * 0.95)]
            percentile_99 = sorted_times[int(n * 0.99)]

    return BenchmarkResult(
        name=name,
        mean_time=mean,
        std_time=std,
        min_time=min(times),
        max_time=max(times),
        iterations=n,
        median_time=median_time,
        percentile_95=percentile_95,
        percentile_99=percentile_99,
        coefficient_of_variation=cv,
        confidence_interval_95=ci_95,
        raw_times=times if store_raw else None,
    )


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string (e.g., "1.23 s", "123 ms", "456 us").
    """
    ms = seconds * 1000
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    elif ms >= 1.0:
        return f"{ms:.3f} ms"
    else:
        return f"{ms * 1000:.1f} us"


def benchmark_backward(
    fn: Callable,
    inputs: List[mx.array],
    iterations: int = 30,
    warmup_iterations: int = 5,
    name: str = "backward",
    loss_fn: Optional[Callable] = None,
    argnums: Optional[List[int]] = None,
) -> BenchmarkResult:
    """Benchmark backward pass (gradient computation).

    Args:
        fn: Forward function to differentiate.
        inputs: List of input tensors requiring gradients.
        iterations: Number of timed iterations.
        warmup_iterations: Warmup iterations.
        name: Benchmark name.
        loss_fn: Function to reduce output to scalar. Defaults to mx.sum.
        argnums: Which arguments to differentiate. Defaults to all.

    Returns:
        BenchmarkResult with backward pass timing.
    """
    loss_fn = loss_fn or mx.sum
    argnums = argnums if argnums is not None else list(range(len(inputs)))

    # Create gradient function
    def compute_loss(*args):
        output = fn(*args)
        return loss_fn(output)

    grad_fn = mx.grad(compute_loss, argnums=argnums)

    def backward_op():
        grads = grad_fn(*inputs)
        # Force evaluation of all gradients
        if isinstance(grads, (list, tuple)):
            mx.eval(*grads)
        else:
            mx.eval(grads)
        return grads

    result = benchmark_fn(backward_op, iterations, warmup_iterations, name)
    if result.metadata is None:
        result.metadata = {}
    result.metadata["type"] = "backward"
    return result


def benchmark_fn_robust(
    fn: Callable,
    iterations: int = 30,
    warmup_iterations: int = 5,
    name: str = "benchmark",
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    store_raw: bool = False,
) -> BenchmarkResult:
    """Benchmark with outlier removal for more stable results.

    Args:
        fn: Function to benchmark.
        iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        name: Name for the benchmark result.
        remove_outliers: Whether to remove outliers.
        outlier_threshold: Standard deviations for outlier detection.
        store_raw: Whether to store raw timing data.

    Returns:
        BenchmarkResult with timing statistics.
    """
    warmup(fn, warmup_iterations)

    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        result = fn()

        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, (list, tuple)):
            arrays = [x for x in result if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)

        mx.synchronize()
        times.append(time.perf_counter() - start)

    # Remove outliers if requested and scipy available
    if remove_outliers and len(times) > 5 and HAS_SCIPY:
        times_arr = np.array(times)
        mean = np.mean(times_arr)
        std = np.std(times_arr)
        if std > 0:
            mask = np.abs(times_arr - mean) < outlier_threshold * std
            times = list(times_arr[mask])

    return compute_statistics(times, name, store_raw=store_raw)


def estimate_time(fn: Callable, quick_iterations: int = 3) -> float:
    """Quickly estimate the execution time of a function.

    Args:
        fn: Function to time.
        quick_iterations: Number of quick iterations for estimation.

    Returns:
        Estimated time in milliseconds.
    """
    warmup(fn, warmup_iterations=1)

    total_time = 0.0
    for _ in range(quick_iterations):
        mx.synchronize()
        start = time.perf_counter()
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, (list, tuple)):
            arrays = [x for x in result if isinstance(x, mx.array)]
            if arrays:
                mx.eval(*arrays)
        mx.synchronize()
        total_time += time.perf_counter() - start

    return (total_time / quick_iterations) * 1000  # Convert to ms


def benchmark_fn_adaptive(
    fn: Callable,
    config: "BenchmarkConfig",
    warmup_iterations: int = 5,
    name: str = "benchmark",
) -> BenchmarkResult:
    """Benchmark with adaptive iteration count based on operation speed.

    Args:
        fn: Function to benchmark.
        config: BenchmarkConfig with adaptive settings.
        warmup_iterations: Number of warmup iterations.
        name: Name for the benchmark result.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Estimate time to determine iteration count
    est_time_ms = estimate_time(fn)
    iterations = config.get_iterations(est_time_ms)

    return benchmark_fn_robust(
        fn,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        name=name,
        remove_outliers=True,
    )
