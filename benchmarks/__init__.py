"""Benchmark suite for mlx-primitives.

This module provides performance benchmarking infrastructure:
- BenchmarkRunner: Run and collect benchmark results
- BenchmarkResult: Container for benchmark metrics
- MemoryProfiler: Track memory usage
- ChartGenerator: Create performance visualization charts
"""

from benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkConfig,
)
from benchmarks.memory import MemoryProfiler
from benchmarks.charts import ChartGenerator

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkConfig",
    "MemoryProfiler",
    "ChartGenerator",
]
