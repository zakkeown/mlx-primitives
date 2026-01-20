"""MLX Primitives Benchmark Suite.

Provides reproducible benchmarks comparing MLX Primitives implementations
against reference implementations and other frameworks.

Usage:
    python -m benchmarks.runner --suite all
    python -m benchmarks.runner --suite attention
    python -m benchmarks.runner --output results.json
"""

from benchmarks.config import BenchmarkConfig
from benchmarks.utils import (
    BenchmarkResult,
    warmup,
    benchmark_fn,
    compute_statistics,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "warmup",
    "benchmark_fn",
    "compute_statistics",
]
