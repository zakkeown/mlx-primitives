"""Parity benchmarking suite for MLXPrimitives vs PyTorch and JAX."""

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig
from benchmarks.parity.runner import ParityBenchmarkRunner
from benchmarks.parity.comparison import (
    ComparisonResult,
    compare_frameworks,
    compute_speedups,
    generate_comparison_table,
)

__all__ = [
    "ParityBenchmarkConfig",
    "ParitySizeConfig",
    "ParityBenchmarkRunner",
    "ComparisonResult",
    "compare_frameworks",
    "compute_speedups",
    "generate_comparison_table",
]
