"""Benchmark suites for MLX Primitives."""

from benchmarks.suites.attention import AttentionBenchmarks
from benchmarks.suites.scan import ScanBenchmarks
from benchmarks.suites.kernels import KernelBenchmarks

__all__ = [
    "AttentionBenchmarks",
    "ScanBenchmarks",
    "KernelBenchmarks",
]
