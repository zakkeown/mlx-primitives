"""Benchmark suites for MLX Primitives."""

from benchmarks.suites.attention import AttentionBenchmarks
from benchmarks.suites.scan import ScanBenchmarks
from benchmarks.suites.kernels import KernelBenchmarks
from benchmarks.suites.layers import LayerBenchmarks
from benchmarks.suites.moe import MoEBenchmarks
from benchmarks.suites.cache import CacheBenchmarks
from benchmarks.suites.generation import GenerationBenchmarks
from benchmarks.suites.training import TrainingBenchmarks
from benchmarks.suites.quantization import QuantizationBenchmarks
from benchmarks.suites.memory import MemoryBenchmarks

__all__ = [
    "AttentionBenchmarks",
    "ScanBenchmarks",
    "KernelBenchmarks",
    "LayerBenchmarks",
    "MoEBenchmarks",
    "CacheBenchmarks",
    "GenerationBenchmarks",
    "TrainingBenchmarks",
    "QuantizationBenchmarks",
    "MemoryBenchmarks",
]
