"""Primitives (scan, gather, scatter) parity benchmarks."""

from typing import Any, Dict, List, Optional

from benchmarks.parity.config import ParityBenchmarkConfig
from benchmarks.parity.runner import BenchmarkResult


class PrimitivesParityBenchmarks:
    """Multi-framework primitives benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        raise NotImplementedError("Stub: PrimitivesParityBenchmarks.run_all")

    def benchmark_associative_scan_add(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_associative_scan_add")

    def benchmark_associative_scan_mul(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_associative_scan_mul")

    def benchmark_associative_scan_ssm(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_associative_scan_ssm")

    def benchmark_selective_scan(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_selective_scan")

    def benchmark_selective_gather(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_selective_gather")

    def benchmark_selective_scatter_add(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_selective_scatter_add")
