"""Quantization parity benchmarks."""

from typing import Any, Dict, List, Optional

from benchmarks.parity.config import ParityBenchmarkConfig
from benchmarks.parity.runner import BenchmarkResult


class QuantizationParityBenchmarks:
    """Multi-framework quantization benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        raise NotImplementedError("Stub: QuantizationParityBenchmarks.run_all")

    def benchmark_int8_quantize(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int8_quantize")

    def benchmark_int8_dequantize(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int8_dequantize")

    def benchmark_int4_quantize(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int4_quantize")

    def benchmark_int4_dequantize(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int4_dequantize")

    def benchmark_int8_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int8_linear")

    def benchmark_int4_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_int4_linear")

    def benchmark_vs_fp32(self, size: str) -> Dict[str, BenchmarkResult]:
        raise NotImplementedError("Stub: benchmark_vs_fp32")
