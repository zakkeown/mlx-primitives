"""Quantization benchmark suite."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from benchmarks.config import BenchmarkConfig, QuantizationSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn


class QuantizationBenchmarks:
    """Benchmark suite for quantization operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[QuantizationSizes] = None,
    ):
        """Initialize quantization benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for quantization benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or QuantizationSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all quantization benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_quantize_benchmarks())
        results.extend(self.run_quantized_linear_benchmarks())
        return results

    def run_quantize_benchmarks(self) -> list[BenchmarkResult]:
        """Run tensor quantization/dequantization benchmarks."""
        results = []

        for m, n, k in self.sizes.matrix_sizes[:4]:
            # INT8 quantize
            result = self._benchmark_quantize_int8(m, n)
            if result:
                results.append(result)

            # INT8 dequantize
            result = self._benchmark_dequantize_int8(m, n)
            if result:
                results.append(result)

            # INT4 quantize
            result = self._benchmark_quantize_int4(m, n)
            if result:
                results.append(result)

        return results

    def run_quantized_linear_benchmarks(self) -> list[BenchmarkResult]:
        """Run quantized linear layer benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for m, n, k in self.sizes.matrix_sizes[:4]:
                seq_len = 128

                # INT8 Linear
                result = self._benchmark_quantized_linear_int8(batch_size, seq_len, k, n)
                if result:
                    results.append(result)

                # INT4 Linear
                result = self._benchmark_quantized_linear_int4(batch_size, seq_len, k, n)
                if result:
                    results.append(result)

                # Compare with regular Linear
                result = self._benchmark_regular_linear(batch_size, seq_len, k, n)
                if result:
                    results.append(result)

        return results

    def _benchmark_quantize_int8(
        self,
        rows: int,
        cols: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark INT8 quantization."""
        try:
            from mlx_primitives.advanced import quantize_tensor
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((rows, cols))

        def fn():
            return quantize_tensor(x, num_bits=8)

        name = f"quantize_int8_{rows}x{cols}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "rows": rows,
            "cols": cols,
            "bits": 8,
            "type": "quantize",
            "operation": "quantize_tensor",
        }
        return result

    def _benchmark_dequantize_int8(
        self,
        rows: int,
        cols: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark INT8 dequantization."""
        try:
            from mlx_primitives.advanced import quantize_tensor, dequantize_tensor
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((rows, cols))
        quantized, scale, zero_point = quantize_tensor(x, num_bits=8)

        def fn():
            return dequantize_tensor(quantized, scale, zero_point)

        name = f"dequantize_int8_{rows}x{cols}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "rows": rows,
            "cols": cols,
            "bits": 8,
            "type": "dequantize",
            "operation": "dequantize_tensor",
        }
        return result

    def _benchmark_quantize_int4(
        self,
        rows: int,
        cols: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark INT4 quantization."""
        try:
            from mlx_primitives.advanced import quantize_tensor
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((rows, cols))

        def fn():
            return quantize_tensor(x, num_bits=4)

        name = f"quantize_int4_{rows}x{cols}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "rows": rows,
            "cols": cols,
            "bits": 4,
            "type": "quantize",
            "operation": "quantize_tensor",
        }
        return result

    def _benchmark_quantized_linear_int8(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark INT8 quantized linear layer."""
        try:
            from mlx_primitives.advanced import QuantizedLinear
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = QuantizedLinear(in_features, out_features, num_bits=8)
        x = mx.random.normal((batch_size, seq_len, in_features))

        def fn():
            return layer(x)

        name = f"linear_int8_b{batch_size}_s{seq_len}_{in_features}x{out_features}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "in_features": in_features,
            "out_features": out_features,
            "bits": 8,
            "type": "quantized_linear",
            "layer": "QuantizedLinear",
        }
        return result

    def _benchmark_quantized_linear_int4(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark INT4 quantized linear layer."""
        try:
            from mlx_primitives.advanced import Int4Linear
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = Int4Linear(in_features, out_features)
        x = mx.random.normal((batch_size, seq_len, in_features))

        def fn():
            return layer(x)

        name = f"linear_int4_b{batch_size}_s{seq_len}_{in_features}x{out_features}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "in_features": in_features,
            "out_features": out_features,
            "bits": 4,
            "type": "quantized_linear",
            "layer": "Int4Linear",
        }
        return result

    def _benchmark_regular_linear(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark regular (fp32) linear layer for comparison."""
        mx.random.seed(self.config.seed)

        layer = nn.Linear(in_features, out_features)
        x = mx.random.normal((batch_size, seq_len, in_features))

        def fn():
            return layer(x)

        name = f"linear_fp32_b{batch_size}_s{seq_len}_{in_features}x{out_features}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "in_features": in_features,
            "out_features": out_features,
            "bits": 32,
            "type": "baseline",
            "layer": "nn.Linear",
        }
        return result
