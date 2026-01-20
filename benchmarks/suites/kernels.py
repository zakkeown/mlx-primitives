"""Fused kernel benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, MatmulSizes
from benchmarks.utils import BenchmarkResult, warmup, benchmark_fn
from benchmarks.baselines.mlx_native import (
    naive_layer_norm,
    naive_rms_norm,
    naive_silu,
    naive_gelu,
    naive_swiglu,
)


class KernelBenchmarks:
    """Benchmark suite for fused kernel operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[MatmulSizes] = None,
    ):
        """Initialize kernel benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for matmul benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or MatmulSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all kernel benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []

        # Run normalization benchmarks
        results.extend(self.run_normalization_benchmarks())

        # Run activation benchmarks
        results.extend(self.run_activation_benchmarks())

        # Run fused operation benchmarks
        results.extend(self.run_fused_benchmarks())

        return results

    def run_normalization_benchmarks(
        self,
        batch_sizes: Optional[list[int]] = None,
        seq_lengths: Optional[list[int]] = None,
        hidden_dims: Optional[list[int]] = None,
    ) -> list[BenchmarkResult]:
        """Run normalization benchmarks.

        Args:
            batch_sizes: Batch sizes to test.
            seq_lengths: Sequence lengths to test.
            hidden_dims: Hidden dimensions to test.

        Returns:
            List of benchmark results.
        """
        batch_sizes = batch_sizes or [1, 4, 8]
        seq_lengths = seq_lengths or [512, 1024, 2048]
        hidden_dims = hidden_dims or [768, 1024, 4096]

        results = []

        for batch in batch_sizes:
            for seq_len in seq_lengths:
                for hidden in hidden_dims:
                    # Layer norm baselines
                    results.append(self._benchmark_naive_layer_norm(batch, seq_len, hidden))

                    # RMS norm baselines
                    results.append(self._benchmark_naive_rms_norm(batch, seq_len, hidden))

                    # Fused versions if available
                    fused_ln = self._benchmark_fused_layer_norm(batch, seq_len, hidden)
                    if fused_ln:
                        results.append(fused_ln)

                    fused_rms = self._benchmark_fused_rms_norm(batch, seq_len, hidden)
                    if fused_rms:
                        results.append(fused_rms)

        return results

    def _benchmark_naive_layer_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> BenchmarkResult:
        """Benchmark naive layer normalization."""
        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.random.normal((hidden_dim,))
        bias = mx.random.normal((hidden_dim,))

        def fn():
            result = naive_layer_norm(x, weight, bias)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"naive_ln_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "baseline",
            },
        )

    def _benchmark_naive_rms_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> BenchmarkResult:
        """Benchmark naive RMS normalization."""
        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.random.normal((hidden_dim,))

        def fn():
            result = naive_rms_norm(x, weight)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"naive_rms_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "baseline",
            },
        )

    def _benchmark_fused_layer_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused layer normalization."""
        try:
            from mlx_primitives.dsl.examples.normalization import layer_norm as fused_layer_norm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.random.normal((hidden_dim,))
        bias = mx.random.normal((hidden_dim,))

        def fn():
            result = fused_layer_norm(x, weight, bias)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"fused_ln_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "optimized",
            },
        )

    def _benchmark_fused_rms_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused RMS normalization."""
        try:
            from mlx_primitives.dsl.examples.normalization import rms_norm as fused_rms_norm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.random.normal((hidden_dim,))

        def fn():
            result = fused_rms_norm(x, weight)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"fused_rms_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "optimized",
            },
        )

    def run_activation_benchmarks(
        self,
        tensor_sizes: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[BenchmarkResult]:
        """Run activation function benchmarks.

        Args:
            tensor_sizes: List of (batch, seq, hidden) tuples to test.

        Returns:
            List of benchmark results.
        """
        tensor_sizes = tensor_sizes or [
            (4, 512, 1024),
            (4, 1024, 2048),
            (8, 2048, 4096),
        ]

        results = []

        for batch, seq, hidden in tensor_sizes:
            # SiLU benchmarks
            results.append(self._benchmark_naive_activation(batch, seq, hidden, "silu"))
            fused_silu = self._benchmark_fused_activation(batch, seq, hidden, "silu")
            if fused_silu:
                results.append(fused_silu)

            # GELU benchmarks
            results.append(self._benchmark_naive_activation(batch, seq, hidden, "gelu"))
            fused_gelu = self._benchmark_fused_activation(batch, seq, hidden, "gelu")
            if fused_gelu:
                results.append(fused_gelu)

        return results

    def _benchmark_naive_activation(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        activation: str,
    ) -> BenchmarkResult:
        """Benchmark naive activation function."""
        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        act_fn = naive_silu if activation == "silu" else naive_gelu

        def fn():
            result = act_fn(x)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"naive_{activation}_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "activation": activation,
                "type": "baseline",
            },
        )

    def _benchmark_fused_activation(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        activation: str,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused activation function."""
        try:
            if activation == "silu":
                from mlx_primitives.dsl.examples.activations import silu as fused_act
            else:
                from mlx_primitives.dsl.examples.activations import gelu as fused_act
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            result = fused_act(x)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"fused_{activation}_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "activation": activation,
                "type": "optimized",
            },
        )

    def run_fused_benchmarks(
        self,
        tensor_sizes: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[BenchmarkResult]:
        """Run fused operation benchmarks (SwiGLU, etc).

        Args:
            tensor_sizes: List of (batch, seq, hidden) tuples to test.

        Returns:
            List of benchmark results.
        """
        tensor_sizes = tensor_sizes or [
            (4, 512, 1024),
            (4, 1024, 2048),
        ]

        results = []

        for batch, seq, hidden in tensor_sizes:
            # SwiGLU benchmarks
            results.append(self._benchmark_naive_swiglu(batch, seq, hidden))
            fused_swiglu = self._benchmark_fused_swiglu(batch, seq, hidden)
            if fused_swiglu:
                results.append(fused_swiglu)

        return results

    def _benchmark_naive_swiglu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> BenchmarkResult:
        """Benchmark naive SwiGLU activation."""
        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        gate = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            result = naive_swiglu(x, gate)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"naive_swiglu_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "baseline",
            },
        )

    def _benchmark_fused_swiglu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused SwiGLU activation."""
        try:
            from mlx_primitives.dsl.examples.activations import fused_silu_mul
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        gate = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            result = fused_silu_mul(gate, x)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"fused_swiglu_b{batch_size}_s{seq_len}_h{hidden_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "type": "optimized",
            },
        )
