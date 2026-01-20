"""Fused kernel benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, MatmulSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn
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

        # Run fused RoPE + attention benchmarks
        results.extend(self.run_fused_rope_benchmarks())

        # Run fused add + norm benchmarks
        results.extend(self.run_fused_add_norm_benchmarks())

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
            return naive_layer_norm(x, weight, bias)

        name = f"naive_ln_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "baseline",
        }
        return result

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
            return naive_rms_norm(x, weight)

        name = f"naive_rms_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "baseline",
        }
        return result

    def _benchmark_fused_layer_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused layer normalization."""
        try:
            from mlx_primitives.kernels.layernorm import fast_layernorm
        except ImportError:
            print(f"  Warning: fast_layernorm not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        gamma = mx.random.normal((hidden_dim,))
        beta = mx.random.normal((hidden_dim,))

        def fn():
            return fast_layernorm(x, gamma, beta)

        name = f"fused_ln_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "optimized",
        }
        return result

    def _benchmark_fused_rms_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused RMS normalization."""
        try:
            from mlx_primitives.kernels.rmsnorm import fast_rmsnorm
        except ImportError:
            print(f"  Warning: fast_rmsnorm not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.random.normal((hidden_dim,))

        def fn():
            return fast_rmsnorm(x, weight)

        name = f"fused_rms_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "optimized",
        }
        return result

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
            return act_fn(x)

        name = f"naive_{activation}_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "type": "baseline",
        }
        return result

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
                from mlx_primitives.kernels.fused_activations import silu as fused_act
            else:
                from mlx_primitives.kernels.fused_activations import gelu as fused_act
        except ImportError:
            print(f"  Warning: fused {activation} not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return fused_act(x)

        name = f"fused_{activation}_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "type": "optimized",
        }
        return result

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
            return naive_swiglu(x, gate)

        name = f"naive_swiglu_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "baseline",
        }
        return result

    def _benchmark_fused_swiglu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused SwiGLU activation."""
        try:
            from mlx_primitives.kernels.swiglu import fast_swiglu
        except ImportError:
            print(f"  Warning: fast_swiglu not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        gate = mx.random.normal((batch_size, seq_len, hidden_dim))
        up = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return fast_swiglu(gate, up)

        name = f"fused_swiglu_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "optimized",
        }
        return result

    def run_fused_rope_benchmarks(self) -> list[BenchmarkResult]:
        """Run fused RoPE + attention benchmarks."""
        results = []

        sizes = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (4, 2048, 8, 64),
        ]

        for batch, seq, heads, dim in sizes:
            result = self._benchmark_fused_rope_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_fused_rope_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused RoPE + attention kernel."""
        try:
            from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn():
            return fused_rope_attention(q, k, v, causal=True)

        name = f"fused_rope_attn_b{batch_size}_s{seq_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "type": "optimized",
            "operation": "fused_rope_attention",
        }
        return result

    def run_fused_add_norm_benchmarks(self) -> list[BenchmarkResult]:
        """Run fused add + normalization benchmarks."""
        results = []

        sizes = [
            (4, 512, 1024),
            (4, 1024, 2048),
            (8, 2048, 4096),
        ]

        for batch, seq, hidden in sizes:
            # Fused add + layer norm
            result = self._benchmark_fused_add_layernorm(batch, seq, hidden)
            if result:
                results.append(result)

            # Fused add + RMS norm
            result = self._benchmark_fused_add_rmsnorm(batch, seq, hidden)
            if result:
                results.append(result)

        return results

    def _benchmark_fused_add_layernorm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused add + layer norm.

        Note: Currently not available as a fused kernel.
        """
        # No fused add + layernorm kernel available yet
        return None

    def _benchmark_fused_add_rmsnorm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark fused add + RMS norm."""
        try:
            from mlx_primitives.kernels.rmsnorm import fast_rmsnorm_residual
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        residual = mx.random.normal((batch_size, seq_len, hidden_dim))
        weight = mx.ones((hidden_dim,))

        def fn():
            return fast_rmsnorm_residual(x, residual, weight)

        name = f"fused_add_rms_b{batch_size}_s{seq_len}_h{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "optimized",
            "operation": "fused_add_rmsnorm",
        }
        return result
