"""Backward pass benchmarks for kernel operations."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, LayerSizes
from benchmarks.utils import BenchmarkResult, benchmark_backward


class KernelsBackwardBenchmarks:
    """Backward pass benchmarks for kernel operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[LayerSizes] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or LayerSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all kernel backward benchmarks."""
        results = []
        results.extend(self.run_normalization_backward())
        results.extend(self.run_activation_backward())
        return results

    def run_normalization_backward(self) -> list[BenchmarkResult]:
        """Benchmark normalization backward passes."""
        results = []

        for hidden_dim in self.sizes.hidden_dims[:3]:
            batch_size = 4
            seq_len = 512

            # RMSNorm backward
            result = self._benchmark_rmsnorm_backward(batch_size, seq_len, hidden_dim)
            if result:
                results.append(result)

            # LayerNorm backward
            result = self._benchmark_layernorm_backward(batch_size, seq_len, hidden_dim)
            if result:
                results.append(result)

        return results

    def run_activation_backward(self) -> list[BenchmarkResult]:
        """Benchmark activation backward passes."""
        results = []

        for hidden_dim in self.sizes.hidden_dims[:3]:
            batch_size = 4
            seq_len = 512

            # SwiGLU backward
            result = self._benchmark_swiglu_backward(batch_size, seq_len, hidden_dim)
            if result:
                results.append(result)

            # GELU backward
            result = self._benchmark_gelu_backward(batch_size, seq_len, hidden_dim)
            if result:
                results.append(result)

            # SiLU backward
            result = self._benchmark_silu_backward(batch_size, seq_len, hidden_dim)
            if result:
                results.append(result)

        return results

    def _benchmark_rmsnorm_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark RMSNorm backward."""
        try:
            from mlx_primitives.layers import RMSNorm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = RMSNorm(hidden_dim)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            return layer(x)

        name = f"rmsnorm_backward_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_backward(
            fn, [x],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "backward",
            "operation": "rmsnorm",
        }
        return result

    def _benchmark_layernorm_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark LayerNorm backward."""
        import mlx.nn as nn

        mx.random.seed(self.config.seed)

        layer = nn.LayerNorm(hidden_dim)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            return layer(x)

        name = f"layernorm_backward_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_backward(
            fn, [x],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "backward",
            "operation": "layernorm",
        }
        return result

    def _benchmark_swiglu_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark SwiGLU backward."""
        try:
            from mlx_primitives.layers import SwiGLU
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = SwiGLU(hidden_dim, hidden_dim * 4)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            return layer(x)

        name = f"swiglu_backward_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_backward(
            fn, [x],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "backward",
            "operation": "swiglu",
        }
        return result

    def _benchmark_gelu_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark GELU backward."""
        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            return mx.nn.gelu(x)

        name = f"gelu_backward_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_backward(
            fn, [x],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "backward",
            "operation": "gelu",
        }
        return result

    def _benchmark_silu_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark SiLU backward."""
        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            return mx.nn.silu(x)

        name = f"silu_backward_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_backward(
            fn, [x],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "type": "backward",
            "operation": "silu",
        }
        return result
