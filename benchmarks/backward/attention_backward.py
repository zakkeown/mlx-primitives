"""Backward pass benchmarks for attention operations."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, AttentionSizes
from benchmarks.utils import BenchmarkResult, benchmark_backward


class AttentionBackwardBenchmarks:
    """Backward pass benchmarks for attention operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[AttentionSizes] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or AttentionSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all attention backward benchmarks."""
        results = []
        results.extend(self.run_flash_attention_backward())
        results.extend(self.run_gqa_backward())
        results.extend(self.run_sliding_window_backward())
        return results

    def run_flash_attention_backward(self) -> list[BenchmarkResult]:
        """Benchmark flash attention backward pass."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            for batch_size in self.sizes.batch_sizes[:2]:
                result = self._benchmark_flash_backward(
                    batch_size, seq_len, self.sizes.num_heads, self.sizes.head_dim
                )
                if result:
                    results.append(result)

        return results

    def run_gqa_backward(self) -> list[BenchmarkResult]:
        """Benchmark GQA backward pass."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            result = self._benchmark_gqa_backward(
                batch_size=2, seq_len=seq_len, num_heads=32, num_kv_heads=8, head_dim=64
            )
            if result:
                results.append(result)

        return results

    def run_sliding_window_backward(self) -> list[BenchmarkResult]:
        """Benchmark sliding window attention backward pass."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            result = self._benchmark_sliding_window_backward(
                batch_size=2, seq_len=seq_len, num_heads=8, head_dim=64, window_size=256
            )
            if result:
                results.append(result)

        return results

    def _benchmark_flash_backward(
        self, batch_size: int, seq_len: int, num_heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark flash attention backward."""
        try:
            from mlx_primitives.attention.flash import flash_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn(q, k, v):
            return flash_attention(q, k, v, causal=True)

        name = f"flash_attn_backward_b{batch_size}_s{seq_len}"
        result = benchmark_backward(
            fn, [q, k, v],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "type": "backward",
            "operation": "flash_attention",
        }
        return result

    def _benchmark_gqa_backward(
        self, batch_size: int, seq_len: int, num_heads: int, num_kv_heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark GQA backward."""
        try:
            from mlx_primitives.attention.grouped_query import grouped_query_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))

        def fn(q, k, v):
            return grouped_query_attention(q, k, v)

        name = f"gqa_backward_b{batch_size}_s{seq_len}_h{num_heads}_kv{num_kv_heads}"
        result = benchmark_backward(
            fn, [q, k, v],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "type": "backward",
            "operation": "gqa",
        }
        return result

    def _benchmark_sliding_window_backward(
        self, batch_size: int, seq_len: int, num_heads: int, head_dim: int, window_size: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark sliding window attention backward."""
        try:
            from mlx_primitives.attention.sliding_window import sliding_window_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn(q, k, v):
            return sliding_window_attention(q, k, v, window_size=window_size)

        name = f"sliding_window_backward_b{batch_size}_s{seq_len}_w{window_size}"
        result = benchmark_backward(
            fn, [q, k, v],
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "window_size": window_size,
            "type": "backward",
            "operation": "sliding_window",
        }
        return result
