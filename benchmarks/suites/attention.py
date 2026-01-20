"""Attention benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, AttentionSizes
from benchmarks.utils import BenchmarkResult, warmup, benchmark_fn
from benchmarks.baselines.mlx_native import naive_attention


class AttentionBenchmarks:
    """Benchmark suite for attention operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[AttentionSizes] = None,
    ):
        """Initialize attention benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for attention benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or AttentionSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all attention benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []

        for seq_len in self.sizes.seq_lengths:
            for batch_size in self.sizes.batch_sizes:
                # Run naive attention baseline
                baseline_result = self._benchmark_naive_attention(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=self.sizes.num_heads,
                    head_dim=self.sizes.head_dim,
                )
                results.append(baseline_result)

                # Run flash attention if available
                flash_result = self._benchmark_flash_attention(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=self.sizes.num_heads,
                    head_dim=self.sizes.head_dim,
                )
                if flash_result is not None:
                    results.append(flash_result)

        return results

    def _benchmark_naive_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> BenchmarkResult:
        """Benchmark naive O(n^2) attention.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.

        Returns:
            Benchmark result.
        """
        mx.random.seed(self.config.seed)

        # Create input tensors
        query = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        key = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        value = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Warmup
        def fn():
            result = naive_attention(query, key, value)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)

        # Benchmark
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"naive_attn_b{batch_size}_s{seq_len}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "type": "baseline",
            },
        )

    def _benchmark_flash_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark flash attention implementation.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.

        Returns:
            Benchmark result if flash attention is available, None otherwise.
        """
        try:
            from mlx_primitives.attention.flash import flash_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Create input tensors
        query = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        key = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        value = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Warmup
        def fn():
            result = flash_attention(query, key, value)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)

        # Benchmark
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"flash_attn_b{batch_size}_s{seq_len}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "type": "optimized",
            },
        )

    def run_scaling_analysis(
        self,
        seq_lengths: Optional[list[int]] = None,
    ) -> list[BenchmarkResult]:
        """Run scaling analysis across sequence lengths.

        Args:
            seq_lengths: Sequence lengths to test.

        Returns:
            List of benchmark results showing scaling behavior.
        """
        seq_lengths = seq_lengths or [128, 256, 512, 1024, 2048, 4096]
        results = []

        for seq_len in seq_lengths:
            # Naive attention
            naive_result = self._benchmark_naive_attention(
                batch_size=1,
                seq_len=seq_len,
                num_heads=8,
                head_dim=64,
            )
            results.append(naive_result)

            # Flash attention
            flash_result = self._benchmark_flash_attention(
                batch_size=1,
                seq_len=seq_len,
                num_heads=8,
                head_dim=64,
            )
            if flash_result:
                results.append(flash_result)

        return results
