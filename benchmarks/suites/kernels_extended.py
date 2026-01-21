"""Extended kernel benchmark suite for RoPE, RMSNorm, GQA, and Pooling.

Benchmarks Metal-optimized kernel implementations against baselines.
"""

from typing import Optional, List
import time

import mlx.core as mx

from benchmarks.config import BenchmarkConfig
from benchmarks.utils import BenchmarkResult, benchmark_fn


class KernelExtendedBenchmarks:
    """Benchmark suite for extended kernel operations."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize extended kernel benchmarks.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()

    def run_all(self) -> List[BenchmarkResult]:
        """Run all extended kernel benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []

        # RoPE benchmarks
        results.extend(self.run_rope_benchmarks())

        # RMSNorm benchmarks
        results.extend(self.run_rmsnorm_benchmarks())

        # GQA benchmarks
        results.extend(self.run_gqa_benchmarks())

        # Pooling benchmarks
        results.extend(self.run_pooling_benchmarks())

        return results

    # =========================================================================
    # RoPE Benchmarks
    # =========================================================================

    def run_rope_benchmarks(
        self,
        configs: Optional[List[dict]] = None,
    ) -> List[BenchmarkResult]:
        """Run RoPE benchmarks.

        Args:
            configs: List of configuration dicts with batch, seq, heads, head_dim.

        Returns:
            List of benchmark results.
        """
        configs = configs or [
            {"batch": 1, "seq": 512, "heads": 32, "head_dim": 128},
            {"batch": 8, "seq": 2048, "heads": 32, "head_dim": 128},
            {"batch": 1, "seq": 8192, "heads": 32, "head_dim": 128},
        ]

        results = []

        for cfg in configs:
            result = self._benchmark_fast_rope(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_fast_rope_qk(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_rope_fallback(**cfg)
            if result:
                results.append(result)

        return results

    def _benchmark_fast_rope(
        self, batch: int, seq: int, heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark fast_rope Metal kernel."""
        try:
            from mlx_primitives.kernels.rope import fast_rope, precompute_rope_cache

            x = mx.random.normal((batch, seq, heads, head_dim))
            cos_cache, sin_cache = precompute_rope_cache(seq, head_dim)
            mx.eval(x, cos_cache, sin_cache)

            def fn():
                return fast_rope(x, cos_cache, sin_cache)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"fast_rope_{batch}x{seq}x{heads}x{head_dim}",
            )
        except Exception as e:
            print(f"fast_rope benchmark failed: {e}")
            return None

    def _benchmark_fast_rope_qk(
        self, batch: int, seq: int, heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark fast_rope_qk fused Metal kernel."""
        try:
            from mlx_primitives.kernels.rope import fast_rope_qk, precompute_rope_cache

            q = mx.random.normal((batch, seq, heads, head_dim))
            k = mx.random.normal((batch, seq, heads, head_dim))
            cos_cache, sin_cache = precompute_rope_cache(seq, head_dim)
            mx.eval(q, k, cos_cache, sin_cache)

            def fn():
                return fast_rope_qk(q, k, cos_cache, sin_cache)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"fast_rope_qk_{batch}x{seq}x{heads}x{head_dim}",
            )
        except Exception as e:
            print(f"fast_rope_qk benchmark failed: {e}")
            return None

    def _benchmark_rope_fallback(
        self, batch: int, seq: int, heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark rope Python fallback."""
        try:
            from mlx_primitives.kernels.rope import rope, precompute_rope_cache

            x = mx.random.normal((batch, seq, heads, head_dim))
            cos_cache, sin_cache = precompute_rope_cache(seq, head_dim)
            mx.eval(x, cos_cache, sin_cache)

            def fn():
                return rope(x, cos_cache, sin_cache, use_metal=False)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"rope_fallback_{batch}x{seq}x{heads}x{head_dim}",
            )
        except Exception as e:
            print(f"rope_fallback benchmark failed: {e}")
            return None

    # =========================================================================
    # RMSNorm Benchmarks
    # =========================================================================

    def run_rmsnorm_benchmarks(
        self,
        configs: Optional[List[dict]] = None,
    ) -> List[BenchmarkResult]:
        """Run RMSNorm benchmarks.

        Args:
            configs: List of configuration dicts with batch, seq, hidden.

        Returns:
            List of benchmark results.
        """
        configs = configs or [
            {"batch": 1, "seq": 512, "hidden": 4096},
            {"batch": 8, "seq": 2048, "hidden": 4096},
            {"batch": 4, "seq": 4096, "hidden": 8192},
        ]

        results = []

        for cfg in configs:
            result = self._benchmark_fast_rmsnorm(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_fast_rmsnorm_residual(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_rmsnorm_fallback(**cfg)
            if result:
                results.append(result)

        return results

    def _benchmark_fast_rmsnorm(
        self, batch: int, seq: int, hidden: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark fast_rmsnorm Metal kernel."""
        try:
            from mlx_primitives.kernels.rmsnorm import fast_rmsnorm

            x = mx.random.normal((batch, seq, hidden))
            weight = mx.random.normal((hidden,))
            mx.eval(x, weight)

            def fn():
                return fast_rmsnorm(x, weight)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"fast_rmsnorm_{batch}x{seq}x{hidden}",
            )
        except Exception as e:
            print(f"fast_rmsnorm benchmark failed: {e}")
            return None

    def _benchmark_fast_rmsnorm_residual(
        self, batch: int, seq: int, hidden: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark fast_rmsnorm_residual fused kernel."""
        try:
            from mlx_primitives.kernels.rmsnorm import fast_rmsnorm_residual

            x = mx.random.normal((batch, seq, hidden))
            residual = mx.random.normal((batch, seq, hidden))
            weight = mx.random.normal((hidden,))
            mx.eval(x, residual, weight)

            def fn():
                return fast_rmsnorm_residual(x, residual, weight)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"fast_rmsnorm_residual_{batch}x{seq}x{hidden}",
            )
        except Exception as e:
            print(f"fast_rmsnorm_residual benchmark failed: {e}")
            return None

    def _benchmark_rmsnorm_fallback(
        self, batch: int, seq: int, hidden: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark rmsnorm Python fallback."""
        try:
            from mlx_primitives.kernels.rmsnorm import rmsnorm

            x = mx.random.normal((batch, seq, hidden))
            weight = mx.random.normal((hidden,))
            mx.eval(x, weight)

            def fn():
                return rmsnorm(x, weight, use_metal=False)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"rmsnorm_fallback_{batch}x{seq}x{hidden}",
            )
        except Exception as e:
            print(f"rmsnorm_fallback benchmark failed: {e}")
            return None

    # =========================================================================
    # GQA Benchmarks
    # =========================================================================

    def run_gqa_benchmarks(
        self,
        configs: Optional[List[dict]] = None,
    ) -> List[BenchmarkResult]:
        """Run GQA benchmarks.

        Args:
            configs: List of configuration dicts with batch, seq, q_heads, kv_heads, head_dim.

        Returns:
            List of benchmark results.
        """
        configs = configs or [
            {"batch": 1, "seq": 512, "q_heads": 32, "kv_heads": 8, "head_dim": 128},
            {"batch": 1, "seq": 512, "q_heads": 64, "kv_heads": 8, "head_dim": 128},
            {"batch": 8, "seq": 2048, "q_heads": 32, "kv_heads": 8, "head_dim": 128},
        ]

        results = []

        for cfg in configs:
            result = self._benchmark_fast_gqa_attention(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_gqa_attention_reference(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_gqa_attention_mlx_sdpa(**cfg)
            if result:
                results.append(result)

        return results

    def _benchmark_fast_gqa_attention(
        self, batch: int, seq: int, q_heads: int, kv_heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark fast_gqa_attention Metal kernel."""
        try:
            from mlx_primitives.kernels.gqa_optimized import fast_gqa_attention

            num_kv_groups = q_heads // kv_heads
            q = mx.random.normal((batch, seq, q_heads, head_dim))
            k = mx.random.normal((batch, seq, kv_heads, head_dim))
            v = mx.random.normal((batch, seq, kv_heads, head_dim))
            mx.eval(q, k, v)

            def fn():
                return fast_gqa_attention(q, k, v, num_kv_groups)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"fast_gqa_{batch}x{seq}x{q_heads}_{kv_heads}x{head_dim}",
            )
        except Exception as e:
            print(f"fast_gqa_attention benchmark failed: {e}")
            return None

    def _benchmark_gqa_attention_reference(
        self, batch: int, seq: int, q_heads: int, kv_heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark gqa_attention_reference (K/V expansion)."""
        try:
            from mlx_primitives.kernels.gqa_optimized import gqa_attention_reference

            num_kv_groups = q_heads // kv_heads
            q = mx.random.normal((batch, seq, q_heads, head_dim))
            k = mx.random.normal((batch, seq, kv_heads, head_dim))
            v = mx.random.normal((batch, seq, kv_heads, head_dim))
            mx.eval(q, k, v)

            def fn():
                return gqa_attention_reference(q, k, v, num_kv_groups)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"gqa_reference_{batch}x{seq}x{q_heads}_{kv_heads}x{head_dim}",
            )
        except Exception as e:
            print(f"gqa_attention_reference benchmark failed: {e}")
            return None

    def _benchmark_gqa_attention_mlx_sdpa(
        self, batch: int, seq: int, q_heads: int, kv_heads: int, head_dim: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark gqa_attention using MLX fast SDPA."""
        try:
            from mlx_primitives.kernels.gqa_optimized import gqa_attention

            num_kv_groups = q_heads // kv_heads
            q = mx.random.normal((batch, seq, q_heads, head_dim))
            k = mx.random.normal((batch, seq, kv_heads, head_dim))
            v = mx.random.normal((batch, seq, kv_heads, head_dim))
            mx.eval(q, k, v)

            def fn():
                return gqa_attention(q, k, v, num_kv_groups, use_mlx_sdpa=True)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"gqa_mlx_sdpa_{batch}x{seq}x{q_heads}_{kv_heads}x{head_dim}",
            )
        except Exception as e:
            print(f"gqa_attention MLX SDPA benchmark failed: {e}")
            return None

    # =========================================================================
    # Pooling Benchmarks
    # =========================================================================

    def run_pooling_benchmarks(
        self,
        configs: Optional[List[dict]] = None,
    ) -> List[BenchmarkResult]:
        """Run Pooling benchmarks.

        Args:
            configs: List of configuration dicts with batch, channels, length, kernel, stride.

        Returns:
            List of benchmark results.
        """
        configs = configs or [
            {"batch": 8, "channels": 256, "length": 1024, "kernel": 3, "stride": 2},
            {"batch": 16, "channels": 512, "length": 512, "kernel": 5, "stride": 2},
        ]

        results = []

        for cfg in configs:
            result = self._benchmark_avg_pool1d(**cfg)
            if result:
                results.append(result)

            result = self._benchmark_max_pool1d(**cfg)
            if result:
                results.append(result)

        return results

    def _benchmark_avg_pool1d(
        self, batch: int, channels: int, length: int, kernel: int, stride: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark AvgPool1d."""
        try:
            from mlx_primitives.layers.pooling import AvgPool1d

            pool = AvgPool1d(kernel_size=kernel, stride=stride)
            x = mx.random.normal((batch, channels, length))
            mx.eval(x)

            def fn():
                return pool(x)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"avg_pool1d_{batch}x{channels}x{length}_k{kernel}_s{stride}",
            )
        except Exception as e:
            print(f"AvgPool1d benchmark failed: {e}")
            return None

    def _benchmark_max_pool1d(
        self, batch: int, channels: int, length: int, kernel: int, stride: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark MaxPool1d."""
        try:
            from mlx_primitives.layers.pooling import MaxPool1d

            pool = MaxPool1d(kernel_size=kernel, stride=stride)
            x = mx.random.normal((batch, channels, length))
            mx.eval(x)

            def fn():
                return pool(x)

            return benchmark_fn(
                fn,
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=f"max_pool1d_{batch}x{channels}x{length}_k{kernel}_s{stride}",
            )
        except Exception as e:
            print(f"MaxPool1d benchmark failed: {e}")
            return None


def run_extended_benchmarks() -> List[BenchmarkResult]:
    """Run all extended kernel benchmarks.

    Returns:
        List of benchmark results.
    """
    benchmarks = KernelExtendedBenchmarks()
    return benchmarks.run_all()


if __name__ == "__main__":
    results = run_extended_benchmarks()

    print("\n" + "=" * 60)
    print("Extended Kernel Benchmarks Results")
    print("=" * 60)

    for result in results:
        print(f"\n{result.name}:")
        print(f"  Mean: {result.mean_time * 1000:.3f} ms")
        print(f"  Std:  {result.std_time * 1000:.3f} ms")
        print(f"  Min:  {result.min_time * 1000:.3f} ms")
        print(f"  Max:  {result.max_time * 1000:.3f} ms")
