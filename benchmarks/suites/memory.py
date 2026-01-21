"""Memory benchmarks for MLX Primitives.

This module benchmarks memory usage patterns for various operations,
helping identify memory-intensive operations and validate memory efficiency.
"""

from typing import Optional, List

import numpy as np
import mlx.core as mx

from benchmarks.config import BenchmarkConfig, AttentionSizes, CacheSizes
from benchmarks.memory import (
    MemoryProfiler,
    MemoryProfile,
    estimate_tensor_memory_mb,
    estimate_attention_memory_mb,
)
from benchmarks.utils import BenchmarkResult


class MemoryBenchmarks:
    """Memory usage benchmarks for MLX Primitives."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        attention_sizes: Optional[AttentionSizes] = None,
        cache_sizes: Optional[CacheSizes] = None,
    ):
        """Initialize memory benchmarks.

        Args:
            config: Benchmark configuration.
            attention_sizes: Attention size configurations.
            cache_sizes: Cache size configurations.
        """
        self.config = config or BenchmarkConfig()
        self.attention_sizes = attention_sizes or AttentionSizes()
        self.cache_sizes = cache_sizes or CacheSizes()
        self.profiler = MemoryProfiler()

    def run_all(self) -> List[BenchmarkResult]:
        """Run all memory benchmarks.

        Returns:
            List of benchmark results with memory metadata.
        """
        results = []
        results.extend(self.run_attention_memory())
        # KV cache and large sequence benchmarks require API updates
        # to match current KVCache implementation
        try:
            results.extend(self.run_kv_cache_memory())
        except Exception as e:
            print(f"Skipping KV cache memory benchmarks: {e}")
        try:
            results.extend(self.run_large_sequence_memory())
        except Exception as e:
            print(f"Skipping large sequence memory benchmarks: {e}")
        return results

    def run_attention_memory(self) -> List[BenchmarkResult]:
        """Benchmark attention memory usage across different sizes.

        Returns:
            List of benchmark results with memory profiling data.
        """
        from mlx_primitives.attention.flash import flash_attention

        results = []

        for seq_len in self.attention_sizes.seq_lengths[:4]:
            for batch_size in self.attention_sizes.batch_sizes[:2]:
                result = self._profile_attention_memory(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=self.attention_sizes.num_heads,
                    head_dim=self.attention_sizes.head_dim,
                )
                if result:
                    results.append(result)

        return results

    def _profile_attention_memory(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Profile memory for a single attention configuration.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Head dimension.

        Returns:
            BenchmarkResult with memory metadata, or None on failure.
        """
        try:
            from mlx_primitives.attention.flash import flash_attention

            np.random.seed(self.config.seed)
            q = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            k = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            v = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            mx.eval(q, k, v)

            def attention_fn():
                out = flash_attention(q, k, v, causal=False)
                mx.eval(out)
                return out

            profile = self.profiler.profile_function(
                attention_fn,
                name=f"attention_b{batch_size}_s{seq_len}",
                warmup=self.config.warmup_iterations,
            )

            # Get theoretical memory estimate
            theoretical_mb = estimate_attention_memory_mb(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )

            return BenchmarkResult(
                name=f"attention_memory_b{batch_size}_s{seq_len}",
                mean_time=0.0,  # Memory benchmark, not timing
                std_time=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=1,
                metadata={
                    "type": "memory",
                    "operation": "attention",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "peak_memory_mb": profile.peak_allocated_mb,
                    "final_memory_mb": profile.final_allocated_mb,
                    "memory_delta_mb": profile.memory_delta_mb,
                    "theoretical_memory_mb": theoretical_mb,
                },
            )

        except Exception as e:
            print(f"Memory profiling failed for attention b{batch_size}_s{seq_len}: {e}")
            return None

    def run_kv_cache_memory(self) -> List[BenchmarkResult]:
        """Benchmark KV cache memory usage.

        Returns:
            List of benchmark results with KV cache memory data.
        """
        results = []

        for seq_len in self.cache_sizes.seq_lengths[:4]:
            result = self._profile_kv_cache_memory(
                batch_size=self.cache_sizes.batch_sizes[0],
                seq_len=seq_len,
                num_heads=self.cache_sizes.num_heads[0],
                head_dim=self.cache_sizes.head_dims[0],
            )
            if result:
                results.append(result)

        return results

    def _profile_kv_cache_memory(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Profile memory for KV cache operations.

        Args:
            batch_size: Batch size.
            seq_len: Maximum sequence length.
            num_heads: Number of attention heads.
            head_dim: Head dimension.

        Returns:
            BenchmarkResult with memory metadata, or None on failure.
        """
        try:
            from mlx_primitives.cache.kv_cache import KVCache

            def cache_fn():
                cache = KVCache(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    max_seq_len=seq_len,
                )

                # Simulate incremental updates
                for i in range(0, seq_len, 32):
                    chunk_len = min(32, seq_len - i)
                    new_k = mx.array(np.random.randn(batch_size, num_heads, chunk_len, head_dim).astype(np.float32))
                    new_v = mx.array(np.random.randn(batch_size, num_heads, chunk_len, head_dim).astype(np.float32))
                    cache.update(new_k, new_v)
                    mx.eval(cache.k_cache, cache.v_cache)

                return cache

            profile = self.profiler.profile_function(
                cache_fn,
                name=f"kv_cache_s{seq_len}",
                warmup=1,
            )

            # Theoretical memory: 2 caches (K and V)
            theoretical_mb = 2 * estimate_tensor_memory_mb(
                shape=(batch_size, num_heads, seq_len, head_dim),
                dtype="float32",
            )

            return BenchmarkResult(
                name=f"kv_cache_memory_s{seq_len}",
                mean_time=0.0,
                std_time=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=1,
                metadata={
                    "type": "memory",
                    "operation": "kv_cache",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "peak_memory_mb": profile.peak_allocated_mb,
                    "final_memory_mb": profile.final_allocated_mb,
                    "memory_delta_mb": profile.memory_delta_mb,
                    "theoretical_memory_mb": theoretical_mb,
                },
            )

        except Exception as e:
            print(f"KV cache memory profiling failed for s{seq_len}: {e}")
            return None

    def run_large_sequence_memory(self) -> List[BenchmarkResult]:
        """Benchmark memory for large sequence scenarios.

        Tests memory behavior at the edge of typical deployment scenarios.

        Returns:
            List of benchmark results for large sequences.
        """
        results = []

        # Test progressively larger sequences
        large_seq_lengths = [2048, 4096, 8192]

        for seq_len in large_seq_lengths:
            result = self._profile_large_sequence_memory(
                batch_size=1,
                seq_len=seq_len,
                num_heads=32,
                head_dim=128,
            )
            if result:
                results.append(result)

        return results

    def _profile_large_sequence_memory(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Profile memory for large sequence attention.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Head dimension.

        Returns:
            BenchmarkResult with memory metadata, or None on failure.
        """
        try:
            from mlx_primitives.attention.flash import flash_attention

            # Calculate expected memory before allocating
            theoretical_mb = estimate_attention_memory_mb(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )

            # Skip if theoretical memory is too large
            if theoretical_mb > 8000:  # 8GB limit
                print(f"Skipping large sequence s{seq_len}: theoretical memory {theoretical_mb:.0f}MB exceeds limit")
                return None

            np.random.seed(self.config.seed)
            q = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            k = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            v = mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32))
            mx.eval(q, k, v)

            def attention_fn():
                out = flash_attention(q, k, v, causal=True)
                mx.eval(out)
                return out

            profile = self.profiler.profile_function(
                attention_fn,
                name=f"large_attention_s{seq_len}",
                warmup=1,
            )

            return BenchmarkResult(
                name=f"large_seq_memory_s{seq_len}",
                mean_time=0.0,
                std_time=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=1,
                metadata={
                    "type": "memory",
                    "operation": "large_sequence_attention",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "peak_memory_mb": profile.peak_allocated_mb,
                    "final_memory_mb": profile.final_allocated_mb,
                    "memory_delta_mb": profile.memory_delta_mb,
                    "theoretical_memory_mb": theoretical_mb,
                },
            )

        except Exception as e:
            print(f"Large sequence memory profiling failed for s{seq_len}: {e}")
            return None

    def run_memory_scaling_test(
        self,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        steps: int = 4,
    ) -> List[BenchmarkResult]:
        """Run memory scaling test across sequence lengths.

        Tests how memory scales as sequence length increases.

        Args:
            base_seq_len: Starting sequence length.
            max_seq_len: Maximum sequence length to test.
            steps: Number of steps between base and max.

        Returns:
            List of benchmark results showing memory scaling.
        """
        from mlx_primitives.attention.flash import flash_attention

        results = []
        seq_lengths = np.linspace(base_seq_len, max_seq_len, steps).astype(int)

        batch_size = 1
        num_heads = 8
        head_dim = 64

        for seq_len in seq_lengths:
            seq_len = int(seq_len)

            theoretical_mb = estimate_attention_memory_mb(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )

            result = BenchmarkResult(
                name=f"memory_scaling_s{seq_len}",
                mean_time=0.0,
                std_time=0.0,
                min_time=0.0,
                max_time=0.0,
                iterations=1,
                metadata={
                    "type": "memory_scaling",
                    "operation": "attention",
                    "seq_len": seq_len,
                    "theoretical_memory_mb": theoretical_mb,
                    "scaling_factor": (seq_len / base_seq_len) ** 2,  # O(n^2) scaling
                },
            )
            results.append(result)

        return results
