"""KV Cache benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, CacheSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn


class CacheBenchmarks:
    """Benchmark suite for KV cache operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[CacheSizes] = None,
    ):
        """Initialize cache benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for cache benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or CacheSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all cache benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_kv_cache_benchmarks())
        results.extend(self.run_paged_attention_benchmarks())
        results.extend(self.run_eviction_benchmarks())
        return results

    def run_kv_cache_benchmarks(self) -> list[BenchmarkResult]:
        """Run KV cache update/retrieval benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for seq_len in self.sizes.seq_lengths[:4]:
                num_heads = 32
                head_dim = 128

                # KV Cache update
                result = self._benchmark_kv_cache_update(batch_size, seq_len, num_heads, head_dim)
                if result:
                    results.append(result)

                # KV Cache retrieval
                result = self._benchmark_kv_cache_get(batch_size, seq_len, num_heads, head_dim)
                if result:
                    results.append(result)

        return results

    def run_paged_attention_benchmarks(self) -> list[BenchmarkResult]:
        """Run paged attention benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for seq_len in self.sizes.seq_lengths[:4]:
                for block_size in self.sizes.block_sizes[:2]:
                    num_heads = 32
                    head_dim = 128

                    result = self._benchmark_paged_attention(
                        batch_size, seq_len, num_heads, head_dim, block_size
                    )
                    if result:
                        results.append(result)

        return results

    def run_eviction_benchmarks(self) -> list[BenchmarkResult]:
        """Run cache eviction policy benchmarks."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            num_heads = 32
            head_dim = 128

            # LRU eviction
            result = self._benchmark_lru_eviction(seq_len, num_heads, head_dim)
            if result:
                results.append(result)

            # Attention-based eviction
            result = self._benchmark_attention_eviction(seq_len, num_heads, head_dim)
            if result:
                results.append(result)

        return results

    def _benchmark_kv_cache_update(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark KV cache update operation."""
        try:
            from mlx_primitives.cache import KVCache, KVCacheConfig
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        config = KVCacheConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=1,
            max_memory_gb=4.0,  # 4GB cache
        )
        cache = KVCache(config)

        # Create sequence
        seq_id = cache.create_sequence()

        # K/V shape: (seq_len, num_heads, head_dim) - no batch dim for cache
        k = mx.random.normal((1, num_heads, head_dim))
        v = mx.random.normal((1, num_heads, head_dim))

        def fn():
            cache.update(seq_id, k, v, layer_idx=0)
            return cache

        name = f"kv_cache_update_b{batch_size}_s{seq_len}"
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
            "type": "cache_update",
            "layer": "KVCache",
        }
        return result

    def _benchmark_kv_cache_get(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark KV cache retrieval operation."""
        try:
            from mlx_primitives.cache import KVCache, KVCacheConfig
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        config = KVCacheConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=1,
            max_memory_gb=4.0,  # 4GB cache
        )
        cache = KVCache(config)

        # Create and populate sequence
        seq_id = cache.create_sequence()
        for _ in range(min(seq_len, 100)):  # Populate with some tokens
            # K/V shape: (seq_len, num_heads, head_dim) - no batch dim
            k = mx.random.normal((1, num_heads, head_dim))
            v = mx.random.normal((1, num_heads, head_dim))
            cache.update(seq_id, k, v, layer_idx=0)

        def fn():
            return cache.get_kv(seq_id, layer_idx=0)  # Single seq_id, not list

        name = f"kv_cache_get_b{batch_size}_s{seq_len}"
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
            "type": "cache_get",
            "layer": "KVCache",
        }
        return result

    def _benchmark_paged_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark paged attention operation."""
        try:
            from mlx_primitives.cache import paged_attention, create_block_table_from_lengths
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Calculate number of blocks needed
        num_blocks = (seq_len + block_size - 1) // block_size
        total_blocks = batch_size * num_blocks

        # Create query, key pool, value pool
        q = mx.random.normal((batch_size, 1, num_heads, head_dim))
        k_pool = mx.random.normal((total_blocks, block_size, num_heads, head_dim))
        v_pool = mx.random.normal((total_blocks, block_size, num_heads, head_dim))

        # Create block tables and lengths
        seq_lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)
        block_tables = create_block_table_from_lengths(seq_lengths, block_size)

        def fn():
            return paged_attention(q, k_pool, v_pool, block_tables, seq_lengths, block_size=block_size)

        name = f"paged_attn_b{batch_size}_s{seq_len}_bs{block_size}"
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
            "block_size": block_size,
            "type": "paged_attention",
            "layer": "paged_attention",
        }
        return result

    def _benchmark_lru_eviction(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark LRU cache eviction policy."""
        try:
            from mlx_primitives.cache import LRUEvictionPolicy
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        policy = LRUEvictionPolicy()

        # Simulate cache accesses - need to create sequences first
        seq_ids = list(range(min(50, seq_len // 10)))
        for seq_id in seq_ids:
            policy.on_create(seq_id)

        def fn():
            for seq_id in seq_ids:
                policy.on_access(seq_id)
            # Get eviction candidates
            return policy.select_for_eviction(seq_ids, 5)

        name = f"lru_eviction_s{seq_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "seq_len": seq_len,
            "num_sequences": len(seq_ids),
            "type": "eviction",
            "policy": "LRU",
        }
        return result

    def _benchmark_attention_eviction(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark attention-based cache eviction policy."""
        try:
            from mlx_primitives.cache import AttentionScoreEvictionPolicy
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        policy = AttentionScoreEvictionPolicy(decay_factor=0.99)

        # Simulate cache accesses with attention scores - need to create sequences first
        seq_ids = list(range(min(50, seq_len // 10)))
        for seq_id in seq_ids:
            policy.on_create(seq_id)
        attention_scores = mx.random.uniform(shape=(len(seq_ids),))

        def fn():
            for seq_id, score in zip(seq_ids, attention_scores.tolist()):
                policy.update_attention_score(seq_id, score)
            return policy.select_for_eviction(seq_ids, 5)

        name = f"attention_eviction_s{seq_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "seq_len": seq_len,
            "num_sequences": len(seq_ids),
            "type": "eviction",
            "policy": "AttentionScore",
        }
        return result
