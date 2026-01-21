"""Cache operations parity benchmarks."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

import mlx.core as mx

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig, DEFAULT_CONFIG
from benchmarks.parity.runner import BenchmarkResult

from mlx_primitives.cache.eviction import LRUEvictionPolicy, FIFOEvictionPolicy
from mlx_primitives.cache.speculative import speculative_verify
from mlx_primitives.cache.paged_attention import paged_attention, create_block_table_from_lengths
from mlx_primitives.cache.block_allocator import BlockConfig, BlockAllocator


class CacheParityBenchmarks:
    """Multi-framework cache benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._size_config = ParitySizeConfig()

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all cache benchmarks.

        Returns:
            Dictionary mapping framework name to list of benchmark results.
        """
        results: Dict[str, List[BenchmarkResult]] = {
            "mlx": [],
        }

        sizes = ["tiny", "small", "medium", "large"]

        # Eviction policy benchmarks
        for size in sizes:
            lru_result = self.benchmark_eviction_lru(size)
            for framework, result in lru_result.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

            fifo_result = self.benchmark_eviction_fifo(size)
            for framework, result in fifo_result.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

            spec_result = self.benchmark_speculative_verification(size)
            for framework, result in spec_result.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

            paged_result = self.benchmark_paged_attention(size)
            for framework, result in paged_result.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

            block_result = self.benchmark_block_allocation(size)
            for framework, result in block_result.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

        return results

    def benchmark_paged_attention(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark paged attention across frameworks.

        Args:
            size: Size configuration name (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        config = self._size_config.get_config("cache", size)
        batch, seq, heads, head_dim, block_size = config

        results: Dict[str, BenchmarkResult] = {}

        # MLX benchmark
        results["mlx"] = self._benchmark_paged_attention_mlx(
            batch, seq, heads, head_dim, block_size, size
        )

        return results

    def _benchmark_paged_attention_mlx(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        size: str,
    ) -> BenchmarkResult:
        """Run paged attention benchmark on MLX."""
        warmup = self.config.warmup_iterations
        iterations = self.config.benchmark_iterations

        np.random.seed(42)

        # Create query tensor for decode mode (single token)
        # Shape: (batch, 1, num_heads, head_dim)
        q = mx.random.normal((batch_size, 1, num_heads, head_dim), dtype=mx.float16)

        # Calculate number of blocks needed for the KV cache
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = batch_size * blocks_per_seq

        # Create block pools: (num_blocks, block_size, num_heads, head_dim)
        k_pool = mx.random.normal((total_blocks, block_size, num_heads, head_dim), dtype=mx.float16)
        v_pool = mx.random.normal((total_blocks, block_size, num_heads, head_dim), dtype=mx.float16)

        # Create block tables using the helper function
        sequence_lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)
        block_tables = create_block_table_from_lengths(sequence_lengths, block_size, blocks_per_seq)

        # Context lengths: number of cached tokens per sequence
        context_lens = mx.full((batch_size,), seq_len, dtype=mx.int32)

        # Scale factor for attention
        scale = 1.0 / (head_dim ** 0.5)

        # Warmup
        for _ in range(warmup):
            output = paged_attention(
                q, k_pool, v_pool, block_tables, context_lens,
                scale=scale, block_size=block_size, causal=True
            )
            mx.eval(output)

        # Timed iterations
        times = []
        for _ in range(iterations):
            mx.eval()  # Sync before timing

            start = time.perf_counter()
            output = paged_attention(
                q, k_pool, v_pool, block_tables, context_lens,
                scale=scale, block_size=block_size, causal=True
            )
            mx.eval(output)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        tokens_per_call = batch_size * seq_len
        return BenchmarkResult(
            name=f"paged_attention_{size}",
            framework="mlx",
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            iterations=len(times),
            throughput=tokens_per_call / float(np.mean(times_arr)),
        )

    def benchmark_block_allocation(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark block allocation across frameworks.

        Args:
            size: Size configuration name (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        config = self._size_config.get_config("cache", size)
        batch, seq, heads, head_dim, block_size = config

        # Calculate number of blocks and operations based on size
        num_blocks = batch * 64
        num_operations = batch * 32

        results: Dict[str, BenchmarkResult] = {}

        # MLX benchmark
        results["mlx"] = self._benchmark_block_allocation_mlx(
            num_blocks, num_operations, heads, head_dim, block_size, size
        )

        return results

    def _benchmark_block_allocation_mlx(
        self,
        num_blocks: int,
        num_operations: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        size: str,
    ) -> BenchmarkResult:
        """Run block allocation benchmark on MLX."""
        warmup = self.config.warmup_iterations
        iterations = self.config.benchmark_iterations

        np.random.seed(42)

        # Create block configuration
        block_config = BlockConfig(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=mx.float16,
        )

        # Pre-generate random data for set_block_data operations
        k_data = mx.random.normal((block_size, num_heads, head_dim), dtype=mx.float16)
        v_data = mx.random.normal((block_size, num_heads, head_dim), dtype=mx.float16)

        # Warmup
        for _ in range(warmup):
            allocator = BlockAllocator(block_config, num_blocks=num_blocks, enable_cow=True)

            allocated_blocks = []
            for _ in range(num_operations):
                blocks = allocator.allocate(count=min(4, allocator.num_free_blocks))
                allocated_blocks.extend(blocks)

                if blocks:
                    allocator.set_block_data(blocks[0], k_data, v_data)
                    allocator.get_block_data(blocks[0])

            allocator.free(allocated_blocks)

        # Timed iterations
        times = []
        for _ in range(iterations):
            allocator = BlockAllocator(block_config, num_blocks=num_blocks, enable_cow=True)

            start = time.perf_counter()

            allocated_blocks = []
            for _ in range(num_operations):
                blocks = allocator.allocate(count=min(4, allocator.num_free_blocks))
                allocated_blocks.extend(blocks)

                if blocks:
                    allocator.set_block_data(blocks[0], k_data, v_data)
                    allocator.get_block_data(blocks[0])

            allocator.free(allocated_blocks)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        return BenchmarkResult(
            name=f"block_allocation_{size}",
            framework="mlx",
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            iterations=len(times),
            throughput=num_operations / float(np.mean(times_arr)),
        )

    def benchmark_eviction_lru(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark LRU eviction policy.

        Args:
            size: Size configuration name (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        config = self._size_config.get_config("cache", size)
        batch, seq, heads, head_dim, block_size = config
        num_sequences = batch * 16  # Scale sequences with batch
        num_accesses = num_sequences * 10

        # Generate access pattern
        np.random.seed(42)
        access_pattern = list(np.random.randint(0, num_sequences, (num_accesses,)))

        results: Dict[str, BenchmarkResult] = {}

        # MLX benchmark
        results["mlx"] = self._benchmark_lru_mlx(
            num_sequences, access_pattern, size
        )

        return results

    def _benchmark_lru_mlx(
        self,
        num_sequences: int,
        access_pattern: List[int],
        size: str,
    ) -> BenchmarkResult:
        """Run LRU eviction benchmark on MLX."""
        warmup = self.config.warmup_iterations
        iterations = self.config.benchmark_iterations

        # Warmup
        for _ in range(warmup):
            policy = LRUEvictionPolicy()
            for i in range(num_sequences):
                policy.on_create(i)
            for seq_id in access_pattern:
                policy.on_access(seq_id)
            policy.select_for_eviction(list(range(num_sequences)), num_sequences // 4)

        # Timed iterations
        times = []
        for _ in range(iterations):
            policy = LRUEvictionPolicy()

            start = time.perf_counter()
            for i in range(num_sequences):
                policy.on_create(i)
            for seq_id in access_pattern:
                policy.on_access(seq_id)
            evicted = policy.select_for_eviction(
                list(range(num_sequences)), num_sequences // 4
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        return BenchmarkResult(
            name=f"lru_eviction_{size}",
            framework="mlx",
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            iterations=len(times),
            throughput=len(access_pattern) / float(np.mean(times_arr)),
        )

    def benchmark_eviction_fifo(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark FIFO eviction policy.

        Args:
            size: Size configuration name (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        config = self._size_config.get_config("cache", size)
        batch, seq, heads, head_dim, block_size = config
        num_sequences = batch * 16
        num_accesses = num_sequences * 10

        np.random.seed(42)
        access_pattern = list(np.random.randint(0, num_sequences, (num_accesses,)))

        results: Dict[str, BenchmarkResult] = {}

        # MLX benchmark
        results["mlx"] = self._benchmark_fifo_mlx(
            num_sequences, access_pattern, size
        )

        return results

    def _benchmark_fifo_mlx(
        self,
        num_sequences: int,
        access_pattern: List[int],
        size: str,
    ) -> BenchmarkResult:
        """Run FIFO eviction benchmark on MLX."""
        warmup = self.config.warmup_iterations
        iterations = self.config.benchmark_iterations

        # Warmup
        for _ in range(warmup):
            policy = FIFOEvictionPolicy()
            for i in range(num_sequences):
                policy.on_create(i)
            for seq_id in access_pattern:
                policy.on_access(seq_id)
            policy.select_for_eviction(list(range(num_sequences)), num_sequences // 4)

        # Timed iterations
        times = []
        for _ in range(iterations):
            policy = FIFOEvictionPolicy()

            start = time.perf_counter()
            for i in range(num_sequences):
                policy.on_create(i)
            for seq_id in access_pattern:
                policy.on_access(seq_id)
            evicted = policy.select_for_eviction(
                list(range(num_sequences)), num_sequences // 4
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        return BenchmarkResult(
            name=f"fifo_eviction_{size}",
            framework="mlx",
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            iterations=len(times),
            throughput=len(access_pattern) / float(np.mean(times_arr)),
        )

    def benchmark_speculative_verification(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark speculative verification.

        Args:
            size: Size configuration name (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        config = self._size_config.get_config("cache", size)
        batch, seq, heads, head_dim, block_size = config
        vocab_size = 32000
        draft_length = min(8, seq // 8)

        results: Dict[str, BenchmarkResult] = {}

        # MLX benchmark
        results["mlx"] = self._benchmark_speculative_mlx(
            batch, draft_length, vocab_size, size
        )

        return results

    def _benchmark_speculative_mlx(
        self,
        batch_size: int,
        draft_length: int,
        vocab_size: int,
        size: str,
    ) -> BenchmarkResult:
        """Run speculative verification benchmark on MLX."""
        warmup = self.config.warmup_iterations
        iterations = self.config.benchmark_iterations

        np.random.seed(42)

        # Generate test data
        draft_tokens = list(np.random.randint(0, vocab_size, (draft_length,)))
        draft_log_probs_np = np.random.randn(draft_length).astype(np.float32) * 0.5
        target_log_probs_np = np.random.randn(draft_length + 1, vocab_size).astype(np.float32)
        target_log_probs_np = target_log_probs_np - np.log(
            np.exp(target_log_probs_np).sum(axis=-1, keepdims=True)
        )

        draft_log_probs = mx.array(draft_log_probs_np)
        target_log_probs = mx.array(target_log_probs_np)

        # Warmup
        for _ in range(warmup):
            mx.random.seed(42)
            speculative_verify(draft_tokens, draft_log_probs, target_log_probs)
            mx.eval()

        # Timed iterations
        times = []
        for _ in range(iterations):
            mx.eval()  # Sync before timing
            mx.random.seed(42)

            start = time.perf_counter()
            accepted, correction = speculative_verify(
                draft_tokens, draft_log_probs, target_log_probs
            )
            mx.eval()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        return BenchmarkResult(
            name=f"speculative_verify_{size}",
            framework="mlx",
            mean_time=float(np.mean(times_arr)),
            std_time=float(np.std(times_arr)),
            min_time=float(np.min(times_arr)),
            max_time=float(np.max(times_arr)),
            iterations=len(times),
            throughput=draft_length / float(np.mean(times_arr)),
        )

    def run_cache_size_scaling(self, operation: str) -> Dict[str, List[BenchmarkResult]]:
        """Run scaling tests for cache sizes.

        Args:
            operation: Operation name (lru, fifo, speculative, paged_attention, block_allocation).

        Returns:
            Dictionary mapping framework to list of results across sizes.
        """
        results: Dict[str, List[BenchmarkResult]] = {"mlx": []}

        sizes = ["tiny", "small", "medium", "large"]

        for size in sizes:
            if operation == "lru":
                size_results = self.benchmark_eviction_lru(size)
            elif operation == "fifo":
                size_results = self.benchmark_eviction_fifo(size)
            elif operation == "speculative":
                size_results = self.benchmark_speculative_verification(size)
            elif operation == "paged_attention":
                size_results = self.benchmark_paged_attention(size)
            elif operation == "block_allocation":
                size_results = self.benchmark_block_allocation(size)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            for framework, result in size_results.items():
                if result is not None:
                    if framework not in results:
                        results[framework] = []
                    results[framework].append(result)

        return results
