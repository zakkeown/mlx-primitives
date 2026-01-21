"""Attention benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, AttentionSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn
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

        # Core attention benchmarks (naive vs flash)
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

        # Extended attention variants
        results.extend(self.run_gqa_benchmarks())
        results.extend(self.run_mqa_benchmarks())
        results.extend(self.run_sliding_window_benchmarks())
        results.extend(self.run_alibi_benchmarks())
        results.extend(self.run_rope_benchmarks())

        # Additional attention variants
        results.extend(self.run_linear_attention_benchmarks())
        results.extend(self.run_sparse_attention_benchmarks())
        results.extend(self.run_chunked_attention_benchmarks())

        # Extended variants
        results.extend(self.run_quantized_kv_cache_benchmarks())
        results.extend(self.run_bigbird_benchmarks())
        results.extend(self.run_cosformer_benchmarks())

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

        def fn():
            return naive_attention(query, key, value)

        # Benchmark (includes warmup)
        name = f"naive_attn_b{batch_size}_s{seq_len}"
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
            "type": "baseline",
        }
        return result

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
            print(f"  Warning: flash_attention not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        # Create input tensors in the format flash_attention expects: (batch, seq, heads, dim)
        # Note: This differs from naive_attention which expects (batch, heads, seq, dim)
        query = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        key = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        value = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn():
            return flash_attention(query, key, value)

        # Benchmark (includes warmup)
        name = f"flash_attn_b{batch_size}_s{seq_len}"
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
        }
        return result

    def run_gqa_benchmarks(self) -> list[BenchmarkResult]:
        """Run Grouped Query Attention benchmarks."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            for batch_size in self.sizes.batch_sizes[:2]:
                result = self._benchmark_gqa(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=32,
                    num_kv_heads=8,
                    head_dim=64,
                )
                if result:
                    results.append(result)

        return results

    def run_mqa_benchmarks(self) -> list[BenchmarkResult]:
        """Run Multi-Query Attention benchmarks."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            for batch_size in self.sizes.batch_sizes[:2]:
                result = self._benchmark_mqa(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=32,
                    head_dim=64,
                )
                if result:
                    results.append(result)

        return results

    def run_sliding_window_benchmarks(self) -> list[BenchmarkResult]:
        """Run sliding window attention benchmarks."""
        results = []

        window_sizes = [128, 256, 512]
        for seq_len in self.sizes.seq_lengths[:3]:
            for window_size in window_sizes:
                if window_size < seq_len:
                    result = self._benchmark_sliding_window(
                        batch_size=2,
                        seq_len=seq_len,
                        num_heads=8,
                        head_dim=64,
                        window_size=window_size,
                    )
                    if result:
                        results.append(result)

        return results

    def run_alibi_benchmarks(self) -> list[BenchmarkResult]:
        """Run ALiBi attention benchmarks."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            result = self._benchmark_alibi(
                batch_size=2,
                seq_len=seq_len,
                num_heads=8,
                head_dim=64,
            )
            if result:
                results.append(result)

        return results

    def run_rope_benchmarks(self) -> list[BenchmarkResult]:
        """Run RoPE (Rotary Position Embedding) benchmarks."""
        results = []

        for seq_len in self.sizes.seq_lengths[:3]:
            # Standard RoPE
            result = self._benchmark_rope(
                batch_size=2,
                seq_len=seq_len,
                num_heads=8,
                head_dim=64,
            )
            if result:
                results.append(result)

        return results

    def _benchmark_gqa(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Grouped Query Attention."""
        try:
            from mlx_primitives.attention.grouped_query import grouped_query_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))

        def fn():
            return grouped_query_attention(q, k, v)

        name = f"gqa_b{batch_size}_s{seq_len}_h{num_heads}_kv{num_kv_heads}"
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
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "type": "optimized",
            "operation": "gqa",
        }
        return result

    def _benchmark_mqa(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Multi-Query Attention (single KV head)."""
        try:
            from mlx_primitives.attention.grouped_query import grouped_query_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, 1, head_dim))
        v = mx.random.normal((batch_size, seq_len, 1, head_dim))

        def fn():
            return grouped_query_attention(q, k, v)

        name = f"mqa_b{batch_size}_s{seq_len}_h{num_heads}"
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
            "num_kv_heads": 1,
            "head_dim": head_dim,
            "type": "optimized",
            "operation": "mqa",
        }
        return result

    def _benchmark_sliding_window(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark sliding window attention."""
        try:
            from mlx_primitives.attention.sliding_window import sliding_window_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn():
            return sliding_window_attention(q, k, v, window_size=window_size)

        name = f"sliding_window_b{batch_size}_s{seq_len}_w{window_size}"
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
            "window_size": window_size,
            "type": "optimized",
            "operation": "sliding_window",
        }
        return result

    def _benchmark_alibi(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark ALiBi attention."""
        try:
            from mlx_primitives.attention.alibi import alibi_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        def fn():
            return alibi_attention(q, k, v)

        name = f"alibi_b{batch_size}_s{seq_len}"
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
            "operation": "alibi",
        }
        return result

    def _benchmark_rope(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark RoPE application."""
        try:
            from mlx_primitives.attention.rope import apply_rope, precompute_freqs_cis
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        # Precompute cos/sin frequencies
        cos, sin, cos_doubled, sin_doubled = precompute_freqs_cis(head_dim, seq_len)

        def fn():
            return apply_rope(q, k, cos, sin, offset=0, cos_doubled=cos_doubled, sin_doubled=sin_doubled)

        name = f"rope_b{batch_size}_s{seq_len}"
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
            "operation": "rope",
        }
        return result

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

    def run_linear_attention_benchmarks(self) -> list[BenchmarkResult]:
        """Run linear attention (O(n)) benchmarks."""
        results = []

        configs = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (2, 2048, 8, 64),
            (4, 1024, 8, 64),
        ]

        for batch, seq, heads, dim in configs:
            # LinearAttention
            result = self._benchmark_linear_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

            # PerformerAttention
            result = self._benchmark_performer_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_linear_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark LinearAttention (O(n) complexity)."""
        try:
            from mlx_primitives.attention.linear import LinearAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        attn = LinearAttention(
            dims=num_heads * head_dim,
            num_heads=num_heads,
        )
        x = mx.random.normal((batch_size, seq_len, num_heads * head_dim))

        def fn():
            return attn(x)

        name = f"linear_attn_b{batch_size}_s{seq_len}"
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
            "operation": "linear_attention",
        }
        return result

    def _benchmark_performer_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark PerformerAttention (FAVOR+)."""
        try:
            from mlx_primitives.attention.linear import PerformerAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        attn = PerformerAttention(
            dims=num_heads * head_dim,
            num_heads=num_heads,
            num_features=head_dim,
        )
        x = mx.random.normal((batch_size, seq_len, num_heads * head_dim))

        def fn():
            return attn(x)

        name = f"performer_attn_b{batch_size}_s{seq_len}"
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
            "operation": "performer_attention",
        }
        return result

    def run_sparse_attention_benchmarks(self) -> list[BenchmarkResult]:
        """Run sparse attention benchmarks."""
        results = []

        configs = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (2, 2048, 8, 64),
        ]

        for batch, seq, heads, dim in configs:
            # BlockSparseAttention
            result = self._benchmark_block_sparse_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

            # LongformerAttention
            result = self._benchmark_longformer_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_block_sparse_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark BlockSparseAttention."""
        try:
            from mlx_primitives.attention.sparse import BlockSparseAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        block_size = 64
        dims = num_heads * head_dim
        attn = BlockSparseAttention(
            dims=dims,
            num_heads=num_heads,
            block_size=block_size,
        )
        x = mx.random.normal((batch_size, seq_len, dims))

        def fn():
            return attn(x)

        name = f"block_sparse_attn_b{batch_size}_s{seq_len}"
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
            "type": "optimized",
            "operation": "block_sparse_attention",
        }
        return result

    def _benchmark_longformer_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark LongformerAttention."""
        try:
            from mlx_primitives.attention.sparse import LongformerAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        window_size = 128
        dims = num_heads * head_dim
        attn = LongformerAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
        )
        x = mx.random.normal((batch_size, seq_len, dims))

        def fn():
            return attn(x)

        name = f"longformer_attn_b{batch_size}_s{seq_len}"
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
            "window_size": window_size,
            "type": "optimized",
            "operation": "longformer_attention",
        }
        return result

    def run_chunked_attention_benchmarks(self) -> list[BenchmarkResult]:
        """Run chunked cross-attention benchmarks."""
        results = []

        configs = [
            (2, 512, 2048, 8, 64),   # (batch, q_len, kv_len, heads, dim)
            (2, 256, 4096, 8, 64),
            (4, 512, 1024, 8, 64),
        ]

        for batch, q_len, kv_len, heads, dim in configs:
            result = self._benchmark_chunked_cross_attention(batch, q_len, kv_len, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_chunked_cross_attention(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark chunked cross-attention for long KV sequences."""
        try:
            from mlx_primitives.attention.chunked import chunked_cross_attention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        q = mx.random.normal((batch_size, q_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, kv_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, kv_len, num_heads, head_dim))

        def fn():
            return chunked_cross_attention(q, k, v, chunk_size=256)

        name = f"chunked_cross_attn_b{batch_size}_q{q_len}_kv{kv_len}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "q_len": q_len,
            "kv_len": kv_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "type": "optimized",
            "operation": "chunked_cross_attention",
        }
        return result

    def run_quantized_kv_cache_benchmarks(self) -> list[BenchmarkResult]:
        """Run quantized KV cache attention benchmarks."""
        results = []

        configs = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (4, 1024, 16, 64),
        ]

        for batch, seq, heads, dim in configs:
            result = self._benchmark_quantized_kv_cache_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_quantized_kv_cache_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark quantized KV cache attention (INT8 KV cache)."""
        try:
            from mlx_primitives.attention.quantized_kv_cache import QuantizedKVCacheAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        dims = num_heads * head_dim
        attn = QuantizedKVCacheAttention(
            dims=dims,
            num_heads=num_heads,
            max_seq_len=seq_len * 2,
            causal=True,
        )
        mx.eval(attn.parameters())

        x = mx.random.normal((batch_size, seq_len, dims))

        def fn():
            return attn(x)

        name = f"quantized_kv_cache_attn_b{batch_size}_s{seq_len}"
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
            "operation": "quantized_kv_cache_attention",
        }
        return result

    def run_bigbird_benchmarks(self) -> list[BenchmarkResult]:
        """Run BigBird attention benchmarks."""
        results = []

        configs = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (2, 2048, 8, 64),
        ]

        for batch, seq, heads, dim in configs:
            result = self._benchmark_bigbird_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_bigbird_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark BigBird attention (random + window + global)."""
        try:
            from mlx_primitives.attention.sparse import BigBirdAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        dims = num_heads * head_dim
        window_size = min(64, seq_len // 4)
        num_global_tokens = min(4, seq_len // 16)

        attn = BigBirdAttention(
            dims=dims,
            num_heads=num_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
        )
        mx.eval(attn.parameters())

        x = mx.random.normal((batch_size, seq_len, dims))

        def fn():
            return attn(x)

        name = f"bigbird_attn_b{batch_size}_s{seq_len}_w{window_size}"
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
            "window_size": window_size,
            "type": "optimized",
            "operation": "bigbird_attention",
        }
        return result

    def run_cosformer_benchmarks(self) -> list[BenchmarkResult]:
        """Run CosFormer attention benchmarks."""
        results = []

        configs = [
            (2, 512, 8, 64),
            (2, 1024, 8, 64),
            (4, 1024, 8, 64),
        ]

        for batch, seq, heads, dim in configs:
            result = self._benchmark_cosformer_attention(batch, seq, heads, dim)
            if result:
                results.append(result)

        return results

    def _benchmark_cosformer_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark CosFormer attention (cos-based reweighting)."""
        try:
            from mlx_primitives.attention.linear import CosFormerAttention
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        dims = num_heads * head_dim
        attn = CosFormerAttention(
            dims=dims,
            num_heads=num_heads,
        )
        mx.eval(attn.parameters())

        x = mx.random.normal((batch_size, seq_len, dims))

        def fn():
            return attn(x)

        name = f"cosformer_attn_b{batch_size}_s{seq_len}"
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
            "operation": "cosformer_attention",
        }
        return result
