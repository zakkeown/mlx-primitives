"""Attention parity benchmarks."""

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig
from benchmarks.parity.runner import BenchmarkResult

# Check for available frameworks
try:
    import mlx.core as mx
    from mlx_primitives.attention.flash import flash_attention
    from mlx_primitives.attention.grouped_query import gqa_attention
    from mlx_primitives.attention.rope import apply_rope, precompute_freqs_cis
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torch.nn.functional as F
    HAS_PYTORCH = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class AttentionParityBenchmarks:
    """Multi-framework attention benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.size_config = ParitySizeConfig()

    def _benchmark_mlx(
        self, fn, name: str, warmup: int = 5, iterations: int = 30
    ) -> BenchmarkResult:
        """Benchmark an MLX function."""
        warmup = warmup or self.config.warmup_iterations
        iterations = iterations or self.config.benchmark_iterations

        # Warmup
        for _ in range(warmup):
            result = fn()
            mx.eval(result)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            mx.eval(result)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult(
            name=name,
            framework="mlx",
            mean_time=statistics.mean(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            iterations=iterations,
        )

    def _benchmark_pytorch(
        self, fn, name: str, warmup: int = 5, iterations: int = 30
    ) -> BenchmarkResult:
        """Benchmark a PyTorch function."""
        warmup = warmup or self.config.warmup_iterations
        iterations = iterations or self.config.benchmark_iterations

        # Warmup
        for _ in range(warmup):
            result = fn()
            if hasattr(result, 'cpu'):
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, 'cpu'):
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult(
            name=name,
            framework="pytorch_mps",
            mean_time=statistics.mean(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            iterations=iterations,
        )

    def _get_attention_config(self, size: str) -> Tuple[int, int, int, int]:
        """Get attention configuration for a size."""
        config = self.size_config.get_config("attention", size)
        if config is None:
            config = (2, 256, 8, 64)  # Default small config
        return config

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all attention parity benchmarks.

        Returns:
            Dictionary mapping framework name to list of benchmark results.
        """
        results: Dict[str, List[BenchmarkResult]] = {
            "mlx": [],
            "pytorch_mps": [],
        }

        sizes = ["small", "medium"]

        for size in sizes:
            # Flash attention forward
            try:
                flash_results = self.benchmark_flash_attention(size)
                for framework, result in flash_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Flash attention benchmark failed for {size}: {e}")

            # Flash attention backward
            try:
                flash_bwd_results = self.benchmark_flash_attention_backward(size)
                for framework, result in flash_bwd_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Flash attention backward benchmark failed for {size}: {e}")

            # GQA
            try:
                gqa_results = self.benchmark_gqa(size)
                for framework, result in gqa_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"GQA benchmark failed for {size}: {e}")

            # MQA
            try:
                mqa_results = self.benchmark_mqa(size)
                for framework, result in mqa_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"MQA benchmark failed for {size}: {e}")

            # Sliding window
            try:
                sw_results = self.benchmark_sliding_window(size)
                for framework, result in sw_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Sliding window benchmark failed for {size}: {e}")

            # Chunked cross attention
            try:
                chunked_results = self.benchmark_chunked_cross(size)
                for framework, result in chunked_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Chunked cross benchmark failed for {size}: {e}")

            # ALiBi
            try:
                alibi_results = self.benchmark_alibi(size)
                for framework, result in alibi_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"ALiBi benchmark failed for {size}: {e}")

            # RoPE
            try:
                rope_results = self.benchmark_rope_variants(size)
                for framework, result in rope_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"RoPE benchmark failed for {size}: {e}")

            # Linear attention
            try:
                linear_results = self.benchmark_linear(size)
                for framework, result in linear_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Linear attention benchmark failed for {size}: {e}")

            # Sparse attention
            try:
                sparse_results = self.benchmark_sparse(size)
                for framework, result in sparse_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Sparse attention benchmark failed for {size}: {e}")

            # Quantized KV cache
            try:
                quant_results = self.benchmark_quantized_kv_cache(size)
                for framework, result in quant_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Quantized KV cache benchmark failed for {size}: {e}")

        return results

    def benchmark_flash_attention(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark flash attention across frameworks.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Dictionary mapping framework to benchmark result.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        results = {}

        # MLX benchmark
        if HAS_MLX:
            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            def mlx_fn():
                return flash_attention(q_mlx, k_mlx, v_mlx, causal=True)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"flash_attention_{size}")

        # PyTorch MPS benchmark
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            k_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            v_torch = torch.randn(batch, heads, seq, head_dim, device=device)

            def pytorch_fn():
                return F.scaled_dot_product_attention(
                    q_torch, k_torch, v_torch, is_causal=True
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"flash_attention_{size}")

        return results

    def benchmark_sliding_window(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark sliding window attention across frameworks.

        Sliding window attention limits attention to a fixed window of tokens,
        reducing complexity from O(n^2) to O(n*w) where w is window size.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        window_size = min(128, seq // 2) if seq >= 256 else seq // 2
        if window_size < 1:
            return {}

        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.sliding_window import sliding_window_attention
            except ImportError:
                return results

            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            def mlx_fn():
                return sliding_window_attention(q_mlx, k_mlx, v_mlx, window_size=window_size, causal=True)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"sliding_window_{size}_w{window_size}")

        # PyTorch MPS doesn't have native sliding window SDPA, so we use a mask-based approach
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            k_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            v_torch = torch.randn(batch, heads, seq, head_dim, device=device)

            # Create sliding window causal mask
            positions = torch.arange(seq, device=device)
            distance = positions[:, None] - positions[None, :]
            mask = (distance >= 0) & (distance <= window_size)
            attn_mask = torch.where(mask, 0.0, float('-inf'))

            def pytorch_fn():
                return F.scaled_dot_product_attention(
                    q_torch, k_torch, v_torch, attn_mask=attn_mask
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"sliding_window_{size}_w{window_size}")

        return results

    def benchmark_chunked_cross(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark chunked cross-attention across frameworks.

        Chunked cross-attention processes long KV sequences in chunks,
        enabling memory-efficient cross-attention for long contexts.
        """
        batch, q_seq, heads, head_dim = self._get_attention_config(size)
        kv_seq = q_seq * 2  # KV sequence is longer
        chunk_size = min(128, kv_seq // 4)
        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.chunked import chunked_cross_attention
            except ImportError:
                return results

            np.random.seed(42)
            q_np = np.random.randn(batch, q_seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, kv_seq, heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, kv_seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            def mlx_fn():
                return chunked_cross_attention(q_mlx, k_mlx, v_mlx, chunk_size=chunk_size)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"chunked_cross_{size}_c{chunk_size}")

        # PyTorch MPS benchmark (standard cross-attention for comparison)
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, q_seq, head_dim, device=device)
            k_torch = torch.randn(batch, heads, kv_seq, head_dim, device=device)
            v_torch = torch.randn(batch, heads, kv_seq, head_dim, device=device)

            def pytorch_fn():
                return F.scaled_dot_product_attention(
                    q_torch, k_torch, v_torch, is_causal=False
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"chunked_cross_{size}_c{chunk_size}")

        return results

    def benchmark_gqa(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark grouped query attention across frameworks.

        GQA uses fewer KV heads than query heads (e.g., 8 Q heads, 2 KV heads).
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        num_kv_heads = max(1, heads // 4)  # Use 1/4 as many KV heads
        results = {}

        # MLX benchmark
        if HAS_MLX:
            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            num_kv_groups = heads // num_kv_heads

            def mlx_fn():
                return gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups=num_kv_groups, causal=True)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"gqa_{size}")

        # PyTorch MPS benchmark (using repeat_kv pattern)
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            k_torch = torch.randn(batch, num_kv_heads, seq, head_dim, device=device)
            v_torch = torch.randn(batch, num_kv_heads, seq, head_dim, device=device)

            # Repeat KV heads to match Q heads
            num_rep = heads // num_kv_heads

            def pytorch_fn():
                # Expand KV heads
                k_exp = k_torch.repeat_interleave(num_rep, dim=1)
                v_exp = v_torch.repeat_interleave(num_rep, dim=1)
                return F.scaled_dot_product_attention(
                    q_torch, k_exp, v_exp, is_causal=True
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"gqa_{size}")

        return results

    def benchmark_mqa(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark multi-query attention (single KV head).

        MQA is an extreme form of GQA where all query heads share a single
        K and V head, providing maximum KV cache memory savings.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        num_kv_heads = 1  # MQA uses single KV head
        results = {}

        # MLX benchmark
        if HAS_MLX:
            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, num_kv_heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            num_kv_groups = heads

            def mlx_fn():
                return gqa_attention(q_mlx, k_mlx, v_mlx, num_kv_groups=num_kv_groups, causal=True)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"mqa_{size}")

        # PyTorch MPS benchmark
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            k_torch = torch.randn(batch, num_kv_heads, seq, head_dim, device=device)
            v_torch = torch.randn(batch, num_kv_heads, seq, head_dim, device=device)

            def pytorch_fn():
                # Broadcast single KV head to all Q heads
                k_exp = k_torch.expand(batch, heads, seq, head_dim)
                v_exp = v_torch.expand(batch, heads, seq, head_dim)
                return F.scaled_dot_product_attention(
                    q_torch, k_exp, v_exp, is_causal=True
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"mqa_{size}")

        return results

    def benchmark_sparse(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark sparse attention (block sparse pattern).

        Block sparse attention attends only to specific blocks, reducing
        computational complexity while maintaining reasonable quality.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        block_size = min(64, seq // 4) if seq >= 256 else 32
        dims = heads * head_dim
        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.sparse import BlockSparseAttention
            except ImportError:
                return results

            np.random.seed(42)
            x_np = np.random.randn(batch, seq, dims).astype(np.float32)

            attn = BlockSparseAttention(
                dims=dims,
                num_heads=heads,
                block_size=block_size,
            )
            mx.eval(attn.parameters())

            x_mlx = mx.array(x_np)
            mx.eval(x_mlx)

            def mlx_fn():
                return attn(x_mlx)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"sparse_{size}_b{block_size}")

        # PyTorch MPS doesn't have native block sparse attention
        # We benchmark standard attention as a comparison baseline
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            x_torch = torch.randn(batch, seq, dims, device=device)
            # Simple self-attention for comparison
            W_qkv = torch.randn(dims, 3 * dims, device=device) * 0.02

            def pytorch_fn():
                qkv = x_torch @ W_qkv
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
                v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"sparse_{size}_baseline")

        return results

    def benchmark_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark linear attention (O(n) complexity).

        Linear attention uses kernel feature maps to achieve O(n) complexity
        instead of O(n^2), enabling efficient processing of long sequences.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        dims = heads * head_dim
        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.linear import LinearAttention
            except ImportError:
                return results

            np.random.seed(42)
            x_np = np.random.randn(batch, seq, dims).astype(np.float32)

            attn = LinearAttention(dims=dims, num_heads=heads)
            mx.eval(attn.parameters())

            x_mlx = mx.array(x_np)
            mx.eval(x_mlx)

            def mlx_fn():
                return attn(x_mlx)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"linear_{size}")

        # PyTorch MPS benchmark - standard attention as comparison
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            x_torch = torch.randn(batch, seq, dims, device=device)
            W_qkv = torch.randn(dims, 3 * dims, device=device) * 0.02

            def pytorch_fn():
                qkv = x_torch @ W_qkv
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
                v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"linear_{size}_baseline")

        return results

    def benchmark_alibi(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark ALiBi (Attention with Linear Biases).

        ALiBi adds a linear position-dependent bias to attention scores,
        providing position information without learned embeddings.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.alibi import alibi_bias
            except ImportError:
                return results

            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            def mlx_fn():
                # Manual attention with ALiBi bias
                q_t = mx.transpose(q_mlx, (0, 2, 1, 3))  # BSHD -> BHSD
                k_t = mx.transpose(k_mlx, (0, 2, 1, 3))
                v_t = mx.transpose(v_mlx, (0, 2, 1, 3))
                scale = 1.0 / math.sqrt(head_dim)
                scores = (q_t @ mx.transpose(k_t, (0, 1, 3, 2))) * scale
                alibi = alibi_bias(seq, seq, heads)
                scores = scores + alibi
                weights = mx.softmax(scores, axis=-1)
                out = weights @ v_t
                return mx.transpose(out, (0, 2, 1, 3))

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"alibi_{size}")

        # PyTorch MPS benchmark
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            k_torch = torch.randn(batch, heads, seq, head_dim, device=device)
            v_torch = torch.randn(batch, heads, seq, head_dim, device=device)

            # Compute ALiBi slopes and bias for PyTorch
            def get_alibi_slopes_torch(n_heads):
                def get_slopes_power_of_2(n):
                    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                    return [start * (start ** i) for i in range(n)]
                if math.log2(n_heads).is_integer():
                    return torch.tensor(get_slopes_power_of_2(n_heads), device=device)
                closest_power = 2 ** math.floor(math.log2(n_heads))
                base = get_slopes_power_of_2(closest_power)
                extra = get_alibi_slopes_torch(2 * closest_power)[::2][:n_heads - closest_power]
                return torch.cat([torch.tensor(base, device=device), extra])

            slopes = get_alibi_slopes_torch(heads)[:, None, None]
            positions = torch.arange(seq, device=device)
            distance = positions[:, None] - positions[None, :]
            alibi_bias_torch = slopes * distance.float()

            def pytorch_fn():
                return F.scaled_dot_product_attention(
                    q_torch, k_torch, v_torch, attn_mask=alibi_bias_torch
                )

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"alibi_{size}")

        return results

    def benchmark_quantized_kv_cache(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark quantized KV cache attention.

        INT8 quantized KV cache provides ~4x memory reduction vs fp32,
        enabling longer context lengths during inference.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        dims = heads * head_dim
        results = {}

        # MLX benchmark
        if HAS_MLX:
            try:
                from mlx_primitives.attention.quantized_kv_cache import QuantizedKVCacheAttention
            except ImportError:
                return results

            np.random.seed(42)
            x_np = np.random.randn(batch, seq, dims).astype(np.float32)

            attn = QuantizedKVCacheAttention(
                dims=dims,
                num_heads=heads,
                max_seq_len=seq * 2,
                causal=True,
            )
            mx.eval(attn.parameters())

            x_mlx = mx.array(x_np)
            mx.eval(x_mlx)

            def mlx_fn():
                return attn(x_mlx)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"quantized_kv_{size}")

        # PyTorch MPS - standard attention as comparison baseline
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            x_torch = torch.randn(batch, seq, dims, device=device)
            W_qkv = torch.randn(dims, 3 * dims, device=device) * 0.02

            def pytorch_fn():
                qkv = x_torch @ W_qkv
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
                k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
                v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"quantized_kv_{size}_baseline")

        return results

    def benchmark_rope_variants(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark RoPE (rotary position embedding) variants.

        Tests the apply_rotary_embedding function which is critical for
        transformer inference performance.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        results = {}

        # MLX benchmark
        if HAS_MLX:
            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            # precompute_freqs_cis returns (cos, sin, cos_doubled, sin_doubled)
            cos, sin, cos_doubled, sin_doubled = precompute_freqs_cis(head_dim, seq * 2)
            mx.eval(q_mlx, k_mlx, cos, sin)

            def mlx_fn():
                # apply_rope takes (q, k, cos, sin) and returns (q_rot, k_rot)
                q_rot, k_rot = apply_rope(q_mlx, k_mlx, cos[:seq], sin[:seq])
                return q_rot, k_rot

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"rope_{size}")

        # PyTorch MPS benchmark
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, seq, heads, head_dim, device=device)
            k_torch = torch.randn(batch, seq, heads, head_dim, device=device)

            # Precompute RoPE frequencies for PyTorch
            def precompute_freqs_torch(dim, max_seq, theta=10000.0):
                freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
                t = torch.arange(max_seq, device=device)
                freqs = torch.outer(t, freqs)
                return torch.polar(torch.ones_like(freqs), freqs)

            freqs_torch = precompute_freqs_torch(head_dim, seq * 2)

            def apply_rope_torch(x, freqs):
                # Split into real and imaginary pairs
                x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
                x_rotated = x_complex * freqs[:x.shape[1], None, :]
                return torch.view_as_real(x_rotated).flatten(-2)

            def pytorch_fn():
                q_rot = apply_rope_torch(q_torch, freqs_torch)
                k_rot = apply_rope_torch(k_torch, freqs_torch)
                return q_rot, k_rot

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"rope_{size}")

        return results

    def benchmark_flash_attention_backward(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark flash attention backward pass (gradient computation).

        The backward pass is critical for training performance and often
        takes longer than the forward pass.
        """
        batch, seq, heads, head_dim = self._get_attention_config(size)
        results = {}

        # MLX benchmark
        if HAS_MLX:
            np.random.seed(42)
            q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
            v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

            q_mlx = mx.array(q_np)
            k_mlx = mx.array(k_np)
            v_mlx = mx.array(v_np)
            mx.eval(q_mlx, k_mlx, v_mlx)

            def mlx_loss_fn(q, k, v):
                return mx.sum(flash_attention(q, k, v, causal=True))

            grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))

            def mlx_fn():
                return grad_fn(q_mlx, k_mlx, v_mlx)

            results["mlx"] = self._benchmark_mlx(mlx_fn, f"flash_attention_backward_{size}")

        # PyTorch MPS benchmark
        if HAS_PYTORCH and "pytorch_mps" in self.config.frameworks:
            device = torch.device("mps")
            q_torch = torch.randn(batch, heads, seq, head_dim, device=device, requires_grad=True)
            k_torch = torch.randn(batch, heads, seq, head_dim, device=device, requires_grad=True)
            v_torch = torch.randn(batch, heads, seq, head_dim, device=device, requires_grad=True)

            def pytorch_fn():
                # Clear gradients
                if q_torch.grad is not None:
                    q_torch.grad.zero_()
                if k_torch.grad is not None:
                    k_torch.grad.zero_()
                if v_torch.grad is not None:
                    v_torch.grad.zero_()

                out = F.scaled_dot_product_attention(
                    q_torch, k_torch, v_torch, is_causal=True
                )
                loss = out.sum()
                loss.backward(retain_graph=True)
                return q_torch.grad, k_torch.grad, v_torch.grad

            results["pytorch_mps"] = self._benchmark_pytorch(pytorch_fn, f"flash_attention_backward_{size}")

        return results

    def run_sequence_scaling(self, operation: str) -> Dict[str, List[BenchmarkResult]]:
        """Run scaling analysis across sequence lengths."""
        results: Dict[str, List[BenchmarkResult]] = {"mlx": [], "pytorch_mps": []}

        for seq_len in self.config.scaling_seq_lengths:
            if seq_len > 4096:  # Skip very long sequences for now
                continue
            try:
                # Create custom config for this seq length
                batch, _, heads, head_dim = self._get_attention_config("small")
                # Run benchmark with this sequence length
                benchmark_results = self._run_single_benchmark(
                    operation, batch, seq_len, heads, head_dim
                )
                for framework, result in benchmark_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Scaling benchmark failed for seq={seq_len}: {e}")

        return results

    def run_batch_scaling(self, operation: str) -> Dict[str, List[BenchmarkResult]]:
        """Run scaling analysis across batch sizes."""
        results: Dict[str, List[BenchmarkResult]] = {"mlx": [], "pytorch_mps": []}

        for batch_size in self.config.scaling_batch_sizes:
            try:
                _, seq, heads, head_dim = self._get_attention_config("small")
                benchmark_results = self._run_single_benchmark(
                    operation, batch_size, seq, heads, head_dim
                )
                for framework, result in benchmark_results.items():
                    if framework in results:
                        results[framework].append(result)
            except Exception as e:
                print(f"Batch scaling benchmark failed for batch={batch_size}: {e}")

        return results

    def _run_single_benchmark(
        self, operation: str, batch: int, seq: int, heads: int, head_dim: int
    ) -> Dict[str, BenchmarkResult]:
        """Run a single benchmark with specific dimensions."""
        if operation == "flash_attention":
            # Create inputs and run
            results = {}

            if HAS_MLX:
                np.random.seed(42)
                q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
                k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
                v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

                q_mlx = mx.array(q_np)
                k_mlx = mx.array(k_np)
                v_mlx = mx.array(v_np)
                mx.eval(q_mlx, k_mlx, v_mlx)

                def mlx_fn():
                    return flash_attention(q_mlx, k_mlx, v_mlx, causal=True)

                results["mlx"] = self._benchmark_mlx(
                    mlx_fn, f"flash_attention_b{batch}_s{seq}"
                )

            return results

        return {}
