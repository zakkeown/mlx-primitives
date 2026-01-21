"""Embeddings parity benchmarks."""

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from benchmarks.parity.config import ParityBenchmarkConfig, DEFAULT_SIZES
from benchmarks.utils import BenchmarkResult

# Framework availability checks
try:
    import torch
    import torch.nn.functional as F
    HAS_PYTORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False
    HAS_MPS = False
    torch = None
    F = None

try:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    jnn = None


class EmbeddingsParityBenchmarks:
    """Multi-framework embeddings benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all embeddings parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            results[f"sinusoidal_{size}"] = self._benchmark_to_list(
                self.benchmark_sinusoidal(size)
            )
            results[f"learned_positional_{size}"] = self._benchmark_to_list(
                self.benchmark_learned_positional(size)
            )
            results[f"rotary_{size}"] = self._benchmark_to_list(
                self.benchmark_rotary(size)
            )
            results[f"alibi_{size}"] = self._benchmark_to_list(
                self.benchmark_alibi(size)
            )
            results[f"relative_positional_{size}"] = self._benchmark_to_list(
                self.benchmark_relative_positional(size)
            )

        return results

    def _benchmark_to_list(self, results: Dict[str, BenchmarkResult]) -> List[BenchmarkResult]:
        """Convert dict of results to list."""
        return list(results.values())

    def _benchmark_mlx(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> BenchmarkResult:
        """Benchmark an MLX function."""
        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            if isinstance(result, mx.array):
                mx.eval(result)

        # Timed iterations
        times = []
        for _ in range(iterations):
            mx.synchronize()
            start = time.perf_counter()
            result = fn()
            if isinstance(result, mx.array):
                mx.eval(result)
            mx.synchronize()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "mlx"},
        )

    def _benchmark_pytorch(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> Optional[BenchmarkResult]:
        """Benchmark a PyTorch function on MPS."""
        if not HAS_PYTORCH or not HAS_MPS:
            return None

        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            fn()
            torch.mps.synchronize()

        # Timed iterations
        times = []
        for _ in range(iterations):
            torch.mps.synchronize()
            start = time.perf_counter()
            fn()
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "pytorch_mps"},
        )

    def _benchmark_jax(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> Optional[BenchmarkResult]:
        """Benchmark a JAX function."""
        if not HAS_JAX:
            return None

        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            times.append(time.perf_counter() - start)

        benchmark_result = BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "jax"},
        )

        # Clear JAX JIT cache to prevent memory accumulation
        jax.clear_caches()

        return benchmark_result

    # =========================================================================
    # Sinusoidal Embeddings
    # =========================================================================

    def benchmark_sinusoidal(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark sinusoidal positional embeddings across frameworks."""
        from mlx_primitives.layers import SinusoidalEmbedding

        config = self.sizes.get_config("embedding", size)
        batch, seq_len, vocab_size, dim = config
        results = {}

        # MLX benchmark
        embed = SinusoidalEmbedding(dims=dim, max_seq_len=seq_len * 2)
        mx.eval(embed._embeddings)

        results["mlx"] = self._benchmark_mlx(
            lambda: embed(seq_len=seq_len),
            name=f"sinusoidal_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            # Precompute frequencies
            positions = torch.arange(seq_len, device="mps")[:, None]
            dims_range = torch.arange(0, dim, 2, device="mps")
            freqs = 10000.0 ** (-dims_range / dim)

            def pytorch_sinusoidal():
                angles = positions * freqs
                sin_emb = torch.sin(angles)
                cos_emb = torch.cos(angles)
                return torch.cat([sin_emb, cos_emb], dim=-1)

            result = self._benchmark_pytorch(
                pytorch_sinusoidal,
                name=f"sinusoidal_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            positions_jax = jnp.arange(seq_len)[:, None]
            dims_range_jax = jnp.arange(0, dim, 2)
            freqs_jax = 10000.0 ** (-dims_range_jax / dim)

            @jax.jit
            def jax_sinusoidal():
                angles = positions_jax * freqs_jax
                sin_emb = jnp.sin(angles)
                cos_emb = jnp.cos(angles)
                return jnp.concatenate([sin_emb, cos_emb], axis=-1)

            result = self._benchmark_jax(
                jax_sinusoidal,
                name=f"sinusoidal_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Learned Positional Embeddings
    # =========================================================================

    def benchmark_learned_positional(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark learned positional embeddings across frameworks."""
        from mlx_primitives.layers import LearnedPositionalEmbedding

        config = self.sizes.get_config("embedding", size)
        batch, seq_len, vocab_size, dim = config
        results = {}

        # MLX benchmark
        embed = LearnedPositionalEmbedding(dims=dim, max_seq_len=seq_len * 2)
        mx.eval(embed.parameters())
        positions_mlx = mx.arange(seq_len)
        mx.eval(positions_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: embed(positions=positions_mlx),
            name=f"learned_positional_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            embed_torch = torch.nn.Embedding(seq_len * 2, dim, device="mps")
            positions_torch = torch.arange(seq_len, device="mps")

            def pytorch_learned():
                return embed_torch(positions_torch)

            result = self._benchmark_pytorch(
                pytorch_learned,
                name=f"learned_positional_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            embed_weights_jax = jnp.array(
                np.random.randn(seq_len * 2, dim).astype(np.float32)
            ) * (dim ** -0.5)
            positions_jax = jnp.arange(seq_len)

            @jax.jit
            def jax_learned():
                return jnp.take(embed_weights_jax, positions_jax, axis=0)

            result = self._benchmark_jax(
                jax_learned,
                name=f"learned_positional_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Rotary Embeddings (RoPE)
    # =========================================================================

    def benchmark_rotary(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark rotary position embeddings across frameworks."""
        from mlx_primitives.layers import RotaryEmbedding

        # Use attention config for rotary (batch, seq, heads, head_dim)
        config = self.sizes.get_config("attention", size)
        batch, seq_len, num_heads, head_dim = config
        results = {}

        # MLX benchmark
        rope = RotaryEmbedding(dims=head_dim, max_seq_len=seq_len * 2)
        mx.eval(rope._freqs_cis)
        q_mlx = mx.random.normal((batch, num_heads, seq_len, head_dim))
        k_mlx = mx.random.normal((batch, num_heads, seq_len, head_dim))
        mx.eval(q_mlx)
        mx.eval(k_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: rope(q_mlx, k_mlx, offset=0),
            name=f"rotary_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            # Precompute frequencies
            freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device="mps") / head_dim))
            positions = torch.arange(seq_len, device="mps")
            freqs_outer = positions[:, None] * freqs[None, :]
            cos_freqs = torch.cos(freqs_outer)  # (seq_len, head_dim/2)
            sin_freqs = torch.sin(freqs_outer)  # (seq_len, head_dim/2)

            q_torch = torch.randn(batch, num_heads, seq_len, head_dim, device="mps")
            k_torch = torch.randn(batch, num_heads, seq_len, head_dim, device="mps")

            def pytorch_rope():
                # Apply rotary to Q
                q_reshaped = q_torch.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
                q0, q1 = q_reshaped[..., 0], q_reshaped[..., 1]
                cos = cos_freqs[None, None, :, :]  # (1, 1, seq_len, head_dim/2)
                sin = sin_freqs[None, None, :, :]
                q_rot0 = q0 * cos - q1 * sin
                q_rot1 = q0 * sin + q1 * cos
                q_rot = torch.stack([q_rot0, q_rot1], dim=-1).reshape(batch, num_heads, seq_len, head_dim)

                # Apply rotary to K
                k_reshaped = k_torch.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
                k0, k1 = k_reshaped[..., 0], k_reshaped[..., 1]
                k_rot0 = k0 * cos - k1 * sin
                k_rot1 = k0 * sin + k1 * cos
                k_rot = torch.stack([k_rot0, k_rot1], dim=-1).reshape(batch, num_heads, seq_len, head_dim)

                return q_rot, k_rot

            result = self._benchmark_pytorch(
                pytorch_rope,
                name=f"rotary_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            freqs_jax = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2) / head_dim))
            positions_jax = jnp.arange(seq_len)
            freqs_outer_jax = positions_jax[:, None] * freqs_jax[None, :]
            cos_freqs_jax = jnp.cos(freqs_outer_jax)
            sin_freqs_jax = jnp.sin(freqs_outer_jax)

            q_jax = jnp.array(np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32))
            k_jax = jnp.array(np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32))

            @jax.jit
            def jax_rope():
                # Apply rotary to Q
                q_reshaped = q_jax.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
                q0, q1 = q_reshaped[..., 0], q_reshaped[..., 1]
                cos = cos_freqs_jax[None, None, :, :]
                sin = sin_freqs_jax[None, None, :, :]
                q_rot0 = q0 * cos - q1 * sin
                q_rot1 = q0 * sin + q1 * cos
                q_rot = jnp.stack([q_rot0, q_rot1], axis=-1).reshape(batch, num_heads, seq_len, head_dim)

                # Apply rotary to K
                k_reshaped = k_jax.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
                k0, k1 = k_reshaped[..., 0], k_reshaped[..., 1]
                k_rot0 = k0 * cos - k1 * sin
                k_rot1 = k0 * sin + k1 * cos
                k_rot = jnp.stack([k_rot0, k_rot1], axis=-1).reshape(batch, num_heads, seq_len, head_dim)

                return q_rot, k_rot

            result = self._benchmark_jax(
                jax_rope,
                name=f"rotary_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # ALiBi Embeddings
    # =========================================================================

    def benchmark_alibi(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark ALiBi (Attention with Linear Biases) across frameworks."""
        from mlx_primitives.layers import AlibiEmbedding

        # Use attention config for alibi (batch, seq, heads, head_dim)
        config = self.sizes.get_config("attention", size)
        batch, seq_len, num_heads, head_dim = config
        results = {}

        # MLX benchmark
        alibi = AlibiEmbedding(num_heads=num_heads)
        mx.eval(alibi._slopes)
        attn_scores_mlx = mx.random.normal((batch, num_heads, seq_len, seq_len))
        mx.eval(attn_scores_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: alibi(attn_scores_mlx),
            name=f"alibi_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            # Compute slopes
            ratio = 2 ** (-8 / num_heads)
            slopes_torch = torch.tensor(
                [ratio ** (i + 1) for i in range(num_heads)],
                device="mps"
            )
            attn_scores_torch = torch.randn(batch, num_heads, seq_len, seq_len, device="mps")

            def pytorch_alibi():
                q_pos = torch.arange(seq_len, device="mps")[:, None]
                k_pos = torch.arange(seq_len, device="mps")[None, :]
                rel_pos = q_pos - k_pos  # (seq_len, seq_len)
                slopes = slopes_torch[:, None, None]  # (num_heads, 1, 1)
                bias = slopes * rel_pos[None, :, :]  # (num_heads, seq_len, seq_len)
                return attn_scores_torch + bias[None, :, :, :]

            result = self._benchmark_pytorch(
                pytorch_alibi,
                name=f"alibi_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            ratio = 2 ** (-8 / num_heads)
            slopes_jax = jnp.array([ratio ** (i + 1) for i in range(num_heads)])
            attn_scores_jax = jnp.array(
                np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)
            )

            @jax.jit
            def jax_alibi():
                q_pos = jnp.arange(seq_len)[:, None]
                k_pos = jnp.arange(seq_len)[None, :]
                rel_pos = q_pos - k_pos
                slopes = slopes_jax[:, None, None]
                bias = slopes * rel_pos[None, :, :]
                return attn_scores_jax + bias[None, :, :, :]

            result = self._benchmark_jax(
                jax_alibi,
                name=f"alibi_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Relative Positional Embeddings (T5-style)
    # =========================================================================

    def benchmark_relative_positional(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark T5-style relative positional embeddings across frameworks."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        # Use attention config for relative pos (batch, seq, heads, head_dim)
        config = self.sizes.get_config("attention", size)
        batch, seq_len, num_heads, head_dim = config
        num_buckets = 32
        max_distance = 128
        results = {}

        # MLX benchmark
        rel_pos = RelativePositionalEmbedding(
            num_heads=num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            bidirectional=True,
        )
        mx.eval(rel_pos.parameters())
        attn_scores_mlx = mx.random.normal((batch, num_heads, seq_len, seq_len))
        mx.eval(attn_scores_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: rel_pos(attn_scores_mlx),
            name=f"relative_positional_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            embed_torch = torch.nn.Embedding(num_buckets, num_heads, device="mps")
            attn_scores_torch = torch.randn(batch, num_heads, seq_len, seq_len, device="mps")

            def relative_position_bucket_torch(rel_pos_tensor, bidirectional=True):
                ret = torch.zeros_like(rel_pos_tensor)
                n_buckets = num_buckets // 2 if bidirectional else num_buckets

                if bidirectional:
                    ret = ret + torch.where(rel_pos_tensor > 0, n_buckets, 0)
                    rel_pos_tensor = torch.abs(rel_pos_tensor)
                else:
                    rel_pos_tensor = torch.clamp(-rel_pos_tensor, min=0)

                max_exact = n_buckets // 2
                is_small = rel_pos_tensor < max_exact

                val_if_large = max_exact + (
                    torch.log(rel_pos_tensor.float() / max_exact)
                    / math.log(max_distance / max_exact)
                    * (n_buckets - max_exact)
                ).long()
                val_if_large = torch.clamp(val_if_large, max=n_buckets - 1)

                ret = ret + torch.where(is_small, rel_pos_tensor, val_if_large)
                return ret.long()

            def pytorch_relative():
                q_pos = torch.arange(seq_len, device="mps")[:, None]
                k_pos = torch.arange(seq_len, device="mps")[None, :]
                rel_pos_tensor = k_pos - q_pos

                buckets = relative_position_bucket_torch(rel_pos_tensor)
                values = embed_torch(buckets.reshape(-1))
                values = values.reshape(seq_len, seq_len, num_heads)
                values = values.permute(2, 0, 1)  # (num_heads, seq_q, seq_k)
                return attn_scores_torch + values[None, :, :, :]

            result = self._benchmark_pytorch(
                pytorch_relative,
                name=f"relative_positional_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            embed_weights_jax = jnp.array(
                np.random.randn(num_buckets, num_heads).astype(np.float32)
            )
            attn_scores_jax = jnp.array(
                np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)
            )

            def relative_position_bucket_jax(rel_pos_arr, bidirectional=True):
                ret = jnp.zeros_like(rel_pos_arr)
                n_buckets = num_buckets // 2 if bidirectional else num_buckets

                if bidirectional:
                    ret = ret + jnp.where(rel_pos_arr > 0, n_buckets, 0)
                    rel_pos_arr = jnp.abs(rel_pos_arr)
                else:
                    rel_pos_arr = jnp.maximum(-rel_pos_arr, 0)

                max_exact = n_buckets // 2
                is_small = rel_pos_arr < max_exact

                val_if_large = max_exact + (
                    jnp.log(rel_pos_arr.astype(jnp.float32) / max_exact)
                    / math.log(max_distance / max_exact)
                    * (n_buckets - max_exact)
                ).astype(jnp.int32)
                val_if_large = jnp.minimum(val_if_large, n_buckets - 1)

                ret = ret + jnp.where(is_small, rel_pos_arr, val_if_large)
                return ret.astype(jnp.int32)

            @jax.jit
            def jax_relative():
                q_pos = jnp.arange(seq_len)[:, None]
                k_pos = jnp.arange(seq_len)[None, :]
                rel_pos_arr = k_pos - q_pos

                buckets = relative_position_bucket_jax(rel_pos_arr)
                values = jnp.take(embed_weights_jax, buckets.reshape(-1), axis=0)
                values = values.reshape(seq_len, seq_len, num_heads)
                values = jnp.transpose(values, (2, 0, 1))
                return attn_scores_jax + values[None, :, :, :]

            result = self._benchmark_jax(
                jax_relative,
                name=f"relative_positional_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results
