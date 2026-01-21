"""Extended JAX Metal baseline benchmarks for comprehensive parity comparison.

This module provides extended JAX Metal implementations for comparing
against MLX implementations across 50+ operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import time

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import nn as jnn
    HAS_JAX = True
    # Check for Metal backend
    HAS_JAX_METAL = any("METAL" in str(d).upper() for d in jax.devices())
except ImportError:
    HAS_JAX = False
    HAS_JAX_METAL = False
    jax = None
    jnp = None
    lax = None
    jnn = None
    np = None


def jax_metal_available() -> bool:
    """Check if JAX with Metal backend is available."""
    return HAS_JAX and HAS_JAX_METAL


@dataclass
class JAXBenchmarkResult:
    """Result from a JAX benchmark."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int


class JAXMetalExtendedBenchmarks:
    """Extended JAX Metal benchmarks covering all 50+ operations."""

    def __init__(self):
        """Initialize JAX Metal benchmarks.

        Raises:
            RuntimeError: If JAX Metal is not available.
        """
        if not jax_metal_available():
            raise RuntimeError("JAX Metal backend is not available")

    def _benchmark(
        self,
        fn,
        iterations: int = 30,
        warmup_iterations: int = 5,
        name: str = "benchmark",
    ) -> JAXBenchmarkResult:
        """Run a benchmark with warmup and timing.

        Args:
            fn: Function to benchmark (should return a JAX array).
            iterations: Number of timed iterations.
            warmup_iterations: Number of warmup iterations.
            name: Benchmark name.

        Returns:
            Benchmark result with timing statistics.
        """
        def sync_result(result):
            """Block until result is ready, handling tuples and lists."""
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, (tuple, list)):
                for r in result:
                    sync_result(r)

        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            sync_result(result)

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            sync_result(result)
            times.append(time.perf_counter() - start)

        return JAXBenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
        )

    # ========== Attention Operations ==========

    def benchmark_flash_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        causal: bool = False,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark flash attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def attention_fn(q, k, v):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if causal:
                mask = jnp.tril(jnp.ones((seq_length, seq_length)))
                scores = jnp.where(mask, scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = attention_fn(q, k, v)

        def fn():
            return attention_fn(q, k, v)

        name = f"jax_flash_attention_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_sliding_window_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark sliding window attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def sliding_window_fn(q, k, v):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            # Create sliding window mask
            i = jnp.arange(seq_length)[:, None]
            j = jnp.arange(seq_length)[None, :]
            mask = jnp.abs(i - j) <= window_size
            scores = jnp.where(mask, scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = sliding_window_fn(q, k, v)

        def fn():
            return sliding_window_fn(q, k, v)

        name = f"jax_sliding_window_b{batch_size}_s{seq_length}_h{num_heads}_w{window_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_chunked_cross_attention(
        self,
        batch_size: int,
        q_seq_length: int,
        kv_seq_length: int,
        num_heads: int,
        head_dim: int,
        chunk_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark chunked cross attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, q_seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, kv_seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, kv_seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def chunked_cross_attention_fn(q, k, v):
            # Process in chunks for memory efficiency
            num_chunks = (kv_seq_length + chunk_size - 1) // chunk_size
            output = jnp.zeros_like(q)
            max_scores = jnp.full((batch_size, num_heads, q_seq_length, 1), -jnp.inf)
            sum_exp = jnp.zeros((batch_size, num_heads, q_seq_length, 1))

            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, kv_seq_length)
                k_chunk = k[:, :, start:end, :]
                v_chunk = v[:, :, start:end, :]

                scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_chunk) * scale
                chunk_max = jnp.max(scores, axis=-1, keepdims=True)
                new_max = jnp.maximum(max_scores, chunk_max)

                exp_scores = jnp.exp(scores - new_max)
                sum_exp = sum_exp * jnp.exp(max_scores - new_max) + jnp.sum(exp_scores, axis=-1, keepdims=True)
                output = output * jnp.exp(max_scores - new_max) + jnp.einsum("bhqk,bhkd->bhqd", exp_scores, v_chunk)
                max_scores = new_max

            return output / sum_exp

        _ = chunked_cross_attention_fn(q, k, v)

        def fn():
            return chunked_cross_attention_fn(q, k, v)

        name = f"jax_chunked_cross_b{batch_size}_q{q_seq_length}_kv{kv_seq_length}_h{num_heads}_c{chunk_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gqa(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark grouped query attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_kv_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_kv_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)
        repeats = num_heads // num_kv_heads

        @jax.jit
        def gqa_fn(q, k, v):
            # Repeat KV heads to match Q heads
            k_expanded = jnp.repeat(k, repeats, axis=1)
            v_expanded = jnp.repeat(v, repeats, axis=1)
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_expanded) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v_expanded)

        _ = gqa_fn(q, k, v)

        def fn():
            return gqa_fn(q, k, v)

        name = f"jax_gqa_b{batch_size}_s{seq_length}_h{num_heads}_kv{num_kv_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_mqa(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark multi-query attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, 1, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, 1, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def mqa_fn(q, k, v):
            # Broadcast single KV head across all Q heads
            scores = jnp.einsum("bhqd,b1kd->bhqk", q, k) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,b1kd->bhqd", weights, v)

        _ = mqa_fn(q, k, v)

        def fn():
            return mqa_fn(q, k, v)

        name = f"jax_mqa_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_sparse_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        sparsity_pattern: str,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark sparse attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        # Create sparsity mask based on pattern
        if sparsity_pattern == "local":
            window = seq_length // 4
            i = jnp.arange(seq_length)[:, None]
            j = jnp.arange(seq_length)[None, :]
            mask = jnp.abs(i - j) <= window
        elif sparsity_pattern == "strided":
            stride = 4
            i = jnp.arange(seq_length)[:, None]
            j = jnp.arange(seq_length)[None, :]
            mask = ((i - j) % stride == 0) | (jnp.abs(i - j) <= 2)
        else:  # "random"
            mask = jax.random.bernoulli(key, 0.25, (seq_length, seq_length))

        @jax.jit
        def sparse_attention_fn(q, k, v, mask):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            scores = jnp.where(mask, scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = sparse_attention_fn(q, k, v, mask)

        def fn():
            return sparse_attention_fn(q, k, v, mask)

        name = f"jax_sparse_attn_b{batch_size}_s{seq_length}_h{num_heads}_{sparsity_pattern}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_linear_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark linear attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))

        @jax.jit
        def linear_attention_fn(q, k, v):
            # ELU+1 feature map
            q_prime = jax.nn.elu(q) + 1
            k_prime = jax.nn.elu(k) + 1
            # Linear attention: phi(Q) @ (phi(K)^T @ V)
            kv = jnp.einsum("bhkd,bhkv->bhdv", k_prime, v)
            output = jnp.einsum("bhqd,bhdv->bhqv", q_prime, kv)
            # Normalize
            k_sum = jnp.sum(k_prime, axis=2, keepdims=True)
            normalizer = jnp.einsum("bhqd,bhkd->bhq", q_prime, k_sum)[..., None]
            return output / (normalizer + 1e-6)

        _ = linear_attention_fn(q, k, v)

        def fn():
            return linear_attention_fn(q, k, v)

        name = f"jax_linear_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_alibi_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark ALiBi attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        # Compute ALiBi slopes
        def get_slopes(n):
            start = 2 ** (-(2 ** -(jnp.log2(n) - 3)))
            return jnp.array([start * (start ** i) for i in range(n)])

        slopes = get_slopes(num_heads)
        positions = jnp.arange(seq_length)
        distances = positions[:, None] - positions[None, :]
        alibi_bias = slopes[:, None, None] * distances[None, :, :]

        @jax.jit
        def alibi_attention_fn(q, k, v, alibi_bias):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale + alibi_bias
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = alibi_attention_fn(q, k, v, alibi_bias)

        def fn():
            return alibi_attention_fn(q, k, v, alibi_bias)

        name = f"jax_alibi_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_quantized_kv_cache_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        bits: int = 8,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark attention with quantized KV cache."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        # Pre-quantized K and V
        k_q = jax.random.randint(key, (batch_size, num_heads, seq_length, head_dim), -127, 128, dtype=jnp.int8)
        v_q = jax.random.randint(key, (batch_size, num_heads, seq_length, head_dim), -127, 128, dtype=jnp.int8)
        k_scale = jnp.array(0.02)
        v_scale = jnp.array(0.02)
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def quantized_kv_attention_fn(q, k_q, v_q, k_scale, v_scale):
            # Dequantize K and V
            k = k_q.astype(jnp.float32) * k_scale
            v = v_q.astype(jnp.float32) * v_scale
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = quantized_kv_attention_fn(q, k_q, v_q, k_scale, v_scale)

        def fn():
            return quantized_kv_attention_fn(q, k_q, v_q, k_scale, v_scale)

        name = f"jax_quantized_kv_attn_b{batch_size}_s{seq_length}_h{num_heads}_bits{bits}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_rope_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark RoPE-integrated attention."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        # Precompute RoPE
        half_dim = head_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        positions = jnp.arange(seq_length)
        angles = positions[:, None] * freqs[None, :]
        cos = jnp.cos(angles)[None, None, :, :]
        sin = jnp.sin(angles)[None, None, :, :]

        @jax.jit
        def rope_attention_fn(q, k, v, cos, sin):
            # Apply RoPE
            q1, q2 = jnp.split(q, 2, axis=-1)
            k1, k2 = jnp.split(k, 2, axis=-1)
            q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
            k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
            # Attention
            scores = jnp.einsum("bhqd,bhkd->bhqk", q_rot, k_rot) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        _ = rope_attention_fn(q, k, v, cos, sin)

        def fn():
            return rope_attention_fn(q, k, v, cos, sin)

        name = f"jax_rope_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_attention_backward(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark attention backward pass."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        def attention_loss(q, k, v):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
            return jnp.sum(output)

        grad_fn = jax.jit(jax.grad(attention_loss, argnums=(0, 1, 2)))

        # Compile once
        _ = grad_fn(q, k, v)

        def fn():
            return grad_fn(q, k, v)

        name = f"jax_attention_backward_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Activation Operations ==========

    def benchmark_swiglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark SwiGLU activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        W_gate = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
        W_up = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02

        @jax.jit
        def swiglu_fn(x, W_gate, W_up):
            gate = jax.nn.silu(x @ W_gate)
            up = x @ W_up
            return gate * up

        # Compile once
        _ = swiglu_fn(x, W_gate, W_up)

        def fn():
            return swiglu_fn(x, W_gate, W_up)

        name = f"jax_swiglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_geglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark GeGLU activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        W_gate = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
        W_up = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02

        @jax.jit
        def geglu_fn(x, W_gate, W_up):
            gate = jax.nn.gelu(x @ W_gate, approximate=False)
            up = x @ W_up
            return gate * up

        # Compile once
        _ = geglu_fn(x, W_gate, W_up)

        def fn():
            return geglu_fn(x, W_gate, W_up)

        name = f"jax_geglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_reglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark ReGLU activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        W_gate = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
        W_up = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02

        @jax.jit
        def reglu_fn(x, W_gate, W_up):
            gate = jax.nn.relu(x @ W_gate)
            up = x @ W_up
            return gate * up

        # Compile once
        _ = reglu_fn(x, W_gate, W_up)

        def fn():
            return reglu_fn(x, W_gate, W_up)

        name = f"jax_reglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_quick_gelu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark QuickGELU activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def quick_gelu_fn(x):
            return x * jax.nn.sigmoid(1.702 * x)

        # Compile once
        _ = quick_gelu_fn(x)

        def fn():
            return quick_gelu_fn(x)

        name = f"jax_quick_gelu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gelu_tanh(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark GELU with tanh approximation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def gelu_tanh_fn(x):
            return jax.nn.gelu(x, approximate=True)

        # Compile once
        _ = gelu_tanh_fn(x)

        def fn():
            return gelu_tanh_fn(x)

        name = f"jax_gelu_tanh_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_mish(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark Mish activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def mish_fn(x):
            # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
            return x * jnp.tanh(jax.nn.softplus(x))

        # Compile once
        _ = mish_fn(x)

        def fn():
            return mish_fn(x)

        name = f"jax_mish_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_squared_relu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark Squared ReLU activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def squared_relu_fn(x):
            return jax.nn.relu(x) ** 2

        # Compile once
        _ = squared_relu_fn(x)

        def fn():
            return squared_relu_fn(x)

        name = f"jax_squared_relu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_hard_swish(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark HardSwish activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def hard_swish_fn(x):
            return x * jnp.clip(x + 3, 0, 6) / 6

        # Compile once
        _ = hard_swish_fn(x)

        def fn():
            return hard_swish_fn(x)

        name = f"jax_hard_swish_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_hard_sigmoid(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark HardSigmoid activation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))

        @jax.jit
        def hard_sigmoid_fn(x):
            return jnp.clip(x + 3, 0, 6) / 6

        # Compile once
        _ = hard_sigmoid_fn(x)

        def fn():
            return hard_sigmoid_fn(x)

        name = f"jax_hard_sigmoid_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Normalization Operations ==========

    def benchmark_rmsnorm(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark RMSNorm."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        weight = jnp.ones(hidden_dim)
        eps = 1e-6

        @jax.jit
        def rmsnorm_fn(x, weight):
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
            return (x / rms) * weight

        # Compile once
        _ = rmsnorm_fn(x, weight)

        def fn():
            return rmsnorm_fn(x, weight)

        name = f"jax_rmsnorm_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_layernorm(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark LayerNorm."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        weight = jnp.ones(hidden_dim)
        bias = jnp.zeros(hidden_dim)
        eps = 1e-5

        @jax.jit
        def layernorm_fn(x, weight, bias):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return weight * (x - mean) / jnp.sqrt(var + eps) + bias

        # Compile once
        _ = layernorm_fn(x, weight, bias)

        def fn():
            return layernorm_fn(x, weight, bias)

        name = f"jax_layernorm_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_groupnorm(
        self,
        batch_size: int,
        channels: int,
        spatial_size: Tuple[int, ...],
        num_groups: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark GroupNorm."""
        key = jax.random.PRNGKey(42)
        # For 2D spatial: (batch, channels, height, width)
        shape = (batch_size, channels) + spatial_size
        x = jax.random.normal(key, shape)
        weight = jnp.ones(channels)
        bias = jnp.zeros(channels)
        eps = 1e-5

        @jax.jit
        def groupnorm_fn(x, weight, bias):
            # Reshape to (batch, num_groups, channels_per_group, *spatial)
            channels_per_group = channels // num_groups
            new_shape = (batch_size, num_groups, channels_per_group) + spatial_size
            x_reshaped = x.reshape(new_shape)

            # Normalize over (channels_per_group, *spatial) dims
            axes = tuple(range(2, len(new_shape)))
            mean = jnp.mean(x_reshaped, axis=axes, keepdims=True)
            var = jnp.var(x_reshaped, axis=axes, keepdims=True)
            x_norm = (x_reshaped - mean) / jnp.sqrt(var + eps)

            # Reshape back and apply affine
            x_norm = x_norm.reshape(shape)
            # Broadcast weight/bias over spatial dims
            weight_shape = (1, channels) + (1,) * len(spatial_size)
            return x_norm * weight.reshape(weight_shape) + bias.reshape(weight_shape)

        # Compile once
        _ = groupnorm_fn(x, weight, bias)

        def fn():
            return groupnorm_fn(x, weight, bias)

        spatial_str = "x".join(str(s) for s in spatial_size)
        name = f"jax_groupnorm_b{batch_size}_c{channels}_s{spatial_str}_g{num_groups}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_instancenorm(
        self,
        batch_size: int,
        channels: int,
        spatial_size: Tuple[int, ...],
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark InstanceNorm."""
        key = jax.random.PRNGKey(42)
        # (batch, channels, *spatial)
        shape = (batch_size, channels) + spatial_size
        x = jax.random.normal(key, shape)
        weight = jnp.ones(channels)
        bias = jnp.zeros(channels)
        eps = 1e-5

        @jax.jit
        def instancenorm_fn(x, weight, bias):
            # Normalize over spatial dims only (each channel independently)
            axes = tuple(range(2, len(shape)))
            mean = jnp.mean(x, axis=axes, keepdims=True)
            var = jnp.var(x, axis=axes, keepdims=True)
            x_norm = (x - mean) / jnp.sqrt(var + eps)

            # Apply affine
            weight_shape = (1, channels) + (1,) * len(spatial_size)
            return x_norm * weight.reshape(weight_shape) + bias.reshape(weight_shape)

        # Compile once
        _ = instancenorm_fn(x, weight, bias)

        def fn():
            return instancenorm_fn(x, weight, bias)

        spatial_str = "x".join(str(s) for s in spatial_size)
        name = f"jax_instancenorm_b{batch_size}_c{channels}_s{spatial_str}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adalayernorm(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark AdaLayerNorm (Adaptive Layer Normalization)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        # Conditioning vector (e.g., from timestep embedding)
        cond = jax.random.normal(key, (batch_size, hidden_dim))
        # Linear projections for scale and shift
        scale_proj = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
        shift_proj = jax.random.normal(key, (hidden_dim, hidden_dim)) * 0.02
        eps = 1e-5

        @jax.jit
        def adaln_fn(x, cond, scale_proj, shift_proj):
            # Layer norm
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / jnp.sqrt(var + eps)

            # Adaptive scale and shift from conditioning
            scale = cond @ scale_proj  # (batch, hidden_dim)
            shift = cond @ shift_proj  # (batch, hidden_dim)

            # Broadcast over sequence dimension
            scale = scale[:, None, :]  # (batch, 1, hidden_dim)
            shift = shift[:, None, :]  # (batch, 1, hidden_dim)

            return x_norm * (1 + scale) + shift

        # Compile once
        _ = adaln_fn(x, cond, scale_proj, shift_proj)

        def fn():
            return adaln_fn(x, cond, scale_proj, shift_proj)

        name = f"jax_adalayernorm_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Fused Operations ==========

    def benchmark_fused_rmsnorm_linear(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        output_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark fused RMSNorm + Linear."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        norm_weight = jnp.ones(hidden_dim)
        linear_weight = jax.random.normal(key, (hidden_dim, output_dim)) * 0.02
        linear_bias = jnp.zeros(output_dim)
        eps = 1e-6

        @jax.jit
        def fused_rmsnorm_linear_fn(x, norm_weight, linear_weight, linear_bias):
            # Fused RMSNorm + Linear in single JIT
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
            x_norm = (x / rms) * norm_weight
            return x_norm @ linear_weight + linear_bias

        # Compile once
        _ = fused_rmsnorm_linear_fn(x, norm_weight, linear_weight, linear_bias)

        def fn():
            return fused_rmsnorm_linear_fn(x, norm_weight, linear_weight, linear_bias)

        name = f"jax_fused_rmsnorm_linear_b{batch_size}_s{seq_length}_d{hidden_dim}_o{output_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_swiglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark fused SwiGLU (Linear -> SiLU gate -> Linear)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        intermediate_dim = hidden_dim * 4
        W_gate = jax.random.normal(key, (hidden_dim, intermediate_dim)) * 0.02
        W_up = jax.random.normal(key, (hidden_dim, intermediate_dim)) * 0.02
        W_down = jax.random.normal(key, (intermediate_dim, hidden_dim)) * 0.02

        @jax.jit
        def fused_swiglu_fn(x, W_gate, W_up, W_down):
            # Fused SwiGLU: down(silu(gate(x)) * up(x))
            gate = jax.nn.silu(x @ W_gate)
            up = x @ W_up
            return (gate * up) @ W_down

        # Compile once
        _ = fused_swiglu_fn(x, W_gate, W_up, W_down)

        def fn():
            return fused_swiglu_fn(x, W_gate, W_up, W_down)

        name = f"jax_fused_swiglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_geglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark fused GeGLU (Linear -> GELU gate -> Linear)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        intermediate_dim = hidden_dim * 4
        W_gate = jax.random.normal(key, (hidden_dim, intermediate_dim)) * 0.02
        W_up = jax.random.normal(key, (hidden_dim, intermediate_dim)) * 0.02
        W_down = jax.random.normal(key, (intermediate_dim, hidden_dim)) * 0.02

        @jax.jit
        def fused_geglu_fn(x, W_gate, W_up, W_down):
            # Fused GeGLU: down(gelu(gate(x)) * up(x))
            gate = jax.nn.gelu(x @ W_gate, approximate=False)
            up = x @ W_up
            return (gate * up) @ W_down

        # Compile once
        _ = fused_geglu_fn(x, W_gate, W_up, W_down)

        def fn():
            return fused_geglu_fn(x, W_gate, W_up, W_down)

        name = f"jax_fused_geglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_rope_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark fused RoPE + Attention in single JIT."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_length, head_dim))
        scale = 1.0 / jnp.sqrt(head_dim)

        # Precompute RoPE
        half_dim = head_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        positions = jnp.arange(seq_length)
        angles = positions[:, None] * freqs[None, :]
        cos = jnp.cos(angles)[None, None, :, :]
        sin = jnp.sin(angles)[None, None, :, :]

        @jax.jit
        def fused_rope_attention_fn(q, k, v, cos, sin):
            # Apply RoPE to Q and K
            q1, q2 = jnp.split(q, 2, axis=-1)
            k1, k2 = jnp.split(k, 2, axis=-1)
            q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
            k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
            # Attention
            scores = jnp.einsum("bhqd,bhkd->bhqk", q_rot, k_rot) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        # Compile once
        _ = fused_rope_attention_fn(q, k, v, cos, sin)

        def fn():
            return fused_rope_attention_fn(q, k, v, cos, sin)

        name = f"jax_fused_rope_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Quantization Operations ==========

    def benchmark_int8_quantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT8 quantization."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (m, n))

        @jax.jit
        def int8_quantize_fn(x):
            scale = jnp.max(jnp.abs(x)) / 127.0
            x_q = jnp.round(x / scale).astype(jnp.int8)
            return x_q, scale

        # Compile once
        _ = int8_quantize_fn(x)

        def fn():
            return int8_quantize_fn(x)

        name = f"jax_int8_quantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int8_dequantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT8 dequantization."""
        key = jax.random.PRNGKey(42)
        x_q = jax.random.randint(key, (m, n), -127, 128, dtype=jnp.int8)
        scale = jnp.array(0.05)

        @jax.jit
        def int8_dequantize_fn(x_q, scale):
            return x_q.astype(jnp.float32) * scale

        # Compile once
        _ = int8_dequantize_fn(x_q, scale)

        def fn():
            return int8_dequantize_fn(x_q, scale)

        name = f"jax_int8_dequantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_quantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT4 quantization."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (m, n))

        @jax.jit
        def int4_quantize_fn(x):
            # INT4 range is [-8, 7]
            scale = jnp.max(jnp.abs(x)) / 7.0
            x_q = jnp.round(x / scale)
            x_q = jnp.clip(x_q, -8, 7).astype(jnp.int8)
            return x_q, scale

        # Compile once
        _ = int4_quantize_fn(x)

        def fn():
            return int4_quantize_fn(x)

        name = f"jax_int4_quantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_dequantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT4 dequantization."""
        key = jax.random.PRNGKey(42)
        x_q = jax.random.randint(key, (m, n), -8, 8, dtype=jnp.int8)
        scale = jnp.array(0.1)

        @jax.jit
        def int4_dequantize_fn(x_q, scale):
            return x_q.astype(jnp.float32) * scale

        # Compile once
        _ = int4_dequantize_fn(x_q, scale)

        def fn():
            return int4_dequantize_fn(x_q, scale)

        name = f"jax_int4_dequantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int8_linear(
        self,
        batch_size: int,
        seq_length: int,
        in_features: int,
        out_features: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT8 linear layer."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, in_features))
        # Pre-quantized weights
        weight_q = jax.random.randint(key, (out_features, in_features), -127, 128, dtype=jnp.int8)
        scale = jnp.array(0.02)
        bias = jax.random.normal(key, (out_features,)) * 0.01

        @jax.jit
        def int8_linear_fn(x, weight_q, scale, bias):
            # Dequantize weights and perform matmul
            weight = weight_q.astype(jnp.float32) * scale
            return x @ weight.T + bias

        # Compile once
        _ = int8_linear_fn(x, weight_q, scale, bias)

        def fn():
            return int8_linear_fn(x, weight_q, scale, bias)

        name = f"jax_int8_linear_b{batch_size}_s{seq_length}_{in_features}x{out_features}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_linear(
        self,
        batch_size: int,
        seq_length: int,
        in_features: int,
        out_features: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark INT4 linear layer."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, in_features))
        # Pre-quantized weights (INT4 stored as INT8)
        weight_q = jax.random.randint(key, (out_features, in_features), -8, 8, dtype=jnp.int8)
        scale = jnp.array(0.1)
        bias = jax.random.normal(key, (out_features,)) * 0.01

        @jax.jit
        def int4_linear_fn(x, weight_q, scale, bias):
            # Dequantize weights and perform matmul
            weight = weight_q.astype(jnp.float32) * scale
            return x @ weight.T + bias

        # Compile once
        _ = int4_linear_fn(x, weight_q, scale, bias)

        def fn():
            return int4_linear_fn(x, weight_q, scale, bias)

        name = f"jax_int4_linear_b{batch_size}_s{seq_length}_{in_features}x{out_features}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Primitive Operations ==========

    def benchmark_associative_scan_add(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark associative scan with addition (cumsum)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, dim))

        @jax.jit
        def scan_add_fn(x):
            return lax.associative_scan(jnp.add, x, axis=1)

        # Compile once
        _ = scan_add_fn(x)

        def fn():
            return scan_add_fn(x)

        name = f"jax_assoc_scan_add_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_associative_scan_mul(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark associative scan with multiplication (cumprod)."""
        key = jax.random.PRNGKey(42)
        # Use smaller values to avoid numerical issues with cumprod
        x = jax.random.uniform(key, (batch_size, seq_length, dim), minval=0.9, maxval=1.1)

        @jax.jit
        def scan_mul_fn(x):
            return lax.associative_scan(jnp.multiply, x, axis=1)

        # Compile once
        _ = scan_mul_fn(x)

        def fn():
            return scan_mul_fn(x)

        name = f"jax_assoc_scan_mul_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_associative_scan_ssm(
        self,
        batch_size: int,
        seq_length: int,
        state_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark associative scan for SSM (state space model).

        SSM recurrence: h_t = A_t * h_{t-1} + B_t * x_t
        Using associative scan with binary operator:
        (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)
        """
        key = jax.random.PRNGKey(42)
        # A coefficients (decay factors, should be < 1 for stability)
        A = jax.random.uniform(key, (batch_size, seq_length, state_dim), minval=0.9, maxval=0.99)
        # B * x term (input contribution)
        Bx = jax.random.normal(key, (batch_size, seq_length, state_dim)) * 0.1

        @jax.jit
        def ssm_scan_fn(A, Bx):
            # Pack (A, Bx) for associative scan
            # The binary operator for SSM: (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)
            def ssm_binary_op(elem1, elem2):
                a1, b1 = elem1
                a2, b2 = elem2
                return (a1 * a2, a2 * b1 + b2)

            init = (A, Bx)
            result = lax.associative_scan(ssm_binary_op, init, axis=1)
            return result[1]  # Return the state (b component)

        # Compile once
        _ = ssm_scan_fn(A, Bx)

        def fn():
            return ssm_scan_fn(A, Bx)

        name = f"jax_assoc_scan_ssm_b{batch_size}_s{seq_length}_d{state_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_scan(
        self,
        batch_size: int,
        seq_length: int,
        d_model: int,
        d_state: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark selective scan (Mamba-style).

        Mamba selective scan: h_t = A_t * h_{t-1} + B_t * x_t, y_t = C_t * h_t
        Where A, B, C are input-dependent (selective).
        """
        key = jax.random.PRNGKey(42)
        # Input
        x = jax.random.normal(key, (batch_size, seq_length, d_model))
        # Selective parameters (input-dependent)
        delta = jax.nn.softplus(jax.random.normal(key, (batch_size, seq_length, d_model)))
        A = -jnp.exp(jax.random.normal(key, (d_model, d_state)))  # Negative for stability
        B = jax.random.normal(key, (batch_size, seq_length, d_state)) * 0.1
        C = jax.random.normal(key, (batch_size, seq_length, d_state)) * 0.1

        @jax.jit
        def selective_scan_fn(x, delta, A, B, C):
            # Discretize: A_bar = exp(delta * A)
            # For simplicity, use first-order approximation
            A_bar = jnp.exp(jnp.einsum("bld,dn->bldn", delta, A))  # (batch, seq, d_model, d_state)
            B_bar = jnp.einsum("bld,bln->bldn", delta, B)  # (batch, seq, d_model, d_state)

            # Reshape for scan
            batch_size, seq_len, d_model = x.shape

            # Use associative scan for the SSM recurrence
            def ssm_op(carry, inputs):
                h = carry  # (batch, d_model, d_state)
                a_t, b_t, x_t, c_t = inputs
                h_new = a_t * h + b_t * x_t[:, :, None]  # (batch, d_model, d_state)
                y_t = jnp.sum(h_new * c_t[:, None, :], axis=-1)  # (batch, d_model)
                return h_new, y_t

            # Transpose for scan over sequence
            A_bar_t = jnp.transpose(A_bar, (1, 0, 2, 3))  # (seq, batch, d_model, d_state)
            B_bar_t = jnp.transpose(B_bar, (1, 0, 2, 3))
            x_t = jnp.transpose(x, (1, 0, 2))  # (seq, batch, d_model)
            C_t = jnp.transpose(C, (1, 0, 2))  # (seq, batch, d_state)

            h0 = jnp.zeros((batch_size, d_model, d_state))
            _, y = lax.scan(ssm_op, h0, (A_bar_t, B_bar_t, x_t, C_t))

            return jnp.transpose(y, (1, 0, 2))  # (batch, seq, d_model)

        # Compile once
        _ = selective_scan_fn(x, delta, A, B, C)

        def fn():
            return selective_scan_fn(x, delta, A, B, C)

        name = f"jax_selective_scan_b{batch_size}_s{seq_length}_d{d_model}_n{d_state}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_gather(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        num_indices: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark selective gather (take_along_axis)."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, dim))
        # Random indices to gather
        indices = jax.random.randint(key, (batch_size, num_indices, dim), 0, seq_length)

        @jax.jit
        def gather_fn(x, indices):
            return jnp.take_along_axis(x, indices, axis=1)

        # Compile once
        _ = gather_fn(x, indices)

        def fn():
            return gather_fn(x, indices)

        name = f"jax_selective_gather_b{batch_size}_s{seq_length}_d{dim}_n{num_indices}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_scatter_add(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        num_indices: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark selective scatter add (segment_sum style)."""
        key = jax.random.PRNGKey(42)
        # Source values to scatter
        values = jax.random.normal(key, (batch_size, num_indices, dim))
        # Target indices for scatter (where to add values)
        indices = jax.random.randint(key, (batch_size, num_indices), 0, seq_length)

        @jax.jit
        def scatter_add_fn(values, indices):
            # Initialize output
            output = jnp.zeros((batch_size, seq_length, dim))
            # Use segment_sum style scatter
            batch_idx = jnp.arange(batch_size)[:, None]
            batch_idx = jnp.broadcast_to(batch_idx, (batch_size, num_indices))
            # Flatten for scatter
            output = output.at[batch_idx.flatten(), indices.flatten()].add(
                values.reshape(-1, dim)
            )
            return output.reshape(batch_size, seq_length, dim)

        # Compile once
        _ = scatter_add_fn(values, indices)

        def fn():
            return scatter_add_fn(values, indices)

        name = f"jax_selective_scatter_add_b{batch_size}_s{seq_length}_d{dim}_n{num_indices}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== MoE Operations ==========

    def benchmark_topk_routing(
        self,
        batch_size: int,
        seq_length: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int = 512,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark top-k routing."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        gate_weight = jax.random.normal(key, (hidden_dim, num_experts)) * 0.02

        @jax.jit
        def topk_routing_fn(x, gate_weight):
            # Compute router logits
            logits = x @ gate_weight  # (batch, seq, num_experts)
            # Get top-k experts
            topk_values, topk_indices = lax.top_k(logits, top_k)
            # Softmax over selected experts
            gate_weights = jax.nn.softmax(topk_values, axis=-1)
            return gate_weights, topk_indices, logits

        # Compile once
        _ = topk_routing_fn(x, gate_weight)

        def fn():
            return topk_routing_fn(x, gate_weight)

        name = f"jax_topk_routing_b{batch_size}_s{seq_length}_e{num_experts}_k{top_k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_expert_dispatch(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int = None,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark expert dispatch (full MoE forward pass)."""
        key = jax.random.PRNGKey(42)
        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim * 4

        n_tokens = batch_size * seq_length
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        # Router weights
        gate_weight = jax.random.normal(key, (hidden_dim, num_experts)) * 0.02
        # Expert weights
        expert_w1 = jax.random.normal(key, (num_experts, hidden_dim, expert_hidden_dim)) * 0.02
        expert_w2 = jax.random.normal(key, (num_experts, expert_hidden_dim, hidden_dim)) * 0.02

        @jax.jit
        def expert_dispatch_fn(x, gate_weight, expert_w1, expert_w2):
            # Flatten input
            x_flat = x.reshape(n_tokens, hidden_dim)

            # Routing
            logits = x_flat @ gate_weight
            topk_values, topk_indices = lax.top_k(logits, top_k)
            gate_weights = jax.nn.softmax(topk_values, axis=-1)

            # Simple dispatch: for each expert, process routed tokens
            output = jnp.zeros_like(x_flat)

            for e in range(num_experts):
                # Find tokens routed to this expert
                expert_mask = (topk_indices == e)
                weights_for_expert = jnp.where(expert_mask, gate_weights, jnp.zeros_like(gate_weights))
                token_weights = weights_for_expert.sum(axis=-1)

                # Expert MLP
                hidden = jax.nn.silu(x_flat @ expert_w1[e])
                expert_out = hidden @ expert_w2[e]

                # Weighted accumulation
                output = output + expert_out * token_weights[:, None]

            return output.reshape(batch_size, seq_length, hidden_dim)

        # Compile once
        _ = expert_dispatch_fn(x, gate_weight, expert_w1, expert_w2)

        def fn():
            return expert_dispatch_fn(x, gate_weight, expert_w1, expert_w2)

        name = f"jax_expert_dispatch_b{batch_size}_s{seq_length}_e{num_experts}_k{top_k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_load_balancing_loss(
        self,
        batch_size: int,
        seq_length: int,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: int = 512,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark load balancing loss computation (GShard auxiliary loss)."""
        key = jax.random.PRNGKey(42)
        # Generate router logits and expert indices
        router_logits = jax.random.normal(key, (batch_size, seq_length, num_experts))
        expert_indices = jax.random.randint(key, (batch_size, seq_length, top_k), 0, num_experts)

        @jax.jit
        def load_balancing_loss_fn(router_logits, expert_indices):
            # Routing probabilities
            router_probs = jax.nn.softmax(router_logits, axis=-1)

            # Count tokens per expert
            total_tokens = batch_size * seq_length * top_k
            expert_counts = jnp.zeros(num_experts)
            for e in range(num_experts):
                expert_counts = expert_counts.at[e].set((expert_indices == e).sum().astype(jnp.float32))

            expert_fraction = expert_counts / total_tokens

            # Mean routing probability per expert
            mean_prob = router_probs.mean(axis=(0, 1))

            # GShard auxiliary loss
            aux_loss = num_experts * (expert_fraction * mean_prob).sum()
            return aux_loss

        # Compile once
        _ = load_balancing_loss_fn(router_logits, expert_indices)

        def fn():
            return load_balancing_loss_fn(router_logits, expert_indices)

        name = f"jax_load_balancing_loss_b{batch_size}_s{seq_length}_e{num_experts}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Pooling Operations ==========

    def benchmark_adaptive_avg_pool1d(
        self,
        batch_size: int,
        channels: int,
        length: int,
        output_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark adaptive average pooling 1D."""
        key = jax.random.PRNGKey(42)
        # (batch, channels, length)
        x = jax.random.normal(key, (batch_size, channels, length))

        @jax.jit
        def adaptive_avg_pool1d_fn(x):
            # Manual adaptive pooling using reshape and mean
            # This avoids reduce_window which isn't supported on Metal
            batch_size, channels, length = x.shape
            # Reshape to (batch, channels, output_size, segment_size)
            segment_size = length // output_size
            x_reshaped = x[:, :, :output_size * segment_size].reshape(
                batch_size, channels, output_size, segment_size
            )
            return jnp.mean(x_reshaped, axis=-1)

        # Compile once
        _ = adaptive_avg_pool1d_fn(x)

        def fn():
            return adaptive_avg_pool1d_fn(x)

        name = f"jax_adaptive_avg_pool1d_b{batch_size}_c{channels}_l{length}_o{output_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_avg_pool2d(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        output_size: Tuple[int, int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark adaptive average pooling 2D."""
        key = jax.random.PRNGKey(42)
        # (batch, channels, height, width)
        x = jax.random.normal(key, (batch_size, channels, height, width))
        out_h, out_w = output_size

        @jax.jit
        def adaptive_avg_pool2d_fn(x):
            # Manual adaptive pooling using reshape and mean
            # This avoids reduce_window which isn't supported on Metal
            b, c, h, w = x.shape
            seg_h = h // out_h
            seg_w = w // out_w
            # Reshape to (batch, channels, out_h, seg_h, out_w, seg_w)
            x_reshaped = x[:, :, :out_h * seg_h, :out_w * seg_w].reshape(
                b, c, out_h, seg_h, out_w, seg_w
            )
            return jnp.mean(x_reshaped, axis=(3, 5))

        # Compile once
        _ = adaptive_avg_pool2d_fn(x)

        def fn():
            return adaptive_avg_pool2d_fn(x)

        name = f"jax_adaptive_avg_pool2d_b{batch_size}_c{channels}_h{height}_w{width}_o{out_h}x{out_w}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_max_pool1d(
        self,
        batch_size: int,
        channels: int,
        length: int,
        output_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark adaptive max pooling 1D."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, channels, length))

        @jax.jit
        def adaptive_max_pool1d_fn(x):
            # Manual adaptive max pooling using reshape and max
            # This avoids reduce_window which isn't supported on Metal
            batch_size, channels, length = x.shape
            segment_size = length // output_size
            x_reshaped = x[:, :, :output_size * segment_size].reshape(
                batch_size, channels, output_size, segment_size
            )
            return jnp.max(x_reshaped, axis=-1)

        # Compile once
        _ = adaptive_max_pool1d_fn(x)

        def fn():
            return adaptive_max_pool1d_fn(x)

        name = f"jax_adaptive_max_pool1d_b{batch_size}_c{channels}_l{length}_o{output_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_max_pool2d(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        output_size: Tuple[int, int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark adaptive max pooling 2D."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, channels, height, width))
        out_h, out_w = output_size

        @jax.jit
        def adaptive_max_pool2d_fn(x):
            # Manual adaptive max pooling using reshape and max
            # This avoids reduce_window which isn't supported on Metal
            b, c, h, w = x.shape
            seg_h = h // out_h
            seg_w = w // out_w
            x_reshaped = x[:, :, :out_h * seg_h, :out_w * seg_w].reshape(
                b, c, out_h, seg_h, out_w, seg_w
            )
            return jnp.max(x_reshaped, axis=(3, 5))

        # Compile once
        _ = adaptive_max_pool2d_fn(x)

        def fn():
            return adaptive_max_pool2d_fn(x)

        name = f"jax_adaptive_max_pool2d_b{batch_size}_c{channels}_h{height}_w{width}_o{out_h}x{out_w}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_global_attention_pooling(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark global attention pooling."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_length, hidden_dim))
        # Attention weights projection
        attn_weight = jax.random.normal(key, (hidden_dim, 1)) * 0.02

        @jax.jit
        def global_attention_pool_fn(x, attn_weight):
            # Compute attention scores
            scores = x @ attn_weight  # (batch, seq_length, 1)
            weights = jax.nn.softmax(scores, axis=1)  # (batch, seq_length, 1)
            # Weighted sum
            return (x * weights).sum(axis=1)  # (batch, hidden_dim)

        # Compile once
        _ = global_attention_pool_fn(x, attn_weight)

        def fn():
            return global_attention_pool_fn(x, attn_weight)

        name = f"jax_global_attn_pool_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gem(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        p: float = 3.0,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark Generalized Mean (GeM) pooling."""
        key = jax.random.PRNGKey(42)
        # Ensure positive values for power operation
        x = jnp.abs(jax.random.normal(key, (batch_size, channels, height, width))) + 1e-6

        @jax.jit
        def gem_fn(x):
            # GeM pooling: (mean(x^p))^(1/p)
            x_pow = jnp.power(x, p)
            mean_pow = jnp.mean(x_pow, axis=(2, 3), keepdims=True)
            return jnp.power(mean_pow, 1.0 / p).squeeze(axis=(2, 3))

        # Compile once
        _ = gem_fn(x)

        def fn():
            return gem_fn(x)

        name = f"jax_gem_b{batch_size}_c{channels}_h{height}_w{width}_p{p}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_spatial_pyramid_pooling(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        levels: List[int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark Spatial Pyramid Pooling."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, channels, height, width))

        @jax.jit
        def spp_fn(x):
            outputs = []
            for level in levels:
                # Adaptive average pool to (level, level)
                stride_h = height // level
                stride_w = width // level
                kernel_h = height - (level - 1) * stride_h
                kernel_w = width - (level - 1) * stride_w

                pooled = lax.reduce_window(
                    x,
                    init_value=0.0,
                    computation=lax.add,
                    window_dimensions=(1, 1, kernel_h, kernel_w),
                    window_strides=(1, 1, stride_h, stride_w),
                    padding="VALID",
                ) / (kernel_h * kernel_w)

                # Flatten spatial dimensions
                pooled_flat = pooled.reshape(batch_size, channels, -1)
                outputs.append(pooled_flat)

            # Concatenate all levels
            return jnp.concatenate(outputs, axis=-1)

        # Compile once
        _ = spp_fn(x)

        def fn():
            return spp_fn(x)

        levels_str = "_".join(str(l) for l in levels)
        name = f"jax_spp_b{batch_size}_c{channels}_h{height}_w{width}_levels{levels_str}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Embedding Operations ==========

    def benchmark_sinusoidal_embedding(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark sinusoidal positional embedding."""
        positions = jnp.arange(seq_length)

        @jax.jit
        def sinusoidal_fn(positions):
            # Geometric frequency spacing
            half_dim = dim // 2
            freqs = jnp.exp(
                -jnp.log(10000.0) * jnp.arange(half_dim) / half_dim
            )
            # Outer product: positions x frequencies
            angles = positions[:, None] * freqs[None, :]  # (seq_length, half_dim)
            # Interleave sin and cos
            sin_emb = jnp.sin(angles)
            cos_emb = jnp.cos(angles)
            return jnp.concatenate([sin_emb, cos_emb], axis=-1)  # (seq_length, dim)

        # Compile once
        _ = sinusoidal_fn(positions)

        def fn():
            return sinusoidal_fn(positions)

        name = f"jax_sinusoidal_emb_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_learned_positional_embedding(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        max_length: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark learned positional embedding."""
        key = jax.random.PRNGKey(42)
        # Embedding table
        embedding_table = jax.random.normal(key, (max_length, dim)) * 0.02
        positions = jnp.arange(seq_length)

        @jax.jit
        def learned_emb_fn(embedding_table, positions):
            # Simple table lookup
            return embedding_table[positions]  # (seq_length, dim)

        # Compile once
        _ = learned_emb_fn(embedding_table, positions)

        def fn():
            return learned_emb_fn(embedding_table, positions)

        name = f"jax_learned_pos_emb_b{batch_size}_s{seq_length}_d{dim}_max{max_length}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_rotary_embedding(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark rotary positional embedding."""
        key = jax.random.PRNGKey(42)
        # Input tensor (batch, seq, num_heads, head_dim)
        x = jax.random.normal(key, (batch_size, seq_length, num_heads, head_dim))

        # Precompute cos and sin for RoPE
        half_dim = head_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        positions = jnp.arange(seq_length)
        angles = positions[:, None] * freqs[None, :]  # (seq_length, half_dim)
        cos = jnp.cos(angles)  # (seq_length, half_dim)
        sin = jnp.sin(angles)  # (seq_length, half_dim)

        @jax.jit
        def rope_fn(x, cos, sin):
            # Split x into two halves
            x1, x2 = jnp.split(x, 2, axis=-1)
            # Reshape cos/sin for broadcasting: (1, seq, 1, half_dim)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            # Apply rotation: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
            rotated = jnp.concatenate([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ], axis=-1)
            return rotated

        # Compile once
        _ = rope_fn(x, cos, sin)

        def fn():
            return rope_fn(x, cos, sin)

        name = f"jax_rope_b{batch_size}_s{seq_length}_h{num_heads}_d{head_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_alibi_embedding(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark ALiBi embedding."""
        # Compute ALiBi slopes
        def get_slopes(n):
            """Get slopes for n heads."""
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(jnp.log2(n) - 3)))
                ratio = start
                return jnp.array([start * (ratio ** i) for i in range(n)])

            if jnp.log2(n) % 1 == 0:
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** jnp.floor(jnp.log2(n))
                slopes = get_slopes_power_of_2(int(closest_power_of_2))
                extra_slopes = get_slopes_power_of_2(int(2 * closest_power_of_2))[0::2][:int(n - closest_power_of_2)]
                return jnp.concatenate([slopes, extra_slopes])

        slopes = get_slopes(num_heads)

        @jax.jit
        def alibi_fn(slopes):
            # Create position bias matrix
            positions = jnp.arange(seq_length)
            # Distance matrix: (seq_length, seq_length)
            distances = positions[:, None] - positions[None, :]
            # ALiBi bias: slopes * distances
            # Shape: (num_heads, seq_length, seq_length)
            bias = slopes[:, None, None] * distances[None, :, :]
            return bias

        # Compile once
        _ = alibi_fn(slopes)

        def fn():
            return alibi_fn(slopes)

        name = f"jax_alibi_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_relative_positional_embedding(
        self,
        batch_size: int,
        q_length: int,
        k_length: int,
        num_heads: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark relative positional embedding."""
        key = jax.random.PRNGKey(42)
        # T5-style relative position embedding table
        num_buckets = 32
        max_distance = 128
        embedding_table = jax.random.normal(key, (num_buckets, num_heads)) * 0.02

        @jax.jit
        def relative_pos_fn(embedding_table):
            # Compute relative positions
            q_pos = jnp.arange(q_length)[:, None]
            k_pos = jnp.arange(k_length)[None, :]
            relative_pos = k_pos - q_pos  # (q_length, k_length)

            # T5-style bucketing
            is_small = jnp.abs(relative_pos) < (num_buckets // 2)

            # For small distances, use linear mapping
            small_bucket = relative_pos + num_buckets // 2

            # For large distances, use logarithmic mapping
            max_exact = num_buckets // 2
            relative_pos_abs = jnp.abs(relative_pos)
            relative_pos_if_large = max_exact + (
                jnp.log(relative_pos_abs / max_exact + 1e-6) /
                jnp.log(max_distance / max_exact) *
                (num_buckets - max_exact)
            ).astype(jnp.int32)
            relative_pos_if_large = jnp.minimum(relative_pos_if_large, num_buckets - 1)

            bucket = jnp.where(is_small, small_bucket, relative_pos_if_large)
            bucket = jnp.clip(bucket, 0, num_buckets - 1)

            # Lookup embeddings: (q_length, k_length, num_heads)
            return embedding_table[bucket]

        # Compile once
        _ = relative_pos_fn(embedding_table)

        def fn():
            return relative_pos_fn(embedding_table)

        name = f"jax_rel_pos_emb_b{batch_size}_q{q_length}_k{k_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Cache Operations ==========

    def benchmark_paged_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark paged attention with block-based KV cache."""
        key = jax.random.PRNGKey(42)
        num_blocks = (seq_length + block_size - 1) // block_size

        # Query for current position
        q = jax.random.normal(key, (batch_size, num_heads, 1, head_dim))
        # Paged KV cache: (num_blocks, batch_size, num_heads, block_size, head_dim)
        k_cache = jax.random.normal(key, (num_blocks, batch_size, num_heads, block_size, head_dim))
        v_cache = jax.random.normal(key, (num_blocks, batch_size, num_heads, block_size, head_dim))
        # Block table: which blocks are used for each sequence
        block_table = jnp.arange(num_blocks)
        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def paged_attention_fn(q, k_cache, v_cache, block_table):
            # Gather K and V from paged cache
            k = k_cache[block_table]  # (num_blocks, batch, heads, block_size, dim)
            v = v_cache[block_table]
            # Reshape to (batch, heads, seq, dim)
            k = k.transpose(1, 2, 0, 3, 4).reshape(batch_size, num_heads, -1, head_dim)
            v = v.transpose(1, 2, 0, 3, 4).reshape(batch_size, num_heads, -1, head_dim)
            # Standard attention
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        # Compile once
        _ = paged_attention_fn(q, k_cache, v_cache, block_table)

        def fn():
            return paged_attention_fn(q, k_cache, v_cache, block_table)

        name = f"jax_paged_attn_b{batch_size}_s{seq_length}_h{num_heads}_blk{block_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_block_allocation(
        self,
        num_sequences: int,
        max_blocks: int,
        block_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark block allocation (bitmap management simulation)."""
        key = jax.random.PRNGKey(42)
        # Free block bitmap (1 = free, 0 = used)
        free_blocks = jax.random.bernoulli(key, 0.5, (max_blocks,)).astype(jnp.int32)
        # Number of blocks to allocate per sequence
        blocks_needed = jax.random.randint(key, (num_sequences,), 1, 5)

        # Pre-compute static allocation sizes for JIT compatibility
        max_alloc_per_seq = 5  # Maximum blocks per sequence

        @jax.jit
        def block_allocation_fn(free_blocks, blocks_needed):
            # Find free block indices
            free_indices = jnp.where(free_blocks == 1, jnp.arange(max_blocks), max_blocks)
            free_indices = jnp.sort(free_indices)

            # Allocate blocks using static-sized outputs
            # Each sequence gets up to max_alloc_per_seq blocks
            def allocate_for_seq(carry, i):
                offset, free_idx = carry
                # Use lax.dynamic_slice with static slice size
                allocated = lax.dynamic_slice(free_idx, (offset,), (max_alloc_per_seq,))
                new_offset = offset + blocks_needed[i]
                return (new_offset, free_idx), allocated

            _, allocations = lax.scan(
                allocate_for_seq,
                (0, free_indices),
                jnp.arange(num_sequences)
            )

            # Flatten allocations and update bitmap
            flat_alloc = allocations.flatten()[:num_sequences * max_alloc_per_seq // 2]  # Take valid portion
            new_free = free_blocks.at[flat_alloc].set(0)
            return new_free, allocations

        # Compile once
        _ = block_allocation_fn(free_blocks, blocks_needed)

        def fn():
            return block_allocation_fn(free_blocks, blocks_needed)

        name = f"jax_block_alloc_nseq{num_sequences}_maxblk{max_blocks}_blksz{block_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_cache_eviction(
        self,
        cache_size: int,
        num_accesses: int,
        policy: str,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark cache eviction policies (LRU/LFU simulation)."""
        key = jax.random.PRNGKey(42)
        # Access timestamps for LRU (higher = more recent)
        timestamps = jax.random.randint(key, (cache_size,), 0, num_accesses)
        # Access counts for LFU
        access_counts = jax.random.randint(key, (cache_size,), 1, 100)
        # Number of entries to evict
        num_evict = cache_size // 4

        policy_idx = 0 if policy == "lru" else 1

        @jax.jit
        def cache_eviction_lru_fn(timestamps, access_counts):
            # Evict entries with lowest timestamps (least recently used)
            _, evict_indices = lax.top_k(-timestamps, num_evict)
            evict_mask = jnp.zeros(cache_size, dtype=jnp.bool_)
            evict_mask = evict_mask.at[evict_indices].set(True)
            return evict_indices, evict_mask

        @jax.jit
        def cache_eviction_lfu_fn(timestamps, access_counts):
            # Evict entries with lowest access counts (least frequently used)
            _, evict_indices = lax.top_k(-access_counts, num_evict)
            evict_mask = jnp.zeros(cache_size, dtype=jnp.bool_)
            evict_mask = evict_mask.at[evict_indices].set(True)
            return evict_indices, evict_mask

        # Select function based on policy (outside JIT)
        cache_eviction_fn = cache_eviction_lru_fn if policy_idx == 0 else cache_eviction_lfu_fn

        # Compile once
        _ = cache_eviction_fn(timestamps, access_counts)

        def fn():
            return cache_eviction_fn(timestamps, access_counts)

        name = f"jax_cache_eviction_sz{cache_size}_acc{num_accesses}_{policy}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_speculative_verification(
        self,
        batch_size: int,
        draft_length: int,
        num_heads: int,
        head_dim: int,
        vocab_size: int = 32000,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark speculative decoding verification."""
        key = jax.random.PRNGKey(42)
        # Draft token probabilities
        draft_probs = jax.random.uniform(key, (batch_size, draft_length, vocab_size))
        draft_probs = draft_probs / draft_probs.sum(axis=-1, keepdims=True)
        # Target model probabilities
        target_probs = jax.random.uniform(key, (batch_size, draft_length, vocab_size))
        target_probs = target_probs / target_probs.sum(axis=-1, keepdims=True)
        # Draft tokens
        draft_tokens = jax.random.randint(key, (batch_size, draft_length), 0, vocab_size)

        @jax.jit
        def speculative_verification_fn(draft_probs, target_probs, draft_tokens):
            # Get probabilities for draft tokens
            batch_idx = jnp.arange(batch_size)[:, None]
            seq_idx = jnp.arange(draft_length)[None, :]
            p_draft = draft_probs[batch_idx, seq_idx, draft_tokens]
            p_target = target_probs[batch_idx, seq_idx, draft_tokens]

            # Compute acceptance probability: min(1, p_target / p_draft)
            accept_prob = jnp.minimum(1.0, p_target / (p_draft + 1e-10))

            # Random values for acceptance decision
            rand_vals = jax.random.uniform(key, accept_prob.shape)
            accepted = rand_vals < accept_prob

            # Find first rejection position
            rejection_mask = ~accepted
            first_rejection = jnp.argmax(rejection_mask.astype(jnp.int32), axis=-1)
            all_accepted = ~rejection_mask.any(axis=-1)
            num_accepted = jnp.where(all_accepted, draft_length, first_rejection)

            return num_accepted, accepted

        # Compile once
        _ = speculative_verification_fn(draft_probs, target_probs, draft_tokens)

        def fn():
            return speculative_verification_fn(draft_probs, target_probs, draft_tokens)

        name = f"jax_spec_verify_b{batch_size}_draft{draft_length}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Generation Operations ==========

    def benchmark_temperature_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        temperature: float,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark temperature sampling."""
        key = jax.random.PRNGKey(42)
        logits = jax.random.normal(key, (batch_size, vocab_size))
        temp = jnp.array(temperature)

        @jax.jit
        def temperature_sampling_fn(logits, temp):
            # Always apply temperature scaling (handles temp=1.0 case naturally)
            return logits / temp

        # Compile once
        _ = temperature_sampling_fn(logits, temp)

        def fn():
            return temperature_sampling_fn(logits, temp)

        name = f"jax_temperature_b{batch_size}_v{vocab_size}_t{temperature}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_top_k_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        k: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark top-k sampling."""
        key = jax.random.PRNGKey(42)
        logits = jax.random.normal(key, (batch_size, vocab_size))

        @jax.jit
        def top_k_sampling_fn(logits):
            # k is captured as a constant, not traced
            # Get top-k values and their threshold
            topk_values, _ = lax.top_k(logits, k)
            threshold = topk_values[:, -1:]  # (batch, 1)
            # Mask values below threshold
            return jnp.where(logits >= threshold, logits, -1e9)

        # Compile once
        _ = top_k_sampling_fn(logits)

        def fn():
            return top_k_sampling_fn(logits)

        name = f"jax_top_k_b{batch_size}_v{vocab_size}_k{k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_top_p_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        p: float,
        warmup: int = 10,
        iterations: int = 100,
    ) -> JAXBenchmarkResult:
        """Benchmark top-p (nucleus) sampling."""
        key = jax.random.PRNGKey(42)
        logits = jax.random.normal(key, (batch_size, vocab_size))
        p_val = jnp.array(p)

        @jax.jit
        def top_p_sampling_fn(logits, p_val):
            # Sort logits descending
            sorted_indices = jnp.argsort(-logits, axis=-1)
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

            # Find cutoff: keep tokens until cumulative prob > p
            shifted = jnp.concatenate([
                jnp.zeros_like(cumulative_probs[:, :1]),
                cumulative_probs[:, :-1]
            ], axis=-1)
            mask = shifted < p_val
            mask = mask.at[:, 0].set(True)  # Always keep at least one token

            # Apply mask
            filtered = jnp.where(mask, sorted_logits, -1e9)
            # Unsort back to original order
            inverse = jnp.argsort(sorted_indices, axis=-1)
            return jnp.take_along_axis(filtered, inverse, axis=-1)

        # Compile once
        _ = top_p_sampling_fn(logits, p_val)

        def fn():
            return top_p_sampling_fn(logits, p_val)

        name = f"jax_top_p_b{batch_size}_v{vocab_size}_p{p}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Utility Methods ==========

    def run_all_benchmarks(
        self,
        size: str = "small",
    ) -> Dict[str, JAXBenchmarkResult]:
        """Run all extended benchmarks.

        Args:
            size: Size configuration ("tiny", "small", "medium", "large").

        Returns:
            Dictionary mapping benchmark names to results.
        """
        # Size configurations
        configs = {
            "tiny": {"batch_size": 1, "seq_length": 64, "hidden_dim": 128, "num_heads": 4, "head_dim": 32},
            "small": {"batch_size": 2, "seq_length": 256, "hidden_dim": 256, "num_heads": 8, "head_dim": 32},
            "medium": {"batch_size": 4, "seq_length": 512, "hidden_dim": 512, "num_heads": 8, "head_dim": 64},
            "large": {"batch_size": 8, "seq_length": 1024, "hidden_dim": 1024, "num_heads": 16, "head_dim": 64},
        }
        cfg = configs.get(size, configs["small"])

        results = {}
        categories = ["attention", "activation", "normalization", "fused", "quantization",
                      "primitive", "moe", "pooling", "embedding", "cache", "generation"]

        for category in categories:
            try:
                category_results = self.run_category_benchmarks(category, size)
                results.update(category_results)
            except Exception as e:
                print(f"Error running {category} benchmarks: {e}")

        return results

    def run_category_benchmarks(
        self,
        category: str,
        size: str = "small",
    ) -> Dict[str, JAXBenchmarkResult]:
        """Run benchmarks for a specific category.

        Args:
            category: Category name (e.g., "attention", "activation").
            size: Size configuration.

        Returns:
            Dictionary mapping benchmark names to results.
        """
        # Size configurations
        configs = {
            "tiny": {"batch_size": 1, "seq_length": 64, "hidden_dim": 128, "num_heads": 4, "head_dim": 32},
            "small": {"batch_size": 2, "seq_length": 256, "hidden_dim": 256, "num_heads": 8, "head_dim": 32},
            "medium": {"batch_size": 4, "seq_length": 512, "hidden_dim": 512, "num_heads": 8, "head_dim": 64},
            "large": {"batch_size": 8, "seq_length": 1024, "hidden_dim": 1024, "num_heads": 16, "head_dim": 64},
        }
        cfg = configs.get(size, configs["small"])
        b, s, d, h, hd = cfg["batch_size"], cfg["seq_length"], cfg["hidden_dim"], cfg["num_heads"], cfg["head_dim"]

        results = {}

        if category == "attention":
            results["flash_attention"] = self.benchmark_flash_attention(b, s, h, hd)
            results["sliding_window_attention"] = self.benchmark_sliding_window_attention(b, s, h, hd, s // 4)
            results["gqa"] = self.benchmark_gqa(b, s, h, h // 2, hd)
            results["mqa"] = self.benchmark_mqa(b, s, h, hd)
            results["linear_attention"] = self.benchmark_linear_attention(b, s, h, hd)
            results["alibi_attention"] = self.benchmark_alibi_attention(b, s, h, hd)
            results["rope_attention"] = self.benchmark_rope_attention(b, s, h, hd)

        elif category == "activation":
            results["swiglu"] = self.benchmark_swiglu(b, s, d)
            results["geglu"] = self.benchmark_geglu(b, s, d)
            results["reglu"] = self.benchmark_reglu(b, s, d)
            results["quick_gelu"] = self.benchmark_quick_gelu(b, s, d)
            results["mish"] = self.benchmark_mish(b, s, d)
            results["hard_swish"] = self.benchmark_hard_swish(b, s, d)

        elif category == "normalization":
            results["rmsnorm"] = self.benchmark_rmsnorm(b, s, d)
            results["layernorm"] = self.benchmark_layernorm(b, s, d)
            results["groupnorm"] = self.benchmark_groupnorm(b, 64, (32, 32), 8)
            results["instancenorm"] = self.benchmark_instancenorm(b, 64, (32, 32))
            results["adalayernorm"] = self.benchmark_adalayernorm(b, s, d)

        elif category == "fused":
            results["fused_rmsnorm_linear"] = self.benchmark_fused_rmsnorm_linear(b, s, d, d)
            results["fused_swiglu"] = self.benchmark_fused_swiglu(b, s, d)
            results["fused_geglu"] = self.benchmark_fused_geglu(b, s, d)
            results["fused_rope_attention"] = self.benchmark_fused_rope_attention(b, s, h, hd)

        elif category == "quantization":
            results["int8_quantize"] = self.benchmark_int8_quantize(d, d)
            results["int8_dequantize"] = self.benchmark_int8_dequantize(d, d)
            results["int4_quantize"] = self.benchmark_int4_quantize(d, d)
            results["int4_dequantize"] = self.benchmark_int4_dequantize(d, d)
            results["int8_linear"] = self.benchmark_int8_linear(b, s, d, d)
            results["int4_linear"] = self.benchmark_int4_linear(b, s, d, d)

        elif category == "primitive":
            results["assoc_scan_add"] = self.benchmark_associative_scan_add(b, s, d)
            results["assoc_scan_mul"] = self.benchmark_associative_scan_mul(b, s, d)
            results["assoc_scan_ssm"] = self.benchmark_associative_scan_ssm(b, s, 16)
            results["selective_scan"] = self.benchmark_selective_scan(b, s // 4, d, 16)
            results["selective_gather"] = self.benchmark_selective_gather(b, s, d, s // 4)
            results["selective_scatter_add"] = self.benchmark_selective_scatter_add(b, s, d, s // 4)

        elif category == "moe":
            results["topk_routing"] = self.benchmark_topk_routing(b, s, 8, 2, d)
            results["expert_dispatch"] = self.benchmark_expert_dispatch(b, s // 4, d, 4, 2)
            results["load_balancing_loss"] = self.benchmark_load_balancing_loss(b, s, 8, 2, d)

        elif category == "pooling":
            results["adaptive_avg_pool1d"] = self.benchmark_adaptive_avg_pool1d(b, 64, 256, 16)
            results["adaptive_avg_pool2d"] = self.benchmark_adaptive_avg_pool2d(b, 64, 64, 64, (8, 8))
            results["global_attention_pooling"] = self.benchmark_global_attention_pooling(b, s, d)
            results["gem"] = self.benchmark_gem(b, 64, 32, 32)
            results["spp"] = self.benchmark_spatial_pyramid_pooling(b, 64, 32, 32, [1, 2, 4])

        elif category == "embedding":
            results["sinusoidal_embedding"] = self.benchmark_sinusoidal_embedding(b, s, d)
            results["learned_positional_embedding"] = self.benchmark_learned_positional_embedding(b, s, d, 2048)
            results["rotary_embedding"] = self.benchmark_rotary_embedding(b, s, h, hd)
            results["alibi_embedding"] = self.benchmark_alibi_embedding(b, s, h)
            results["relative_positional_embedding"] = self.benchmark_relative_positional_embedding(b, s, s, h, hd)

        elif category == "cache":
            results["paged_attention"] = self.benchmark_paged_attention(b, s, h, hd, 16)
            results["block_allocation"] = self.benchmark_block_allocation(16, 128, 16)
            results["cache_eviction_lru"] = self.benchmark_cache_eviction(256, 1000, "lru")
            results["cache_eviction_lfu"] = self.benchmark_cache_eviction(256, 1000, "lfu")
            results["speculative_verification"] = self.benchmark_speculative_verification(b, 8, h, hd)

        elif category == "generation":
            results["temperature_sampling"] = self.benchmark_temperature_sampling(b, 32000, 0.7)
            results["top_k_sampling"] = self.benchmark_top_k_sampling(b, 32000, 50)
            results["top_p_sampling"] = self.benchmark_top_p_sampling(b, 32000, 0.9)

        return results
