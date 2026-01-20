"""JAX Metal baseline implementations for performance comparison.

This module provides JAX implementations running on Apple Metal
for comparing against MLX implementations.
"""

import time
from typing import Callable, Optional
from dataclasses import dataclass

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
    # Check for GPU/Metal backend
    _devices = jax.devices()
    HAS_METAL = any("gpu" in str(d).lower() or "metal" in str(d).lower() for d in _devices)
except ImportError:
    HAS_JAX = False
    HAS_METAL = False
    jax = None
    jnp = None
    lax = None
    np = None


def jax_available() -> bool:
    """Check if JAX with Metal/GPU is available."""
    return HAS_JAX and HAS_METAL


@dataclass
class JAXBenchmarkResult:
    """Result from a JAX benchmark."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int


class JAXMetalBenchmarks:
    """JAX Metal baseline benchmarks for comparison with MLX."""

    def __init__(self):
        """Initialize JAX Metal benchmarks.

        Raises:
            RuntimeError: If JAX with Metal is not available.
        """
        if not jax_available():
            raise RuntimeError("JAX with Metal/GPU not available")

    def _benchmark(
        self,
        fn: Callable,
        iterations: int = 30,
        warmup_iterations: int = 5,
        name: str = "benchmark",
    ) -> JAXBenchmarkResult:
        """Run a benchmark with warmup and timing.

        Args:
            fn: Function to benchmark (should be jitted).
            iterations: Number of timed iterations.
            warmup_iterations: Number of warmup iterations.
            name: Benchmark name.

        Returns:
            Benchmark result with timing statistics.
        """
        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, tuple):
                for r in result:
                    if hasattr(r, "block_until_ready"):
                        r.block_until_ready()

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, tuple):
                for r in result:
                    if hasattr(r, "block_until_ready"):
                        r.block_until_ready()
            times.append(time.perf_counter() - start)

        return JAXBenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
        )

    def benchmark_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        causal: bool = False,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX attention.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            causal: Whether to use causal masking.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        # JAX uses (batch, seq_len, num_heads, head_dim) or we transpose
        q = jax.random.normal(key, (batch_size, num_heads, seq_len, head_dim))
        k = jax.random.normal(key, (batch_size, num_heads, seq_len, head_dim))
        v = jax.random.normal(key, (batch_size, num_heads, seq_len, head_dim))

        scale = 1.0 / jnp.sqrt(head_dim)

        @jax.jit
        def attention_fn(q, k, v):
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if causal:
                mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                scores = jnp.where(mask, scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        # Compile once
        _ = attention_fn(q, k, v)

        def fn():
            return attention_fn(q, k, v)

        name = f"jax_attention_b{batch_size}_s{seq_len}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_layer_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX LayerNorm.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
        weight = jnp.ones(hidden_dim)
        bias = jnp.zeros(hidden_dim)
        eps = 1e-5

        @jax.jit
        def layer_norm_fn(x, weight, bias):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return weight * (x - mean) / jnp.sqrt(var + eps) + bias

        # Compile once
        _ = layer_norm_fn(x, weight, bias)

        def fn():
            return layer_norm_fn(x, weight, bias)

        name = f"jax_layernorm_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_associative_scan(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX associative scan (parallel prefix sum).

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            dim: Feature dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        x = jax.random.normal(key, (batch_size, seq_len, dim))

        @jax.jit
        def scan_fn(x):
            return lax.associative_scan(jnp.add, x, axis=1)

        # Compile once
        _ = scan_fn(x)

        def fn():
            return scan_fn(x)

        name = f"jax_assoc_scan_b{batch_size}_s{seq_len}_d{dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_matmul(
        self,
        m: int,
        n: int,
        k: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX matrix multiplication.

        Args:
            m: First matrix rows.
            n: Second matrix columns.
            k: Shared dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        a = jax.random.normal(key, (m, k))
        b = jax.random.normal(key, (k, n))

        @jax.jit
        def matmul_fn(a, b):
            return jnp.matmul(a, b)

        # Compile once
        _ = matmul_fn(a, b)

        def fn():
            return matmul_fn(a, b)

        name = f"jax_matmul_{m}x{k}x{n}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_softmax(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX softmax.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            vocab_size: Vocabulary size.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))

        @jax.jit
        def softmax_fn(x):
            return jax.nn.softmax(x, axis=-1)

        # Compile once
        _ = softmax_fn(logits)

        def fn():
            return softmax_fn(logits)

        name = f"jax_softmax_b{batch_size}_s{seq_len}_v{vocab_size}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_gelu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX GELU activation.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

        @jax.jit
        def gelu_fn(x):
            return jax.nn.gelu(x, approximate=True)

        # Compile once
        _ = gelu_fn(x)

        def fn():
            return gelu_fn(x)

        name = f"jax_gelu_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_silu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> JAXBenchmarkResult:
        """Benchmark JAX SiLU (Swish) activation.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        key = jax.random.PRNGKey(42)

        x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

        @jax.jit
        def silu_fn(x):
            return jax.nn.silu(x)

        # Compile once
        _ = silu_fn(x)

        def fn():
            return silu_fn(x)

        name = f"jax_silu_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)
