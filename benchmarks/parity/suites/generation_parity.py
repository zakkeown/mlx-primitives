"""Generation/sampling parity benchmarks."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

import mlx.core as mx

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
    from jax import random as jrandom
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    jnn = None
    jrandom = None


class GenerationParityBenchmarks:
    """Multi-framework generation benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all generation parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            results[f"temperature_sampling_{size}"] = self._benchmark_to_list(
                self.benchmark_temperature_sampling(size)
            )
            results[f"top_k_sampling_{size}"] = self._benchmark_to_list(
                self.benchmark_top_k_sampling(size)
            )
            results[f"top_p_sampling_{size}"] = self._benchmark_to_list(
                self.benchmark_top_p_sampling(size)
            )
            results[f"combined_sampling_{size}"] = self._benchmark_to_list(
                self.benchmark_combined_sampling(size)
            )

        # Add scaling analysis
        scaling_results = self.run_vocab_size_scaling("combined")
        for key, value in scaling_results.items():
            results[key] = value

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
    # Temperature Sampling
    # =========================================================================

    def benchmark_temperature_sampling(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark temperature-scaled sampling across frameworks."""
        from mlx_primitives.generation.samplers import apply_temperature

        config = self.sizes.get_config("generation", size)
        batch, vocab_size = config
        temperature = 0.7
        results = {}

        # MLX benchmark
        logits_mlx = mx.random.normal((batch, vocab_size))
        mx.eval(logits_mlx)

        def mlx_temperature_sample():
            scaled = apply_temperature(logits_mlx, temperature)
            return mx.random.categorical(scaled)

        results["mlx"] = self._benchmark_mlx(
            mlx_temperature_sample,
            name=f"temperature_sampling_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            logits_torch = torch.randn(batch, vocab_size, device="mps")

            def pytorch_temperature_sample():
                scaled = logits_torch / temperature
                probs = F.softmax(scaled, dim=-1)
                return torch.multinomial(probs, num_samples=1)

            result = self._benchmark_pytorch(
                pytorch_temperature_sample,
                name=f"temperature_sampling_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            logits_jax = jnp.array(np.random.randn(batch, vocab_size).astype(np.float32))
            key = jrandom.PRNGKey(42)

            @jax.jit
            def jax_temperature_sample():
                scaled = logits_jax / temperature
                return jrandom.categorical(key, scaled, axis=-1)

            result = self._benchmark_jax(
                jax_temperature_sample,
                name=f"temperature_sampling_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Top-K Sampling
    # =========================================================================

    def benchmark_top_k_sampling(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark top-k sampling across frameworks."""
        from mlx_primitives.generation.samplers import apply_top_k

        config = self.sizes.get_config("generation", size)
        batch, vocab_size = config
        k = 50
        results = {}

        # MLX benchmark
        logits_mlx = mx.random.normal((batch, vocab_size))
        mx.eval(logits_mlx)

        def mlx_top_k_sample():
            filtered = apply_top_k(logits_mlx, k)
            return mx.random.categorical(filtered)

        results["mlx"] = self._benchmark_mlx(
            mlx_top_k_sample,
            name=f"top_k_sampling_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            logits_torch = torch.randn(batch, vocab_size, device="mps")

            def pytorch_top_k_sample():
                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(logits_torch, k, dim=-1)
                # Create mask for non-top-k values
                filtered = torch.full_like(logits_torch, float('-inf'))
                filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                probs = F.softmax(filtered, dim=-1)
                return torch.multinomial(probs, num_samples=1)

            result = self._benchmark_pytorch(
                pytorch_top_k_sample,
                name=f"top_k_sampling_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            logits_jax = jnp.array(np.random.randn(batch, vocab_size).astype(np.float32))
            key = jrandom.PRNGKey(42)

            @jax.jit
            def jax_top_k_sample():
                # Get k-th largest value
                sorted_logits = jnp.sort(logits_jax, axis=-1)
                threshold = sorted_logits[:, -k : -k + 1]
                # Mask values below threshold
                filtered = jnp.where(logits_jax >= threshold, logits_jax, float('-inf'))
                return jrandom.categorical(key, filtered, axis=-1)

            result = self._benchmark_jax(
                jax_top_k_sample,
                name=f"top_k_sampling_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Top-P (Nucleus) Sampling
    # =========================================================================

    def benchmark_top_p_sampling(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark top-p (nucleus) sampling across frameworks."""
        from mlx_primitives.generation.samplers import apply_top_p

        config = self.sizes.get_config("generation", size)
        batch, vocab_size = config
        p = 0.9
        results = {}

        # MLX benchmark
        logits_mlx = mx.random.normal((batch, vocab_size))
        mx.eval(logits_mlx)

        def mlx_top_p_sample():
            filtered = apply_top_p(logits_mlx, p)
            return mx.random.categorical(filtered)

        results["mlx"] = self._benchmark_mlx(
            mlx_top_p_sample,
            name=f"top_p_sampling_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            logits_torch = torch.randn(batch, vocab_size, device="mps")

            def pytorch_top_p_sample():
                # Sort by descending probability
                sorted_indices = torch.argsort(logits_torch, dim=-1, descending=True)
                sorted_logits = torch.gather(logits_torch, -1, sorted_indices)

                # Compute cumulative probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Find cutoff
                shifted_cumulative = torch.cat(
                    [torch.zeros(batch, 1, device="mps"), cumulative_probs[:, :-1]],
                    dim=-1
                )
                sorted_mask = shifted_cumulative < p

                # Ensure at least the top token is kept
                sorted_mask[:, 0] = True

                # Set filtered positions to -inf
                sorted_logits = torch.where(sorted_mask, sorted_logits, torch.tensor(float('-inf'), device="mps"))

                # Unsort back to original order
                inverse_indices = torch.argsort(sorted_indices, dim=-1)
                filtered = torch.gather(sorted_logits, -1, inverse_indices)

                probs = F.softmax(filtered, dim=-1)
                return torch.multinomial(probs, num_samples=1)

            result = self._benchmark_pytorch(
                pytorch_top_p_sample,
                name=f"top_p_sampling_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            logits_jax = jnp.array(np.random.randn(batch, vocab_size).astype(np.float32))
            key = jrandom.PRNGKey(42)

            @jax.jit
            def jax_top_p_sample():
                # Sort by descending probability
                sorted_indices = jnp.argsort(logits_jax, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(logits_jax, sorted_indices, axis=-1)

                # Compute cumulative probabilities in fp32 for stability
                sorted_probs = jnn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

                # Find cutoff
                shifted_cumulative = jnp.concatenate(
                    [jnp.zeros((batch, 1)), cumulative_probs[:, :-1]],
                    axis=-1
                )
                sorted_mask = shifted_cumulative < p

                # Ensure at least the top token is kept
                first_token_mask = jnp.zeros_like(sorted_mask).at[:, 0].set(True)
                sorted_mask = jnp.logical_or(sorted_mask, first_token_mask)

                # Set filtered positions to -inf
                sorted_logits = jnp.where(sorted_mask, sorted_logits, float('-inf'))

                # Unsort back to original order
                inverse_indices = jnp.argsort(sorted_indices, axis=-1)
                filtered = jnp.take_along_axis(sorted_logits, inverse_indices, axis=-1)

                return jrandom.categorical(key, filtered, axis=-1)

            result = self._benchmark_jax(
                jax_top_p_sample,
                name=f"top_p_sampling_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Combined Sampling (Temperature + Top-K + Top-P)
    # =========================================================================

    def benchmark_combined_sampling(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark combined sampling (temp + top-k + top-p) across frameworks."""
        from mlx_primitives.generation.samplers import (
            apply_temperature,
            apply_top_k,
            apply_top_p,
        )

        config = self.sizes.get_config("generation", size)
        batch, vocab_size = config
        temperature = 0.7
        k = 50
        p = 0.9
        results = {}

        # MLX benchmark
        logits_mlx = mx.random.normal((batch, vocab_size))
        mx.eval(logits_mlx)

        def mlx_combined_sample():
            scaled = apply_temperature(logits_mlx, temperature)
            top_k_filtered = apply_top_k(scaled, k)
            top_p_filtered = apply_top_p(top_k_filtered, p)
            return mx.random.categorical(top_p_filtered)

        results["mlx"] = self._benchmark_mlx(
            mlx_combined_sample,
            name=f"combined_sampling_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            logits_torch = torch.randn(batch, vocab_size, device="mps")

            def pytorch_combined_sample():
                # Temperature
                scaled = logits_torch / temperature

                # Top-k
                top_k_values, top_k_indices = torch.topk(scaled, k, dim=-1)
                top_k_filtered = torch.full_like(scaled, float('-inf'))
                top_k_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

                # Top-p
                sorted_indices = torch.argsort(top_k_filtered, dim=-1, descending=True)
                sorted_logits = torch.gather(top_k_filtered, -1, sorted_indices)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                shifted_cumulative = torch.cat(
                    [torch.zeros(batch, 1, device="mps"), cumulative_probs[:, :-1]],
                    dim=-1
                )
                sorted_mask = shifted_cumulative < p
                sorted_mask[:, 0] = True
                sorted_logits = torch.where(sorted_mask, sorted_logits, torch.tensor(float('-inf'), device="mps"))
                inverse_indices = torch.argsort(sorted_indices, dim=-1)
                filtered = torch.gather(sorted_logits, -1, inverse_indices)

                probs = F.softmax(filtered, dim=-1)
                return torch.multinomial(probs, num_samples=1)

            result = self._benchmark_pytorch(
                pytorch_combined_sample,
                name=f"combined_sampling_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            logits_jax = jnp.array(np.random.randn(batch, vocab_size).astype(np.float32))
            key = jrandom.PRNGKey(42)

            @jax.jit
            def jax_combined_sample():
                # Temperature
                scaled = logits_jax / temperature

                # Top-k
                sorted_logits = jnp.sort(scaled, axis=-1)
                threshold = sorted_logits[:, -k : -k + 1]
                top_k_filtered = jnp.where(scaled >= threshold, scaled, float('-inf'))

                # Top-p
                sorted_indices = jnp.argsort(top_k_filtered, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(top_k_filtered, sorted_indices, axis=-1)
                sorted_probs = jnn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                shifted_cumulative = jnp.concatenate(
                    [jnp.zeros((batch, 1)), cumulative_probs[:, :-1]],
                    axis=-1
                )
                sorted_mask = shifted_cumulative < p
                first_token_mask = jnp.zeros_like(sorted_mask).at[:, 0].set(True)
                sorted_mask = jnp.logical_or(sorted_mask, first_token_mask)
                sorted_logits = jnp.where(sorted_mask, sorted_logits, float('-inf'))
                inverse_indices = jnp.argsort(sorted_indices, axis=-1)
                filtered = jnp.take_along_axis(sorted_logits, inverse_indices, axis=-1)

                return jrandom.categorical(key, filtered, axis=-1)

            result = self._benchmark_jax(
                jax_combined_sample,
                name=f"combined_sampling_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Vocabulary Size Scaling Analysis
    # =========================================================================

    def run_vocab_size_scaling(self, operation: str) -> Dict[str, List[BenchmarkResult]]:
        """Run scaling analysis across vocabulary sizes."""
        from mlx_primitives.generation.samplers import (
            apply_temperature,
            apply_top_k,
            apply_top_p,
        )

        batch = 8
        vocab_sizes = [1000, 10000, 32000, 50000, 100000, 128000]
        temperature = 0.7
        k = 50
        p = 0.9
        results = {}

        for vocab_size in vocab_sizes:
            key_name = f"vocab_scaling_{vocab_size}"
            results[key_name] = []

            # MLX benchmark
            logits_mlx = mx.random.normal((batch, vocab_size))
            mx.eval(logits_mlx)

            def mlx_combined():
                scaled = apply_temperature(logits_mlx, temperature)
                top_k_filtered = apply_top_k(scaled, k)
                top_p_filtered = apply_top_p(top_k_filtered, p)
                return mx.random.categorical(top_p_filtered)

            mlx_result = self._benchmark_mlx(
                mlx_combined,
                name=f"combined_vocab_{vocab_size}_mlx",
            )
            results[key_name].append(mlx_result)

            # PyTorch MPS benchmark
            if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
                logits_torch = torch.randn(batch, vocab_size, device="mps")

                def pytorch_combined():
                    scaled = logits_torch / temperature
                    top_k_values, top_k_indices = torch.topk(scaled, k, dim=-1)
                    top_k_filtered = torch.full_like(scaled, float('-inf'))
                    top_k_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

                    sorted_indices = torch.argsort(top_k_filtered, dim=-1, descending=True)
                    sorted_logits = torch.gather(top_k_filtered, -1, sorted_indices)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    shifted_cumulative = torch.cat(
                        [torch.zeros(batch, 1, device="mps"), cumulative_probs[:, :-1]],
                        dim=-1
                    )
                    sorted_mask = shifted_cumulative < p
                    sorted_mask[:, 0] = True
                    sorted_logits = torch.where(sorted_mask, sorted_logits, torch.tensor(float('-inf'), device="mps"))
                    inverse_indices = torch.argsort(sorted_indices, dim=-1)
                    filtered = torch.gather(sorted_logits, -1, inverse_indices)

                    probs = F.softmax(filtered, dim=-1)
                    return torch.multinomial(probs, num_samples=1)

                pytorch_result = self._benchmark_pytorch(
                    pytorch_combined,
                    name=f"combined_vocab_{vocab_size}_pytorch_mps",
                )
                if pytorch_result:
                    results[key_name].append(pytorch_result)

            # JAX benchmark
            if HAS_JAX and self.config.include_jax:
                np.random.seed(42)
                logits_jax = jnp.array(np.random.randn(batch, vocab_size).astype(np.float32))
                key = jrandom.PRNGKey(42)

                @jax.jit
                def jax_combined():
                    scaled = logits_jax / temperature
                    sorted_logits_k = jnp.sort(scaled, axis=-1)
                    threshold = sorted_logits_k[:, -k : -k + 1]
                    top_k_filtered = jnp.where(scaled >= threshold, scaled, float('-inf'))

                    sorted_indices = jnp.argsort(top_k_filtered, axis=-1)[:, ::-1]
                    sorted_logits = jnp.take_along_axis(top_k_filtered, sorted_indices, axis=-1)
                    sorted_probs = jnn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
                    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                    shifted_cumulative = jnp.concatenate(
                        [jnp.zeros((batch, 1)), cumulative_probs[:, :-1]],
                        axis=-1
                    )
                    sorted_mask = shifted_cumulative < p
                    first_token_mask = jnp.zeros_like(sorted_mask).at[:, 0].set(True)
                    sorted_mask = jnp.logical_or(sorted_mask, first_token_mask)
                    sorted_logits = jnp.where(sorted_mask, sorted_logits, float('-inf'))
                    inverse_indices = jnp.argsort(sorted_indices, axis=-1)
                    filtered = jnp.take_along_axis(sorted_logits, inverse_indices, axis=-1)

                    return jrandom.categorical(key, filtered, axis=-1)

                jax_result = self._benchmark_jax(
                    jax_combined,
                    name=f"combined_vocab_{vocab_size}_jax",
                )
                if jax_result:
                    results[key_name].append(jax_result)

        return results
