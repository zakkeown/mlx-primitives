"""Primitives (scan, gather, scatter) parity benchmarks."""

import time
from typing import Any, Dict, List, Optional

import mlx.core as mx

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig
from benchmarks.parity.runner import BenchmarkResult

# Check for PyTorch availability
try:
    import torch
    HAS_PYTORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False
    HAS_MPS = False

# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _benchmark_mlx_fn(fn, iterations: int, warmup: int = 5) -> tuple:
    """Benchmark an MLX function, returning (mean, std, min, max)."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)

    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
        mx.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    import statistics
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, min(times), max(times)


def _benchmark_torch_fn(fn, iterations: int, warmup: int = 5) -> tuple:
    """Benchmark a PyTorch function, returning (mean, std, min, max)."""
    if not HAS_PYTORCH:
        return 0.0, 0.0, 0.0, 0.0

    # Warmup
    for _ in range(warmup):
        result = fn()
        if HAS_MPS:
            torch.mps.synchronize()

    times = []
    for _ in range(iterations):
        if HAS_MPS:
            torch.mps.synchronize()
        start = time.perf_counter()
        result = fn()
        if HAS_MPS:
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    import statistics
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, min(times), max(times)


def _benchmark_jax_fn(fn, iterations: int, warmup: int = 5) -> tuple:
    """Benchmark a JAX function, returning (mean, std, min, max)."""
    if not HAS_JAX:
        return 0.0, 0.0, 0.0, 0.0

    # Warmup
    for _ in range(warmup):
        result = fn()
        if isinstance(result, jax.Array):
            result.block_until_ready()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if isinstance(result, jax.Array):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    import statistics
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, min(times), max(times)


class PrimitivesParityBenchmarks:
    """Multi-framework primitives benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.size_config = ParitySizeConfig()
        self.iterations = self.config.benchmark_iterations
        self.warmup = self.config.warmup_iterations
        self.sizes = ["tiny", "small", "medium", "large"]

        # Set up PyTorch device
        if HAS_PYTORCH and HAS_MPS:
            self.torch_device = torch.device("mps")
        elif HAS_PYTORCH:
            self.torch_device = torch.device("cpu")
        else:
            self.torch_device = None

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all primitives parity benchmarks."""
        results: Dict[str, List[BenchmarkResult]] = {
            "mlx": [],
            "pytorch_mps": [],
            "jax_metal": [],
        }

        for size in self.sizes:
            # Associative scan add (cumsum)
            scan_add_results = self.benchmark_associative_scan_add(size)
            for framework, result in scan_add_results.items():
                if result is not None:
                    results[framework].append(result)

            # Associative scan mul (cumprod)
            scan_mul_results = self.benchmark_associative_scan_mul(size)
            for framework, result in scan_mul_results.items():
                if result is not None:
                    results[framework].append(result)

            # SSM scan
            ssm_results = self.benchmark_associative_scan_ssm(size)
            for framework, result in ssm_results.items():
                if result is not None:
                    results[framework].append(result)

            # Selective scan (Mamba)
            selective_results = self.benchmark_selective_scan(size)
            for framework, result in selective_results.items():
                if result is not None:
                    results[framework].append(result)

            # Gather
            gather_results = self.benchmark_selective_gather(size)
            for framework, result in gather_results.items():
                if result is not None:
                    results[framework].append(result)

            # Scatter add
            scatter_results = self.benchmark_selective_scatter_add(size)
            for framework, result in scatter_results.items():
                if result is not None:
                    results[framework].append(result)

        return results

    def benchmark_associative_scan_add(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark cumulative sum across frameworks."""
        from mlx_primitives.primitives import associative_scan

        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        def mlx_fn():
            return associative_scan(x_mlx, operator="add", axis=1)

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"cumsum_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, dim, device=self.torch_device)

            def torch_fn():
                return torch.cumsum(x_torch, dim=1)

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_fn, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"cumsum_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            key = jax.random.PRNGKey(42)
            x_jax = jax.random.normal(key, (batch, seq, dim))

            @jax.jit
            def jax_fn():
                return jnp.cumsum(x_jax, axis=1)

            mean, std, min_t, max_t = _benchmark_jax_fn(jax_fn, self.iterations, self.warmup)
            results["jax_metal"] = BenchmarkResult(
                name=f"cumsum_{size}",
                framework="jax_metal",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results

    def benchmark_associative_scan_mul(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark cumulative product across frameworks."""
        from mlx_primitives.primitives import associative_scan

        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config

        results = {}

        # MLX benchmark - use values close to 1 to avoid overflow
        mx.random.seed(42)
        x_mlx = 0.99 + 0.02 * mx.random.uniform((batch, seq, dim))
        mx.eval(x_mlx)

        def mlx_fn():
            return associative_scan(x_mlx, operator="mul", axis=1)

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"cumprod_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = 0.99 + 0.02 * torch.rand(batch, seq, dim, device=self.torch_device)

            def torch_fn():
                return torch.cumprod(x_torch, dim=1)

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_fn, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"cumprod_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            key = jax.random.PRNGKey(42)
            x_jax = 0.99 + 0.02 * jax.random.uniform(key, (batch, seq, dim))

            @jax.jit
            def jax_fn():
                return jnp.cumprod(x_jax, axis=1)

            mean, std, min_t, max_t = _benchmark_jax_fn(jax_fn, self.iterations, self.warmup)
            results["jax_metal"] = BenchmarkResult(
                name=f"cumprod_{size}",
                framework="jax_metal",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results

    def benchmark_associative_scan_ssm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark SSM scan h[t] = A[t]*h[t-1] + x[t] across frameworks."""
        from mlx_primitives.primitives import associative_scan

        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, dim))
        # A should be in (0, 1) for stability (decay factors)
        A_mlx = 0.9 + 0.09 * mx.random.uniform((batch, seq, dim))
        mx.eval(x_mlx, A_mlx)

        def mlx_fn():
            return associative_scan(x_mlx, operator="ssm", A=A_mlx, axis=1)

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"ssm_scan_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark - implement sequential SSM scan as reference
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, dim, device=self.torch_device)
            A_torch = 0.9 + 0.09 * torch.rand(batch, seq, dim, device=self.torch_device)

            def torch_ssm_scan():
                """Sequential SSM scan in PyTorch."""
                h = torch.zeros(batch, dim, device=self.torch_device)
                outputs = []
                for t in range(seq):
                    h = A_torch[:, t, :] * h + x_torch[:, t, :]
                    outputs.append(h)
                return torch.stack(outputs, dim=1)

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_ssm_scan, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"ssm_scan_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        # JAX benchmark - using lax.scan for SSM
        if HAS_JAX and self.config.include_jax:
            key = jax.random.PRNGKey(42)
            key1, key2 = jax.random.split(key)
            x_jax = jax.random.normal(key1, (batch, seq, dim))
            A_jax = 0.9 + 0.09 * jax.random.uniform(key2, (batch, seq, dim))

            @jax.jit
            def jax_ssm_scan():
                """SSM scan using jax.lax.scan."""
                def step(h, inputs):
                    a, x = inputs
                    h_new = a * h + x
                    return h_new, h_new

                # Transpose for scan: (batch, seq, dim) -> (seq, batch, dim)
                A_t = jnp.transpose(A_jax, (1, 0, 2))
                x_t = jnp.transpose(x_jax, (1, 0, 2))

                h0 = jnp.zeros((batch, dim))
                _, outputs = jax.lax.scan(step, h0, (A_t, x_t))

                # Transpose back: (seq, batch, dim) -> (batch, seq, dim)
                return jnp.transpose(outputs, (1, 0, 2))

            mean, std, min_t, max_t = _benchmark_jax_fn(jax_ssm_scan, self.iterations, self.warmup)
            results["jax_metal"] = BenchmarkResult(
                name=f"ssm_scan_{size}",
                framework="jax_metal",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results

    def benchmark_selective_scan(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Mamba-style selective scan across frameworks."""
        from mlx_primitives.primitives import selective_scan

        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config
        d_state = 16  # Typical state dimension for Mamba
        d_inner = dim * 2  # Expansion factor

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, d_inner))
        delta_mlx = mx.abs(mx.random.normal((batch, seq, d_inner))) + 0.01
        A_mlx = -mx.abs(mx.random.normal((d_inner, d_state)))  # Negative for stability
        B_mlx = mx.random.normal((batch, seq, d_state)) * 0.1
        C_mlx = mx.random.normal((batch, seq, d_state)) * 0.1
        D_mlx = mx.random.normal((d_inner,)) * 0.1
        mx.eval(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx)

        def mlx_fn():
            return selective_scan(x_mlx, delta_mlx, A_mlx, B_mlx, C_mlx, D_mlx)

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"selective_scan_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark - implement reference selective scan
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, d_inner, device=self.torch_device)
            delta_torch = torch.abs(torch.randn(batch, seq, d_inner, device=self.torch_device)) + 0.01
            A_torch = -torch.abs(torch.randn(d_inner, d_state, device=self.torch_device))
            B_torch = torch.randn(batch, seq, d_state, device=self.torch_device) * 0.1
            C_torch = torch.randn(batch, seq, d_state, device=self.torch_device) * 0.1
            D_torch = torch.randn(d_inner, device=self.torch_device) * 0.1

            def torch_selective_scan():
                """Reference selective scan in PyTorch."""
                # Discretize
                delta_A = delta_torch.unsqueeze(-1) * A_torch.unsqueeze(0).unsqueeze(0)
                A_bar = torch.exp(delta_A)
                B_bar = delta_torch.unsqueeze(-1) * B_torch.unsqueeze(2)

                # Sequential scan
                h = torch.zeros(batch, d_inner, d_state, device=self.torch_device)
                outputs = []
                for t in range(seq):
                    h = A_bar[:, t] * h + B_bar[:, t] * x_torch[:, t].unsqueeze(-1)
                    y_t = torch.sum(h * C_torch[:, t].unsqueeze(1), dim=-1)
                    outputs.append(y_t)

                y = torch.stack(outputs, dim=1)
                return y + x_torch * D_torch.unsqueeze(0).unsqueeze(0)

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_selective_scan, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"selective_scan_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results

    def benchmark_selective_gather(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark gather/indexing operations across frameworks."""
        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config
        num_indices = seq // 4  # Gather 25% of elements

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, dim))
        indices_mlx = mx.random.randint(0, seq, (batch, num_indices))
        mx.eval(x_mlx, indices_mlx)

        def mlx_fn():
            # Gather along sequence dimension
            return mx.take_along_axis(x_mlx, indices_mlx[:, :, None], axis=1)

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"gather_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, dim, device=self.torch_device)
            indices_torch = torch.randint(0, seq, (batch, num_indices), device=self.torch_device)

            def torch_fn():
                return torch.gather(x_torch, 1, indices_torch.unsqueeze(-1).expand(-1, -1, dim))

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_fn, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"gather_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            key = jax.random.PRNGKey(42)
            key1, key2 = jax.random.split(key)
            x_jax = jax.random.normal(key1, (batch, seq, dim))
            indices_jax = jax.random.randint(key2, (batch, num_indices), 0, seq)

            @jax.jit
            def jax_fn():
                return jnp.take_along_axis(x_jax, indices_jax[:, :, None], axis=1)

            mean, std, min_t, max_t = _benchmark_jax_fn(jax_fn, self.iterations, self.warmup)
            results["jax_metal"] = BenchmarkResult(
                name=f"gather_{size}",
                framework="jax_metal",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results

    def benchmark_selective_scatter_add(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark scatter_add operations across frameworks."""
        config = self.size_config.get_config("scan", size)
        if config is None:
            return {}
        batch, seq, dim = config
        num_updates = seq // 4  # Scatter 25% of elements

        results = {}

        # MLX benchmark - use scatter approach
        mx.random.seed(42)
        x_mlx = mx.zeros((batch, seq, dim))
        indices_mlx = mx.random.randint(0, seq, (batch, num_updates))
        updates_mlx = mx.random.normal((batch, num_updates, dim))
        mx.eval(x_mlx, indices_mlx, updates_mlx)

        def mlx_fn():
            # Scatter add using advanced indexing
            # MLX doesn't have direct scatter_add, use loop or at[].add
            result = x_mlx.at[mx.arange(batch)[:, None], indices_mlx].add(updates_mlx)
            return result

        mean, std, min_t, max_t = _benchmark_mlx_fn(mlx_fn, self.iterations, self.warmup)
        results["mlx"] = BenchmarkResult(
            name=f"scatter_add_{size}",
            framework="mlx",
            mean_time=mean,
            std_time=std,
            min_time=min_t,
            max_time=max_t,
            iterations=self.iterations,
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and self.config.include_pytorch:
            torch.manual_seed(42)
            indices_torch = torch.randint(0, seq, (batch, num_updates), device=self.torch_device)
            updates_torch = torch.randn(batch, num_updates, dim, device=self.torch_device)

            def torch_fn():
                x = torch.zeros(batch, seq, dim, device=self.torch_device)
                return x.scatter_add(1, indices_torch.unsqueeze(-1).expand(-1, -1, dim), updates_torch)

            mean, std, min_t, max_t = _benchmark_torch_fn(torch_fn, self.iterations, self.warmup)
            results["pytorch_mps"] = BenchmarkResult(
                name=f"scatter_add_{size}",
                framework="pytorch_mps",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            key = jax.random.PRNGKey(42)
            key1, key2 = jax.random.split(key)
            indices_jax = jax.random.randint(key1, (batch, num_updates), 0, seq)
            updates_jax = jax.random.normal(key2, (batch, num_updates, dim))

            @jax.jit
            def jax_fn():
                x = jnp.zeros((batch, seq, dim))
                # JAX scatter add using segment_sum or at[].add
                return x.at[jnp.arange(batch)[:, None], indices_jax].add(updates_jax)

            mean, std, min_t, max_t = _benchmark_jax_fn(jax_fn, self.iterations, self.warmup)
            results["jax_metal"] = BenchmarkResult(
                name=f"scatter_add_{size}",
                framework="jax_metal",
                mean_time=mean,
                std_time=std,
                min_time=min_t,
                max_time=max_t,
                iterations=self.iterations,
            )

        return results
