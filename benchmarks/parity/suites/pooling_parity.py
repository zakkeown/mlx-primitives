"""Pooling parity benchmarks."""

import time
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig
from benchmarks.parity.runner import BenchmarkResult

# Check for PyTorch availability
try:
    import torch
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


def _benchmark_pytorch_fn(
    fn: Callable,
    iterations: int = 10,
    warmup_iterations: int = 3,
    name: str = "benchmark",
) -> BenchmarkResult:
    """Benchmark a PyTorch function.

    Args:
        fn: Function to benchmark (should return a torch.Tensor).
        iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        name: Name for the benchmark result.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Warmup
    for _ in range(warmup_iterations):
        result = fn()
        if HAS_PYTORCH and isinstance(result, torch.Tensor):
            torch.mps.synchronize()

    times = []
    for _ in range(iterations):
        if HAS_PYTORCH:
            torch.mps.synchronize()

        start = time.perf_counter()
        result = fn()

        if HAS_PYTORCH and isinstance(result, torch.Tensor):
            torch.mps.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return BenchmarkResult(
        name=name,
        framework="pytorch_mps",
        mean_time=mean_time,
        std_time=std_time,
        min_time=min(times),
        max_time=max(times),
        iterations=iterations,
    )


def _benchmark_mlx_fn(
    fn: Callable,
    iterations: int = 10,
    warmup_iterations: int = 3,
    name: str = "benchmark",
) -> BenchmarkResult:
    """Benchmark an MLX function.

    Args:
        fn: Function to benchmark (should return an mx.array).
        iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        name: Name for the benchmark result.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Warmup
    for _ in range(warmup_iterations):
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

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return BenchmarkResult(
        name=name,
        framework="mlx",
        mean_time=mean_time,
        std_time=std_time,
        min_time=min(times),
        max_time=max(times),
        iterations=iterations,
    )


class PoolingParityBenchmarks:
    """Multi-framework pooling benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = ParitySizeConfig()
        self._warmup = self.config.warmup_iterations
        self._iterations = self.config.benchmark_iterations

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all pooling parity benchmarks.

        Returns:
            Dictionary mapping framework names to lists of benchmark results.
        """
        results: Dict[str, List[BenchmarkResult]] = {
            "mlx": [],
            "pytorch_mps": [],
        }

        sizes = ["tiny", "small", "medium", "large"]

        for size in sizes:
            # AdaptiveAvgPool1d
            mlx_result, pytorch_result = self.benchmark_adaptive_avg_pool1d(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # AdaptiveAvgPool2d
            mlx_result, pytorch_result = self.benchmark_adaptive_avg_pool2d(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # AdaptiveMaxPool1d
            mlx_result, pytorch_result = self.benchmark_adaptive_max_pool1d(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # AdaptiveMaxPool2d
            mlx_result, pytorch_result = self.benchmark_adaptive_max_pool2d(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # GlobalAttentionPooling
            mlx_result, pytorch_result = self.benchmark_global_attention_pooling(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # GeM
            mlx_result, pytorch_result = self.benchmark_gem(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

            # SpatialPyramidPooling
            mlx_result, pytorch_result = self.benchmark_spp(size)
            results["mlx"].append(mlx_result)
            if pytorch_result:
                results["pytorch_mps"].append(pytorch_result)

        return results

    def benchmark_adaptive_avg_pool1d(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark AdaptiveAvgPool1d.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import AdaptiveAvgPool1d

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        length = width  # Use width as 1D length
        output_size = 8

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX benchmark
        pool_mlx = AdaptiveAvgPool1d(output_size)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return pool_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"adaptive_avg_pool1d_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                return F.adaptive_avg_pool1d(x_torch, output_size)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"adaptive_avg_pool1d_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_adaptive_avg_pool2d(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark AdaptiveAvgPool2d.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import AdaptiveAvgPool2d

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        output_size = (7, 7)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX benchmark
        pool_mlx = AdaptiveAvgPool2d(output_size)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return pool_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"adaptive_avg_pool2d_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                return F.adaptive_avg_pool2d(x_torch, output_size)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"adaptive_avg_pool2d_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_adaptive_max_pool1d(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark AdaptiveMaxPool1d.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import AdaptiveMaxPool1d

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        length = width
        output_size = 8

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX benchmark
        pool_mlx = AdaptiveMaxPool1d(output_size)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return pool_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"adaptive_max_pool1d_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                return F.adaptive_max_pool1d(x_torch, output_size)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"adaptive_max_pool1d_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_adaptive_max_pool2d(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark AdaptiveMaxPool2d.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import AdaptiveMaxPool2d

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        output_size = (7, 7)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX benchmark
        pool_mlx = AdaptiveMaxPool2d(output_size)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return pool_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"adaptive_max_pool2d_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                return F.adaptive_max_pool2d(x_torch, output_size)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"adaptive_max_pool2d_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_global_attention_pooling(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark GlobalAttentionPooling.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import GlobalAttentionPooling

        # Use activation config for sequence-based pooling
        config = self.sizes.get_config("activation", size)
        batch, seq, dims = config
        hidden_dims = dims // 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dims).astype(np.float32)

        # MLX benchmark
        pool_mlx = GlobalAttentionPooling(dims, hidden_dims)
        mx.eval(pool_mlx.parameters())
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return pool_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"global_attention_pooling_{size}",
        )

        # PyTorch benchmark - implement equivalent attention pooling
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            attention_layers = pool_mlx.attention.layers
            W1_np = np.array(attention_layers[0].weight).T
            b1_np = np.array(attention_layers[0].bias)
            W2_np = np.array(attention_layers[2].weight).T

            x_torch = torch.from_numpy(x_np).to("mps")
            W1 = torch.from_numpy(W1_np).to("mps")
            b1 = torch.from_numpy(b1_np).to("mps")
            W2 = torch.from_numpy(W2_np).to("mps")

            def pytorch_fn():
                hidden = torch.tanh(x_torch @ W1 + b1)
                scores = (hidden @ W2).squeeze(-1)
                weights = torch.softmax(scores, dim=-1)
                return torch.sum(x_torch * weights.unsqueeze(-1), dim=1)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"global_attention_pooling_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_gem(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark GeM (Generalized Mean) pooling.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import GeM

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        p = 3.0
        eps = 1e-6

        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + eps

        # MLX benchmark
        gem_mlx = GeM(p=p, eps=eps, learnable=False)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return gem_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"gem_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                x_clamped = x_torch.clamp(min=eps)
                x_pow = x_clamped.pow(p)
                mean_pow = x_pow.mean(dim=(2, 3), keepdim=True)
                return mean_pow.pow(1.0 / p)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"gem_{size}",
            )

        return mlx_result, pytorch_result

    def benchmark_spp(
        self, size: str
    ) -> Tuple[BenchmarkResult, Optional[BenchmarkResult]]:
        """Benchmark SpatialPyramidPooling.

        Args:
            size: Size configuration (tiny, small, medium, large).

        Returns:
            Tuple of (MLX result, PyTorch result).
        """
        from mlx_primitives.layers.pooling import SpatialPyramidPooling

        config = self.sizes.get_config("pooling", size)
        batch, channels, height, width = config
        levels = [1, 2, 4]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX benchmark
        spp_mlx = SpatialPyramidPooling(output_sizes=levels)
        x_mlx = mx.array(x_np)

        def mlx_fn():
            return spp_mlx(x_mlx)

        mlx_result = _benchmark_mlx_fn(
            mlx_fn,
            iterations=self._iterations,
            warmup_iterations=self._warmup,
            name=f"spp_{size}",
        )

        # PyTorch benchmark
        pytorch_result = None
        if HAS_PYTORCH and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")

            def pytorch_fn():
                pooled = []
                for level in levels:
                    p = F.adaptive_avg_pool2d(x_torch, (level, level))
                    p = p.reshape(batch, -1)
                    pooled.append(p)
                return torch.cat(pooled, dim=1)

            pytorch_result = _benchmark_pytorch_fn(
                pytorch_fn,
                iterations=self._iterations,
                warmup_iterations=self._warmup,
                name=f"spp_{size}",
            )

        return mlx_result, pytorch_result
