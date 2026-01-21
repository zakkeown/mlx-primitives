"""Normalization parity benchmarks."""

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
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


class NormalizationParityBenchmarks:
    """Multi-framework normalization benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all normalization parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            results[f"rmsnorm_{size}"] = self._benchmark_to_list(self.benchmark_rmsnorm(size))
            results[f"layernorm_{size}"] = self._benchmark_to_list(self.benchmark_layernorm(size))
            results[f"groupnorm_{size}"] = self._benchmark_to_list(self.benchmark_groupnorm(size))
            results[f"instancenorm_{size}"] = self._benchmark_to_list(self.benchmark_instancenorm(size))
            results[f"adalayernorm_{size}"] = self._benchmark_to_list(self.benchmark_adalayernorm(size))

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

    def benchmark_rmsnorm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark RMSNorm across MLX, PyTorch MPS, and JAX Metal."""
        from mlx_primitives.layers import RMSNorm

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        results = {}

        # MLX benchmark
        rmsnorm = RMSNorm(hidden)
        mx.eval(rmsnorm.parameters())
        x_mlx = mx.random.normal((batch, seq, hidden))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: rmsnorm(x_mlx),
            name=f"rmsnorm_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, hidden, device="mps")
            weight_torch = torch.ones(hidden, device="mps")
            eps = 1e-6

            def pytorch_rmsnorm():
                rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + eps)
                return (x_torch / rms) * weight_torch

            result = self._benchmark_pytorch(
                pytorch_rmsnorm,
                name=f"rmsnorm_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_layernorm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark LayerNorm across MLX, PyTorch MPS, and JAX Metal."""
        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        results = {}

        # MLX benchmark
        layernorm = nn.LayerNorm(hidden)
        mx.eval(layernorm.parameters())
        x_mlx = mx.random.normal((batch, seq, hidden))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: layernorm(x_mlx),
            name=f"layernorm_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, hidden, device="mps")
            weight_torch = torch.ones(hidden, device="mps")
            bias_torch = torch.zeros(hidden, device="mps")
            eps = 1e-5

            def pytorch_layernorm():
                return F.layer_norm(x_torch, (hidden,), weight_torch, bias_torch, eps)

            result = self._benchmark_pytorch(
                pytorch_layernorm,
                name=f"layernorm_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_groupnorm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark GroupNorm across MLX, PyTorch MPS, and JAX Metal."""
        from mlx_primitives.layers import GroupNorm

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        # GroupNorm uses NCHW format
        num_channels = hidden
        height, width = 8, 8
        num_groups = 8

        # Ensure divisibility
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2

        results = {}

        # MLX benchmark
        groupnorm = GroupNorm(num_groups, num_channels)
        mx.eval(groupnorm.parameters())
        x_mlx = mx.random.normal((batch, num_channels, height, width))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: groupnorm(x_mlx),
            name=f"groupnorm_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, num_channels, height, width, device="mps")
            weight_torch = torch.ones(num_channels, device="mps")
            bias_torch = torch.zeros(num_channels, device="mps")
            eps = 1e-5

            def pytorch_groupnorm():
                return F.group_norm(x_torch, num_groups, weight_torch, bias_torch, eps)

            result = self._benchmark_pytorch(
                pytorch_groupnorm,
                name=f"groupnorm_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_instancenorm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark InstanceNorm across MLX, PyTorch MPS, and JAX Metal."""
        from mlx_primitives.layers import InstanceNorm

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        # InstanceNorm uses NCHW format
        num_features = hidden
        height, width = 8, 8
        results = {}

        # MLX benchmark
        instancenorm = InstanceNorm(num_features)
        mx.eval(instancenorm.parameters())
        x_mlx = mx.random.normal((batch, num_features, height, width))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: instancenorm(x_mlx),
            name=f"instancenorm_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, num_features, height, width, device="mps")
            weight_torch = torch.ones(num_features, device="mps")
            bias_torch = torch.zeros(num_features, device="mps")
            eps = 1e-5

            def pytorch_instancenorm():
                return F.instance_norm(x_torch, weight=weight_torch, bias=bias_torch, eps=eps)

            result = self._benchmark_pytorch(
                pytorch_instancenorm,
                name=f"instancenorm_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_adalayernorm(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark AdaLayerNorm across MLX, PyTorch MPS, and JAX Metal."""
        from mlx_primitives.layers import AdaLayerNorm

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        cond_dims = hidden // 2
        eps = 1e-6
        results = {}

        # MLX benchmark
        adaln = AdaLayerNorm(hidden, cond_dims, eps=eps)
        mx.eval(adaln.parameters())
        x_mlx = mx.random.normal((batch, seq, hidden))
        cond_mlx = mx.random.normal((batch, cond_dims))
        mx.eval(x_mlx, cond_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: adaln(x_mlx, cond_mlx),
            name=f"adalayernorm_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, hidden, device="mps")
            cond_torch = torch.randn(batch, cond_dims, device="mps")

            # Get projection weights from MLX model
            proj_weight_np = np.array(adaln.proj.weight)
            proj_bias_np = np.array(adaln.proj.bias)
            proj_weight_torch = torch.from_numpy(proj_weight_np).to("mps")
            proj_bias_torch = torch.from_numpy(proj_bias_np).to("mps")

            def pytorch_adalayernorm():
                # LayerNorm without affine
                mean = x_torch.mean(dim=-1, keepdim=True)
                var = x_torch.var(dim=-1, keepdim=True, unbiased=False)
                x_norm = (x_torch - mean) / torch.sqrt(var + eps)

                # Get scale/shift from conditioning
                scale_shift = F.linear(cond_torch, proj_weight_torch, proj_bias_torch)
                scale, shift = scale_shift.chunk(2, dim=-1)
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)

                return x_norm * (1 + scale) + shift

            result = self._benchmark_pytorch(
                pytorch_adalayernorm,
                name=f"adalayernorm_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results
