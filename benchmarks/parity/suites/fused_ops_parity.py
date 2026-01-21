"""Fused operations parity benchmarks."""

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


class FusedOpsParityBenchmarks:
    """Multi-framework fused operations benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all fused operations parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            results[f"fused_rmsnorm_linear_{size}"] = self._benchmark_to_list(
                self.benchmark_fused_rmsnorm_linear(size)
            )
            results[f"fused_swiglu_{size}"] = self._benchmark_to_list(
                self.benchmark_fused_swiglu(size)
            )
            results[f"fused_geglu_{size}"] = self._benchmark_to_list(
                self.benchmark_fused_geglu(size)
            )
            results[f"fused_rope_attention_{size}"] = self._benchmark_to_list(
                self.benchmark_fused_rope_attention(size)
            )
            results[f"fused_vs_separate_{size}"] = self._benchmark_to_list(
                self.benchmark_vs_separate_ops(size)
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

    def benchmark_fused_rmsnorm_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark fused RMSNorm+Linear across MLX, PyTorch MPS."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        out_features = hidden * 4
        eps = 1e-5

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, hidden))
        norm_weight = mx.ones((hidden,))
        linear_weight = mx.random.normal((out_features, hidden)) * 0.02
        mx.eval(x_mlx, norm_weight, linear_weight)

        results["mlx"] = self._benchmark_mlx(
            lambda: fused_rmsnorm_linear(x_mlx, norm_weight, linear_weight, eps=eps),
            name=f"fused_rmsnorm_linear_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, hidden, device="mps")
            norm_weight_torch = torch.ones(hidden, device="mps")
            linear_weight_torch = torch.randn(out_features, hidden, device="mps") * 0.02

            def pytorch_fused_rmsnorm_linear():
                # RMSNorm
                rms = torch.sqrt(torch.mean(x_torch**2, dim=-1, keepdim=True) + eps)
                norm_x = (x_torch / rms) * norm_weight_torch
                # Linear
                return F.linear(norm_x, linear_weight_torch)

            result = self._benchmark_pytorch(
                pytorch_fused_rmsnorm_linear,
                name=f"fused_rmsnorm_linear_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_fused_swiglu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark fused SwiGLU across MLX, PyTorch MPS."""
        from mlx_primitives.kernels.fused_activations import fused_swiglu

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        hidden_dim = dim

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, dim))
        W_gate = mx.random.normal((hidden_dim, dim)) * 0.02
        W_up = mx.random.normal((hidden_dim, dim)) * 0.02
        mx.eval(x_mlx, W_gate, W_up)

        results["mlx"] = self._benchmark_mlx(
            lambda: fused_swiglu(x_mlx, W_gate, W_up),
            name=f"fused_swiglu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, dim, device="mps")
            W_gate_torch = torch.randn(hidden_dim, dim, device="mps") * 0.02
            W_up_torch = torch.randn(hidden_dim, dim, device="mps") * 0.02

            def pytorch_swiglu():
                gate = F.silu(x_torch @ W_gate_torch.T)
                up = x_torch @ W_up_torch.T
                return gate * up

            result = self._benchmark_pytorch(
                pytorch_swiglu,
                name=f"fused_swiglu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_fused_geglu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark fused GeGLU across MLX, PyTorch MPS."""
        from mlx_primitives.kernels.fused_activations import fused_geglu

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        hidden_dim = dim

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, dim))
        W_gate = mx.random.normal((hidden_dim, dim)) * 0.02
        W_up = mx.random.normal((hidden_dim, dim)) * 0.02
        mx.eval(x_mlx, W_gate, W_up)

        results["mlx"] = self._benchmark_mlx(
            lambda: fused_geglu(x_mlx, W_gate, W_up),
            name=f"fused_geglu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            torch.manual_seed(42)
            x_torch = torch.randn(batch, seq, dim, device="mps")
            W_gate_torch = torch.randn(hidden_dim, dim, device="mps") * 0.02
            W_up_torch = torch.randn(hidden_dim, dim, device="mps") * 0.02

            def pytorch_geglu():
                gate = F.gelu(x_torch @ W_gate_torch.T, approximate="tanh")
                up = x_torch @ W_up_torch.T
                return gate * up

            result = self._benchmark_pytorch(
                pytorch_geglu,
                name=f"fused_geglu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_fused_rope_attention(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark fused RoPE+Attention across MLX, PyTorch MPS."""
        from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

        config = self.sizes.get_config("attention", size)
        batch, seq, heads, head_dim = config
        scale = 1.0 / (head_dim**0.5)

        results = {}

        # MLX benchmark
        mx.random.seed(42)
        q = mx.random.normal((batch, seq, heads, head_dim))
        k = mx.random.normal((batch, seq, heads, head_dim))
        v = mx.random.normal((batch, seq, heads, head_dim))
        mx.eval(q, k, v)

        results["mlx"] = self._benchmark_mlx(
            lambda: fused_rope_attention(q, k, v, scale=scale, causal=True),
            name=f"fused_rope_attention_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            torch.manual_seed(42)
            q_torch = torch.randn(batch, seq, heads, head_dim, device="mps")
            k_torch = torch.randn(batch, seq, heads, head_dim, device="mps")
            v_torch = torch.randn(batch, seq, heads, head_dim, device="mps")

            # Precompute RoPE components
            positions = torch.arange(seq, device="mps", dtype=torch.float32)
            inv_freq = 1.0 / (
                10000.0
                ** (torch.arange(0, head_dim, 2, device="mps", dtype=torch.float32) / head_dim)
            )
            angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
            cos = torch.cos(angles)
            sin = torch.sin(angles)

            def apply_rope(x, cos, sin):
                x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
                cos_expanded = cos[None, :, None, :]
                sin_expanded = sin[None, :, None, :]
                return torch.cat(
                    [
                        x1 * cos_expanded - x2 * sin_expanded,
                        x1 * sin_expanded + x2 * cos_expanded,
                    ],
                    dim=-1,
                )

            def pytorch_rope_attention():
                q_rot = apply_rope(q_torch, cos, sin)
                k_rot = apply_rope(k_torch, cos, sin)

                # Transpose to (batch, heads, seq, dim) for SDPA
                q_t = q_rot.transpose(1, 2)
                k_t = k_rot.transpose(1, 2)
                v_t = v_torch.transpose(1, 2)

                out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
                return out.transpose(1, 2)

            result = self._benchmark_pytorch(
                pytorch_rope_attention,
                name=f"fused_rope_attention_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def benchmark_vs_separate_ops(self, size: str) -> Dict[str, BenchmarkResult]:
        """Compare fused ops vs separate operations for speedup analysis."""
        from mlx_primitives.kernels.fused_norm_linear import fused_rmsnorm_linear

        config = self.sizes.get_config("normalization", size)
        batch, seq, hidden = config
        out_features = hidden * 4
        eps = 1e-5

        results = {}

        mx.random.seed(42)
        x_mlx = mx.random.normal((batch, seq, hidden))
        norm_weight = mx.ones((hidden,))
        linear_weight = mx.random.normal((out_features, hidden)) * 0.02
        mx.eval(x_mlx, norm_weight, linear_weight)

        # Fused version
        results["fused"] = self._benchmark_mlx(
            lambda: fused_rmsnorm_linear(x_mlx, norm_weight, linear_weight, eps=eps),
            name=f"rmsnorm_linear_fused_{size}",
        )

        # Separate operations version
        def separate_ops():
            rms = mx.sqrt(mx.mean(x_mlx * x_mlx, axis=-1, keepdims=True) + eps)
            norm_x = x_mlx / rms * norm_weight
            return norm_x @ linear_weight.T

        results["separate"] = self._benchmark_mlx(
            separate_ops,
            name=f"rmsnorm_linear_separate_{size}",
        )

        return results
