"""PyTorch MPS baseline implementations for performance comparison.

This module provides PyTorch implementations running on Apple Metal (MPS backend)
for comparing against MLX implementations.
"""

import time
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    torch = None
    F = None
    np = None


def pytorch_available() -> bool:
    """Check if PyTorch with MPS is available."""
    return HAS_TORCH and HAS_MPS


@dataclass
class PyTorchBenchmarkResult:
    """Result from a PyTorch benchmark."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int


class PyTorchMPSBenchmarks:
    """PyTorch MPS baseline benchmarks for comparison with MLX."""

    def __init__(self, device: str = "mps"):
        """Initialize PyTorch MPS benchmarks.

        Args:
            device: Device to use ("mps" for Apple Metal).

        Raises:
            RuntimeError: If PyTorch with MPS is not available.
        """
        if not pytorch_available():
            raise RuntimeError("PyTorch with MPS not available")
        self.device = torch.device(device)

    def _benchmark(
        self,
        fn: Callable,
        iterations: int = 30,
        warmup_iterations: int = 5,
        name: str = "benchmark",
    ) -> PyTorchBenchmarkResult:
        """Run a benchmark with warmup and timing.

        Args:
            fn: Function to benchmark.
            iterations: Number of timed iterations.
            warmup_iterations: Number of warmup iterations.
            name: Benchmark name.

        Returns:
            Benchmark result with timing statistics.
        """
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

        return PyTorchBenchmarkResult(
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
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch scaled dot-product attention.

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
        torch.manual_seed(42)

        # PyTorch expects (batch, num_heads, seq_len, head_dim)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        name = f"pytorch_attention_b{batch_size}_s{seq_len}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_layer_norm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch LayerNorm.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        weight = torch.ones(hidden_dim, device=self.device)
        bias = torch.zeros(hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.layer_norm(x, [hidden_dim], weight, bias)

        name = f"pytorch_layernorm_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_matmul(
        self,
        m: int,
        n: int,
        k: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch matrix multiplication.

        Args:
            m: First matrix rows.
            n: Second matrix columns.
            k: Shared dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        a = torch.randn(m, k, device=self.device)
        b = torch.randn(k, n, device=self.device)

        def fn():
            with torch.no_grad():
                return torch.matmul(a, b)

        name = f"pytorch_matmul_{m}x{k}x{n}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_softmax(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch softmax.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            vocab_size: Vocabulary size.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)

        def fn():
            with torch.no_grad():
                return F.softmax(logits, dim=-1)

        name = f"pytorch_softmax_b{batch_size}_s{seq_len}_v{vocab_size}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_gelu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch GELU activation.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.gelu(x, approximate="tanh")

        name = f"pytorch_gelu_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_silu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch SiLU (Swish) activation.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.silu(x)

        name = f"pytorch_silu_b{batch_size}_s{seq_len}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, name=name)

    def benchmark_linear(
        self,
        batch_size: int,
        seq_len: int,
        in_features: int,
        out_features: int,
        iterations: int = 30,
    ) -> PyTorchBenchmarkResult:
        """Benchmark PyTorch Linear layer.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            in_features: Input features.
            out_features: Output features.
            iterations: Number of benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, in_features, device=self.device)
        layer = torch.nn.Linear(in_features, out_features, device=self.device)

        def fn():
            with torch.no_grad():
                return layer(x)

        name = f"pytorch_linear_b{batch_size}_s{seq_len}_{in_features}x{out_features}"
        return self._benchmark(fn, iterations=iterations, name=name)
