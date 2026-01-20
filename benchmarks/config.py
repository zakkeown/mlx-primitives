"""Benchmark configuration constants."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark suite.

    Attributes:
        warmup_iterations: Number of warmup iterations before timing.
        benchmark_iterations: Number of timed iterations.
        min_time_seconds: Minimum time to run benchmarks.
        timeout_seconds: Maximum time per benchmark.
        seed: Random seed for reproducibility.
    """

    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    min_time_seconds: float = 0.1
    timeout_seconds: float = 60.0
    seed: int = 42


@dataclass
class AttentionSizes:
    """Common attention tensor sizes for benchmarking.

    Each size is (batch, seq_len, num_heads, head_dim).
    """

    small: Tuple[int, ...] = (1, 128, 8, 64)
    medium: Tuple[int, ...] = (2, 512, 8, 64)
    large: Tuple[int, ...] = (2, 2048, 8, 64)
    xlarge: Tuple[int, ...] = (4, 4096, 16, 64)
    # LLaMA-style configurations
    llama_7b: Tuple[int, ...] = (1, 2048, 32, 128)
    llama_13b: Tuple[int, ...] = (1, 2048, 40, 128)


@dataclass
class ScanSizes:
    """Common scan tensor sizes for benchmarking.

    Each size is (batch, seq_len, state_dim).
    """

    small: Tuple[int, ...] = (4, 128, 32)
    medium: Tuple[int, ...] = (8, 512, 64)
    large: Tuple[int, ...] = (4, 2048, 128)
    mamba_style: Tuple[int, ...] = (1, 2048, 16)  # d_state=16 typical for Mamba


@dataclass
class MatmulSizes:
    """Common matrix multiplication sizes for benchmarking.

    Each size is (M, N, K) for A(M,K) @ B(K,N).
    """

    small: Tuple[int, ...] = (512, 512, 512)
    medium: Tuple[int, ...] = (2048, 2048, 2048)
    large: Tuple[int, ...] = (4096, 4096, 4096)
