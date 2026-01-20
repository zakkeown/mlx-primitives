"""Benchmark configuration constants."""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark suite.

    Attributes:
        warmup_iterations: Number of warmup iterations before timing.
        benchmark_iterations: Number of timed iterations (used when not adaptive).
        min_time_seconds: Minimum time to run benchmarks.
        timeout_seconds: Maximum time per benchmark.
        seed: Random seed for reproducibility.
        adaptive_iterations: Whether to use adaptive iteration counts.
        fast_iterations: Iterations for fast ops (<1ms).
        medium_iterations: Iterations for medium ops (1-10ms).
        slow_iterations: Iterations for slow ops (>10ms).
        fast_threshold_ms: Threshold for fast ops in milliseconds.
        medium_threshold_ms: Threshold for medium ops in milliseconds.
    """

    warmup_iterations: int = 5
    benchmark_iterations: int = 30
    min_time_seconds: float = 0.1
    timeout_seconds: float = 60.0
    seed: int = 42

    # Adaptive iteration settings
    adaptive_iterations: bool = True
    fast_iterations: int = 100  # For ops < 1ms
    medium_iterations: int = 50  # For ops 1-10ms
    slow_iterations: int = 30  # For ops > 10ms
    fast_threshold_ms: float = 1.0
    medium_threshold_ms: float = 10.0

    def get_iterations(self, estimated_time_ms: float) -> int:
        """Get appropriate iteration count based on estimated runtime.

        Args:
            estimated_time_ms: Estimated operation time in milliseconds.

        Returns:
            Number of iterations to run.
        """
        if not self.adaptive_iterations:
            return self.benchmark_iterations

        if estimated_time_ms < self.fast_threshold_ms:
            return self.fast_iterations
        elif estimated_time_ms < self.medium_threshold_ms:
            return self.medium_iterations
        else:
            return self.slow_iterations


@dataclass
class RegressionConfig:
    """Configuration for regression detection.

    Attributes:
        threshold_percent: Regression threshold percentage (default 10%).
        confidence_level: Statistical confidence level (default 0.95).
        min_iterations: Minimum iterations for statistical validity.
        custom_thresholds: Per-operation threshold overrides.
    """

    threshold_percent: float = 10.0
    confidence_level: float = 0.95
    min_iterations: int = 30

    # Per-operation threshold overrides (pattern -> threshold)
    custom_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "moe_*": 15.0,  # MoE operations are more variable
        "ssm_*": 15.0,  # SSM has variable performance
        "*_backward": 12.0,  # Backward passes are more variable
    })

    def get_threshold(self, benchmark_name: str) -> float:
        """Get threshold for a specific benchmark.

        Args:
            benchmark_name: Name of the benchmark.

        Returns:
            Threshold percentage for this benchmark.
        """
        import fnmatch
        for pattern, threshold in self.custom_thresholds.items():
            if fnmatch.fnmatch(benchmark_name, pattern):
                return threshold
        return self.threshold_percent


@dataclass
class AttentionSizes:
    """Common attention tensor sizes for benchmarking.

    Provides both named configurations and iterable attributes for benchmarks.
    """

    # Named configurations: (batch, seq_len, num_heads, head_dim)
    small: Tuple[int, ...] = (1, 128, 8, 64)
    medium: Tuple[int, ...] = (2, 512, 8, 64)
    large: Tuple[int, ...] = (2, 2048, 8, 64)
    xlarge: Tuple[int, ...] = (4, 4096, 16, 64)
    # LLaMA-style configurations
    llama_7b: Tuple[int, ...] = (1, 2048, 32, 128)
    llama_13b: Tuple[int, ...] = (1, 2048, 40, 128)

    # Iterable attributes for benchmark loops
    seq_lengths: Tuple[int, ...] = (128, 512, 1024, 2048)
    batch_sizes: Tuple[int, ...] = (1, 2, 4)
    num_heads: int = 8
    head_dim: int = 64


@dataclass
class ScanSizes:
    """Common scan tensor sizes for benchmarking.

    Provides both named configurations and iterable attributes for benchmarks.
    """

    # Named configurations: (batch, seq_len, state_dim)
    small: Tuple[int, ...] = (4, 128, 32)
    medium: Tuple[int, ...] = (8, 512, 64)
    large: Tuple[int, ...] = (4, 2048, 128)
    mamba_style: Tuple[int, ...] = (1, 2048, 16)  # d_state=16 typical for Mamba

    # Iterable attributes for benchmark loops
    seq_lengths: Tuple[int, ...] = (128, 512, 1024, 2048)
    feature_dims: Tuple[int, ...] = (32, 64, 128)
    batch_size: int = 4


@dataclass
class MatmulSizes:
    """Common matrix multiplication sizes for benchmarking.

    Each size is (M, N, K) for A(M,K) @ B(K,N).
    """

    small: Tuple[int, ...] = (512, 512, 512)
    medium: Tuple[int, ...] = (2048, 2048, 2048)
    large: Tuple[int, ...] = (4096, 4096, 4096)


@dataclass
class LayerSizes:
    """Common layer sizes for benchmarking normalization, pooling, embeddings.

    Attributes:
        batch_sizes: Batch sizes to test.
        hidden_dims: Hidden dimensions for normalization layers.
        seq_lengths: Sequence lengths for embeddings.
        vocab_sizes: Vocabulary sizes for embeddings.
        num_groups: Number of groups for GroupNorm.
    """

    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    hidden_dims: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    seq_lengths: Tuple[int, ...] = (128, 512, 1024, 2048)
    vocab_sizes: Tuple[int, ...] = (32000, 50257, 128256)  # LLaMA, GPT-2, LLaMA-3
    num_groups: Tuple[int, ...] = (8, 16, 32)

    # Named configurations: (batch, seq_len, hidden_dim)
    small: Tuple[int, ...] = (1, 128, 512)
    medium: Tuple[int, ...] = (4, 512, 1024)
    large: Tuple[int, ...] = (8, 1024, 2048)


@dataclass
class MoESizes:
    """Common MoE sizes for benchmarking.

    Attributes:
        num_experts: Number of experts to test.
        top_k: Top-k values for routing.
        hidden_dims: Hidden dimensions.
        expert_dims: Expert intermediate dimensions.
    """

    num_experts: Tuple[int, ...] = (4, 8, 16, 32)
    top_k: Tuple[int, ...] = (1, 2, 4)
    hidden_dims: Tuple[int, ...] = (512, 1024, 2048, 4096)
    expert_dims: Tuple[int, ...] = (1024, 2048, 4096, 8192)

    # Named configurations: (batch, seq_len, num_experts, top_k, hidden_dim, expert_dim)
    small: Tuple[int, ...] = (1, 128, 4, 2, 512, 1024)
    medium: Tuple[int, ...] = (2, 512, 8, 2, 1024, 2048)
    large: Tuple[int, ...] = (4, 1024, 16, 2, 2048, 4096)
    mixtral_style: Tuple[int, ...] = (1, 2048, 8, 2, 4096, 14336)


@dataclass
class CacheSizes:
    """Common KV cache sizes for benchmarking.

    Attributes:
        batch_sizes: Batch sizes.
        seq_lengths: Cached sequence lengths.
        num_heads: Number of attention heads.
        head_dims: Head dimensions.
        block_sizes: Block sizes for paged attention.
    """

    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    seq_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    num_heads: Tuple[int, ...] = (8, 32, 40)
    head_dims: Tuple[int, ...] = (64, 128)
    block_sizes: Tuple[int, ...] = (16, 32, 64, 128)

    # Named configurations: (batch, seq_len, num_heads, head_dim)
    small: Tuple[int, ...] = (1, 512, 8, 64)
    medium: Tuple[int, ...] = (4, 2048, 32, 128)
    large: Tuple[int, ...] = (8, 4096, 40, 128)


@dataclass
class QuantizationSizes:
    """Common sizes for quantization benchmarks.

    Attributes:
        matrix_sizes: Matrix dimensions (M, N, K).
        batch_sizes: Batch sizes for quantized linear.
    """

    matrix_sizes: Tuple[Tuple[int, ...], ...] = (
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    )
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)

    # Named configurations for quantized linear: (batch, seq_len, in_features, out_features)
    small: Tuple[int, ...] = (1, 128, 512, 512)
    medium: Tuple[int, ...] = (4, 512, 2048, 2048)
    large: Tuple[int, ...] = (8, 1024, 4096, 4096)


@dataclass
class GenerationSizes:
    """Common sizes for generation/sampling benchmarks.

    Attributes:
        batch_sizes: Batch sizes.
        vocab_sizes: Vocabulary sizes.
        top_k_values: Top-k values for sampling.
        top_p_values: Top-p values for nucleus sampling.
    """

    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16, 32)
    vocab_sizes: Tuple[int, ...] = (32000, 50257, 128256)
    top_k_values: Tuple[int, ...] = (10, 50, 100)
    top_p_values: Tuple[float, ...] = (0.9, 0.95, 0.99)

    # Named configurations: (batch, vocab_size)
    small: Tuple[int, ...] = (1, 32000)
    medium: Tuple[int, ...] = (8, 50257)
    large: Tuple[int, ...] = (32, 128256)


@dataclass
class TrainingSizes:
    """Common sizes for training utility benchmarks.

    Attributes:
        param_counts: Parameter counts for EMA, gradient clipping.
        batch_sizes: Batch sizes for gradient accumulation.
    """

    param_counts: Tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16, 32)

    # Named configurations
    small: int = 10_000
    medium: int = 1_000_000
    large: int = 100_000_000
