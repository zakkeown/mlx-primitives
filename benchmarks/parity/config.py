"""Configuration for parity benchmarks."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ParityBenchmarkConfig:
    """Configuration for parity benchmarks.

    Attributes:
        include_pytorch: Whether to include PyTorch benchmarks.
        include_jax: Whether to include JAX benchmarks.
        include_numpy: Whether to include NumPy baselines.
        frameworks: List of frameworks to benchmark.
        profile_memory: Whether to profile memory usage.
        memory_warmup_iterations: Warmup iterations for memory profiling.
        warmup_iterations: Number of warmup iterations.
        benchmark_iterations: Number of timed iterations.
        timeout_seconds: Timeout per benchmark in seconds.
        scaling_seq_lengths: Sequence lengths for scaling analysis.
        scaling_batch_sizes: Batch sizes for scaling analysis.
        scaling_model_sizes: Model sizes for scaling analysis.
        regression_threshold: Threshold for regression detection (fraction).
    """

    include_pytorch: bool = True
    include_jax: bool = True
    include_numpy: bool = False
    frameworks: List[str] = field(default_factory=lambda: ["mlx", "pytorch_mps", "jax_metal"])

    profile_memory: bool = True
    memory_warmup_iterations: int = 3

    warmup_iterations: int = 5
    benchmark_iterations: int = 30
    timeout_seconds: int = 60

    scaling_seq_lengths: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)
    scaling_batch_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    scaling_model_sizes: Tuple[str, ...] = ("small", "medium", "large", "xlarge")

    regression_threshold: float = 0.10  # 10% regression threshold


@dataclass
class ParitySizeConfig:
    """Size configurations for different operation categories.

    Each attribute is a tuple defining (batch, seq, heads, head_dim) or similar
    depending on the operation type.
    """

    # Attention configurations: (batch, seq, heads, head_dim)
    attention_tiny: Tuple[int, int, int, int] = (1, 64, 4, 32)
    attention_small: Tuple[int, int, int, int] = (2, 256, 8, 64)
    attention_medium: Tuple[int, int, int, int] = (4, 1024, 16, 64)
    attention_large: Tuple[int, int, int, int] = (8, 4096, 32, 128)

    # Activation configurations: (batch, seq, dim)
    activation_tiny: Tuple[int, int, int] = (1, 64, 256)
    activation_small: Tuple[int, int, int] = (4, 256, 1024)
    activation_medium: Tuple[int, int, int] = (8, 1024, 2048)
    activation_large: Tuple[int, int, int] = (16, 4096, 4096)

    # Normalization configurations: (batch, seq, hidden)
    normalization_tiny: Tuple[int, int, int] = (1, 64, 256)
    normalization_small: Tuple[int, int, int] = (4, 256, 1024)
    normalization_medium: Tuple[int, int, int] = (8, 1024, 2048)
    normalization_large: Tuple[int, int, int] = (16, 4096, 4096)

    # Quantization configurations: (m, n, k)
    quantization_tiny: Tuple[int, int, int] = (64, 64, 64)
    quantization_small: Tuple[int, int, int] = (256, 256, 256)
    quantization_medium: Tuple[int, int, int] = (1024, 1024, 1024)
    quantization_large: Tuple[int, int, int] = (4096, 4096, 4096)

    # MoE configurations: (batch, seq, dim, experts, top_k)
    moe_tiny: Tuple[int, int, int, int, int] = (1, 32, 128, 4, 2)
    moe_small: Tuple[int, int, int, int, int] = (2, 128, 512, 8, 2)
    moe_medium: Tuple[int, int, int, int, int] = (4, 512, 1024, 16, 2)
    moe_large: Tuple[int, int, int, int, int] = (8, 2048, 2048, 32, 2)

    # Pooling configurations: (batch, channels, height, width)
    pooling_tiny: Tuple[int, int, int, int] = (1, 32, 16, 16)
    pooling_small: Tuple[int, int, int, int] = (4, 64, 32, 32)
    pooling_medium: Tuple[int, int, int, int] = (8, 128, 64, 64)
    pooling_large: Tuple[int, int, int, int] = (16, 256, 128, 128)

    # Embedding configurations: (batch, seq, vocab_size, dim)
    embedding_tiny: Tuple[int, int, int, int] = (1, 32, 1000, 64)
    embedding_small: Tuple[int, int, int, int] = (4, 128, 10000, 256)
    embedding_medium: Tuple[int, int, int, int] = (8, 512, 50000, 512)
    embedding_large: Tuple[int, int, int, int] = (16, 2048, 100000, 1024)

    # Scan/primitive configurations: (batch, seq, dim)
    scan_tiny: Tuple[int, int, int] = (1, 64, 32)
    scan_small: Tuple[int, int, int] = (4, 256, 64)
    scan_medium: Tuple[int, int, int] = (8, 1024, 128)
    scan_large: Tuple[int, int, int] = (16, 4096, 256)

    # Cache configurations: (batch, seq, heads, head_dim, block_size)
    cache_tiny: Tuple[int, int, int, int, int] = (1, 64, 4, 32, 16)
    cache_small: Tuple[int, int, int, int, int] = (2, 256, 8, 64, 32)
    cache_medium: Tuple[int, int, int, int, int] = (4, 1024, 16, 64, 64)
    cache_large: Tuple[int, int, int, int, int] = (8, 4096, 32, 128, 128)

    # Generation/sampling configurations: (batch, vocab_size)
    generation_tiny: Tuple[int, int] = (1, 1000)
    generation_small: Tuple[int, int] = (4, 10000)
    generation_medium: Tuple[int, int] = (8, 50000)
    generation_large: Tuple[int, int] = (16, 100000)

    def get_config(self, category: str, size: str) -> Tuple:
        """Get configuration for a category and size.

        Args:
            category: Operation category (attention, activation, etc.)
            size: Size name (tiny, small, medium, large)

        Returns:
            Configuration tuple.
        """
        attr_name = f"{category}_{size}"
        return getattr(self, attr_name, None)


# Default configurations
DEFAULT_CONFIG = ParityBenchmarkConfig()
DEFAULT_SIZES = ParitySizeConfig()
