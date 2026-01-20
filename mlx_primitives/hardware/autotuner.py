"""Runtime auto-tuning for optimal tiling configurations.

This module provides infrastructure for automatically discovering
optimal tiling configurations through micro-benchmarking.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.hardware.detection import ChipInfo, get_chip_info

logger = logging.getLogger(__name__)
from mlx_primitives.hardware.tiling import (
    DataType,
    OperationType,
    ProblemSize,
    TilingConfig,
    classify_problem_size,
    dtype_to_enum,
)
from mlx_primitives.hardware.tiling_database import TilingDatabase, get_tiling_database


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        config: Configuration that was tested.
        time_ms: Average execution time in milliseconds.
        gflops: Achieved GFLOPS (if applicable).
        throughput_gbps: Memory throughput in GB/s.
        valid: Whether the result is valid (no errors).
    """

    config: TilingConfig
    time_ms: float
    gflops: float = 0.0
    throughput_gbps: float = 0.0
    valid: bool = True


class AutoTuner:
    """Runtime auto-tuning for optimal tiling configurations.

    Uses micro-benchmarks to discover optimal configurations for
    specific operations on the current hardware.

    Example:
        >>> tuner = AutoTuner()
        >>> # Auto-tune for a specific attention workload
        >>> config = tuner.tune_attention(
        ...     batch_size=1, seq_len=2048, num_heads=8, head_dim=64
        ... )
        >>> print(f"Optimal config: {config.block_m}x{config.block_n}")
    """

    def __init__(
        self,
        database: Optional[TilingDatabase] = None,
        chip_info: Optional[ChipInfo] = None,
        warmup_iters: int = 3,
        benchmark_iters: int = 10,
        timeout_ms: float = 5000.0,
    ):
        """Initialize the auto-tuner.

        Args:
            database: Tiling database for saving results.
            chip_info: Hardware info (auto-detected if None).
            warmup_iters: Number of warmup iterations.
            benchmark_iters: Number of timed iterations.
            timeout_ms: Maximum time per configuration.
        """
        self.database = database or get_tiling_database()
        self.chip_info = chip_info or get_chip_info()
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.timeout_ms = timeout_ms

    def get_search_space(
        self,
        operation: OperationType,
        problem_shape: tuple,
        dtype: DataType = DataType.FP32,
    ) -> List[TilingConfig]:
        """Generate candidate configurations based on constraints.

        Args:
            operation: Operation type.
            problem_shape: Problem dimensions.
            dtype: Data type.

        Returns:
            List of valid candidate configurations.
        """
        max_shared = self.chip_info.max_threadgroup_memory  # 32KB
        max_threads = self.chip_info.max_threads_per_threadgroup  # 1024

        candidates = []

        if operation in (
            OperationType.ATTENTION,
            OperationType.FLASH_ATTENTION,
            OperationType.SLIDING_WINDOW,
        ):
            # Attention: (batch, seq, heads, head_dim)
            head_dim = problem_shape[-1] if len(problem_shape) >= 4 else 64

            # Block size candidates
            block_m_options = [16, 32, 48, 64, 128]
            block_n_options = [16, 32, 48, 64]

            for block_m in block_m_options:
                for block_n in block_n_options:
                    # Calculate shared memory
                    padded_dim = head_dim + 4
                    shared_mem = 2 * block_n * padded_dim * 4  # K + V tiles

                    if shared_mem > max_shared:
                        continue

                    threads = min(block_m * 8, max_threads)

                    config = TilingConfig(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=head_dim,
                        threads_per_threadgroup=threads,
                        shared_memory_bytes=shared_mem,
                        use_padding=True,
                        padding_elements=4,
                    )

                    if config.validate(max_shared):
                        candidates.append(config)

        elif operation == OperationType.MATMUL:
            # Matmul: (M, N, K) or similar
            block_m_options = [32, 64, 128]
            block_n_options = [32, 64, 128]
            block_k_options = [16, 32, 64]

            for block_m in block_m_options:
                for block_n in block_n_options:
                    for block_k in block_k_options:
                        shared_mem = (block_m * block_k + block_k * block_n) * 4

                        if shared_mem > max_shared:
                            continue

                        config = TilingConfig(
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            threads_per_threadgroup=256,
                            shared_memory_bytes=shared_mem,
                        )

                        if config.validate(max_shared):
                            candidates.append(config)

        elif operation in (OperationType.SCAN, OperationType.SSM_SCAN):
            # Scan: block size must be power of 2
            for block_size in [128, 256, 512, 1024]:
                if block_size > max_threads:
                    continue

                shared_mem = 2 * block_size * 4  # A and h arrays

                if shared_mem > max_shared:
                    continue

                config = TilingConfig(
                    block_m=block_size,
                    block_n=1,
                    block_k=1,
                    threads_per_threadgroup=block_size,
                    num_simd_groups=block_size // 32,
                    shared_memory_bytes=shared_mem,
                )

                candidates.append(config)

        return candidates

    def benchmark_config(
        self,
        config: TilingConfig,
        kernel_fn: Callable,
        inputs: List[mx.array],
        expected_flops: Optional[int] = None,
        expected_bytes: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark a single configuration.

        Args:
            config: Configuration to test.
            kernel_fn: Kernel function that accepts (inputs, config).
            inputs: Input tensors.
            expected_flops: Expected FLOPs (for GFLOPS calculation).
            expected_bytes: Expected memory bytes (for bandwidth).

        Returns:
            BenchmarkResult with timing and throughput metrics.
        """
        try:
            # Warmup
            for _ in range(self.warmup_iters):
                _ = kernel_fn(*inputs, config=config)
                mx.eval(_)

            # Timed runs
            times = []
            for _ in range(self.benchmark_iters):
                mx.synchronize()
                start = time.perf_counter()

                result = kernel_fn(*inputs, config=config)
                mx.eval(result)
                mx.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

                # Check timeout
                if sum(times) > self.timeout_ms:
                    break

            avg_time = sum(times) / len(times)

            # Calculate throughput
            gflops = 0.0
            if expected_flops:
                gflops = expected_flops / (avg_time / 1000) / 1e9

            throughput = 0.0
            if expected_bytes:
                throughput = expected_bytes / (avg_time / 1000) / 1e9

            return BenchmarkResult(
                config=config,
                time_ms=avg_time,
                gflops=gflops,
                throughput_gbps=throughput,
                valid=True,
            )

        except Exception as e:
            logger.debug(
                f"Benchmark failed for config (block_m={config.block_m}, "
                f"block_n={config.block_n}, block_k={config.block_k}): {e}"
            )
            return BenchmarkResult(
                config=config,
                time_ms=float("inf"),
                valid=False,
            )

    def tune(
        self,
        operation: OperationType,
        kernel_fn: Callable,
        inputs: List[mx.array],
        expected_flops: Optional[int] = None,
        expected_bytes: Optional[int] = None,
        save_result: bool = True,
    ) -> TilingConfig:
        """Run auto-tuning and return best configuration.

        Args:
            operation: Operation type.
            kernel_fn: Kernel function to benchmark.
            inputs: Input tensors.
            expected_flops: Expected FLOPs per invocation.
            expected_bytes: Expected memory bytes per invocation.
            save_result: Whether to save result to database.

        Returns:
            Best TilingConfig found.
        """
        # Determine problem shape and size
        problem_shape = tuple(inputs[0].shape)
        dtype = dtype_to_enum(inputs[0].dtype)
        problem_size = classify_problem_size(problem_shape, operation)

        # Get candidate configurations
        candidates = self.get_search_space(operation, problem_shape, dtype)

        if not candidates:
            # Return default if no candidates
            return self.database.get_config(
                operation,
                self.chip_info.family,
                self.chip_info.tier,
                problem_size,
                dtype,
            )

        # Benchmark all candidates
        results = []
        for config in candidates:
            result = self.benchmark_config(
                config, kernel_fn, inputs, expected_flops, expected_bytes
            )
            if result.valid:
                results.append(result)

        if not results:
            # All failed, return default
            return self.database.get_config(
                operation,
                self.chip_info.family,
                self.chip_info.tier,
                problem_size,
                dtype,
            )

        # Select best (lowest time)
        best = min(results, key=lambda r: r.time_ms)

        # Save result
        if save_result:
            self.database.save_tuned_config(
                operation=operation,
                chip_family=self.chip_info.family,
                chip_tier=self.chip_info.tier,
                problem_size=problem_size,
                dtype=dtype,
                config=best.config,
            )

        return best.config

    def tune_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        kernel_fn: Callable,
        causal: bool = True,
    ) -> TilingConfig:
        """Auto-tune for attention workload.

        Benchmarks the provided kernel function with different tiling configurations
        to find the optimal one for this workload.

        IMPORTANT: The kernel_fn MUST actually use the config parameter to control
        tiling behavior. Passing a kernel that ignores the config will produce
        meaningless results.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            head_dim: Head dimension.
            kernel_fn: The actual tiled attention kernel to benchmark. Must accept
                signature (q, k, v, config=TilingConfig) and use the config to
                control block sizes. The kernel should use config.block_m,
                config.block_n, and config.block_k for tiling.
            causal: Whether attention is causal.

        Returns:
            Optimal TilingConfig for this workload.

        Example:
            >>> def my_flash_attention(q, k, v, config=None):
            ...     # Kernel that actually uses config.block_m, config.block_n
            ...     return tiled_attention_impl(q, k, v, config)
            >>> tuner = AutoTuner()
            >>> config = tuner.tune_attention(
            ...     batch_size=1, seq_len=2048, num_heads=8, head_dim=64,
            ...     kernel_fn=my_flash_attention,
            ... )

        Raises:
            ValueError: If kernel_fn is None.
        """
        if kernel_fn is None:
            raise ValueError(
                "kernel_fn is required. You must provide an actual tiled attention "
                "kernel that uses the config parameter. Passing None or a kernel "
                "that ignores config will produce meaningless auto-tuning results."
            )

        # Create test inputs
        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        mx.eval(q, k, v)

        # Calculate expected FLOPs for attention
        # QK^T: 2 * batch * heads * seq^2 * dim
        # softmax: ~5 * batch * heads * seq^2
        # AV: 2 * batch * heads * seq^2 * dim
        qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        expected_flops = qk_flops + av_flops

        return self.tune(
            operation=OperationType.FLASH_ATTENTION,
            kernel_fn=kernel_fn,
            inputs=[q, k, v],
            expected_flops=expected_flops,
        )


# Global tuner instance with thread-safe initialization
_autotuner: Optional[AutoTuner] = None
_autotuner_lock = threading.Lock()


def get_autotuner() -> AutoTuner:
    """Get the global auto-tuner instance.

    Thread-safe singleton accessor.

    Returns:
        Singleton AutoTuner instance.
    """
    global _autotuner
    if _autotuner is None:
        with _autotuner_lock:
            # Double-check locking pattern
            if _autotuner is None:
                _autotuner = AutoTuner()
    return _autotuner


def auto_tune_for_workload(
    operation: OperationType,
    problem_shape: tuple,
    dtype: mx.Dtype = mx.float32,
    kernel_fn: Optional[Callable] = None,
) -> TilingConfig:
    """Auto-tune for a specific workload.

    If a kernel function is provided, runs actual benchmarks.
    Otherwise, returns the best known configuration.

    Args:
        operation: Operation type.
        problem_shape: Problem dimensions.
        dtype: Data type.
        kernel_fn: Optional kernel function for benchmarking.

    Returns:
        Optimal or best-known TilingConfig.
    """
    tuner = get_autotuner()

    if kernel_fn is not None:
        # Create dummy inputs based on shape
        inputs = [mx.random.normal(problem_shape)]
        mx.eval(*inputs)
        return tuner.tune(operation, kernel_fn, inputs)
    else:
        # Return best known config
        data_type = dtype_to_enum(dtype)
        problem_size = classify_problem_size(problem_shape, operation)
        chip_info = get_chip_info()

        return tuner.database.get_config(
            operation,
            chip_info.family,
            chip_info.tier,
            problem_size,
            data_type,
        )
