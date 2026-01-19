"""Tests for runtime auto-tuning infrastructure."""

import pytest
import mlx.core as mx

from mlx_primitives.hardware.autotuner import (
    AutoTuner,
    BenchmarkResult,
    auto_tune_for_workload,
    get_autotuner,
)
from mlx_primitives.hardware.detection import get_chip_info
from mlx_primitives.hardware.tiling import (
    DataType,
    OperationType,
    ProblemSize,
    TilingConfig,
)
from mlx_primitives.hardware.tiling_database import get_tiling_database


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_default_values(self):
        """Test BenchmarkResult default values."""
        config = TilingConfig(block_m=64, block_n=64)
        result = BenchmarkResult(config=config, time_ms=1.5)

        assert result.config == config
        assert result.time_ms == 1.5
        assert result.gflops == 0.0
        assert result.throughput_gbps == 0.0
        assert result.valid is True

    def test_with_metrics(self):
        """Test BenchmarkResult with performance metrics."""
        config = TilingConfig(block_m=64, block_n=64)
        result = BenchmarkResult(
            config=config,
            time_ms=0.5,
            gflops=500.0,
            throughput_gbps=200.0,
            valid=True,
        )

        assert result.time_ms == 0.5
        assert result.gflops == 500.0
        assert result.throughput_gbps == 200.0

    def test_invalid_result(self):
        """Test BenchmarkResult for failed benchmark."""
        config = TilingConfig(block_m=64, block_n=64)
        result = BenchmarkResult(
            config=config,
            time_ms=float("inf"),
            valid=False,
        )

        assert result.valid is False
        assert result.time_ms == float("inf")


class TestAutoTuner:
    """Tests for AutoTuner class."""

    def test_initialization_defaults(self):
        """Test AutoTuner with default parameters."""
        tuner = AutoTuner()

        assert tuner.database is not None
        assert tuner.chip_info is not None
        assert tuner.warmup_iters == 3
        assert tuner.benchmark_iters == 10
        assert tuner.timeout_ms == 5000.0

    def test_initialization_custom_params(self):
        """Test AutoTuner with custom parameters."""
        database = get_tiling_database()
        chip_info = get_chip_info()

        tuner = AutoTuner(
            database=database,
            chip_info=chip_info,
            warmup_iters=5,
            benchmark_iters=20,
            timeout_ms=10000.0,
        )

        assert tuner.database is database
        assert tuner.chip_info is chip_info
        assert tuner.warmup_iters == 5
        assert tuner.benchmark_iters == 20
        assert tuner.timeout_ms == 10000.0

    def test_get_search_space_attention(self):
        """Test search space generation for attention."""
        tuner = AutoTuner()
        problem_shape = (1, 512, 8, 64)  # batch, seq, heads, head_dim

        candidates = tuner.get_search_space(
            OperationType.FLASH_ATTENTION,
            problem_shape,
            DataType.FP32,
        )

        assert len(candidates) > 0
        for config in candidates:
            assert config.block_m > 0
            assert config.block_n > 0
            assert config.block_k == 64  # head_dim
            assert config.validate()

    def test_get_search_space_matmul(self):
        """Test search space generation for matmul."""
        tuner = AutoTuner()
        problem_shape = (1024, 1024, 1024)

        candidates = tuner.get_search_space(
            OperationType.MATMUL,
            problem_shape,
            DataType.FP32,
        )

        assert len(candidates) > 0
        for config in candidates:
            assert config.block_m > 0
            assert config.block_n > 0
            assert config.block_k > 0
            assert config.validate()

    def test_get_search_space_scan(self):
        """Test search space generation for scan operations."""
        tuner = AutoTuner()
        problem_shape = (1024, 256)

        candidates = tuner.get_search_space(
            OperationType.SCAN,
            problem_shape,
            DataType.FP32,
        )

        assert len(candidates) > 0
        for config in candidates:
            # Scan uses block_m for block size
            assert config.block_m > 0
            # Block size should be power of 2
            assert (config.block_m & (config.block_m - 1)) == 0

    def test_get_search_space_sliding_window(self):
        """Test search space generation for sliding window attention."""
        tuner = AutoTuner()
        problem_shape = (1, 2048, 8, 64)

        candidates = tuner.get_search_space(
            OperationType.SLIDING_WINDOW,
            problem_shape,
            DataType.FP32,
        )

        assert len(candidates) > 0

    def test_benchmark_config_simple_kernel(self):
        """Test benchmarking a simple kernel."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=3)
        config = TilingConfig(block_m=64, block_n=64)

        # Simple kernel function that ignores config
        def simple_kernel(x, config=None):
            return x * 2.0

        inputs = [mx.random.normal((64, 64))]
        mx.eval(*inputs)

        result = tuner.benchmark_config(config, simple_kernel, inputs)

        assert result.valid is True
        assert result.time_ms > 0
        assert result.config == config

    def test_benchmark_config_with_flops(self):
        """Test benchmarking with FLOPS calculation."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=3)
        config = TilingConfig(block_m=64, block_n=64)

        def matmul_kernel(a, b, config=None):
            return mx.matmul(a, b)

        a = mx.random.normal((64, 64))
        b = mx.random.normal((64, 64))
        mx.eval(a, b)

        # 2 * M * N * K FLOPs for matmul
        expected_flops = 2 * 64 * 64 * 64

        result = tuner.benchmark_config(
            config, matmul_kernel, [a, b], expected_flops=expected_flops
        )

        assert result.valid is True
        assert result.gflops > 0

    def test_benchmark_config_failing_kernel(self):
        """Test benchmarking a kernel that raises an exception."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=3)
        config = TilingConfig(block_m=64, block_n=64)

        def failing_kernel(x, config=None):
            raise RuntimeError("Kernel failed")

        inputs = [mx.random.normal((64, 64))]
        mx.eval(*inputs)

        result = tuner.benchmark_config(config, failing_kernel, inputs)

        assert result.valid is False
        assert result.time_ms == float("inf")

    def test_tune_returns_best_config(self):
        """Test that tune returns the best configuration."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=2)

        def simple_kernel(x, config=None):
            return x + 1.0

        inputs = [mx.random.normal((128, 128))]
        mx.eval(*inputs)

        config = tuner.tune(
            operation=OperationType.MATMUL,
            kernel_fn=simple_kernel,
            inputs=inputs,
            save_result=False,
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        assert config.block_n > 0

    def test_tune_with_no_valid_candidates(self):
        """Test tune falls back to database when all candidates fail."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=2)

        def failing_kernel(x, config=None):
            raise RuntimeError("Always fails")

        inputs = [mx.random.normal((128, 128))]
        mx.eval(*inputs)

        # Should return database default when all candidates fail
        config = tuner.tune(
            operation=OperationType.MATMUL,
            kernel_fn=failing_kernel,
            inputs=inputs,
            save_result=False,
        )

        assert isinstance(config, TilingConfig)

    def test_tune_attention_convenience(self):
        """Test tune_attention convenience method."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=2)

        config = tuner.tune_attention(
            batch_size=1,
            seq_len=128,
            num_heads=4,
            head_dim=32,
            causal=True,
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        assert config.block_n > 0

    def test_tune_attention_non_causal(self):
        """Test tune_attention with non-causal attention."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=2)

        config = tuner.tune_attention(
            batch_size=2,
            seq_len=64,
            num_heads=8,
            head_dim=64,
            causal=False,
        )

        assert isinstance(config, TilingConfig)


class TestGlobalAutoTuner:
    """Tests for global auto-tuner functions."""

    def test_get_autotuner_singleton(self):
        """Test that get_autotuner returns singleton."""
        tuner1 = get_autotuner()
        tuner2 = get_autotuner()

        assert tuner1 is tuner2

    def test_get_autotuner_is_autotuner(self):
        """Test that get_autotuner returns AutoTuner instance."""
        tuner = get_autotuner()
        assert isinstance(tuner, AutoTuner)

    def test_auto_tune_for_workload_without_kernel(self):
        """Test auto_tune_for_workload returns database config without kernel."""
        config = auto_tune_for_workload(
            operation=OperationType.MATMUL,
            problem_shape=(512, 512, 512),
            dtype=mx.float32,
            kernel_fn=None,
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0

    def test_auto_tune_for_workload_with_kernel(self):
        """Test auto_tune_for_workload runs benchmarks with kernel."""
        def simple_kernel(x, config=None):
            return x * 2.0

        config = auto_tune_for_workload(
            operation=OperationType.MATMUL,
            problem_shape=(64, 64),
            dtype=mx.float32,
            kernel_fn=simple_kernel,
        )

        assert isinstance(config, TilingConfig)

    def test_auto_tune_for_workload_attention(self):
        """Test auto_tune_for_workload for attention operation."""
        config = auto_tune_for_workload(
            operation=OperationType.FLASH_ATTENTION,
            problem_shape=(1, 256, 8, 64),
            dtype=mx.float32,
        )

        assert isinstance(config, TilingConfig)

    def test_auto_tune_for_workload_fp16(self):
        """Test auto_tune_for_workload with FP16 dtype."""
        config = auto_tune_for_workload(
            operation=OperationType.MATMUL,
            problem_shape=(256, 256, 256),
            dtype=mx.float16,
        )

        assert isinstance(config, TilingConfig)


class TestSearchSpaceConstraints:
    """Tests for search space constraint validation."""

    def test_attention_respects_shared_memory_limit(self):
        """Test attention configs respect shared memory limit."""
        tuner = AutoTuner()
        max_shared = tuner.chip_info.max_threadgroup_memory

        candidates = tuner.get_search_space(
            OperationType.FLASH_ATTENTION,
            (1, 1024, 16, 128),
            DataType.FP32,
        )

        for config in candidates:
            assert config.shared_memory_bytes <= max_shared

    def test_matmul_respects_thread_limit(self):
        """Test matmul configs respect thread limit."""
        tuner = AutoTuner()
        max_threads = tuner.chip_info.max_threads_per_threadgroup

        candidates = tuner.get_search_space(
            OperationType.MATMUL,
            (512, 512, 512),
            DataType.FP32,
        )

        for config in candidates:
            assert config.threads_per_threadgroup <= max_threads

    def test_scan_block_sizes_are_power_of_2(self):
        """Test scan block sizes are powers of 2."""
        tuner = AutoTuner()

        candidates = tuner.get_search_space(
            OperationType.SCAN,
            (2048, 128),
            DataType.FP32,
        )

        for config in candidates:
            block_size = config.block_m
            # Check power of 2: n & (n-1) == 0 for powers of 2
            assert block_size > 0
            assert (block_size & (block_size - 1)) == 0


class TestBenchmarkEdgeCases:
    """Tests for benchmark edge cases."""

    def test_benchmark_empty_inputs(self):
        """Test benchmarking with minimal inputs."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=1)
        config = TilingConfig(block_m=16, block_n=16)

        def identity_kernel(x, config=None):
            return x

        inputs = [mx.zeros((1, 1))]
        mx.eval(*inputs)

        result = tuner.benchmark_config(config, identity_kernel, inputs)
        assert result.valid is True

    def test_benchmark_large_inputs(self):
        """Test benchmarking with larger inputs."""
        tuner = AutoTuner(warmup_iters=1, benchmark_iters=2)
        config = TilingConfig(block_m=128, block_n=128)

        def sum_kernel(x, config=None):
            return mx.sum(x)

        inputs = [mx.random.normal((512, 512))]
        mx.eval(*inputs)

        result = tuner.benchmark_config(config, sum_kernel, inputs)
        assert result.valid is True
        assert result.time_ms > 0

    def test_benchmark_timeout_handling(self):
        """Test that timeout is respected during benchmarking."""
        # Very short timeout
        tuner = AutoTuner(
            warmup_iters=1,
            benchmark_iters=1000,  # Many iterations
            timeout_ms=1.0,  # Very short timeout
        )
        config = TilingConfig(block_m=64, block_n=64)

        def slow_kernel(x, config=None):
            # Do some work
            for _ in range(10):
                x = mx.matmul(x, x.T)
            return x

        inputs = [mx.random.normal((128, 128))]
        mx.eval(*inputs)

        result = tuner.benchmark_config(config, slow_kernel, inputs)
        # Should complete (either valid or timeout) without hanging
        assert isinstance(result, BenchmarkResult)
