"""Scan/cumsum benchmark suite."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, ScanSizes
from benchmarks.utils import BenchmarkResult, warmup, benchmark_fn
from benchmarks.baselines.mlx_native import naive_cumsum


class ScanBenchmarks:
    """Benchmark suite for scan/cumsum operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[ScanSizes] = None,
    ):
        """Initialize scan benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for scan benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or ScanSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all scan benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []

        for seq_len in self.sizes.seq_lengths:
            for feature_dim in self.sizes.feature_dims:
                # Run native cumsum baseline
                baseline_result = self._benchmark_native_cumsum(
                    seq_len=seq_len,
                    feature_dim=feature_dim,
                    batch_size=self.sizes.batch_size,
                )
                results.append(baseline_result)

                # Run associative scan if available
                scan_result = self._benchmark_associative_scan(
                    seq_len=seq_len,
                    feature_dim=feature_dim,
                    batch_size=self.sizes.batch_size,
                )
                if scan_result is not None:
                    results.append(scan_result)

        return results

    def _benchmark_native_cumsum(
        self,
        seq_len: int,
        feature_dim: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark native MLX cumsum.

        Args:
            seq_len: Sequence length.
            feature_dim: Feature dimension.
            batch_size: Batch size.

        Returns:
            Benchmark result.
        """
        mx.random.seed(self.config.seed)

        # Create input tensor
        x = mx.random.normal((batch_size, seq_len, feature_dim))

        # Warmup
        def fn():
            result = naive_cumsum(x, axis=1)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)

        # Benchmark
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"native_cumsum_s{seq_len}_d{feature_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "seq_len": seq_len,
                "feature_dim": feature_dim,
                "batch_size": batch_size,
                "type": "baseline",
            },
        )

    def _benchmark_associative_scan(
        self,
        seq_len: int,
        feature_dim: int,
        batch_size: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark associative scan implementation.

        Args:
            seq_len: Sequence length.
            feature_dim: Feature dimension.
            batch_size: Batch size.

        Returns:
            Benchmark result if associative scan is available, None otherwise.
        """
        try:
            from mlx_primitives.scan.associative import associative_scan
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Create input tensor
        x = mx.random.normal((batch_size, seq_len, feature_dim))

        # Warmup
        def fn():
            # Use add operator for cumsum equivalent
            result = associative_scan(x, op="add", axis=1)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)

        # Benchmark
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"assoc_scan_s{seq_len}_d{feature_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "seq_len": seq_len,
                "feature_dim": feature_dim,
                "batch_size": batch_size,
                "type": "optimized",
            },
        )

    def run_ssm_benchmark(
        self,
        seq_lengths: Optional[list[int]] = None,
        state_dim: int = 16,
    ) -> list[BenchmarkResult]:
        """Run SSM-specific scan benchmarks.

        Tests the scan operation used in state space models (Mamba-style).

        Args:
            seq_lengths: Sequence lengths to test.
            state_dim: State dimension for SSM.

        Returns:
            List of benchmark results.
        """
        seq_lengths = seq_lengths or [256, 512, 1024, 2048, 4096]
        results = []

        for seq_len in seq_lengths:
            # SSM scan with state
            ssm_result = self._benchmark_ssm_scan(
                seq_len=seq_len,
                state_dim=state_dim,
                batch_size=4,
            )
            if ssm_result:
                results.append(ssm_result)

        return results

    def _benchmark_ssm_scan(
        self,
        seq_len: int,
        state_dim: int,
        batch_size: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark SSM-style scan operation.

        Args:
            seq_len: Sequence length.
            state_dim: State dimension.
            batch_size: Batch size.

        Returns:
            Benchmark result if available, None otherwise.
        """
        try:
            from mlx_primitives.scan.associative import associative_scan
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Create SSM-style inputs
        # A: state transition [batch, seq, state_dim]
        # B: input projection [batch, seq, state_dim]
        A = mx.random.normal((batch_size, seq_len, state_dim))
        B = mx.random.normal((batch_size, seq_len, state_dim))

        # Warmup
        def fn():
            # SSM scan combines state transitions
            result = associative_scan(A * B, op="add", axis=1)
            mx.eval(result)
            return result

        warmup(fn, self.config.warmup_iterations)

        # Benchmark
        result = benchmark_fn(fn, self.config.benchmark_iterations)

        return BenchmarkResult(
            name=f"ssm_scan_s{seq_len}_d{state_dim}",
            mean_time=result.mean_time,
            std_time=result.std_time,
            min_time=result.min_time,
            max_time=result.max_time,
            iterations=result.iterations,
            metadata={
                "seq_len": seq_len,
                "state_dim": state_dim,
                "batch_size": batch_size,
                "type": "ssm",
            },
        )
