"""Scan/cumsum benchmark suite."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from benchmarks.config import BenchmarkConfig, ScanSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn
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

        # Extended SSM/Mamba benchmarks
        results.extend(self.run_ssm_benchmark())
        results.extend(self.run_selective_scan_benchmarks())
        results.extend(self.run_mamba_block_benchmarks())
        results.extend(self.run_s4_layer_benchmarks())

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

        def fn():
            return naive_cumsum(x, axis=1)

        # Benchmark (includes warmup)
        name = f"native_cumsum_s{seq_len}_d{feature_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "batch_size": batch_size,
            "type": "baseline",
        }
        return result

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
            from mlx_primitives.primitives.scan import associative_scan
        except ImportError:
            print(f"  Warning: associative_scan not available, skipping optimized benchmark")
            return None

        mx.random.seed(self.config.seed)

        # Create input tensor
        x = mx.random.normal((batch_size, seq_len, feature_dim))

        def fn():
            # Use add operator for cumsum equivalent
            return associative_scan(x, operator="add", axis=1)

        # Benchmark (includes warmup)
        name = f"assoc_scan_s{seq_len}_d{feature_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "batch_size": batch_size,
            "type": "optimized",
        }
        return result

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
            from mlx_primitives.primitives.scan import associative_scan
        except ImportError:
            print(f"  Warning: associative_scan not available, skipping SSM benchmark")
            return None

        mx.random.seed(self.config.seed)

        # Create SSM-style inputs
        # A: state transition [batch, seq, state_dim]
        # B: input projection [batch, seq, state_dim]
        A = mx.random.normal((batch_size, seq_len, state_dim))
        B = mx.random.normal((batch_size, seq_len, state_dim))

        def fn():
            # SSM scan combines state transitions
            return associative_scan(A * B, operator="add", axis=1)

        # Benchmark (includes warmup)
        name = f"ssm_scan_s{seq_len}_d{state_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "seq_len": seq_len,
            "state_dim": state_dim,
            "batch_size": batch_size,
            "type": "ssm",
        }
        return result

    def run_selective_scan_benchmarks(self) -> list[BenchmarkResult]:
        """Run selective scan (Mamba-style) benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        seq_lengths = [256, 512, 1024, 2048]
        hidden_dims = [256, 512, 1024]

        for seq_len in seq_lengths[:3]:
            for hidden_dim in hidden_dims[:2]:
                result = self._benchmark_selective_scan(
                    batch_size=2,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    state_dim=16,
                )
                if result:
                    results.append(result)

        return results

    def run_mamba_block_benchmarks(self) -> list[BenchmarkResult]:
        """Run MambaBlock benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        seq_lengths = [256, 512, 1024]
        hidden_dims = [256, 512]

        for seq_len in seq_lengths:
            for hidden_dim in hidden_dims:
                result = self._benchmark_mamba_block(
                    batch_size=2,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                )
                if result:
                    results.append(result)

        return results

    def run_s4_layer_benchmarks(self) -> list[BenchmarkResult]:
        """Run S4 (Structured State Space) layer benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        seq_lengths = [256, 512, 1024, 2048]
        hidden_dims = [128, 256, 512]

        for seq_len in seq_lengths[:3]:
            for hidden_dim in hidden_dims[:2]:
                result = self._benchmark_s4_layer(
                    batch_size=2,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    state_dim=64,
                )
                if result:
                    results.append(result)

        return results

    def _benchmark_selective_scan(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        state_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Mamba-style selective scan.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            state_dim: State dimension (N in Mamba).

        Returns:
            Benchmark result if available, None otherwise.
        """
        try:
            from mlx_primitives.advanced.ssm import selective_scan
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Selective scan inputs matching Mamba architecture
        # x: (batch, seq_len, hidden_dim)
        # delta: (batch, seq_len, hidden_dim) - discretization step
        # A: (hidden_dim, state_dim) - state matrix
        # B: (batch, seq_len, state_dim) - input projection
        # C: (batch, seq_len, state_dim) - output projection
        # D: (hidden_dim,) - skip connection
        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        delta = nn.softplus(mx.random.normal((batch_size, seq_len, hidden_dim)))
        A = -mx.exp(mx.random.normal((hidden_dim, state_dim)))
        B = mx.random.normal((batch_size, seq_len, state_dim))
        C = mx.random.normal((batch_size, seq_len, state_dim))
        D = mx.random.normal((hidden_dim,))

        def fn():
            return selective_scan(x, delta, A, B, C, D)

        name = f"selective_scan_b{batch_size}_s{seq_len}_d{hidden_dim}_n{state_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "state_dim": state_dim,
            "type": "optimized",
            "operation": "selective_scan",
        }
        return result

    def _benchmark_mamba_block(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        expand: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
    ) -> Optional[BenchmarkResult]:
        """Benchmark full MambaBlock.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            expand: Expansion factor for inner dimension.
            state_dim: State dimension.
            conv_kernel: Convolution kernel size.

        Returns:
            Benchmark result if available, None otherwise.
        """
        try:
            from mlx_primitives.advanced.ssm import MambaBlock
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        block = MambaBlock(
            dims=hidden_dim,
            expand=expand,
            d_state=state_dim,
            d_conv=conv_kernel,
        )
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return block(x)

        name = f"mamba_block_b{batch_size}_s{seq_len}_d{hidden_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "expand": expand,
            "state_dim": state_dim,
            "conv_kernel": conv_kernel,
            "type": "optimized",
            "operation": "mamba_block",
        }
        return result

    def _benchmark_s4_layer(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        state_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark S4 (Structured State Space) layer.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            hidden_dim: Hidden dimension.
            state_dim: State dimension.

        Returns:
            Benchmark result if available, None otherwise.
        """
        try:
            from mlx_primitives.advanced.ssm import S4Layer
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = S4Layer(
            dims=hidden_dim,
            d_state=state_dim,
        )
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"s4_layer_b{batch_size}_s{seq_len}_d{hidden_dim}_n{state_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "state_dim": state_dim,
            "type": "optimized",
            "operation": "s4_layer",
        }
        return result

    def run_parallel_scan_benchmarks(self) -> list[BenchmarkResult]:
        """Run parallel scan operation benchmarks.

        Returns:
            List of benchmark results comparing sequential vs parallel.
        """
        results = []
        seq_lengths = [512, 1024, 2048, 4096, 8192]

        for seq_len in seq_lengths:
            # Benchmark parallel associative scan
            result = self._benchmark_parallel_scan(
                batch_size=4,
                seq_len=seq_len,
                feature_dim=64,
            )
            if result:
                results.append(result)

        return results

    def _benchmark_parallel_scan(
        self,
        batch_size: int,
        seq_len: int,
        feature_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark parallel scan with work-efficient algorithm.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            feature_dim: Feature dimension.

        Returns:
            Benchmark result if available, None otherwise.
        """
        try:
            from mlx_primitives.primitives.scan import parallel_associative_scan
        except ImportError:
            # Fall back to standard associative scan
            try:
                from mlx_primitives.primitives.scan import associative_scan
                parallel_associative_scan = lambda x, op, axis: associative_scan(x, op=op, axis=axis)
            except ImportError:
                return None

        mx.random.seed(self.config.seed)

        x = mx.random.normal((batch_size, seq_len, feature_dim))

        def fn():
            return parallel_associative_scan(x, operator="add", axis=1)

        name = f"parallel_scan_b{batch_size}_s{seq_len}_d{feature_dim}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "type": "optimized",
            "operation": "parallel_scan",
        }
        return result

    def run_scaling_analysis(
        self,
        seq_lengths: Optional[list[int]] = None,
    ) -> list[BenchmarkResult]:
        """Run scaling analysis across sequence lengths for all scan variants.

        Args:
            seq_lengths: Sequence lengths to test.

        Returns:
            List of benchmark results showing scaling behavior.
        """
        seq_lengths = seq_lengths or [128, 256, 512, 1024, 2048, 4096, 8192]
        results = []

        for seq_len in seq_lengths:
            # Native cumsum baseline
            baseline = self._benchmark_native_cumsum(
                seq_len=seq_len,
                feature_dim=64,
                batch_size=4,
            )
            results.append(baseline)

            # Associative scan
            assoc = self._benchmark_associative_scan(
                seq_len=seq_len,
                feature_dim=64,
                batch_size=4,
            )
            if assoc:
                results.append(assoc)

            # SSM scan
            ssm = self._benchmark_ssm_scan(
                seq_len=seq_len,
                state_dim=16,
                batch_size=4,
            )
            if ssm:
                results.append(ssm)

            # Selective scan (if available)
            selective = self._benchmark_selective_scan(
                batch_size=2,
                seq_len=seq_len,
                hidden_dim=256,
                state_dim=16,
            )
            if selective:
                results.append(selective)

        return results
