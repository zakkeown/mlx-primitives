"""Main runner for parity benchmarks."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from benchmarks.parity.config import ParityBenchmarkConfig, ParitySizeConfig, DEFAULT_CONFIG
from benchmarks.parity.comparison import (
    ComparisonResult,
    compare_frameworks,
    generate_comparison_table,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    framework: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None  # ops/sec or tokens/sec


class ParityBenchmarkRunner:
    """Runner for multi-framework parity benchmarks."""

    SUITE_NAMES = [
        "attention",
        "activations",
        "normalization",
        "fused_ops",
        "quantization",
        "primitives",
        "moe",
        "pooling",
        "embeddings",
        "cache",
        "generation",
        "all",
    ]

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        """Initialize the parity benchmark runner.

        Args:
            config: Benchmark configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self.size_config = ParitySizeConfig()
        self._suites: Dict[str, Any] = {}

    def _init_suites(self) -> None:
        """Lazily initialize benchmark suites."""
        if self._suites:
            return

        from benchmarks.parity.suites import (
            AttentionParityBenchmarks,
            ActivationParityBenchmarks,
            NormalizationParityBenchmarks,
            FusedOpsParityBenchmarks,
            QuantizationParityBenchmarks,
            PrimitivesParityBenchmarks,
            MoEParityBenchmarks,
            PoolingParityBenchmarks,
            EmbeddingsParityBenchmarks,
            CacheParityBenchmarks,
            GenerationParityBenchmarks,
        )

        self._suites = {
            "attention": AttentionParityBenchmarks(self.config),
            "activations": ActivationParityBenchmarks(self.config),
            "normalization": NormalizationParityBenchmarks(self.config),
            "fused_ops": FusedOpsParityBenchmarks(self.config),
            "quantization": QuantizationParityBenchmarks(self.config),
            "primitives": PrimitivesParityBenchmarks(self.config),
            "moe": MoEParityBenchmarks(self.config),
            "pooling": PoolingParityBenchmarks(self.config),
            "embeddings": EmbeddingsParityBenchmarks(self.config),
            "cache": CacheParityBenchmarks(self.config),
            "generation": GenerationParityBenchmarks(self.config),
        }

    def run_suite(self, name: str) -> Dict[str, List[BenchmarkResult]]:
        """Run a specific parity benchmark suite.

        Args:
            name: Suite name (attention, activations, etc.)

        Returns:
            Dictionary mapping framework to list of benchmark results.
        """
        self._init_suites()

        if name not in self._suites:
            raise ValueError(f"Unknown suite: {name}. Available: {list(self._suites.keys())}")

        suite = self._suites[name]
        return suite.run_all()

    def run_all(self) -> Dict[str, Dict[str, List[BenchmarkResult]]]:
        """Run all parity benchmark suites.

        Returns:
            Nested dictionary: {suite_name: {framework: [results]}}
        """
        self._init_suites()

        all_results = {}
        for name, suite in self._suites.items():
            print(f"Running {name} benchmarks...")
            all_results[name] = suite.run_all()

        return all_results

    def run_with_memory_profile(
        self, suite_name: str
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks with memory usage tracking.

        Args:
            suite_name: Name of the suite to benchmark.

        Returns:
            Dictionary with results including memory profiling data.
        """
        raise NotImplementedError("Stub: run_with_memory_profile")

    def run_scaling_analysis(
        self, operation: str, scale_dimension: str
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run scaling analysis for an operation.

        Args:
            operation: Operation name to analyze.
            scale_dimension: Dimension to scale (seq_length, batch_size, etc.)

        Returns:
            Dictionary with scaling analysis results.
        """
        raise NotImplementedError("Stub: run_scaling_analysis")

    def generate_comparison_report(self, results: Dict) -> str:
        """Generate markdown comparison report.

        Args:
            results: Benchmark results dictionary.

        Returns:
            Markdown formatted report string.
        """
        raise NotImplementedError("Stub: generate_comparison_report")

    def export_results(
        self, results: Dict, path: Path, format: str = "json"
    ) -> None:
        """Export results to JSON or CSV.

        Args:
            results: Benchmark results dictionary.
            path: Output file path.
            format: Output format (json or csv).
        """
        raise NotImplementedError("Stub: export_results")


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point for parity benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run MLXPrimitives parity benchmarks against PyTorch and JAX"
    )
    parser.add_argument(
        "--suite",
        choices=ParityBenchmarkRunner.SUITE_NAMES,
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["mlx", "pytorch_mps", "jax_metal"],
        default=["mlx", "pytorch_mps", "jax_metal"],
        help="Frameworks to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable memory profiling",
    )
    parser.add_argument(
        "--scaling-analysis",
        action="store_true",
        help="Run scaling analysis",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Generate markdown report at path",
    )
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        help="Compare against baseline JSON results",
    )

    parsed = parser.parse_args(args)

    config = ParityBenchmarkConfig(
        frameworks=parsed.frameworks,
        profile_memory=parsed.memory_profile,
    )
    runner = ParityBenchmarkRunner(config)

    if parsed.suite == "all":
        results = runner.run_all()
    else:
        results = {parsed.suite: runner.run_suite(parsed.suite)}

    if parsed.output:
        runner.export_results(results, parsed.output)
        print(f"Results exported to {parsed.output}")

    if parsed.markdown_report:
        report = runner.generate_comparison_report(results)
        parsed.markdown_report.write_text(report)
        print(f"Report written to {parsed.markdown_report}")


if __name__ == "__main__":
    main()
