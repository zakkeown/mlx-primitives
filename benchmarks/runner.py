"""Benchmark runner for MLX Primitives."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig
from benchmarks.suites import AttentionBenchmarks, ScanBenchmarks, KernelBenchmarks
from benchmarks.reports.table_formatter import print_results_table, print_comparison_table
from benchmarks.reports.json_exporter import export_to_json
from benchmarks.utils import BenchmarkResult


class BenchmarkRunner:
    """Main benchmark runner orchestrating all suites."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self.attention = AttentionBenchmarks(config=self.config)
        self.scan = ScanBenchmarks(config=self.config)
        self.kernels = KernelBenchmarks(config=self.config)

    def run_suite(self, suite_name: str) -> list[BenchmarkResult]:
        """Run a specific benchmark suite.

        Args:
            suite_name: Name of the suite to run ('attention', 'scan', 'kernels', 'all').

        Returns:
            List of benchmark results.
        """
        results: list[BenchmarkResult] = []

        if suite_name in ("attention", "all"):
            print("Running attention benchmarks...")
            results.extend(self.attention.run_all())

        if suite_name in ("scan", "all"):
            print("Running scan benchmarks...")
            results.extend(self.scan.run_all())

        if suite_name in ("kernels", "all"):
            print("Running kernel benchmarks...")
            results.extend(self.kernels.run_all())

        return results

    def run_scaling_analysis(self) -> list[BenchmarkResult]:
        """Run scaling analysis benchmarks.

        Returns:
            List of benchmark results showing scaling behavior.
        """
        results: list[BenchmarkResult] = []

        print("Running attention scaling analysis...")
        results.extend(self.attention.run_scaling_analysis())

        print("Running SSM scan benchmarks...")
        results.extend(self.scan.run_ssm_benchmark())

        return results

    def get_device_info(self) -> dict[str, str]:
        """Get device information for metadata.

        Returns:
            Dictionary with device info.
        """
        return {
            "device": str(mx.default_device()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for benchmark runner.

    Args:
        args: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="MLX Primitives Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.runner --suite all
  python -m benchmarks.runner --suite attention --output results.json
  python -m benchmarks.runner --scaling
  python -m benchmarks.runner --suite kernels --warmup 5 --iterations 20
        """,
    )

    parser.add_argument(
        "--suite",
        choices=["all", "attention", "scan", "kernels"],
        default="all",
        help="Which benchmark suite to run (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path for JSON results",
    )

    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling analysis benchmarks",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress table output (only write JSON if specified)",
    )

    parsed_args = parser.parse_args(args)

    # Create config
    config = BenchmarkConfig(
        warmup_iterations=parsed_args.warmup,
        benchmark_iterations=parsed_args.iterations,
        seed=parsed_args.seed,
    )

    # Create runner
    runner = BenchmarkRunner(config=config)

    # Run benchmarks
    if parsed_args.scaling:
        results = runner.run_scaling_analysis()
    else:
        results = runner.run_suite(parsed_args.suite)

    if not results:
        print("No benchmark results collected.")
        return 1

    # Print table output
    if not parsed_args.quiet:
        device_info = runner.get_device_info()
        title = f"MLX Primitives Benchmark Suite\nDevice: {device_info['device']} | Date: {device_info['date']}"
        print_results_table(results, title=title)

        # Print comparison if we have both baseline and optimized
        baselines = [r for r in results if r.metadata.get("type") == "baseline"]
        optimized = [r for r in results if r.metadata.get("type") == "optimized"]
        if baselines and optimized:
            print_comparison_table(optimized, baselines)

    # Export to JSON if requested
    if parsed_args.output:
        device_info = runner.get_device_info()
        export_to_json(
            results,
            parsed_args.output,
            metadata={
                **device_info,
                "suite": parsed_args.suite,
                "warmup_iterations": config.warmup_iterations,
                "benchmark_iterations": config.benchmark_iterations,
            },
        )
        print(f"Results exported to {parsed_args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
