"""Benchmark runner for MLX Primitives."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig
from benchmarks.suites import (
    AttentionBenchmarks,
    ScanBenchmarks,
    KernelBenchmarks,
    LayerBenchmarks,
    MoEBenchmarks,
    CacheBenchmarks,
    GenerationBenchmarks,
    TrainingBenchmarks,
    QuantizationBenchmarks,
)
from benchmarks.reports.table_formatter import print_results_table, print_comparison_table
from benchmarks.reports.json_exporter import export_to_json
from benchmarks.utils import BenchmarkResult

# Available suites mapping
SUITE_NAMES = [
    "attention",
    "scan",
    "kernels",
    "layers",
    "moe",
    "cache",
    "generation",
    "training",
    "quantization",
    "all",
]


class BenchmarkRunner:
    """Main benchmark runner orchestrating all suites."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()

        # Core suites
        self.attention = AttentionBenchmarks(config=self.config)
        self.scan = ScanBenchmarks(config=self.config)
        self.kernels = KernelBenchmarks(config=self.config)

        # New suites
        self.layers = LayerBenchmarks(config=self.config)
        self.moe = MoEBenchmarks(config=self.config)
        self.cache = CacheBenchmarks(config=self.config)
        self.generation = GenerationBenchmarks(config=self.config)
        self.training = TrainingBenchmarks(config=self.config)
        self.quantization = QuantizationBenchmarks(config=self.config)

        # External baselines (lazy loaded)
        self._pytorch_baselines = None
        self._jax_baselines = None

    @property
    def pytorch_baselines(self):
        """Lazy load PyTorch MPS baselines."""
        if self._pytorch_baselines is None:
            try:
                from benchmarks.baselines.pytorch_mps import PyTorchMPSBenchmarks, pytorch_available
                if pytorch_available():
                    self._pytorch_baselines = PyTorchMPSBenchmarks()
            except Exception:
                pass
        return self._pytorch_baselines

    @property
    def jax_baselines(self):
        """Lazy load JAX Metal baselines."""
        if self._jax_baselines is None:
            try:
                from benchmarks.baselines.jax_metal import JAXMetalBenchmarks, jax_available
                if jax_available():
                    self._jax_baselines = JAXMetalBenchmarks()
            except Exception:
                pass
        return self._jax_baselines

    def run_suite(self, suite_name: str) -> list[BenchmarkResult]:
        """Run a specific benchmark suite.

        Args:
            suite_name: Name of the suite to run.

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

        if suite_name in ("layers", "all"):
            print("Running layer benchmarks...")
            results.extend(self.layers.run_all())

        if suite_name in ("moe", "all"):
            print("Running MoE benchmarks...")
            results.extend(self.moe.run_all())

        if suite_name in ("cache", "all"):
            print("Running cache benchmarks...")
            results.extend(self.cache.run_all())

        if suite_name in ("generation", "all"):
            print("Running generation benchmarks...")
            results.extend(self.generation.run_all())

        if suite_name in ("training", "all"):
            print("Running training benchmarks...")
            results.extend(self.training.run_all())

        if suite_name in ("quantization", "all"):
            print("Running quantization benchmarks...")
            results.extend(self.quantization.run_all())

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

    def run_with_baselines(
        self,
        suite_name: str = "attention",
        include_pytorch: bool = True,
        include_jax: bool = True,
    ) -> dict[str, list]:
        """Run benchmarks with external framework baselines for comparison.

        Args:
            suite_name: Suite to run (attention, kernels, etc.).
            include_pytorch: Include PyTorch MPS comparison.
            include_jax: Include JAX Metal comparison.

        Returns:
            Dictionary with results from each framework.
        """
        results = {
            "mlx": self.run_suite(suite_name),
        }

        if include_pytorch and self.pytorch_baselines:
            print("Running PyTorch MPS baselines...")
            pytorch_results = self._run_pytorch_baselines(suite_name)
            if pytorch_results:
                results["pytorch_mps"] = pytorch_results

        if include_jax and self.jax_baselines:
            print("Running JAX Metal baselines...")
            jax_results = self._run_jax_baselines(suite_name)
            if jax_results:
                results["jax_metal"] = jax_results

        return results

    def _run_pytorch_baselines(self, suite_name: str) -> list:
        """Run PyTorch baseline benchmarks for a suite."""
        if not self.pytorch_baselines:
            return []

        results = []
        iterations = self.config.benchmark_iterations

        if suite_name in ("attention", "all"):
            # Attention benchmarks
            for seq_len in [128, 512, 1024, 2048]:
                result = self.pytorch_baselines.benchmark_attention(
                    batch_size=1, seq_len=seq_len, num_heads=8, head_dim=64,
                    iterations=iterations,
                )
                results.append(result)

        if suite_name in ("kernels", "all"):
            # Layer norm benchmarks
            for hidden_dim in [512, 1024, 2048]:
                result = self.pytorch_baselines.benchmark_layer_norm(
                    batch_size=4, seq_len=512, hidden_dim=hidden_dim,
                    iterations=iterations,
                )
                results.append(result)

            # GELU benchmarks
            for hidden_dim in [512, 1024, 2048]:
                result = self.pytorch_baselines.benchmark_gelu(
                    batch_size=4, seq_len=512, hidden_dim=hidden_dim,
                    iterations=iterations,
                )
                results.append(result)

        return results

    def _run_jax_baselines(self, suite_name: str) -> list:
        """Run JAX baseline benchmarks for a suite."""
        if not self.jax_baselines:
            return []

        results = []
        iterations = self.config.benchmark_iterations

        if suite_name in ("attention", "all"):
            # Attention benchmarks
            for seq_len in [128, 512, 1024, 2048]:
                result = self.jax_baselines.benchmark_attention(
                    batch_size=1, seq_len=seq_len, num_heads=8, head_dim=64,
                    iterations=iterations,
                )
                results.append(result)

        if suite_name in ("scan", "all"):
            # Associative scan benchmarks
            for seq_len in [128, 512, 1024, 2048]:
                result = self.jax_baselines.benchmark_associative_scan(
                    batch_size=4, seq_len=seq_len, dim=64,
                    iterations=iterations,
                )
                results.append(result)

        if suite_name in ("kernels", "all"):
            # Layer norm benchmarks
            for hidden_dim in [512, 1024, 2048]:
                result = self.jax_baselines.benchmark_layer_norm(
                    batch_size=4, seq_len=512, hidden_dim=hidden_dim,
                    iterations=iterations,
                )
                results.append(result)

        return results

    def get_device_info(self) -> dict[str, str]:
        """Get device information for metadata.

        Returns:
            Dictionary with device info.
        """
        info = {
            "device": str(mx.default_device()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add external framework availability
        info["pytorch_available"] = str(self.pytorch_baselines is not None)
        info["jax_available"] = str(self.jax_baselines is not None)

        return info


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
  python -m benchmarks.runner --suite kernels --warmup 5 --iterations 30
  python -m benchmarks.runner --suite attention --compare-pytorch --compare-jax
  python -m benchmarks.runner --suite all --adaptive-iterations
        """,
    )

    parser.add_argument(
        "--suite",
        choices=SUITE_NAMES,
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
        default=5,
        help="Number of warmup iterations (default: 5)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30)",
    )

    parser.add_argument(
        "--adaptive-iterations",
        action="store_true",
        help="Use adaptive iteration counts based on operation speed",
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

    parser.add_argument(
        "--compare-pytorch",
        action="store_true",
        help="Include PyTorch MPS baseline comparison",
    )

    parser.add_argument(
        "--compare-jax",
        action="store_true",
        help="Include JAX Metal baseline comparison",
    )

    parsed_args = parser.parse_args(args)

    # Create config
    config = BenchmarkConfig(
        warmup_iterations=parsed_args.warmup,
        benchmark_iterations=parsed_args.iterations,
        seed=parsed_args.seed,
        adaptive_iterations=parsed_args.adaptive_iterations,
    )

    # Create runner
    runner = BenchmarkRunner(config=config)

    # Run benchmarks
    if parsed_args.scaling:
        results = runner.run_scaling_analysis()
    elif parsed_args.compare_pytorch or parsed_args.compare_jax:
        all_results = runner.run_with_baselines(
            suite_name=parsed_args.suite,
            include_pytorch=parsed_args.compare_pytorch,
            include_jax=parsed_args.compare_jax,
        )
        # Flatten results for output
        results = all_results.get("mlx", [])
        # Print comparison tables for each framework
        if not parsed_args.quiet:
            for framework, framework_results in all_results.items():
                if framework != "mlx" and framework_results:
                    print(f"\n{framework.upper()} Baseline Results:")
                    for r in framework_results:
                        print(f"  {r.name}: {r.mean_time*1000:.3f}ms")
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
        baselines = [r for r in results if r.metadata and r.metadata.get("type") == "baseline"]
        optimized = [r for r in results if r.metadata and r.metadata.get("type") == "optimized"]
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
                "adaptive_iterations": config.adaptive_iterations,
            },
        )
        print(f"Results exported to {parsed_args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
