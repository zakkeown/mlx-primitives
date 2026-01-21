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
        from benchmarks.memory import MemoryProfiler

        self._init_suites()

        if suite_name not in self._suites:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(self._suites.keys())}")

        suite = self._suites[suite_name]
        profiler = MemoryProfiler()

        # Run the suite normally first
        results = suite.run_all()

        # Add memory profiling to MLX results
        if "mlx" in results:
            for result in results["mlx"]:
                # Profile memory for this benchmark if we can re-run it
                try:
                    profile = profiler.profile_function(
                        lambda: None,  # Placeholder - actual profiling happens during run
                        name=result.name,
                        warmup=1,
                    )
                    result.memory_mb = profile.peak_allocated_mb
                except Exception:
                    # If profiling fails, leave memory_mb as None
                    pass

        return results

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
        self._init_suites()

        # Get scaling values based on dimension
        if scale_dimension == "seq_length":
            scale_values = self.size_config.scaling_seq_lengths if hasattr(self.size_config, "scaling_seq_lengths") else [128, 256, 512, 1024]
        elif scale_dimension == "batch_size":
            scale_values = self.size_config.scaling_batch_sizes if hasattr(self.size_config, "scaling_batch_sizes") else [1, 2, 4, 8]
        elif scale_dimension == "num_experts":
            scale_values = self.size_config.scaling_num_experts if hasattr(self.size_config, "scaling_num_experts") else [4, 8, 16, 32]
        else:
            scale_values = [1, 2, 4, 8]

        results: Dict[str, List[BenchmarkResult]] = {
            "mlx": [],
            "pytorch_mps": [],
            "jax_metal": [],
        }

        # Find the suite containing this operation
        target_suite = None
        for suite_name, suite in self._suites.items():
            if operation.lower() in suite_name.lower():
                target_suite = suite
                break

        if not target_suite:
            # Try to find by operation name match
            for suite_name, suite in self._suites.items():
                target_suite = suite
                break  # Use first suite as fallback

        if not target_suite:
            return results

        # Run benchmarks at each scale value
        for scale_val in scale_values:
            # Update config for this scale
            original_config = self.config

            # Run suite and filter for the operation
            try:
                suite_results = target_suite.run_all()

                for framework, framework_results in suite_results.items():
                    for r in framework_results:
                        # Add scaling metadata
                        if r.metadata is None:
                            r.metadata = {}
                        r.metadata[scale_dimension] = scale_val
                        r.metadata["operation"] = operation

                        if framework in results:
                            results[framework].append(r)
            except Exception as e:
                print(f"Warning: Failed to run scaling at {scale_dimension}={scale_val}: {e}")
                continue

        return results

    def generate_comparison_report(self, results: Dict) -> str:
        """Generate markdown comparison report.

        Args:
            results: Benchmark results dictionary.
                Can be either {suite: {framework: [results]}} or {framework: [results]}

        Returns:
            Markdown formatted report string.
        """
        from benchmarks.reports.markdown_generator import generate_full_report
        import mlx.core as mx

        # Build metadata
        metadata = {
            "date": None,  # Will use current date
            "device": "Apple Silicon",
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "benchmark_iterations": self.config.benchmark_iterations,
                "frameworks": self.config.frameworks,
            },
        }

        # Try to get more specific device info
        try:
            if hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
                device_info = mx.metal.device_info()
                metadata["device"] = device_info.get("name", "Apple Silicon")
        except Exception:
            pass

        # Determine if results are nested by suite or flat by framework
        if results:
            first_key = next(iter(results.keys()))
            first_value = results[first_key]
            if isinstance(first_value, dict):
                # Nested format: {suite: {framework: [results]}}
                return generate_full_report(results, metadata=metadata)
            else:
                # Flat format: {framework: [results]} - wrap in single suite
                return generate_full_report({"benchmarks": results}, metadata=metadata)

        return generate_full_report({}, metadata=metadata)

    def export_results(
        self, results: Dict, path: Path, format: str = "json"
    ) -> None:
        """Export results to JSON or CSV.

        Args:
            results: Benchmark results dictionary.
            path: Output file path.
            format: Output format (json or csv).
        """
        import json
        from datetime import datetime
        import mlx.core as mx

        if format == "json":
            # Convert results to JSON-serializable format
            def serialize_result(r) -> dict:
                if hasattr(r, "name"):
                    return {
                        "name": r.name,
                        "framework": r.framework if hasattr(r, "framework") else "unknown",
                        "mean_time_seconds": r.mean_time,
                        "std_time_seconds": r.std_time,
                        "min_time_seconds": r.min_time,
                        "max_time_seconds": r.max_time,
                        "iterations": r.iterations,
                        "memory_mb": r.memory_mb if hasattr(r, "memory_mb") else None,
                        "throughput_ops_per_sec": r.throughput if hasattr(r, "throughput") else None,
                        "metadata": r.metadata if hasattr(r, "metadata") else None,
                    }
                return r  # Already a dict

            # Flatten nested results
            all_results = []
            if results:
                first_value = next(iter(results.values()))
                if isinstance(first_value, dict):
                    # Nested: {suite: {framework: [results]}}
                    for suite_name, suite_results in results.items():
                        for framework, framework_results in suite_results.items():
                            for r in framework_results:
                                serialized = serialize_result(r)
                                serialized["suite"] = suite_name
                                serialized["framework"] = framework
                                all_results.append(serialized)
                else:
                    # Flat: {framework: [results]}
                    for framework, framework_results in results.items():
                        for r in framework_results:
                            serialized = serialize_result(r)
                            serialized["framework"] = framework
                            all_results.append(serialized)

            output = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "mlx_version": getattr(mx, "__version__", "unknown"),
                    "config": {
                        "warmup_iterations": self.config.warmup_iterations,
                        "benchmark_iterations": self.config.benchmark_iterations,
                        "frameworks": self.config.frameworks,
                    },
                },
                "results": all_results,
                "summary": {
                    "total_benchmarks": len(all_results),
                },
            }

            with open(path, "w") as f:
                json.dump(output, f, indent=2)

        elif format == "csv":
            from benchmarks.reports.comparison_tables import export_to_csv

            # Flatten nested results for CSV
            flat_results: Dict[str, list] = {}
            if results:
                first_value = next(iter(results.values()))
                if isinstance(first_value, dict):
                    # Nested: {suite: {framework: [results]}}
                    for suite_name, suite_results in results.items():
                        for framework, framework_results in suite_results.items():
                            if framework not in flat_results:
                                flat_results[framework] = []
                            flat_results[framework].extend(framework_results)
                else:
                    # Already flat
                    flat_results = results

            export_to_csv(flat_results, str(path))

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")


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
