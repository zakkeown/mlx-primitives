"""Benchmark runner for mlx-primitives.

Provides infrastructure for running reproducible benchmarks and collecting metrics.
"""

import time
import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any
from pathlib import Path
import statistics

import mlx.core as mx


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    name: str
    warmup_runs: int = 3
    benchmark_runs: int = 10
    min_time_seconds: float = 0.1  # Minimum total benchmark time
    gc_between_runs: bool = True
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    config: Dict[str, Any]

    # Timing metrics (in milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    # Throughput metrics
    throughput: Optional[float] = None
    throughput_unit: Optional[str] = None

    # Memory metrics (in MB)
    peak_memory_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None

    # System info
    system_info: Dict[str, str] = field(default_factory=dict)

    # Raw timings
    raw_timings_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(**data)


class BenchmarkRunner:
    """Run benchmarks and collect results."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, str]:
        """Collect system information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        }

        # Try to get Apple Silicon chip info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["cpu"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Get memory info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["total_memory_gb"] = f"{mem_bytes / (1024**3):.1f}"
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        return info

    def benchmark(
        self,
        func: Callable,
        config: BenchmarkConfig,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        throughput_items: Optional[int] = None,
        throughput_unit: str = "items/sec",
    ) -> BenchmarkResult:
        """Run a benchmark.

        Args:
            func: Function to benchmark (should call mx.eval internally)
            config: Benchmark configuration
            setup: Optional setup function called before each run
            teardown: Optional teardown function called after each run
            throughput_items: Number of items processed per run (for throughput calc)
            throughput_unit: Unit for throughput measurement

        Returns:
            BenchmarkResult with timing metrics
        """
        mx.random.seed(config.seed)

        # Warmup runs
        for _ in range(config.warmup_runs):
            if setup:
                setup()
            func()
            if teardown:
                teardown()

        # Benchmark runs
        timings = []
        total_time = 0.0
        run_count = 0

        while run_count < config.benchmark_runs or total_time < config.min_time_seconds:
            if config.gc_between_runs:
                mx.synchronize()

            if setup:
                setup()

            start = time.perf_counter()
            func()
            mx.synchronize()  # Ensure GPU work is complete
            end = time.perf_counter()

            if teardown:
                teardown()

            elapsed_ms = (end - start) * 1000
            timings.append(elapsed_ms)
            total_time += (end - start)
            run_count += 1

            # Safety limit
            if run_count > 1000:
                break

        # Compute statistics
        result = BenchmarkResult(
            name=config.name,
            config={"warmup_runs": config.warmup_runs, "benchmark_runs": run_count},
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            min_ms=min(timings),
            max_ms=max(timings),
            median_ms=statistics.median(timings),
            raw_timings_ms=timings,
            system_info=self.system_info,
        )

        # Compute throughput if items specified
        if throughput_items is not None:
            result.throughput = throughput_items / (result.mean_ms / 1000)
            result.throughput_unit = throughput_unit

        self.results.append(result)
        return result

    def benchmark_scaling(
        self,
        func_factory: Callable[[int], Callable],
        scales: List[int],
        config_factory: Callable[[int], BenchmarkConfig],
        throughput_factory: Optional[Callable[[int], int]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarks across different scales.

        Args:
            func_factory: Function that creates benchmark function for given scale
            scales: List of scale values to test
            config_factory: Function that creates config for given scale
            throughput_factory: Function that returns throughput items for given scale

        Returns:
            List of BenchmarkResults
        """
        results = []

        for scale in scales:
            func = func_factory(scale)
            config = config_factory(scale)
            throughput_items = throughput_factory(scale) if throughput_factory else None

            result = self.benchmark(func, config, throughput_items=throughput_items)
            results.append(result)

        return results

    def compare(
        self,
        implementations: Dict[str, Callable],
        config: BenchmarkConfig,
        setup: Optional[Callable] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple implementations.

        Args:
            implementations: Dict of name -> function pairs
            config: Benchmark configuration (name will be overridden)
            setup: Optional setup function

        Returns:
            Dict of name -> BenchmarkResult
        """
        results = {}

        for name, func in implementations.items():
            impl_config = BenchmarkConfig(
                name=f"{config.name}_{name}",
                warmup_runs=config.warmup_runs,
                benchmark_runs=config.benchmark_runs,
                min_time_seconds=config.min_time_seconds,
                seed=config.seed,
            )

            result = self.benchmark(func, impl_config, setup=setup)
            results[name] = result

        return results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        if self.output_dir is None:
            raise ValueError("No output directory specified")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename

        data = {
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_results(self, filename: str = "benchmark_results.json"):
        """Load results from JSON file."""
        if self.output_dir is None:
            raise ValueError("No output directory specified")

        filepath = self.output_dir / filename

        with open(filepath) as f:
            data = json.load(f)

        self.system_info = data["system_info"]
        self.results = [BenchmarkResult.from_dict(r) for r in data["results"]]

    def print_results(self, results: Optional[List[BenchmarkResult]] = None):
        """Print benchmark results in a formatted table."""
        results = results or self.results

        if not results:
            print("No benchmark results to display.")
            return

        # Header
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"\nSystem: {self.system_info.get('cpu', 'Unknown')}")
        print(f"Memory: {self.system_info.get('total_memory_gb', 'Unknown')} GB")
        print(f"MLX Version: {self.system_info.get('mlx_version', 'Unknown')}")
        print()

        # Results table
        print(f"{'Name':<40} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Throughput':<20}")
        print("-" * 96)

        for r in results:
            throughput_str = f"{r.throughput:.2f} {r.throughput_unit}" if r.throughput else "N/A"
            print(f"{r.name:<40} {r.mean_ms:<12.3f} {r.std_ms:<12.3f} {r.min_ms:<12.3f} {throughput_str:<20}")

        print()

    def print_comparison(self, results: Dict[str, BenchmarkResult], baseline: str):
        """Print comparison results with speedup relative to baseline."""
        if baseline not in results:
            print(f"Baseline '{baseline}' not found in results")
            return

        baseline_result = results[baseline]

        print("\n" + "=" * 80)
        print(f"COMPARISON (baseline: {baseline})")
        print("=" * 80)
        print()

        print(f"{'Implementation':<30} {'Mean (ms)':<12} {'Speedup':<12}")
        print("-" * 54)

        for name, result in sorted(results.items(), key=lambda x: x[1].mean_ms):
            speedup = baseline_result.mean_ms / result.mean_ms
            speedup_str = f"{speedup:.2f}x" if name != baseline else "(baseline)"
            print(f"{name:<30} {result.mean_ms:<12.3f} {speedup_str:<12}")

        print()


def timed(func: Callable, warmup: int = 3, runs: int = 10) -> float:
    """Simple timing utility.

    Args:
        func: Function to time (should include mx.eval)
        warmup: Number of warmup iterations
        runs: Number of timed iterations

    Returns:
        Mean time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Time
    timings = []
    for _ in range(runs):
        mx.synchronize()
        start = time.perf_counter()
        func()
        mx.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    return statistics.mean(timings)
