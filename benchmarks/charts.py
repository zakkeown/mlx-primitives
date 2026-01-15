"""Chart generation for benchmark visualization.

Creates performance charts from benchmark results. Supports both matplotlib
(for detailed charts) and ASCII (for terminal output).
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from benchmarks.runner import BenchmarkResult


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    title: str
    xlabel: str
    ylabel: str
    width: int = 800
    height: int = 600
    show_legend: bool = True
    log_scale_x: bool = False
    log_scale_y: bool = False


class ChartGenerator:
    """Generate charts from benchmark results."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize chart generator.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self._matplotlib_available = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def bar_chart(
        self,
        results: Dict[str, BenchmarkResult],
        config: ChartConfig,
        metric: str = "mean_ms",
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Create a bar chart comparing implementations.

        Args:
            results: Dict of name -> BenchmarkResult
            config: Chart configuration
            metric: Metric to plot (mean_ms, min_ms, throughput, etc.)
            filename: Output filename (without extension)

        Returns:
            Path to saved chart or None
        """
        if not self._matplotlib_available:
            print("matplotlib not available, generating ASCII chart instead")
            return self.ascii_bar_chart(results, config, metric)

        import matplotlib.pyplot as plt

        names = list(results.keys())
        values = [getattr(results[n], metric, 0) for n in names]

        fig, ax = plt.subplots(figsize=(config.width / 100, config.height / 100))

        bars = ax.bar(names, values)

        # Color bars by relative performance
        if values:
            min_val = min(v for v in values if v > 0)
            colors = ["green" if v == min_val else "steelblue" for v in values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)

        if config.log_scale_y:
            ax.set_yscale("log")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if filename and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)

        plt.show()
        plt.close()
        return None

    def line_chart(
        self,
        data: Dict[str, List[Tuple[float, float]]],
        config: ChartConfig,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Create a line chart showing scaling behavior.

        Args:
            data: Dict of series_name -> [(x, y), ...] points
            config: Chart configuration
            filename: Output filename

        Returns:
            Path to saved chart or None
        """
        if not self._matplotlib_available:
            print("matplotlib not available, generating ASCII chart instead")
            return self.ascii_line_chart(data, config)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(config.width / 100, config.height / 100))

        for name, points in data.items():
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            ax.plot(x_vals, y_vals, marker="o", label=name)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)

        if config.log_scale_x:
            ax.set_xscale("log")
        if config.log_scale_y:
            ax.set_yscale("log")

        if config.show_legend:
            ax.legend()

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if filename and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)

        plt.show()
        plt.close()
        return None

    def scaling_chart(
        self,
        results: List[BenchmarkResult],
        scale_param: str,
        config: ChartConfig,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Create a chart showing performance scaling.

        Args:
            results: List of BenchmarkResults at different scales
            scale_param: Name of the scale parameter in result.config
            config: Chart configuration
            filename: Output filename

        Returns:
            Path to saved chart or None
        """
        # Group results by implementation name
        series: Dict[str, List[Tuple[float, float]]] = {}

        for result in results:
            # Extract base name (remove scale suffix if present)
            name_parts = result.name.rsplit("_", 1)
            base_name = name_parts[0] if len(name_parts) > 1 else result.name

            scale = result.config.get(scale_param, 0)

            if base_name not in series:
                series[base_name] = []
            series[base_name].append((scale, result.mean_ms))

        # Sort points by scale
        for name in series:
            series[name].sort(key=lambda p: p[0])

        return self.line_chart(series, config, filename)

    def memory_chart(
        self,
        memory_data: List[Dict[str, Any]],
        scale_param: str,
        config: ChartConfig,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Create a chart showing memory scaling.

        Args:
            memory_data: List of dicts with scale and memory info
            scale_param: Name of the scale parameter
            config: Chart configuration
            filename: Output filename

        Returns:
            Path to saved chart or None
        """
        data = {
            "Peak Memory": [
                (d[scale_param], d.get("peak_memory_mb", 0))
                for d in memory_data
            ]
        }

        return self.line_chart(data, config, filename)

    def ascii_bar_chart(
        self,
        results: Dict[str, BenchmarkResult],
        config: ChartConfig,
        metric: str = "mean_ms",
    ) -> None:
        """Generate ASCII bar chart for terminal output.

        Args:
            results: Dict of name -> BenchmarkResult
            config: Chart configuration
            metric: Metric to plot
        """
        print(f"\n{config.title}")
        print("=" * 60)

        names = list(results.keys())
        values = [getattr(results[n], metric, 0) for n in names]

        if not values or max(values) == 0:
            print("No data to display")
            return

        max_val = max(values)
        max_name_len = max(len(n) for n in names)
        bar_width = 40

        for name, val in zip(names, values):
            bar_len = int((val / max_val) * bar_width) if max_val > 0 else 0
            bar = "█" * bar_len
            print(f"{name:<{max_name_len}} | {bar} {val:.2f}")

        print(f"\n{config.xlabel}: {config.ylabel}")

    def ascii_line_chart(
        self,
        data: Dict[str, List[Tuple[float, float]]],
        config: ChartConfig,
    ) -> None:
        """Generate ASCII representation of line chart data.

        Args:
            data: Dict of series_name -> [(x, y), ...] points
            config: Chart configuration
        """
        print(f"\n{config.title}")
        print("=" * 60)

        for name, points in data.items():
            print(f"\n{name}:")
            print("-" * 40)
            print(f"{'X':<15} | {'Y':<15}")
            print("-" * 40)
            for x, y in sorted(points):
                print(f"{x:<15.2f} | {y:<15.2f}")

        print(f"\nX-axis: {config.xlabel}")
        print(f"Y-axis: {config.ylabel}")

    def speedup_chart(
        self,
        results: Dict[str, BenchmarkResult],
        baseline: str,
        config: ChartConfig,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Create a speedup comparison chart.

        Args:
            results: Dict of name -> BenchmarkResult
            baseline: Name of baseline implementation
            config: Chart configuration
            filename: Output filename

        Returns:
            Path to saved chart or None
        """
        if baseline not in results:
            print(f"Baseline '{baseline}' not found")
            return None

        baseline_time = results[baseline].mean_ms

        # Calculate speedups
        speedup_results = {}
        for name, result in results.items():
            speedup_result = BenchmarkResult(
                name=name,
                config=result.config,
                mean_ms=baseline_time / result.mean_ms,  # This is speedup, not time
                std_ms=0,
                min_ms=0,
                max_ms=0,
                median_ms=0,
            )
            speedup_results[name] = speedup_result

        # Create bar chart with speedup as metric
        speedup_config = ChartConfig(
            title=config.title,
            xlabel="Implementation",
            ylabel="Speedup (x)",
            width=config.width,
            height=config.height,
        )

        return self.bar_chart(speedup_results, speedup_config, "mean_ms", filename)

    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_file: Optional[str] = None,
    ) -> str:
        """Generate a markdown report from benchmark results.

        Args:
            results: List of benchmark results
            output_file: Optional file to save report

        Returns:
            Markdown report string
        """
        lines = [
            "# Benchmark Report",
            "",
            "## System Information",
            "",
        ]

        if results and results[0].system_info:
            info = results[0].system_info
            lines.extend([
                f"- **Platform**: {info.get('platform', 'Unknown')}",
                f"- **CPU**: {info.get('cpu', 'Unknown')}",
                f"- **Memory**: {info.get('total_memory_gb', 'Unknown')} GB",
                f"- **MLX Version**: {info.get('mlx_version', 'Unknown')}",
                "",
            ])

        lines.extend([
            "## Results",
            "",
            "| Benchmark | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |",
            "|-----------|-----------|----------|----------|----------|",
        ])

        for r in results:
            lines.append(
                f"| {r.name} | {r.mean_ms:.3f} | {r.std_ms:.3f} | "
                f"{r.min_ms:.3f} | {r.max_ms:.3f} |"
            )

        lines.append("")

        report = "\n".join(lines)

        if output_file and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / output_file
            with open(filepath, "w") as f:
                f.write(report)

        return report
