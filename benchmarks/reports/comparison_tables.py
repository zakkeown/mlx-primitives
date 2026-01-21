"""Comparison table generation for MLX vs PyTorch vs JAX."""

import csv
from typing import Any, Dict, List, Optional

from benchmarks.reports.markdown_generator import format_time, format_speedup


class ComparisonTableGenerator:
    """Generate comparison tables for multi-framework benchmarks."""

    def __init__(self, results: Dict[str, Any]):
        """Initialize the table generator.

        Args:
            results: Benchmark results dictionary mapping framework to results list.
        """
        self.results = results

    def _get_results_list(self, framework: str) -> List[Any]:
        """Get results list for a framework."""
        return self.results.get(framework, self.results.get(f"{framework}_mps", []))

    def _extract_field(self, result: Any, field: str, default: Any = None) -> Any:
        """Extract field from result (handles both objects and dicts)."""
        if hasattr(result, field):
            return getattr(result, field)
        elif isinstance(result, dict):
            return result.get(field, default)
        return default

    def generate_throughput_table(self) -> str:
        """Generate throughput (ops/sec or tokens/sec) comparison table.

        Returns:
            Markdown formatted throughput table.
        """
        lines = [
            "### Throughput Comparison\n",
            "| Benchmark | MLX (ops/s) | PyTorch (ops/s) | JAX (ops/s) |",
            "|-----------|-------------|-----------------|-------------|",
        ]

        mlx_results = self._get_results_list("mlx")
        pytorch_results = self._get_results_list("pytorch")
        jax_results = self._get_results_list("jax")

        # Build lookups by name
        def build_lookup(results: List[Any]) -> Dict[str, Any]:
            return {self._extract_field(r, "name", ""): r for r in results}

        mlx_lookup = build_lookup(mlx_results)
        pytorch_lookup = build_lookup(pytorch_results)
        jax_lookup = build_lookup(jax_results)

        for name, mlx_r in mlx_lookup.items():
            mlx_throughput = self._extract_field(mlx_r, "throughput_ops_per_sec")
            mlx_str = f"{mlx_throughput:,.0f}" if mlx_throughput else "-"

            pytorch_r = pytorch_lookup.get(name)
            pytorch_throughput = self._extract_field(pytorch_r, "throughput_ops_per_sec") if pytorch_r else None
            pytorch_str = f"{pytorch_throughput:,.0f}" if pytorch_throughput else "-"

            jax_r = jax_lookup.get(name)
            jax_throughput = self._extract_field(jax_r, "throughput_ops_per_sec") if jax_r else None
            jax_str = f"{jax_throughput:,.0f}" if jax_throughput else "-"

            lines.append(f"| {name} | {mlx_str} | {pytorch_str} | {jax_str} |")

        lines.append("")
        return "\n".join(lines)

    def generate_latency_table(self) -> str:
        """Generate latency comparison table (mean, p50, p95, p99).

        Returns:
            Markdown formatted latency table.
        """
        lines = [
            "### Latency Comparison\n",
            "| Benchmark | Mean | Median (p50) | p95 | p99 |",
            "|-----------|------|--------------|-----|-----|",
        ]

        mlx_results = self._get_results_list("mlx")

        for r in mlx_results:
            name = self._extract_field(r, "name", "unknown")
            mean_time = self._extract_field(r, "mean_time", 0)
            median_time = self._extract_field(r, "median_time")
            p95 = self._extract_field(r, "percentile_95")
            p99 = self._extract_field(r, "percentile_99")

            mean_str = format_time(mean_time) if mean_time else "-"
            median_str = format_time(median_time) if median_time else "-"
            p95_str = format_time(p95) if p95 else "-"
            p99_str = format_time(p99) if p99 else "-"

            lines.append(f"| {name} | {mean_str} | {median_str} | {p95_str} | {p99_str} |")

        lines.append("")
        return "\n".join(lines)

    def generate_memory_table(self) -> str:
        """Generate memory usage comparison table.

        Returns:
            Markdown formatted memory table.
        """
        lines = [
            "### Memory Usage Comparison\n",
            "| Benchmark | MLX (MB) | PyTorch (MB) | JAX (MB) | MLX/PyTorch | MLX/JAX |",
            "|-----------|----------|--------------|----------|-------------|---------|",
        ]

        mlx_results = self._get_results_list("mlx")
        pytorch_results = self._get_results_list("pytorch")
        jax_results = self._get_results_list("jax")

        def build_lookup(results: List[Any]) -> Dict[str, Any]:
            return {self._extract_field(r, "name", ""): r for r in results}

        mlx_lookup = build_lookup(mlx_results)
        pytorch_lookup = build_lookup(pytorch_results)
        jax_lookup = build_lookup(jax_results)

        for name, mlx_r in mlx_lookup.items():
            mlx_mem = self._extract_field(mlx_r, "memory_mb")
            mlx_str = f"{mlx_mem:.1f}" if mlx_mem else "-"

            pytorch_r = pytorch_lookup.get(name)
            pytorch_mem = self._extract_field(pytorch_r, "memory_mb") if pytorch_r else None
            pytorch_str = f"{pytorch_mem:.1f}" if pytorch_mem else "-"

            jax_r = jax_lookup.get(name)
            jax_mem = self._extract_field(jax_r, "memory_mb") if jax_r else None
            jax_str = f"{jax_mem:.1f}" if jax_mem else "-"

            # Compute ratios
            ratio_pytorch = f"{mlx_mem / pytorch_mem:.2f}x" if mlx_mem and pytorch_mem else "-"
            ratio_jax = f"{mlx_mem / jax_mem:.2f}x" if mlx_mem and jax_mem else "-"

            lines.append(f"| {name} | {mlx_str} | {pytorch_str} | {jax_str} | {ratio_pytorch} | {ratio_jax} |")

        lines.append("")
        return "\n".join(lines)

    def generate_speedup_table(self) -> str:
        """Generate MLX speedup over PyTorch/JAX table.

        Returns:
            Markdown formatted speedup table.
        """
        from benchmarks.parity.comparison import compare_frameworks

        lines = [
            "### Speedup Comparison\n",
            "| Benchmark | vs PyTorch | vs JAX |",
            "|-----------|------------|--------|",
        ]

        mlx_results = self._get_results_list("mlx")
        pytorch_results = self._get_results_list("pytorch")
        jax_results = self._get_results_list("jax")

        comparison_results = compare_frameworks(mlx_results, pytorch_results, jax_results)

        for r in comparison_results:
            vs_pytorch = format_speedup(r.mlx_vs_pytorch_speedup) if r.mlx_vs_pytorch_speedup else "-"
            vs_jax = format_speedup(r.mlx_vs_jax_speedup) if r.mlx_vs_jax_speedup else "-"
            lines.append(f"| {r.operation}_{r.size_config} | {vs_pytorch} | {vs_jax} |")

        lines.append("")
        return "\n".join(lines)

    def generate_backward_table(self) -> str:
        """Generate backward pass comparison table.

        Returns:
            Markdown formatted backward pass table.
        """
        lines = [
            "### Backward Pass Comparison\n",
            "| Benchmark | MLX | PyTorch | JAX | vs PyTorch | vs JAX |",
            "|-----------|-----|---------|-----|------------|--------|",
        ]

        # Filter for backward pass results
        def filter_backward(results: List[Any]) -> List[Any]:
            backward_results = []
            for r in results:
                metadata = self._extract_field(r, "metadata", {}) or {}
                if metadata.get("type") == "backward":
                    backward_results.append(r)
            return backward_results

        mlx_results = filter_backward(self._get_results_list("mlx"))
        pytorch_results = filter_backward(self._get_results_list("pytorch"))
        jax_results = filter_backward(self._get_results_list("jax"))

        if not mlx_results:
            return "### Backward Pass Comparison\n\nNo backward pass benchmarks available.\n"

        def build_lookup(results: List[Any]) -> Dict[str, Any]:
            return {self._extract_field(r, "name", ""): r for r in results}

        mlx_lookup = build_lookup(mlx_results)
        pytorch_lookup = build_lookup(pytorch_results)
        jax_lookup = build_lookup(jax_results)

        for name, mlx_r in mlx_lookup.items():
            mlx_time = self._extract_field(mlx_r, "mean_time", 0)
            mlx_str = format_time(mlx_time)

            pytorch_r = pytorch_lookup.get(name)
            pytorch_time = self._extract_field(pytorch_r, "mean_time") if pytorch_r else None
            pytorch_str = format_time(pytorch_time) if pytorch_time else "-"

            jax_r = jax_lookup.get(name)
            jax_time = self._extract_field(jax_r, "mean_time") if jax_r else None
            jax_str = format_time(jax_time) if jax_time else "-"

            # Compute speedups
            vs_pytorch = format_speedup(pytorch_time / mlx_time) if pytorch_time and mlx_time else "-"
            vs_jax = format_speedup(jax_time / mlx_time) if jax_time and mlx_time else "-"

            lines.append(f"| {name} | {mlx_str} | {pytorch_str} | {jax_str} | {vs_pytorch} | {vs_jax} |")

        lines.append("")
        return "\n".join(lines)

    def generate_scaling_table(self, dimension: str) -> str:
        """Generate scaling analysis table.

        Args:
            dimension: Scaling dimension (seq_length, batch_size, etc.).

        Returns:
            Markdown formatted scaling table.
        """
        lines = [
            f"### Scaling by {dimension.replace('_', ' ').title()}\n",
            f"| {dimension} | MLX | PyTorch | JAX | MLX Scaling |",
            "|------------|-----|---------|-----|-------------|",
        ]

        mlx_results = self._get_results_list("mlx")

        # Group results by dimension value
        grouped: Dict[Any, Dict[str, Any]] = {}
        for r in mlx_results:
            metadata = self._extract_field(r, "metadata", {}) or {}
            dim_value = metadata.get(dimension, metadata.get("seq_len", metadata.get("batch_size")))
            if dim_value is not None:
                grouped[dim_value] = {"mlx": r}

        # Add pytorch and jax results
        for framework in ["pytorch", "jax"]:
            for r in self._get_results_list(framework):
                metadata = self._extract_field(r, "metadata", {}) or {}
                dim_value = metadata.get(dimension, metadata.get("seq_len", metadata.get("batch_size")))
                if dim_value in grouped:
                    grouped[dim_value][framework] = r

        prev_mlx_time = None
        for dim_value in sorted(grouped.keys()):
            data = grouped[dim_value]

            mlx_r = data.get("mlx")
            mlx_time = self._extract_field(mlx_r, "mean_time", 0) if mlx_r else 0
            mlx_str = format_time(mlx_time) if mlx_time else "-"

            pytorch_r = data.get("pytorch")
            pytorch_time = self._extract_field(pytorch_r, "mean_time") if pytorch_r else None
            pytorch_str = format_time(pytorch_time) if pytorch_time else "-"

            jax_r = data.get("jax")
            jax_time = self._extract_field(jax_r, "mean_time") if jax_r else None
            jax_str = format_time(jax_time) if jax_time else "-"

            # Calculate scaling factor
            if prev_mlx_time and prev_mlx_time > 0 and mlx_time > 0:
                scaling = mlx_time / prev_mlx_time
                scaling_str = f"{scaling:.2f}x"
            else:
                scaling_str = "-"

            lines.append(f"| {dim_value} | {mlx_str} | {pytorch_str} | {jax_str} | {scaling_str} |")
            prev_mlx_time = mlx_time

        lines.append("")
        return "\n".join(lines)

    def generate_all_tables(self) -> str:
        """Generate all comparison tables.

        Returns:
            Combined markdown with all tables.
        """
        sections = [
            "# Benchmark Comparison Tables\n",
            self.generate_throughput_table(),
            self.generate_latency_table(),
            self.generate_memory_table(),
            self.generate_speedup_table(),
            self.generate_backward_table(),
            self.generate_scaling_table("seq_length"),
        ]
        return "\n".join(sections)


def create_html_comparison_chart(
    results: Dict[str, List[Any]],
    title: str,
) -> str:
    """Create HTML chart for visual comparison.

    Args:
        results: Benchmark results mapping framework to results list.
        title: Chart title.

    Returns:
        HTML string with embedded SVG chart.
    """
    if not results:
        return f"<div><h3>{title}</h3><p>No data available</p></div>"

    # Get framework data
    frameworks = list(results.keys())
    colors = {
        "mlx": "#4CAF50",
        "pytorch": "#FF5722",
        "pytorch_mps": "#FF5722",
        "jax": "#2196F3",
        "jax_metal": "#2196F3",
    }

    # Collect benchmark names and times
    all_names: set = set()
    framework_times: Dict[str, Dict[str, float]] = {f: {} for f in frameworks}

    for framework, framework_results in results.items():
        for r in framework_results:
            name = r.name if hasattr(r, "name") else r.get("name", "")
            mean_time = (
                r.mean_time
                if hasattr(r, "mean_time")
                else r.get("mean_time", r.get("mean_time_seconds", 0))
            )
            if name and mean_time:
                all_names.add(name)
                framework_times[framework][name] = mean_time * 1000  # Convert to ms

    names = sorted(all_names)[:10]  # Limit to 10 benchmarks for readability

    if not names:
        return f"<div><h3>{title}</h3><p>No benchmark data</p></div>"

    # Calculate chart dimensions
    chart_width = 600
    chart_height = 40 * len(names) + 80
    bar_height = 15
    max_time = max(
        t
        for times in framework_times.values()
        for t in times.values()
        if t > 0
    ) if any(framework_times.values()) else 1

    # Generate SVG
    svg_lines = [
        f'<svg viewBox="0 0 {chart_width} {chart_height}" xmlns="http://www.w3.org/2000/svg">',
        f'  <text x="{chart_width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>',
    ]

    # Legend
    legend_x = 50
    for framework in frameworks:
        color = colors.get(framework, "#999999")
        svg_lines.append(
            f'  <rect x="{legend_x}" y="35" width="15" height="10" fill="{color}"/>'
        )
        svg_lines.append(
            f'  <text x="{legend_x + 20}" y="44" font-size="10">{framework}</text>'
        )
        legend_x += 100

    # Bars
    y_offset = 60
    for name in names:
        # Label
        svg_lines.append(
            f'  <text x="5" y="{y_offset + bar_height}" font-size="10" text-anchor="start">{name[:20]}</text>'
        )

        bar_y = y_offset
        for framework in frameworks:
            time_ms = framework_times[framework].get(name, 0)
            if time_ms > 0:
                bar_width = (time_ms / max_time) * 400
                color = colors.get(framework, "#999999")
                svg_lines.append(
                    f'  <rect x="150" y="{bar_y}" width="{bar_width}" height="{bar_height - 2}" fill="{color}" opacity="0.8"/>'
                )
                svg_lines.append(
                    f'  <text x="{155 + bar_width}" y="{bar_y + bar_height - 4}" font-size="8">{time_ms:.2f}ms</text>'
                )
            bar_y += bar_height

        y_offset += 40

    svg_lines.append("</svg>")

    return "\n".join(svg_lines)


def export_to_csv(
    results: Dict[str, List[Any]],
    filepath: str,
) -> None:
    """Export comparison results to CSV.

    Args:
        results: Benchmark results mapping framework to results list.
        filepath: Output CSV file path.
    """
    if not results:
        # Write empty CSV with headers
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "framework", "mean_time_s", "std_time_s", "min_time_s", "max_time_s", "memory_mb", "throughput_ops_s"])
        return

    rows = []
    for framework, framework_results in results.items():
        for r in framework_results:
            # Handle both object and dict formats
            if hasattr(r, "name"):
                row = {
                    "name": r.name,
                    "framework": framework,
                    "mean_time_s": r.mean_time,
                    "std_time_s": r.std_time,
                    "min_time_s": r.min_time,
                    "max_time_s": r.max_time,
                    "memory_mb": r.memory_mb if hasattr(r, "memory_mb") else None,
                    "throughput_ops_s": r.throughput_ops_per_sec if hasattr(r, "throughput_ops_per_sec") else None,
                }
            else:
                row = {
                    "name": r.get("name", ""),
                    "framework": framework,
                    "mean_time_s": r.get("mean_time", r.get("mean_time_seconds", 0)),
                    "std_time_s": r.get("std_time", r.get("std_time_seconds", 0)),
                    "min_time_s": r.get("min_time", r.get("min_time_seconds", 0)),
                    "max_time_s": r.get("max_time", r.get("max_time_seconds", 0)),
                    "memory_mb": r.get("memory_mb"),
                    "throughput_ops_s": r.get("throughput_ops_per_sec"),
                }
            rows.append(row)

    # Write CSV
    fieldnames = ["name", "framework", "mean_time_s", "std_time_s", "min_time_s", "max_time_s", "memory_mb", "throughput_ops_s"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Replace None with empty string for CSV
            cleaned_row = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(cleaned_row)
