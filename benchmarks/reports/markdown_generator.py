"""Markdown report generation for parity benchmarks."""

from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_summary_table(results: Dict[str, List[Any]]) -> str:
    """Generate markdown summary table.

    Args:
        results: Dictionary mapping framework to benchmark results.

    Returns:
        Markdown formatted summary table.
    """
    if not results:
        return "No results available.\n"

    lines = [
        "## Summary\n",
        "| Framework | Benchmarks | Avg Time | Min Time | Max Time |",
        "|-----------|------------|----------|----------|----------|",
    ]

    for framework, benchmark_results in sorted(results.items()):
        if not benchmark_results:
            continue

        num_benchmarks = len(benchmark_results)
        times = []
        for r in benchmark_results:
            mean_time = (
                r.mean_time
                if hasattr(r, "mean_time")
                else r.get("mean_time", r.get("mean_time_seconds", 0))
            )
            if mean_time > 0:
                times.append(mean_time)

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            lines.append(
                f"| {framework} | {num_benchmarks} | {format_time(avg_time)} | {format_time(min_time)} | {format_time(max_time)} |"
            )
        else:
            lines.append(f"| {framework} | {num_benchmarks} | - | - | - |")

    lines.append("")
    return "\n".join(lines)


def generate_operation_table(
    operation: str,
    mlx_results: List[Any],
    pytorch_results: Optional[List[Any]] = None,
    jax_results: Optional[List[Any]] = None,
) -> str:
    """Generate detailed operation comparison table.

    Args:
        operation: Operation name.
        mlx_results: MLX benchmark results.
        pytorch_results: PyTorch benchmark results.
        jax_results: JAX benchmark results.

    Returns:
        Markdown formatted operation table.
    """
    if not mlx_results:
        return f"### {operation}\n\nNo MLX results available.\n"

    # Import comparison functions
    from benchmarks.parity.comparison import compare_frameworks

    # Compare frameworks
    comparison_results = compare_frameworks(mlx_results, pytorch_results, jax_results)

    lines = [
        f"### {operation}\n",
        "| Size | MLX | PyTorch | JAX | vs PyTorch | vs JAX |",
        "|------|-----|---------|-----|------------|--------|",
    ]

    for r in comparison_results:
        mlx_str = format_time(r.mlx_time)
        pytorch_str = format_time(r.pytorch_time) if r.pytorch_time else "-"
        jax_str = format_time(r.jax_time) if r.jax_time else "-"
        vs_pytorch = (
            format_speedup(r.mlx_vs_pytorch_speedup)
            if r.mlx_vs_pytorch_speedup
            else "-"
        )
        vs_jax = format_speedup(r.mlx_vs_jax_speedup) if r.mlx_vs_jax_speedup else "-"
        lines.append(
            f"| {r.size_config} | {mlx_str} | {pytorch_str} | {jax_str} | {vs_pytorch} | {vs_jax} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_scaling_table(scaling_data: Dict[str, Any]) -> str:
    """Generate scaling analysis table.

    Args:
        scaling_data: Scaling benchmark data with keys:
            - dimension: Scaling dimension name
            - x_values: Scaling dimension values
            - mlx_times: MLX execution times
            - pytorch_times: PyTorch execution times (optional)
            - jax_times: JAX execution times (optional)

    Returns:
        Markdown formatted scaling table.
    """
    if not scaling_data or not scaling_data.get("x_values"):
        return "### Scaling Analysis\n\nNo scaling data available.\n"

    dimension = scaling_data.get("dimension", "Size")
    x_values = scaling_data.get("x_values", [])
    mlx_times = scaling_data.get("mlx_times", [])
    pytorch_times = scaling_data.get("pytorch_times", [])
    jax_times = scaling_data.get("jax_times", [])

    has_pytorch = len(pytorch_times) == len(x_values) and any(pytorch_times)
    has_jax = len(jax_times) == len(x_values) and any(jax_times)

    # Build header
    header = f"| {dimension.replace('_', ' ').title()} | MLX"
    if has_pytorch:
        header += " | PyTorch"
    if has_jax:
        header += " | JAX"
    header += " | MLX Scaling |"

    separator = "|" + "----|" * (3 + int(has_pytorch) + int(has_jax))

    lines = [
        "### Scaling Analysis\n",
        header,
        separator,
    ]

    prev_mlx_time = None
    for i, x_val in enumerate(x_values):
        mlx_time = mlx_times[i] if i < len(mlx_times) else 0

        row = f"| {x_val} | {format_time(mlx_time)}"

        if has_pytorch:
            pt_time = pytorch_times[i] if i < len(pytorch_times) else 0
            row += f" | {format_time(pt_time) if pt_time else '-'}"

        if has_jax:
            jax_time = jax_times[i] if i < len(jax_times) else 0
            row += f" | {format_time(jax_time) if jax_time else '-'}"

        # Calculate scaling factor
        if prev_mlx_time and prev_mlx_time > 0 and mlx_time > 0:
            scaling_factor = mlx_time / prev_mlx_time
            row += f" | {scaling_factor:.2f}x"
        else:
            row += " | -"

        row += " |"
        lines.append(row)
        prev_mlx_time = mlx_time

    lines.append("")
    return "\n".join(lines)


def generate_full_report(
    all_results: Dict[str, Dict[str, List[Any]]],
    metadata: Optional[Dict[str, Any]] = None,
    include_scaling: bool = True,
) -> str:
    """Generate complete markdown benchmark report.

    Args:
        all_results: All benchmark results {suite: {framework: [results]}}.
        metadata: Optional metadata (date, config, etc.).
        include_scaling: Whether to include scaling analysis.

    Returns:
        Complete markdown report string.
    """
    from benchmarks.parity.comparison import (
        compare_frameworks,
        compute_speedups,
        generate_scaling_chart_data,
    )

    lines = ["# MLX Primitives Benchmark Report\n"]

    # Add metadata header
    if metadata:
        lines.append(f"**Date:** {metadata.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
        if "device" in metadata:
            lines.append(f"**Device:** {metadata['device']}\n")
        if "mlx_version" in metadata:
            lines.append(f"**MLX Version:** {metadata['mlx_version']}\n")
        lines.append("")
    else:
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Generate table of contents
    lines.append("## Table of Contents\n")
    lines.append("- [Summary](#summary)")
    for suite_name in sorted(all_results.keys()):
        anchor = suite_name.lower().replace("_", "-")
        lines.append(f"- [{suite_name.replace('_', ' ').title()}](#{anchor})")
    if include_scaling:
        lines.append("- [Scaling Analysis](#scaling-analysis)")
    lines.append("\n")

    # Generate overall summary
    all_framework_results: Dict[str, List[Any]] = {}
    for suite_results in all_results.values():
        for framework, results in suite_results.items():
            if framework not in all_framework_results:
                all_framework_results[framework] = []
            all_framework_results[framework].extend(results)

    lines.append(generate_summary_table(all_framework_results))

    # Compute overall speedups
    mlx_results = all_framework_results.get("mlx", [])
    pytorch_results = all_framework_results.get("pytorch_mps", all_framework_results.get("pytorch", []))
    jax_results = all_framework_results.get("jax_metal", all_framework_results.get("jax", []))

    if mlx_results:
        comparison_results = compare_frameworks(mlx_results, pytorch_results, jax_results)
        speedups = compute_speedups(comparison_results)

        if speedups.get("mean_speedup_vs_pytorch") or speedups.get("mean_speedup_vs_jax"):
            lines.append("### Overall Speedups\n")
            if speedups.get("mean_speedup_vs_pytorch"):
                lines.append(f"- **Mean vs PyTorch:** {format_speedup(speedups['mean_speedup_vs_pytorch'])}")
                lines.append(f"- **Median vs PyTorch:** {format_speedup(speedups['median_speedup_vs_pytorch'])}")
            if speedups.get("mean_speedup_vs_jax"):
                lines.append(f"- **Mean vs JAX:** {format_speedup(speedups['mean_speedup_vs_jax'])}")
                lines.append(f"- **Median vs JAX:** {format_speedup(speedups['median_speedup_vs_jax'])}")
            lines.append("")

    # Generate per-suite sections
    for suite_name in sorted(all_results.keys()):
        suite_results = all_results[suite_name]
        lines.append(f"## {suite_name.replace('_', ' ').title()}\n")

        suite_mlx = suite_results.get("mlx", [])
        suite_pytorch = suite_results.get("pytorch_mps", suite_results.get("pytorch", []))
        suite_jax = suite_results.get("jax_metal", suite_results.get("jax", []))

        if suite_mlx:
            lines.append(
                generate_operation_table(suite_name, suite_mlx, suite_pytorch, suite_jax)
            )

    # Generate scaling analysis
    if include_scaling and all_framework_results:
        scaling_data = generate_scaling_chart_data(all_framework_results)
        if scaling_data.get("x_values"):
            lines.append(generate_scaling_table(scaling_data))

    return "\n".join(lines)


def generate_regression_report(
    regressions: List[Dict[str, Any]],
    baseline_info: Dict[str, Any],
) -> str:
    """Generate regression detection report.

    Args:
        regressions: List of detected regressions from detect_regressions().
        baseline_info: Information about the baseline (commit, date, etc.).

    Returns:
        Markdown formatted regression report.
    """
    lines = ["## Regression Report\n"]

    # Add baseline info
    if baseline_info:
        if "commit" in baseline_info:
            lines.append(f"**Baseline:** commit {baseline_info['commit']}")
        if "date" in baseline_info:
            lines.append(f" ({baseline_info['date']})")
        lines.append("\n")

    if not regressions:
        lines.append("**Status:** No significant performance changes detected.\n")
        return "\n".join(lines)

    # Separate regressions and improvements
    actual_regressions = [r for r in regressions if r.get("is_regression")]
    improvements = [r for r in regressions if r.get("is_improvement")]

    # Status line
    if actual_regressions:
        lines.append(f"**Status:** {len(actual_regressions)} regression(s) detected\n")
    elif improvements:
        lines.append(f"**Status:** {len(improvements)} improvement(s) detected\n")
    else:
        lines.append("**Status:** No significant changes\n")

    # Regressions table
    if actual_regressions:
        lines.append("### Regressions\n")
        lines.append("| Operation | Size | Change | Current | Baseline |")
        lines.append("|-----------|------|--------|---------|----------|")

        for r in sorted(actual_regressions, key=lambda x: -x.get("percent_change", 0)):
            pct = r.get("percent_change", 0)
            current = r.get("current_time", 0)
            baseline = r.get("baseline_time", 0)
            lines.append(
                f"| {r.get('operation', '-')} | {r.get('size_config', '-')} | "
                f"+{pct:.1f}% | {format_time(current)} | {format_time(baseline)} |"
            )
        lines.append("")

    # Improvements table
    if improvements:
        lines.append("### Improvements\n")
        lines.append("| Operation | Size | Change | Current | Baseline |")
        lines.append("|-----------|------|--------|---------|----------|")

        for r in sorted(improvements, key=lambda x: x.get("percent_change", 0)):
            pct = r.get("percent_change", 0)
            current = r.get("current_time", 0)
            baseline = r.get("baseline_time", 0)
            lines.append(
                f"| {r.get('operation', '-')} | {r.get('size_config', '-')} | "
                f"{pct:.1f}% | {format_time(current)} | {format_time(baseline)} |"
            )
        lines.append("")

    # Summary
    lines.append("### Summary\n")
    lines.append(f"- **Total changes detected:** {len(regressions)}")
    lines.append(f"- **Regressions:** {len(actual_regressions)}")
    lines.append(f"- **Improvements:** {len(improvements)}")

    if regressions:
        avg_change = sum(r.get("percent_change", 0) for r in regressions) / len(regressions)
        lines.append(f"- **Average change:** {avg_change:+.1f}%")

    lines.append("")
    return "\n".join(lines)


def format_time(seconds: float, precision: int = 3) -> str:
    """Format time value with appropriate units.

    Args:
        seconds: Time in seconds.
        precision: Decimal precision.

    Returns:
        Formatted time string (e.g., "1.234 ms" or "567.8 µs").
    """
    if seconds >= 1.0:
        return f"{seconds:.{precision}f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.{precision}f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.{precision}f} µs"
    else:
        return f"{seconds * 1e9:.{precision}f} ns"


def format_speedup(speedup: float) -> str:
    """Format speedup value.

    Args:
        speedup: Speedup factor (>1 = faster).

    Returns:
        Formatted speedup string with color indicator.
    """
    if speedup >= 2.0:
        return f"**{speedup:.2f}x** ✅"
    elif speedup >= 1.0:
        return f"{speedup:.2f}x ✅"
    elif speedup >= 0.8:
        return f"{speedup:.2f}x ⚠️"
    else:
        return f"{speedup:.2f}x ❌"
