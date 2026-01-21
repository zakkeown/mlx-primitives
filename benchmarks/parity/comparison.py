"""Framework comparison utilities for parity benchmarks."""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ComparisonResult:
    """Result of comparing frameworks for one benchmark.

    Attributes:
        operation: Operation name.
        size_config: Size configuration string.
        mlx_time: MLX execution time in seconds.
        pytorch_time: PyTorch MPS execution time (if available).
        jax_time: JAX Metal execution time (if available).
        mlx_vs_pytorch_speedup: MLX speedup over PyTorch (>1 = MLX faster).
        mlx_vs_jax_speedup: MLX speedup over JAX (>1 = MLX faster).
        mlx_memory_mb: MLX memory usage in MB (if profiled).
        pytorch_memory_mb: PyTorch memory usage in MB (if profiled).
        jax_memory_mb: JAX memory usage in MB (if profiled).
    """

    operation: str
    size_config: str
    mlx_time: float
    pytorch_time: Optional[float] = None
    jax_time: Optional[float] = None
    mlx_vs_pytorch_speedup: Optional[float] = None
    mlx_vs_jax_speedup: Optional[float] = None
    mlx_memory_mb: Optional[float] = None
    pytorch_memory_mb: Optional[float] = None
    jax_memory_mb: Optional[float] = None


def _extract_config_key(name: str) -> Tuple[str, str]:
    """Extract (operation, size_config) from benchmark name.

    Handles patterns like:
    - "rmsnorm_small_mlx" -> ("rmsnorm", "small")
    - "flash_attention_b2_s128" -> ("flash_attention", "b2_s128")
    - "attention" -> ("attention", "default")
    """
    # Try to split off framework suffix first (mlx, pytorch_mps, jax_metal)
    for suffix in ("_mlx", "_pytorch_mps", "_jax_metal", "_pytorch", "_jax"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Now extract operation and size config
    # Common patterns: op_size, op_bN_sM, op_small/medium/large
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        # Check if second part looks like a size config
        size_part = parts[1]
        if size_part in ("small", "medium", "large", "tiny", "huge"):
            return (parts[0], size_part)
        # Check for batch/seq pattern like "b2" or "s128"
        if size_part.startswith(("b", "s")) and any(c.isdigit() for c in size_part):
            return (parts[0], size_part)

    # Try splitting off more parts for complex names like "op_b2_s128"
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        # Pattern: op_bN_sM
        if parts[1].startswith("b") and parts[2].startswith("s"):
            return (parts[0], f"{parts[1]}_{parts[2]}")

    # Fallback: use full name as operation, "default" as size
    return (name, "default")


def compare_frameworks(
    mlx_results: List[Any],
    pytorch_results: Optional[List[Any]] = None,
    jax_results: Optional[List[Any]] = None,
) -> List[ComparisonResult]:
    """Compare benchmark results across frameworks.

    Args:
        mlx_results: List of MLX benchmark results.
        pytorch_results: Optional list of PyTorch benchmark results.
        jax_results: Optional list of JAX benchmark results.

    Returns:
        List of ComparisonResult objects.
    """
    if not mlx_results:
        return []

    # Build lookup dictionaries by config key
    def build_lookup(results: Optional[List[Any]]) -> Dict[Tuple[str, str], Any]:
        if not results:
            return {}
        lookup = {}
        for r in results:
            name = r.name if hasattr(r, "name") else r.get("name", "")
            key = _extract_config_key(name)
            lookup[key] = r
        return lookup

    mlx_lookup = build_lookup(mlx_results)
    pytorch_lookup = build_lookup(pytorch_results)
    jax_lookup = build_lookup(jax_results)

    comparison_results = []

    for key, mlx_result in mlx_lookup.items():
        operation, size_config = key

        # Get MLX timing
        mlx_time = (
            mlx_result.mean_time
            if hasattr(mlx_result, "mean_time")
            else mlx_result.get("mean_time", mlx_result.get("mean_time_seconds", 0))
        )
        mlx_memory = (
            mlx_result.memory_mb
            if hasattr(mlx_result, "memory_mb")
            else mlx_result.get("memory_mb")
        )

        # Get PyTorch timing if available
        pytorch_time = None
        pytorch_memory = None
        mlx_vs_pytorch = None
        pytorch_result = pytorch_lookup.get(key)
        if pytorch_result:
            pytorch_time = (
                pytorch_result.mean_time
                if hasattr(pytorch_result, "mean_time")
                else pytorch_result.get(
                    "mean_time", pytorch_result.get("mean_time_seconds", 0)
                )
            )
            pytorch_memory = (
                pytorch_result.memory_mb
                if hasattr(pytorch_result, "memory_mb")
                else pytorch_result.get("memory_mb")
            )
            if mlx_time > 0 and pytorch_time > 0:
                mlx_vs_pytorch = pytorch_time / mlx_time

        # Get JAX timing if available
        jax_time = None
        jax_memory = None
        mlx_vs_jax = None
        jax_result = jax_lookup.get(key)
        if jax_result:
            jax_time = (
                jax_result.mean_time
                if hasattr(jax_result, "mean_time")
                else jax_result.get("mean_time", jax_result.get("mean_time_seconds", 0))
            )
            jax_memory = (
                jax_result.memory_mb
                if hasattr(jax_result, "memory_mb")
                else jax_result.get("memory_mb")
            )
            if mlx_time > 0 and jax_time > 0:
                mlx_vs_jax = jax_time / mlx_time

        comparison_results.append(
            ComparisonResult(
                operation=operation,
                size_config=size_config,
                mlx_time=mlx_time,
                pytorch_time=pytorch_time,
                jax_time=jax_time,
                mlx_vs_pytorch_speedup=mlx_vs_pytorch,
                mlx_vs_jax_speedup=mlx_vs_jax,
                mlx_memory_mb=mlx_memory,
                pytorch_memory_mb=pytorch_memory,
                jax_memory_mb=jax_memory,
            )
        )

    return comparison_results


def compute_speedups(
    comparison_results: List[ComparisonResult],
) -> Dict[str, Any]:
    """Compute aggregate speedup statistics.

    Args:
        comparison_results: List of comparison results.

    Returns:
        Dictionary with speedup statistics:
        - mean_speedup_vs_pytorch
        - median_speedup_vs_pytorch
        - mean_speedup_vs_jax
        - median_speedup_vs_jax
        - per_operation_speedups
    """
    if not comparison_results:
        return {
            "mean_speedup_vs_pytorch": None,
            "median_speedup_vs_pytorch": None,
            "min_speedup_vs_pytorch": None,
            "max_speedup_vs_pytorch": None,
            "mean_speedup_vs_jax": None,
            "median_speedup_vs_jax": None,
            "min_speedup_vs_jax": None,
            "max_speedup_vs_jax": None,
            "per_operation_speedups": {},
        }

    # Collect speedups
    pytorch_speedups = [
        r.mlx_vs_pytorch_speedup
        for r in comparison_results
        if r.mlx_vs_pytorch_speedup is not None
    ]
    jax_speedups = [
        r.mlx_vs_jax_speedup
        for r in comparison_results
        if r.mlx_vs_jax_speedup is not None
    ]

    # Compute aggregate stats for PyTorch
    mean_pytorch = statistics.mean(pytorch_speedups) if pytorch_speedups else None
    median_pytorch = statistics.median(pytorch_speedups) if pytorch_speedups else None
    min_pytorch = min(pytorch_speedups) if pytorch_speedups else None
    max_pytorch = max(pytorch_speedups) if pytorch_speedups else None

    # Compute aggregate stats for JAX
    mean_jax = statistics.mean(jax_speedups) if jax_speedups else None
    median_jax = statistics.median(jax_speedups) if jax_speedups else None
    min_jax = min(jax_speedups) if jax_speedups else None
    max_jax = max(jax_speedups) if jax_speedups else None

    # Collect per-operation speedups
    per_operation: Dict[str, Dict[str, Optional[float]]] = {}
    for r in comparison_results:
        op_key = f"{r.operation}_{r.size_config}"
        per_operation[op_key] = {
            "pytorch": r.mlx_vs_pytorch_speedup,
            "jax": r.mlx_vs_jax_speedup,
        }

    return {
        "mean_speedup_vs_pytorch": mean_pytorch,
        "median_speedup_vs_pytorch": median_pytorch,
        "min_speedup_vs_pytorch": min_pytorch,
        "max_speedup_vs_pytorch": max_pytorch,
        "mean_speedup_vs_jax": mean_jax,
        "median_speedup_vs_jax": median_jax,
        "min_speedup_vs_jax": min_jax,
        "max_speedup_vs_jax": max_jax,
        "per_operation_speedups": per_operation,
    }


def generate_comparison_table(
    results: List[ComparisonResult],
    format: str = "markdown",
) -> str:
    """Generate comparison table in specified format.

    Args:
        results: List of comparison results.
        format: Output format (markdown, html, csv).

    Returns:
        Formatted table string.
    """
    if not results:
        return ""

    # Import formatting helpers
    from benchmarks.reports.markdown_generator import format_time, format_speedup

    if format == "markdown":
        lines = [
            "| Operation | Size | MLX | PyTorch | JAX | vs PyTorch | vs JAX |",
            "|-----------|------|-----|---------|-----|------------|--------|",
        ]
        for r in results:
            mlx_str = format_time(r.mlx_time)
            pytorch_str = format_time(r.pytorch_time) if r.pytorch_time else "-"
            jax_str = format_time(r.jax_time) if r.jax_time else "-"
            vs_pytorch = (
                format_speedup(r.mlx_vs_pytorch_speedup)
                if r.mlx_vs_pytorch_speedup
                else "-"
            )
            vs_jax = (
                format_speedup(r.mlx_vs_jax_speedup) if r.mlx_vs_jax_speedup else "-"
            )
            lines.append(
                f"| {r.operation} | {r.size_config} | {mlx_str} | {pytorch_str} | {jax_str} | {vs_pytorch} | {vs_jax} |"
            )
        return "\n".join(lines)

    elif format == "html":
        rows = ["<table>", "<thead><tr>"]
        headers = ["Operation", "Size", "MLX", "PyTorch", "JAX", "vs PyTorch", "vs JAX"]
        for h in headers:
            rows.append(f"<th>{h}</th>")
        rows.append("</tr></thead><tbody>")

        for r in results:
            rows.append("<tr>")
            rows.append(f"<td>{r.operation}</td>")
            rows.append(f"<td>{r.size_config}</td>")
            rows.append(f"<td>{format_time(r.mlx_time)}</td>")
            rows.append(
                f"<td>{format_time(r.pytorch_time) if r.pytorch_time else '-'}</td>"
            )
            rows.append(f"<td>{format_time(r.jax_time) if r.jax_time else '-'}</td>")
            rows.append(
                f"<td>{format_speedup(r.mlx_vs_pytorch_speedup) if r.mlx_vs_pytorch_speedup else '-'}</td>"
            )
            rows.append(
                f"<td>{format_speedup(r.mlx_vs_jax_speedup) if r.mlx_vs_jax_speedup else '-'}</td>"
            )
            rows.append("</tr>")

        rows.append("</tbody></table>")
        return "\n".join(rows)

    elif format == "csv":
        lines = ["operation,size,mlx_time_s,pytorch_time_s,jax_time_s,vs_pytorch,vs_jax"]
        for r in results:
            pytorch_time = r.pytorch_time if r.pytorch_time else ""
            jax_time = r.jax_time if r.jax_time else ""
            vs_pytorch = r.mlx_vs_pytorch_speedup if r.mlx_vs_pytorch_speedup else ""
            vs_jax = r.mlx_vs_jax_speedup if r.mlx_vs_jax_speedup else ""
            lines.append(
                f"{r.operation},{r.size_config},{r.mlx_time},{pytorch_time},{jax_time},{vs_pytorch},{vs_jax}"
            )
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'markdown', 'html', or 'csv'.")


def generate_scaling_chart_data(
    scaling_results: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """Generate data structure for scaling charts.

    Args:
        scaling_results: Dictionary of scaling benchmark results keyed by framework.
            Expected format: {"mlx": [...], "pytorch_mps": [...], "jax_metal": [...]}

    Returns:
        JSON-serializable data for plotting:
        - dimension: Scaling dimension name
        - x_values: Scaling dimension values
        - mlx_times: MLX execution times
        - pytorch_times: PyTorch execution times
        - jax_times: JAX execution times
    """
    if not scaling_results:
        return {
            "dimension": "unknown",
            "x_values": [],
            "mlx_times": [],
            "pytorch_times": [],
            "jax_times": [],
        }

    # Determine the scaling dimension from metadata
    dimension = "seq_length"  # Default
    mlx_results = scaling_results.get("mlx", scaling_results.get("mlx_metal", []))

    if mlx_results:
        first_result = mlx_results[0]
        if hasattr(first_result, "metadata"):
            metadata = first_result.metadata or {}
        elif isinstance(first_result, dict):
            metadata = first_result.get("metadata", {}) or {}
        else:
            metadata = {}
        if metadata:
            if "batch_size" in metadata and "seq_len" not in metadata:
                dimension = "batch_size"
            elif "seq_len" in metadata or "seq_length" in metadata:
                dimension = "seq_length"
            elif "num_experts" in metadata:
                dimension = "num_experts"

    # Extract x values and times for each framework
    def extract_data(
        results: Optional[List[Any]],
    ) -> Tuple[List[float], List[float]]:
        if not results:
            return [], []
        x_vals = []
        times = []
        for r in results:
            if hasattr(r, "metadata"):
                metadata = r.metadata or {}
            elif isinstance(r, dict):
                metadata = r.get("metadata", {}) or {}
            else:
                metadata = {}
            # Try various metadata keys for the dimension value
            x_val = metadata.get(
                dimension,
                metadata.get(
                    "seq_len",
                    metadata.get(
                        "batch_size", metadata.get("num_experts", len(x_vals) + 1)
                    ),
                ),
            )
            mean_time = (
                r.mean_time
                if hasattr(r, "mean_time")
                else r.get("mean_time", r.get("mean_time_seconds", 0))
            )
            x_vals.append(x_val)
            times.append(mean_time)
        return x_vals, times

    mlx_x, mlx_times = extract_data(mlx_results)
    pytorch_results = scaling_results.get(
        "pytorch_mps", scaling_results.get("pytorch", [])
    )
    _, pytorch_times = extract_data(pytorch_results)
    jax_results = scaling_results.get("jax_metal", scaling_results.get("jax", []))
    _, jax_times = extract_data(jax_results)

    return {
        "dimension": dimension,
        "x_values": mlx_x,
        "mlx_times": mlx_times,
        "pytorch_times": pytorch_times,
        "jax_times": jax_times,
    }


def detect_regressions(
    current_results: List[ComparisonResult],
    baseline_results: List[ComparisonResult],
    threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """Detect performance regressions against baseline.

    Args:
        current_results: Current benchmark results.
        baseline_results: Baseline results to compare against.
        threshold: Regression threshold (fraction, e.g., 0.10 = 10%).

    Returns:
        List of detected regressions with details.
    """
    if not current_results or not baseline_results:
        return []

    # Build baseline lookup by (operation, size_config)
    baseline_lookup = {
        (r.operation, r.size_config): r for r in baseline_results
    }

    regressions = []

    for current in current_results:
        key = (current.operation, current.size_config)
        baseline = baseline_lookup.get(key)

        if not baseline:
            continue

        # Compare MLX times
        if current.mlx_time > 0 and baseline.mlx_time > 0:
            percent_change = (current.mlx_time - baseline.mlx_time) / baseline.mlx_time

            is_regression = percent_change > threshold
            is_improvement = percent_change < -threshold

            if is_regression or is_improvement:
                regressions.append({
                    "operation": current.operation,
                    "size_config": current.size_config,
                    "current_time": current.mlx_time,
                    "baseline_time": baseline.mlx_time,
                    "percent_change": percent_change * 100,  # Convert to percentage
                    "is_regression": is_regression,
                    "is_improvement": is_improvement,
                    "threshold_percent": threshold * 100,
                })

    return regressions


def compute_memory_comparison(
    comparison_results: List[ComparisonResult],
) -> Dict[str, Any]:
    """Compute memory usage comparison statistics.

    Args:
        comparison_results: List of comparison results with memory data.

    Returns:
        Dictionary with memory comparison statistics.
    """
    if not comparison_results:
        return {
            "total_mlx_memory_mb": 0.0,
            "total_pytorch_memory_mb": 0.0,
            "total_jax_memory_mb": 0.0,
            "mean_memory_ratio_vs_pytorch": None,
            "mean_memory_ratio_vs_jax": None,
            "per_operation_memory": {},
        }

    # Collect memory data
    mlx_memory_total = 0.0
    pytorch_memory_total = 0.0
    jax_memory_total = 0.0

    pytorch_ratios = []
    jax_ratios = []
    per_operation: Dict[str, Dict[str, Optional[float]]] = {}

    for r in comparison_results:
        op_key = f"{r.operation}_{r.size_config}"
        per_op: Dict[str, Optional[float]] = {
            "mlx_mb": r.mlx_memory_mb,
            "pytorch_mb": r.pytorch_memory_mb,
            "jax_mb": r.jax_memory_mb,
            "ratio_vs_pytorch": None,
            "ratio_vs_jax": None,
        }

        if r.mlx_memory_mb is not None:
            mlx_memory_total += r.mlx_memory_mb

        if r.pytorch_memory_mb is not None:
            pytorch_memory_total += r.pytorch_memory_mb
            if r.mlx_memory_mb and r.pytorch_memory_mb > 0:
                ratio = r.mlx_memory_mb / r.pytorch_memory_mb
                pytorch_ratios.append(ratio)
                per_op["ratio_vs_pytorch"] = ratio

        if r.jax_memory_mb is not None:
            jax_memory_total += r.jax_memory_mb
            if r.mlx_memory_mb and r.jax_memory_mb > 0:
                ratio = r.mlx_memory_mb / r.jax_memory_mb
                jax_ratios.append(ratio)
                per_op["ratio_vs_jax"] = ratio

        per_operation[op_key] = per_op

    mean_pytorch_ratio = statistics.mean(pytorch_ratios) if pytorch_ratios else None
    mean_jax_ratio = statistics.mean(jax_ratios) if jax_ratios else None

    return {
        "total_mlx_memory_mb": mlx_memory_total,
        "total_pytorch_memory_mb": pytorch_memory_total,
        "total_jax_memory_mb": jax_memory_total,
        "mean_memory_ratio_vs_pytorch": mean_pytorch_ratio,
        "mean_memory_ratio_vs_jax": mean_jax_ratio,
        "per_operation_memory": per_operation,
    }
