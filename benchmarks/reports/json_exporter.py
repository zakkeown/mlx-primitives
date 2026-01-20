"""JSON export functionality for benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from benchmarks.utils import BenchmarkResult


def export_to_json(
    results: list[BenchmarkResult],
    output_path: Union[str, Path],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Export benchmark results to JSON format.

    Args:
        results: List of benchmark results to export.
        output_path: Path to write the JSON file.
        metadata: Optional metadata to include (device info, date, etc.).
    """
    output_path = Path(output_path)

    # Build export structure
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            **(metadata or {}),
        },
        "results": [_result_to_dict(r) for r in results],
    }

    # Compute summary statistics
    if results:
        times = [r.mean_time for r in results]
        export_data["summary"] = {
            "total_benchmarks": len(results),
            "total_time_seconds": sum(times),
            "fastest_benchmark": min(results, key=lambda r: r.mean_time).name,
            "slowest_benchmark": max(results, key=lambda r: r.mean_time).name,
        }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)


def _result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Convert a BenchmarkResult to a dictionary."""
    return {
        "name": result.name,
        "mean_time_seconds": result.mean_time,
        "std_time_seconds": result.std_time,
        "min_time_seconds": result.min_time,
        "max_time_seconds": result.max_time,
        "iterations": result.iterations,
        "metadata": result.metadata,
    }


def load_from_json(input_path: Union[str, Path]) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    """Load benchmark results from a JSON file.

    Args:
        input_path: Path to the JSON file to load.

    Returns:
        Tuple of (results list, metadata dict).
    """
    input_path = Path(input_path)

    with open(input_path) as f:
        data = json.load(f)

    results = []
    for r in data.get("results", []):
        results.append(BenchmarkResult(
            name=r["name"],
            mean_time=r["mean_time_seconds"],
            std_time=r["std_time_seconds"],
            min_time=r["min_time_seconds"],
            max_time=r["max_time_seconds"],
            iterations=r["iterations"],
            metadata=r.get("metadata", {}),
        ))

    return results, data.get("metadata", {})


def compare_json_results(
    current_path: Union[str, Path],
    baseline_path: Union[str, Path],
) -> dict[str, Any]:
    """Compare two JSON result files and return regression analysis.

    Args:
        current_path: Path to current results.
        baseline_path: Path to baseline results.

    Returns:
        Dictionary with comparison analysis.
    """
    current_results, current_meta = load_from_json(current_path)
    baseline_results, baseline_meta = load_from_json(baseline_path)

    # Build lookup tables
    current_by_name = {r.name: r for r in current_results}
    baseline_by_name = {r.name: r for r in baseline_results}

    comparisons = []
    regressions = []
    improvements = []

    for name in set(current_by_name.keys()) & set(baseline_by_name.keys()):
        curr = current_by_name[name]
        base = baseline_by_name[name]

        # Calculate percentage change (positive = slower/regression)
        pct_change = ((curr.mean_time - base.mean_time) / base.mean_time) * 100

        comparison = {
            "name": name,
            "current_mean": curr.mean_time,
            "baseline_mean": base.mean_time,
            "percent_change": pct_change,
            "is_regression": pct_change > 5.0,  # >5% slower
            "is_improvement": pct_change < -5.0,  # >5% faster
        }
        comparisons.append(comparison)

        if comparison["is_regression"]:
            regressions.append(name)
        elif comparison["is_improvement"]:
            improvements.append(name)

    return {
        "current_metadata": current_meta,
        "baseline_metadata": baseline_meta,
        "comparisons": comparisons,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }
