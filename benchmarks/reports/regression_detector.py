"""Statistical regression detection for benchmarks.

This module provides tools for detecting statistically significant performance
regressions between benchmark runs using Welch's t-test.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    np = None
    stats = None


@dataclass
class RegressionResult:
    """Result of regression analysis for one benchmark."""

    name: str
    current_mean: float
    current_std: float
    baseline_mean: float
    baseline_std: float
    percent_change: float
    t_statistic: float
    p_value: float
    is_regression: bool
    is_improvement: bool
    confidence_interval: Tuple[float, float]


def detect_regressions(
    current_results: List[dict],
    baseline_results: List[dict],
    threshold_percent: float = 10.0,
    confidence_level: float = 0.95,
) -> Tuple[List[RegressionResult], bool, bool]:
    """Detect statistically significant performance regressions.

    Uses Welch's t-test to determine if performance difference is significant.

    Args:
        current_results: Current benchmark results.
        baseline_results: Baseline benchmark results.
        threshold_percent: Regression threshold percentage.
        confidence_level: Statistical confidence level (e.g., 0.95 for 95%).

    Returns:
        Tuple of (regression results list, has_regression, has_improvement).
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for regression detection. Install with: pip install scipy")

    baseline_by_name = {r["name"]: r for r in baseline_results}

    results = []
    has_regression = False
    has_improvement = False

    for current in current_results:
        name = current["name"]
        if name not in baseline_by_name:
            continue

        baseline = baseline_by_name[name]

        # Extract timing data
        curr_mean = current.get("mean_time_seconds", current.get("mean_time", 0))
        curr_std = current.get("std_time_seconds", current.get("std_time", curr_mean * 0.1))
        curr_n = current.get("iterations", 10)

        base_mean = baseline.get("mean_time_seconds", baseline.get("mean_time", 0))
        base_std = baseline.get("std_time_seconds", baseline.get("std_time", base_mean * 0.1))
        base_n = baseline.get("iterations", 10)

        # Skip if we don't have valid data
        if curr_mean <= 0 or base_mean <= 0:
            continue

        # Percent change (positive = slower = regression)
        pct_change = ((curr_mean - base_mean) / base_mean) * 100

        # Welch's t-test for unequal variances
        # Ensure we have positive standard deviations for the test
        curr_std = max(curr_std, curr_mean * 0.01)
        base_std = max(base_std, base_mean * 0.01)

        t_stat = (curr_mean - base_mean) / np.sqrt(
            (curr_std**2 / curr_n) + (base_std**2 / base_n)
        )

        # Degrees of freedom (Welch-Satterthwaite)
        numerator = (curr_std**2 / curr_n + base_std**2 / base_n) ** 2
        denominator = (curr_std**4 / (curr_n**2 * (curr_n - 1))) + (
            base_std**4 / (base_n**2 * (base_n - 1))
        )
        df = numerator / denominator if denominator > 0 else curr_n + base_n - 2

        # Two-sided p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Confidence interval for the difference
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        se_diff = np.sqrt(curr_std**2 / curr_n + base_std**2 / base_n)
        ci_low = (curr_mean - base_mean) - t_crit * se_diff
        ci_high = (curr_mean - base_mean) + t_crit * se_diff

        # Is this a significant regression or improvement?
        is_significant = p_value < (1 - confidence_level)
        is_regression_flag = pct_change > threshold_percent and is_significant
        is_improvement_flag = pct_change < -threshold_percent and is_significant

        if is_regression_flag:
            has_regression = True
        if is_improvement_flag:
            has_improvement = True

        results.append(
            RegressionResult(
                name=name,
                current_mean=curr_mean,
                current_std=curr_std,
                baseline_mean=base_mean,
                baseline_std=base_std,
                percent_change=pct_change,
                t_statistic=t_stat,
                p_value=p_value,
                is_regression=is_regression_flag,
                is_improvement=is_improvement_flag,
                confidence_interval=(ci_low, ci_high),
            )
        )

    return results, has_regression, has_improvement


def generate_report(
    results: List[RegressionResult],
    has_regression: bool,
    has_improvement: bool,
    threshold_percent: float = 10.0,
) -> str:
    """Generate markdown report of regression analysis.

    Args:
        results: List of regression analysis results.
        has_regression: Whether any regressions were detected.
        has_improvement: Whether any improvements were detected.
        threshold_percent: Threshold used for detection.

    Returns:
        Markdown formatted report string.
    """
    report = ["## Benchmark Regression Report\n"]

    if has_regression:
        report.append("**:x: STATUS: REGRESSION DETECTED**\n\n")
    elif has_improvement:
        report.append("**:white_check_mark: STATUS: Performance improved!**\n\n")
    else:
        report.append("**:white_check_mark: STATUS: No significant changes**\n\n")

    report.append(f"Threshold: {threshold_percent}% with 95% confidence\n\n")

    # Regressions table
    regressions = [r for r in results if r.is_regression]
    if regressions:
        report.append("### :warning: Regressions\n\n")
        report.append("| Benchmark | Change | p-value | Current | Baseline |\n")
        report.append("|-----------|--------|---------|---------|----------|\n")
        for r in sorted(regressions, key=lambda x: -x.percent_change):
            report.append(
                f"| `{r.name}` | **+{r.percent_change:.1f}%** | {r.p_value:.4f} | "
                f"{r.current_mean*1000:.3f}ms | {r.baseline_mean*1000:.3f}ms |\n"
            )
        report.append("\n")

    # Improvements table
    improvements = [r for r in results if r.is_improvement]
    if improvements:
        report.append("### :rocket: Improvements\n\n")
        report.append("| Benchmark | Change | p-value | Current | Baseline |\n")
        report.append("|-----------|--------|---------|---------|----------|\n")
        for r in sorted(improvements, key=lambda x: x.percent_change):
            report.append(
                f"| `{r.name}` | **{r.percent_change:.1f}%** | {r.p_value:.4f} | "
                f"{r.current_mean*1000:.3f}ms | {r.baseline_mean*1000:.3f}ms |\n"
            )
        report.append("\n")

    # Summary statistics
    if results:
        changes = [r.percent_change for r in results]
        report.append("### Summary\n\n")
        report.append(f"- **Benchmarks compared:** {len(results)}\n")
        report.append(f"- **Regressions:** {len(regressions)}\n")
        report.append(f"- **Improvements:** {len(improvements)}\n")
        report.append(f"- **Mean change:** {sum(changes)/len(changes):.1f}%\n")

    return "".join(report)


def generate_json_report(
    results: List[RegressionResult],
    has_regression: bool,
    has_improvement: bool,
) -> dict:
    """Generate JSON report of regression analysis.

    Args:
        results: List of regression analysis results.
        has_regression: Whether any regressions were detected.
        has_improvement: Whether any improvements were detected.

    Returns:
        Dictionary with regression analysis data.
    """
    return {
        "status": "regression" if has_regression else ("improvement" if has_improvement else "stable"),
        "has_regression": has_regression,
        "has_improvement": has_improvement,
        "total_benchmarks": len(results),
        "regressions": [
            {
                "name": r.name,
                "percent_change": r.percent_change,
                "p_value": r.p_value,
                "current_mean_ms": r.current_mean * 1000,
                "baseline_mean_ms": r.baseline_mean * 1000,
            }
            for r in results
            if r.is_regression
        ],
        "improvements": [
            {
                "name": r.name,
                "percent_change": r.percent_change,
                "p_value": r.p_value,
                "current_mean_ms": r.current_mean * 1000,
                "baseline_mean_ms": r.baseline_mean * 1000,
            }
            for r in results
            if r.is_improvement
        ],
    }


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for regression detector CLI.

    Args:
        args: Command line arguments.

    Returns:
        Exit code (0 for no regression, 1 for regression detected).
    """
    parser = argparse.ArgumentParser(
        description="Detect performance regressions in benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.reports.regression_detector \\
    --current current.json --baseline baseline.json

  python -m benchmarks.reports.regression_detector \\
    --current current.json --baseline baseline.json \\
    --threshold 15 --confidence 0.99 --output report.md
        """,
    )

    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current benchmark results JSON",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline benchmark results JSON",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percentage (default: 10)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Statistical confidence level (default: 0.95)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for markdown report",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Output file for JSON report",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output",
    )

    parsed_args = parser.parse_args(args)

    # Check for scipy
    if not HAS_SCIPY:
        print("Error: scipy is required for regression detection.")
        print("Install with: pip install scipy")
        return 1

    # Load results
    if not parsed_args.current.exists():
        print(f"Error: Current results file not found: {parsed_args.current}")
        return 1

    if not parsed_args.baseline.exists():
        if not parsed_args.quiet:
            print(f"Warning: Baseline file not found: {parsed_args.baseline}")
            print("Skipping regression detection (no baseline to compare)")
        return 0

    with open(parsed_args.current) as f:
        current_data = json.load(f)

    with open(parsed_args.baseline) as f:
        baseline_data = json.load(f)

    # Extract benchmark results from JSON structure
    current_results = current_data.get("benchmarks", current_data.get("results", []))
    baseline_results = baseline_data.get("benchmarks", baseline_data.get("results", []))

    if not current_results:
        print("Error: No benchmark results found in current file")
        return 1

    if not baseline_results:
        print("Warning: No baseline results to compare")
        return 0

    # Detect regressions
    results, has_regression, has_improvement = detect_regressions(
        current_results,
        baseline_results,
        threshold_percent=parsed_args.threshold,
        confidence_level=parsed_args.confidence,
    )

    # Generate reports
    report = generate_report(results, has_regression, has_improvement, parsed_args.threshold)

    if not parsed_args.quiet:
        print(report)

    if parsed_args.output:
        parsed_args.output.write_text(report)
        if not parsed_args.quiet:
            print(f"Report written to {parsed_args.output}")

    if parsed_args.json_output:
        json_report = generate_json_report(results, has_regression, has_improvement)
        with open(parsed_args.json_output, "w") as f:
            json.dump(json_report, f, indent=2)
        if not parsed_args.quiet:
            print(f"JSON report written to {parsed_args.json_output}")

    # Exit with error code if regression detected
    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
