"""Markdown report generation for parity benchmarks."""

from typing import Any, Dict, List, Optional


def generate_summary_table(results: Dict[str, List[Any]]) -> str:
    """Generate markdown summary table.

    Args:
        results: Dictionary mapping framework to benchmark results.

    Returns:
        Markdown formatted summary table.
    """
    raise NotImplementedError("Stub: generate_summary_table")


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
    raise NotImplementedError("Stub: generate_operation_table")


def generate_scaling_table(scaling_data: Dict[str, Any]) -> str:
    """Generate scaling analysis table.

    Args:
        scaling_data: Scaling benchmark data.

    Returns:
        Markdown formatted scaling table.
    """
    raise NotImplementedError("Stub: generate_scaling_table")


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
    raise NotImplementedError("Stub: generate_full_report")


def generate_regression_report(
    regressions: List[Dict[str, Any]],
    baseline_info: Dict[str, Any],
) -> str:
    """Generate regression detection report.

    Args:
        regressions: List of detected regressions.
        baseline_info: Information about the baseline.

    Returns:
        Markdown formatted regression report.
    """
    raise NotImplementedError("Stub: generate_regression_report")


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
