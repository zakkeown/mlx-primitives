"""Framework comparison utilities for parity benchmarks."""

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
    raise NotImplementedError("Stub: compare_frameworks")


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
    raise NotImplementedError("Stub: compute_speedups")


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
    raise NotImplementedError("Stub: generate_comparison_table")


def generate_scaling_chart_data(
    scaling_results: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """Generate data structure for scaling charts.

    Args:
        scaling_results: Dictionary of scaling benchmark results.

    Returns:
        JSON-serializable data for plotting:
        - x_values: Scaling dimension values
        - mlx_times: MLX execution times
        - pytorch_times: PyTorch execution times
        - jax_times: JAX execution times
    """
    raise NotImplementedError("Stub: generate_scaling_chart_data")


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
    raise NotImplementedError("Stub: detect_regressions")


def compute_memory_comparison(
    comparison_results: List[ComparisonResult],
) -> Dict[str, Any]:
    """Compute memory usage comparison statistics.

    Args:
        comparison_results: List of comparison results with memory data.

    Returns:
        Dictionary with memory comparison statistics.
    """
    raise NotImplementedError("Stub: compute_memory_comparison")
