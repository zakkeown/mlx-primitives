"""ASCII table formatting for benchmark results."""

from typing import Optional

from benchmarks.utils import BenchmarkResult


def print_results_table(
    results: list[BenchmarkResult],
    title: str = "MLX Primitives Benchmark Suite",
    baseline_name: Optional[str] = None,
) -> None:
    """Print benchmark results as an ASCII table.

    Args:
        results: List of benchmark results.
        title: Title to display above the table.
        baseline_name: If provided, compute speedups relative to this benchmark.
    """
    if not results:
        print("No benchmark results to display.")
        return

    # Find baseline for speedup calculation
    baseline_mean: Optional[float] = None
    if baseline_name:
        for r in results:
            if r.name == baseline_name:
                baseline_mean = r.mean_time
                break

    # Print header
    print()
    print(title)
    print("=" * 70)

    # Column headers
    if baseline_mean is not None:
        print(f"{'Benchmark':<24} | {'Mean (ms)':>12} | {'Std (ms)':>12} | {'Speedup':>10}")
        print("-" * 70)
    else:
        print(f"{'Benchmark':<24} | {'Mean (ms)':>12} | {'Std (ms)':>12} | {'Iterations':>10}")
        print("-" * 70)

    # Print each result
    for r in results:
        mean_ms = r.mean_time * 1000
        std_ms = r.std_time * 1000

        if baseline_mean is not None:
            speedup = baseline_mean / r.mean_time if r.mean_time > 0 else 0.0
            speedup_str = f"{speedup:.2f}x"
            print(f"{r.name:<24} | {mean_ms:>12.3f} | {std_ms:>12.3f} | {speedup_str:>10}")
        else:
            print(f"{r.name:<24} | {mean_ms:>12.3f} | {std_ms:>12.3f} | {r.iterations:>10}")

    print("=" * 70)
    print()


def print_comparison_table(
    implementation_results: list[BenchmarkResult],
    baseline_results: list[BenchmarkResult],
    title: str = "Implementation vs Baseline Comparison",
) -> None:
    """Print a side-by-side comparison of implementation vs baseline.

    Args:
        implementation_results: Results from the implementation being tested.
        baseline_results: Results from the baseline/reference implementation.
        title: Title to display above the table.
    """
    if not implementation_results or not baseline_results:
        print("Missing results for comparison.")
        return

    # Build lookup by extracting the config suffix (e.g., "b2_s512" from "naive_attn_b2_s512")
    def extract_config_key(name: str) -> str:
        """Extract the configuration part of benchmark name for matching.

        Examples:
            naive_attn_b2_s512 -> attn_b2_s512
            flash_attn_b2_s512 -> attn_b2_s512
            native_cumsum_s128_d32 -> cumsum_s128_d32
            assoc_scan_s128_d32 -> scan_s128_d32
            naive_ln_b1_s512_h768 -> ln_b1_s512_h768
            fused_ln_b1_s512_h768 -> ln_b1_s512_h768
        """
        # Remove common prefixes that indicate implementation type
        prefixes_to_strip = [
            "naive_", "flash_", "fused_", "native_", "assoc_",
            "impl_", "ref_", "baseline_", "mlx_", "torch_", "optimized_"
        ]
        for prefix in prefixes_to_strip:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    impl_by_key = {extract_config_key(r.name): r for r in implementation_results}
    baseline_by_key = {extract_config_key(r.name): r for r in baseline_results}

    # Find matching benchmarks
    common_keys = set(impl_by_key.keys()) & set(baseline_by_key.keys())

    if not common_keys:
        print("No matching benchmarks found for comparison.")
        print(f"Implementation keys: {list(impl_by_key.keys())}")
        print(f"Baseline keys: {list(baseline_by_key.keys())}")
        return

    # Print header
    print()
    print(title)
    print("=" * 90)
    print(f"{'Benchmark':<20} | {'Impl (ms)':>12} | {'Base (ms)':>12} | {'Speedup':>10} | {'Status':>12}")
    print("-" * 90)

    # Print comparisons
    for key in sorted(common_keys):
        impl = impl_by_key[key]
        base = baseline_by_key[key]

        impl_ms = impl.mean_time * 1000
        base_ms = base.mean_time * 1000

        speedup = base.mean_time / impl.mean_time if impl.mean_time > 0 else 0.0
        speedup_str = f"{speedup:.2f}x"

        # Status based on speedup
        if speedup >= 2.0:
            status = "EXCELLENT"
        elif speedup >= 1.2:
            status = "GOOD"
        elif speedup >= 0.9:
            status = "PARITY"
        else:
            status = "SLOWER"

        print(f"{key:<20} | {impl_ms:>12.3f} | {base_ms:>12.3f} | {speedup_str:>10} | {status:>12}")

    print("=" * 90)

    # Summary statistics
    speedups = []
    for key in common_keys:
        impl = impl_by_key[key]
        base = baseline_by_key[key]
        if impl.mean_time > 0:
            speedups.append(base.mean_time / impl.mean_time)

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        print(f"\nSummary: Avg={avg_speedup:.2f}x  Max={max_speedup:.2f}x  Min={min_speedup:.2f}x")
    print()


def format_memory_usage(bytes_used: int) -> str:
    """Format memory usage in human-readable form."""
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024 * 1024:
        return f"{bytes_used / 1024:.2f} KB"
    elif bytes_used < 1024 * 1024 * 1024:
        return f"{bytes_used / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_used / (1024 * 1024 * 1024):.2f} GB"
