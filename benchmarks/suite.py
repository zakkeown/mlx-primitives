"""Benchmark suite for mlx-primitives.

Run with: python -m benchmarks.suite [options]

Examples:
    python -m benchmarks.suite --all                # Run all benchmarks
    python -m benchmarks.suite --attention          # Run attention benchmarks
    python -m benchmarks.suite --layers             # Run layer benchmarks
    python -m benchmarks.suite --quick              # Quick run with fewer iterations
"""

import argparse
from pathlib import Path
from typing import List, Dict

import mlx.core as mx
import mlx.nn as nn

from benchmarks.runner import BenchmarkRunner, BenchmarkConfig, timed
from benchmarks.memory import MemoryProfiler, estimate_attention_memory_mb
from benchmarks.charts import ChartGenerator, ChartConfig


# ============================================================================
# Attention Benchmarks
# ============================================================================


def benchmark_attention(runner: BenchmarkRunner, quick: bool = False):
    """Run attention benchmarks."""
    from mlx_primitives.attention import (
        FlashAttention,
        GroupedQueryAttention,
        MultiQueryAttention,
        SlidingWindowAttention,
        LinearAttention,
    )

    print("\n" + "=" * 60)
    print("ATTENTION BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 20
    warmup = 2 if quick else 5

    # Benchmark configuration
    batch_size = 2
    num_heads = 8
    head_dim = 64
    dims = num_heads * head_dim

    # Test different sequence lengths
    seq_lens = [128, 256, 512, 1024] if quick else [128, 256, 512, 1024, 2048, 4096]

    results = []

    for seq_len in seq_lens:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        # FlashAttention
        flash = FlashAttention(dims=dims, num_heads=num_heads)

        def run_flash():
            out = flash(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"FlashAttention_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(
            run_flash,
            config,
            throughput_items=batch_size * seq_len,
            throughput_unit="tokens/sec"
        )
        results.append(result)
        print(f"  FlashAttention: {result.mean_ms:.3f} ms")

        # GQA
        gqa = GroupedQueryAttention(dims=dims, num_heads=num_heads, num_kv_heads=2)

        def run_gqa():
            out = gqa(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"GQA_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_gqa, config)
        results.append(result)
        print(f"  GQA: {result.mean_ms:.3f} ms")

        # MQA
        mqa = MultiQueryAttention(dims=dims, num_heads=num_heads)

        def run_mqa():
            out = mqa(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"MQA_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_mqa, config)
        results.append(result)
        print(f"  MQA: {result.mean_ms:.3f} ms")

        # Linear Attention
        linear = LinearAttention(dims=dims, num_heads=num_heads)

        def run_linear():
            out = linear(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"LinearAttention_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_linear, config)
        results.append(result)
        print(f"  LinearAttention: {result.mean_ms:.3f} ms")

    return results


def benchmark_attention_scaling(runner: BenchmarkRunner, quick: bool = False):
    """Benchmark attention scaling with sequence length."""
    from mlx_primitives.attention import FlashAttention, LinearAttention

    print("\n" + "=" * 60)
    print("ATTENTION SCALING BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 15
    warmup = 2 if quick else 3

    batch_size = 1
    num_heads = 8
    head_dim = 64
    dims = num_heads * head_dim

    seq_lens = [256, 512, 1024, 2048] if quick else [256, 512, 1024, 2048, 4096, 8192]

    flash_times = []
    linear_times = []

    for seq_len in seq_lens:
        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        # FlashAttention
        flash = FlashAttention(dims=dims, num_heads=num_heads)

        def run_flash():
            out = flash(x)
            mx.eval(out)

        time_ms = timed(run_flash, warmup=warmup, runs=runs)
        flash_times.append((seq_len, time_ms))
        print(f"FlashAttention seq={seq_len}: {time_ms:.3f} ms")

        # LinearAttention
        linear = LinearAttention(dims=dims, num_heads=num_heads)

        def run_linear():
            out = linear(x)
            mx.eval(out)

        time_ms = timed(run_linear, warmup=warmup, runs=runs)
        linear_times.append((seq_len, time_ms))
        print(f"LinearAttention seq={seq_len}: {time_ms:.3f} ms")

    return {
        "FlashAttention": flash_times,
        "LinearAttention": linear_times,
    }


# ============================================================================
# Layer Benchmarks
# ============================================================================


def benchmark_layers(runner: BenchmarkRunner, quick: bool = False):
    """Run layer benchmarks."""
    from mlx_primitives.layers import (
        RMSNorm,
        SwiGLU,
        GeGLU,
        AdaptiveAvgPool1d,
        GeM,
    )

    print("\n" + "=" * 60)
    print("LAYER BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 20
    warmup = 2 if quick else 5

    results = []

    # RMSNorm benchmark
    print("\nRMSNorm:")
    print("-" * 40)

    for dims in [256, 512, 1024, 2048]:
        x = mx.random.normal((4, 512, dims))
        mx.eval(x)

        rmsnorm = RMSNorm(dims=dims)

        def run_rmsnorm():
            out = rmsnorm(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"RMSNorm_dim{dims}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_rmsnorm, config)
        results.append(result)
        print(f"  dims={dims}: {result.mean_ms:.3f} ms")

    # SwiGLU benchmark
    print("\nSwiGLU:")
    print("-" * 40)

    for hidden_dim in [512, 1024, 2048, 4096]:
        x = mx.random.normal((4, 256, 256))
        mx.eval(x)

        swiglu = SwiGLU(256, hidden_dim)

        def run_swiglu():
            out = swiglu(x)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"SwiGLU_hidden{hidden_dim}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_swiglu, config)
        results.append(result)
        print(f"  hidden_dim={hidden_dim}: {result.mean_ms:.3f} ms")

    # SwiGLU vs GeGLU comparison
    print("\nSwiGLU vs GeGLU:")
    print("-" * 40)

    x = mx.random.normal((4, 256, 512))
    mx.eval(x)

    swiglu = SwiGLU(512, 2048)
    geglu = GeGLU(512, 2048)

    implementations = {
        "SwiGLU": lambda: mx.eval(swiglu(x)),
        "GeGLU": lambda: mx.eval(geglu(x)),
    }

    comparison = runner.compare(
        implementations,
        BenchmarkConfig(name="GLU_comparison", warmup_runs=warmup, benchmark_runs=runs)
    )
    runner.print_comparison(comparison, baseline="SwiGLU")

    return results


def benchmark_normalization(runner: BenchmarkRunner, quick: bool = False):
    """Benchmark normalization layers."""
    from mlx_primitives.layers import RMSNorm, GroupNorm, InstanceNorm

    print("\n" + "=" * 60)
    print("NORMALIZATION BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 20
    warmup = 2 if quick else 5

    batch_size = 4
    seq_len = 512
    dims = 768

    x = mx.random.normal((batch_size, seq_len, dims))
    mx.eval(x)

    # Compare normalizations
    rmsnorm = RMSNorm(dims=dims)
    layernorm = nn.LayerNorm(dims=dims)

    implementations = {
        "RMSNorm": lambda: mx.eval(rmsnorm(x)),
        "LayerNorm": lambda: mx.eval(layernorm(x)),
    }

    config = BenchmarkConfig(name="Normalization", warmup_runs=warmup, benchmark_runs=runs)
    comparison = runner.compare(implementations, config)

    runner.print_comparison(comparison, baseline="LayerNorm")

    return comparison


# ============================================================================
# Training Benchmarks
# ============================================================================


def benchmark_training(runner: BenchmarkRunner, quick: bool = False):
    """Run training utility benchmarks."""
    from mlx_primitives.training import (
        EMA,
        GradientClipper,
        GradientAccumulator,
        CosineAnnealingLR,
        WarmupCosineScheduler,
    )
    import mlx.nn as nn
    import mlx.optimizers as optim

    print("\n" + "=" * 60)
    print("TRAINING BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 20
    warmup = 2 if quick else 5

    results = []

    # EMA benchmark
    print("\nEMA Update Speed:")
    print("-" * 40)

    for model_params in [1000, 10000, 100000]:
        # Create model with approximate param count
        dim = int((model_params // 2) ** 0.5)
        model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, dim),
        )

        ema = EMA(model, decay=0.999)

        def run_ema_update():
            ema.update(step=0)
            mx.eval(ema.shadow_params)

        config = BenchmarkConfig(
            name=f"EMA_params{model_params}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_ema_update, config)
        results.append(result)
        print(f"  ~{model_params} params: {result.mean_ms:.3f} ms")

    # Gradient Clipping benchmark
    print("\nGradient Clipping:")
    print("-" * 40)

    clipper = GradientClipper(max_norm=1.0)

    for grad_size in [1000, 10000, 100000]:
        dim = int(grad_size ** 0.5)
        grads = {"weight": mx.random.normal((dim, dim))}
        mx.eval(grads["weight"])

        def run_clip():
            clipped, _ = clipper.clip_by_norm(grads)
            mx.eval(clipped["weight"])

        config = BenchmarkConfig(
            name=f"GradClip_{grad_size}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_clip, config)
        results.append(result)
        print(f"  {grad_size} elements: {result.mean_ms:.3f} ms")

    # LR Scheduler overhead
    print("\nLR Scheduler Overhead:")
    print("-" * 40)

    schedulers = {
        "Cosine": CosineAnnealingLR(base_lr=0.01, T_max=1000),
        "WarmupCosine": WarmupCosineScheduler(base_lr=0.01, warmup_steps=100, total_steps=1000),
    }

    for name, scheduler in schedulers.items():
        def run_scheduler():
            for step in range(100):
                _ = scheduler.get_lr(step)

        config = BenchmarkConfig(
            name=f"Scheduler_{name}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_scheduler, config)
        results.append(result)
        print(f"  {name} (100 calls): {result.mean_ms:.3f} ms")

    return results


# ============================================================================
# Advanced Benchmarks
# ============================================================================


def benchmark_advanced(runner: BenchmarkRunner, quick: bool = False):
    """Run advanced primitive benchmarks."""
    from mlx_primitives.advanced import (
        TopKRouter,
        MoELayer,
        selective_scan,
        quantize_tensor,
        dequantize_tensor,
    )

    print("\n" + "=" * 60)
    print("ADVANCED BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 20
    warmup = 2 if quick else 5

    results = []

    # MoE Router benchmark
    print("\nMoE Router:")
    print("-" * 40)

    for num_experts in [4, 8, 16]:
        router = TopKRouter(dims=512, num_experts=num_experts, top_k=2)

        x = mx.random.normal((4, 128, 512))
        mx.eval(x)

        def run_router():
            weights, indices, _ = router(x)
            mx.eval(weights, indices)

        config = BenchmarkConfig(
            name=f"TopKRouter_experts{num_experts}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_router, config)
        results.append(result)
        print(f"  {num_experts} experts: {result.mean_ms:.3f} ms")

    # Selective Scan (Mamba) benchmark
    print("\nSelective Scan (Mamba):")
    print("-" * 40)

    for seq_len in [128, 256, 512, 1024] if quick else [128, 256, 512, 1024, 2048]:
        batch = 4
        d_inner = 256
        d_state = 16

        x = mx.random.normal((batch, seq_len, d_inner))
        A = mx.random.normal((d_inner, d_state)) * 0.1
        B = mx.random.normal((batch, seq_len, d_state))
        C = mx.random.normal((batch, seq_len, d_state))
        delta = mx.abs(mx.random.normal((batch, seq_len, d_inner))) + 0.1
        mx.eval(x, A, B, C, delta)

        def run_ssm():
            out = selective_scan(x, delta, A, B, C)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"SelectiveScan_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_ssm, config)
        results.append(result)
        print(f"  seq_len={seq_len}: {result.mean_ms:.3f} ms")

    # Quantization benchmark
    print("\nQuantization:")
    print("-" * 40)

    for size in [256, 512, 1024]:
        x = mx.random.normal((size, size))
        mx.eval(x)

        def run_quantize():
            q, scale, zp = quantize_tensor(x, num_bits=8)
            restored = dequantize_tensor(q, scale, zp)
            mx.eval(restored)

        config = BenchmarkConfig(
            name=f"Quantize8bit_{size}x{size}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
        result = runner.benchmark(run_quantize, config)
        results.append(result)
        print(f"  {size}x{size}: {result.mean_ms:.3f} ms")

    return results


# ============================================================================
# Vs-Naive Comparison Benchmarks
# ============================================================================


def naive_attention(q, k, v, scale, causal=False):
    """Naive O(n^2) attention implementation."""
    # Standard dot-product attention
    scores = mx.matmul(q, k.swapaxes(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
        scores = scores + mask

    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v)


def naive_rmsnorm(x, weight, eps=1e-6):
    """Naive RMSNorm implementation."""
    rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def naive_swiglu(x, w1, w2, w3):
    """Naive SwiGLU implementation."""
    return mx.sigmoid(x @ w1) * (x @ w1) * (x @ w2)


def benchmark_vs_naive(runner: BenchmarkRunner, quick: bool = False):
    """Benchmark our implementations vs naive baselines."""
    from mlx_primitives.attention import FlashAttention, LinearAttention
    from mlx_primitives.layers import RMSNorm, SwiGLU

    print("\n" + "=" * 60)
    print("VS-NAIVE COMPARISON BENCHMARKS")
    print("=" * 60)

    runs = 5 if quick else 15
    warmup = 2 if quick else 5

    # Attention comparison
    print("\nAttention: FlashAttention vs Naive")
    print("-" * 50)

    for seq_len in [256, 512, 1024] if quick else [256, 512, 1024, 2048]:
        batch_size = 2
        num_heads = 8
        head_dim = 64
        dims = num_heads * head_dim
        scale = head_dim ** -0.5

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        flash = FlashAttention(dims=dims, num_heads=num_heads, causal=True)

        # For naive attention, we need Q, K, V
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        mx.eval(q, k, v)

        def run_flash():
            out = flash(x)
            mx.eval(out)

        def run_naive():
            out = naive_attention(q, k, v, scale, causal=True)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"Attention_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )

        comparison = runner.compare(
            {"FlashAttention": run_flash, "NaiveAttention": run_naive},
            config
        )

        flash_ms = comparison["FlashAttention"].mean_ms
        naive_ms = comparison["NaiveAttention"].mean_ms
        speedup = naive_ms / flash_ms
        print(f"  seq={seq_len}: Flash={flash_ms:.3f}ms, Naive={naive_ms:.3f}ms, Speedup={speedup:.2f}x")

    # RMSNorm comparison
    print("\nRMSNorm: Ours vs Naive")
    print("-" * 50)

    for dims in [512, 1024, 2048]:
        x = mx.random.normal((4, 512, dims))
        weight = mx.ones((dims,))
        mx.eval(x, weight)

        rmsnorm = RMSNorm(dims=dims)

        def run_ours():
            out = rmsnorm(x)
            mx.eval(out)

        def run_naive():
            out = naive_rmsnorm(x, weight)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"RMSNorm_dim{dims}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )

        comparison = runner.compare(
            {"Ours": run_ours, "Naive": run_naive},
            config
        )

        ours_ms = comparison["Ours"].mean_ms
        naive_ms = comparison["Naive"].mean_ms
        speedup = naive_ms / ours_ms
        print(f"  dims={dims}: Ours={ours_ms:.3f}ms, Naive={naive_ms:.3f}ms, Speedup={speedup:.2f}x")

    # Linear vs Quadratic attention scaling
    print("\nLinear Attention vs Standard Attention (Scaling)")
    print("-" * 50)

    for seq_len in [512, 1024, 2048, 4096] if not quick else [512, 1024, 2048]:
        batch_size = 1
        num_heads = 8
        head_dim = 64
        dims = num_heads * head_dim
        scale = head_dim ** -0.5

        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        linear = LinearAttention(dims=dims, num_heads=num_heads)

        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        mx.eval(q, k, v)

        def run_linear():
            out = linear(x)
            mx.eval(out)

        def run_standard():
            out = naive_attention(q, k, v, scale, causal=False)
            mx.eval(out)

        config = BenchmarkConfig(
            name=f"LinearVsStandard_seq{seq_len}",
            warmup_runs=warmup,
            benchmark_runs=runs,
        )

        comparison = runner.compare(
            {"LinearAttention": run_linear, "StandardAttention": run_standard},
            config
        )

        linear_ms = comparison["LinearAttention"].mean_ms
        standard_ms = comparison["StandardAttention"].mean_ms
        speedup = standard_ms / linear_ms
        print(f"  seq={seq_len}: Linear={linear_ms:.3f}ms, Standard={standard_ms:.3f}ms, Speedup={speedup:.2f}x")


# ============================================================================
# Memory Benchmarks
# ============================================================================


def benchmark_memory(quick: bool = False):
    """Run memory benchmarks."""
    from mlx_primitives.attention import FlashAttention, LinearAttention

    print("\n" + "=" * 60)
    print("MEMORY BENCHMARKS")
    print("=" * 60)

    profiler = MemoryProfiler()

    batch_size = 1
    num_heads = 8
    head_dim = 64
    dims = num_heads * head_dim

    seq_lens = [512, 1024, 2048] if quick else [512, 1024, 2048, 4096, 8192]

    print("\nTheoretical Memory Estimates (attention matrix):")
    print("-" * 50)
    for seq_len in seq_lens:
        estimates = estimate_attention_memory_mb(
            batch_size, seq_len, num_heads, head_dim
        )
        print(f"  seq_len={seq_len}: {estimates['attention_matrix_mb']:.2f} MB (attention matrix)")
        print(f"                   {estimates['total_mb']:.2f} MB (total)")

    print("\nActual Memory Usage (FlashAttention):")
    print("-" * 50)

    for seq_len in seq_lens:
        x = mx.random.normal((batch_size, seq_len, dims))
        mx.eval(x)

        flash = FlashAttention(dims=dims, num_heads=num_heads)

        def run_flash():
            out = flash(x)
            mx.eval(out)

        profile = profiler.profile_function(run_flash, name=f"flash_seq{seq_len}")
        print(f"  seq_len={seq_len}: peak={profile.peak_allocated_mb:.2f} MB")

    return profiler


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run mlx-primitives benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--attention", action="store_true", help="Run attention benchmarks")
    parser.add_argument("--layers", action="store_true", help="Run layer benchmarks")
    parser.add_argument("--training", action="store_true", help="Run training benchmarks")
    parser.add_argument("--advanced", action="store_true", help="Run advanced benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory benchmarks")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmarks")
    parser.add_argument("--vs-naive", action="store_true", dest="vs_naive", help="Run vs-naive comparison benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--charts", action="store_true", help="Generate charts")

    args = parser.parse_args()

    # Default to all if no specific benchmark selected
    if not any([args.all, args.attention, args.layers, args.training, args.advanced, args.memory, args.scaling, args.vs_naive]):
        args.all = True

    output_dir = Path(args.output)
    runner = BenchmarkRunner(output_dir=str(output_dir))
    chart_gen = ChartGenerator(output_dir=str(output_dir / "charts"))

    print("\nMLX-Primitives Benchmark Suite")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output: {output_dir}")
    print()

    all_results = []

    if args.all or args.attention:
        results = benchmark_attention(runner, quick=args.quick)
        all_results.extend(results)

    if args.all or args.scaling:
        scaling_data = benchmark_attention_scaling(runner, quick=args.quick)

        if args.charts:
            chart_gen.line_chart(
                scaling_data,
                ChartConfig(
                    title="Attention Scaling with Sequence Length",
                    xlabel="Sequence Length",
                    ylabel="Time (ms)",
                    log_scale_x=True,
                    log_scale_y=True,
                ),
                filename="attention_scaling"
            )

    if args.all or args.layers:
        results = benchmark_layers(runner, quick=args.quick)
        all_results.extend(results)

        norm_comparison = benchmark_normalization(runner, quick=args.quick)

    if args.all or args.training:
        results = benchmark_training(runner, quick=args.quick)
        all_results.extend(results)

    if args.all or args.advanced:
        results = benchmark_advanced(runner, quick=args.quick)
        all_results.extend(results)

    if args.all or args.vs_naive:
        benchmark_vs_naive(runner, quick=args.quick)

    if args.all or args.memory:
        benchmark_memory(quick=args.quick)

    # Save results
    if all_results:
        runner.print_results()
        runner.save_results()

        if args.charts:
            report = chart_gen.generate_report(all_results, "benchmark_report.md")
            print(f"\nReport saved to {output_dir / 'charts' / 'benchmark_report.md'}")

    print("\nBenchmark suite complete!")


if __name__ == "__main__":
    main()
