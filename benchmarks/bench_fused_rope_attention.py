"""Benchmark fused RoPE + Flash Attention vs separate operations.

Run with: python benchmarks/bench_fused_rope_attention.py

Compares:
1. Separate: rope(Q), rope(K), flash_attention(Q_rot, K_rot, V)
2. Fused: fused_rope_attention(Q, K, V)
"""

import math
import time
from typing import Tuple

import mlx.core as mx


def benchmark_separate(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cos: mx.array,
    sin: mx.array,
    scale: float,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> float:
    """Benchmark separate RoPE + attention."""
    from mlx_primitives.kernels.rope import rope
    from mlx_primitives.attention.flash import flash_attention_forward

    # Warmup
    for _ in range(n_warmup):
        q_rot = rope(q, cos, sin)
        k_rot = rope(k, cos, sin)
        out = flash_attention_forward(q_rot, k_rot, v, scale, causal=True)
        mx.eval(out)

    mx.synchronize()
    start = time.perf_counter()

    for _ in range(n_iters):
        q_rot = rope(q, cos, sin)
        k_rot = rope(k, cos, sin)
        out = flash_attention_forward(q_rot, k_rot, v, scale, causal=True)
        mx.eval(out)

    mx.synchronize()
    return (time.perf_counter() - start) / n_iters


def benchmark_fused(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cos: mx.array,
    sin: mx.array,
    scale: float,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> float:
    """Benchmark fused RoPE + attention with precomputed cache."""
    from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

    # Warmup
    for _ in range(n_warmup):
        out = fused_rope_attention(q, k, v, scale, cos_cache=cos, sin_cache=sin, causal=True)
        mx.eval(out)

    mx.synchronize()
    start = time.perf_counter()

    for _ in range(n_iters):
        out = fused_rope_attention(q, k, v, scale, cos_cache=cos, sin_cache=sin, causal=True)
        mx.eval(out)

    mx.synchronize()
    return (time.perf_counter() - start) / n_iters


def run_benchmarks(quick: bool = False):
    """Run comparative benchmarks."""
    from mlx_primitives.kernels.rope import precompute_rope_cache

    configs = [
        # (batch, seq, heads, head_dim)
        (2, 128, 8, 64),     # Small
        (2, 256, 8, 64),     # Medium-small
        (2, 512, 12, 64),    # Medium
        (2, 1024, 12, 64),   # Large
    ]

    if not quick:
        configs.extend([
            (2, 2048, 12, 128),  # Very large
            (4, 4096, 16, 64),   # Extra large
        ])

    n_warmup = 3 if quick else 5
    n_iters = 10 if quick else 20

    print("=" * 70)
    print("Fused RoPE + Flash Attention Benchmarks")
    print("=" * 70)
    print()
    print(f"{'Config':<25} {'Separate':<12} {'Fused':<12} {'Speedup':<12}")
    print("-" * 70)

    total_speedup = 0
    n_configs = 0

    for batch, seq, heads, dim in configs:
        mx.random.seed(42)

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))
        cos, sin = precompute_rope_cache(seq * 2, dim)
        scale = 1.0 / math.sqrt(dim)
        mx.eval(q, k, v, cos, sin)

        try:
            t_sep = benchmark_separate(q, k, v, cos, sin, scale, n_warmup, n_iters) * 1000
            t_fused = benchmark_fused(q, k, v, cos, sin, scale, n_warmup, n_iters) * 1000
            speedup = t_sep / t_fused

            config_str = f"({batch}, {seq}, {heads}, {dim})"
            print(f"{config_str:<25} {t_sep:<12.3f} {t_fused:<12.3f} {speedup:<12.2f}x")

            total_speedup += speedup
            n_configs += 1

        except Exception as e:
            config_str = f"({batch}, {seq}, {heads}, {dim})"
            print(f"{config_str:<25} Error: {e}")

    print("-" * 70)
    if n_configs > 0:
        print(f"{'Average:':<25} {'':<12} {'':<12} {total_speedup/n_configs:<12.2f}x")
    print()


def run_scaling_benchmark(quick: bool = False):
    """Benchmark scaling with sequence length."""
    from mlx_primitives.kernels.rope import precompute_rope_cache
    from mlx_primitives.kernels.fused_rope_attention import fused_rope_attention

    print("=" * 70)
    print("Sequence Length Scaling")
    print("=" * 70)
    print()

    batch = 2
    heads = 8
    dim = 64
    scale = 1.0 / math.sqrt(dim)

    seq_lens = [64, 128, 256, 512, 1024] if quick else [64, 128, 256, 512, 1024, 2048, 4096]
    n_warmup = 3 if quick else 5
    n_iters = 10 if quick else 20

    print(f"{'Seq Length':<15} {'Time (ms)':<15} {'Throughput (Mtok/s)':<20}")
    print("-" * 50)

    for seq in seq_lens:
        mx.random.seed(42)

        q = mx.random.normal((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim))
        v = mx.random.normal((batch, seq, heads, dim))
        cos, sin = precompute_rope_cache(seq * 2, dim)
        mx.eval(q, k, v, cos, sin)

        try:
            t_fused = benchmark_fused(q, k, v, cos, sin, scale, n_warmup, n_iters) * 1000
            throughput = (batch * seq) / (t_fused / 1000) / 1e6  # Million tokens per second

            print(f"{seq:<15} {t_fused:<15.3f} {throughput:<20.2f}")

        except Exception as e:
            print(f"{seq:<15} Error: {e}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark fused RoPE + attention")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    args = parser.parse_args()

    run_benchmarks(quick=args.quick)
    run_scaling_benchmark(quick=args.quick)
