"""
FlashAttention Example - Memory-efficient attention for long sequences.

Demonstrates FlashAttention's O(n) memory usage compared to standard attention's O(n^2).
Shows both BSHD and BHSD tensor layouts.

Usage:
    python flash_attention.py
    python flash_attention.py --seq_len 4096 --batch_size 4
    python flash_attention.py --layout BHSD
"""

import argparse
import time

import mlx.core as mx

from mlx_primitives import flash_attention


def benchmark_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    layout: str,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark flash attention with given parameters."""

    # Create tensors based on layout
    if layout == "BSHD":
        shape = (batch_size, seq_len, num_heads, head_dim)
    else:  # BHSD
        shape = (batch_size, num_heads, seq_len, head_dim)

    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)

    # Warmup
    for _ in range(num_warmup):
        out = flash_attention(q, k, v, causal=True, layout=layout)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = flash_attention(q, k, v, causal=True, layout=layout)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "shape": shape,
    }


def main():
    parser = argparse.ArgumentParser(description="FlashAttention example")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--layout",
        type=str,
        default="BSHD",
        choices=["BSHD", "BHSD"],
        help="Tensor layout",
    )
    args = parser.parse_args()

    print("FlashAttention Example")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq length:  {args.seq_len}")
    print(f"  Num heads:   {args.num_heads}")
    print(f"  Head dim:    {args.head_dim}")
    print(f"  Layout:      {args.layout}")
    print()

    # Run benchmark
    results = benchmark_attention(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        layout=args.layout,
    )

    print(f"Results:")
    print(f"  Shape:    {results['shape']}")
    print(f"  Mean:     {results['mean_ms']:.2f} ms")
    print(f"  Min:      {results['min_ms']:.2f} ms")
    print(f"  Max:      {results['max_ms']:.2f} ms")
    print()

    # Memory estimation
    # Standard attention would need O(batch * heads * seq^2) for attention weights
    # Flash attention only needs O(batch * seq * heads * head_dim) for output
    standard_mem_mb = (
        args.batch_size * args.num_heads * args.seq_len * args.seq_len * 4 / 1e6
    )
    flash_mem_mb = (
        args.batch_size * args.seq_len * args.num_heads * args.head_dim * 4 / 1e6
    )

    print(f"Memory comparison (attention weights only):")
    print(f"  Standard attention: ~{standard_mem_mb:.1f} MB")
    print(f"  Flash attention:    ~{flash_mem_mb:.1f} MB")
    print(f"  Reduction:          {standard_mem_mb / flash_mem_mb:.1f}x")


if __name__ == "__main__":
    main()
