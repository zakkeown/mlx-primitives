"""
Sliding Window Attention Example - Bounded context attention.

Sliding window attention limits each token to attending only within a fixed window,
useful for long sequences where full attention is too expensive or for models
like Mistral that use this pattern.

Usage:
    python sliding_window.py
    python sliding_window.py --seq_len 4096 --window_size 256
"""

import argparse
import time

import mlx.core as mx

from mlx_primitives import SlidingWindowAttention


def benchmark_sliding_window(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    window_size: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark sliding window attention."""

    # Create attention module
    attn = SlidingWindowAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        window_size=window_size,
        causal=True,
    )

    # Create input tensors (BSHD layout)
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)

    # Warmup
    for _ in range(num_warmup):
        out = attn(q, k, v)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = attn(q, k, v)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "shape": shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Sliding Window Attention example")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--window_size", type=int, default=256, help="One-sided window size"
    )
    args = parser.parse_args()

    print("Sliding Window Attention Example")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Seq length:   {args.seq_len}")
    print(f"  Num heads:    {args.num_heads}")
    print(f"  Head dim:     {args.head_dim}")
    print(f"  Window size:  {args.window_size}")
    print()

    # Run benchmark
    results = benchmark_sliding_window(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        window_size=args.window_size,
    )

    print(f"Results:")
    print(f"  Shape:    {results['shape']}")
    print(f"  Mean:     {results['mean_ms']:.2f} ms")
    print(f"  Min:      {results['min_ms']:.2f} ms")
    print(f"  Max:      {results['max_ms']:.2f} ms")
    print()

    # Context window explanation
    total_context = 2 * args.window_size + 1
    print(f"Context window:")
    print(f"  Each token attends to {total_context} tokens")
    print(f"  ({args.window_size} before + self + {args.window_size} after)")
    print(f"  With causal masking: up to {args.window_size + 1} tokens")


if __name__ == "__main__":
    main()
