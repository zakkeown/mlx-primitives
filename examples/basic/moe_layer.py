"""
Mixture of Experts (MoE) Example - Sparse expert routing.

MoE layers route each token to a subset of expert networks, enabling
larger model capacity without proportional compute increase. This is
the architecture used in models like Mixtral and GPT-4.

Usage:
    python moe_layer.py
    python moe_layer.py --num_experts 16 --top_k 2
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.advanced.moe import MoELayer, TopKRouter


def benchmark_moe(
    batch_size: int,
    seq_len: int,
    dims: int,
    num_experts: int,
    top_k: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark MoE layer."""

    # Create MoE layer
    moe = MoELayer(
        dims=dims,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dims=dims * 4,
    )

    # Create input
    x = mx.random.normal((batch_size, seq_len, dims))

    # Warmup
    for _ in range(num_warmup):
        result = moe(x)
        mx.eval(result.output)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = moe(x)
        mx.eval(result.output)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "aux_loss": result.aux_loss.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Mixture of Experts example")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--dims", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument(
        "--top_k", type=int, default=2, help="Number of experts per token"
    )
    args = parser.parse_args()

    print("Mixture of Experts (MoE) Example")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Seq length:    {args.seq_len}")
    print(f"  Dims:          {args.dims}")
    print(f"  Num experts:   {args.num_experts}")
    print(f"  Top-K:         {args.top_k}")
    print()

    # Calculate efficiency metrics
    total_params_dense = args.dims * (args.dims * 4) * 2  # Dense MLP
    total_params_moe = args.dims * (args.dims * 4) * 2 * args.num_experts
    active_params_moe = args.dims * (args.dims * 4) * 2 * args.top_k

    print(f"Parameter comparison:")
    print(f"  Dense MLP params:     {total_params_dense:,}")
    print(f"  MoE total params:     {total_params_moe:,}")
    print(f"  MoE active params:    {active_params_moe:,} (per token)")
    print(f"  Capacity increase:    {total_params_moe / total_params_dense:.1f}x")
    print(f"  Compute increase:     {active_params_moe / total_params_dense:.1f}x")
    print()

    # Run benchmark
    results = benchmark_moe(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dims=args.dims,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    print(f"Results:")
    print(f"  Mean time:    {results['mean_ms']:.2f} ms")
    print(f"  Min time:     {results['min_ms']:.2f} ms")
    print(f"  Max time:     {results['max_ms']:.2f} ms")
    print(f"  Aux loss:     {results['aux_loss']:.4f}")
    print()

    # Demonstrate router behavior
    print("Router behavior demonstration:")
    moe = MoELayer(
        dims=args.dims,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_dims=args.dims * 4,
    )

    x = mx.random.normal((1, 4, args.dims))  # Small batch for visualization
    result = moe(x)

    # Get router logits to show expert selection
    router_probs = mx.softmax(result.router_logits, axis=-1)
    mx.eval(router_probs)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {result.output.shape}")
    print(f"  Router logits shape: {result.router_logits.shape}")
    print(f"\n  Sample token expert probabilities:")
    probs = router_probs[0, 0].tolist()
    for i, p in enumerate(probs):
        bar = "#" * int(p * 20)
        print(f"    Expert {i}: {p:.3f} {bar}")


if __name__ == "__main__":
    main()
