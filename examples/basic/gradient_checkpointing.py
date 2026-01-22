"""
Gradient Checkpointing Example - Memory-efficient training.

Gradient checkpointing trades compute for memory by not storing intermediate
activations during the forward pass, recomputing them during backward.

Note: In MLX with lazy evaluation, traditional checkpointing doesn't provide
the same memory benefits as in eager frameworks like PyTorch. This example
shows the API for compatibility and demonstrates MLX-specific patterns.

Usage:
    python gradient_checkpointing.py
"""

import argparse

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives import checkpoint


class MLPBlock(nn.Module):
    """Simple MLP block for demonstration."""

    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.linear1 = nn.Linear(dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.linear1(x))
        return self.linear2(h)


class TransformerBlock(nn.Module):
    """Simplified transformer block."""

    def __init__(self, dims: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)

        # Simplified self-attention
        self.qkv = nn.Linear(dims, 3 * dims)
        self.proj = nn.Linear(dims, dims)
        self.num_heads = num_heads
        self.head_dim = dims // num_heads

        # MLP
        self.mlp = MLPBlock(dims, int(dims * mlp_ratio))

    def attention(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.proj(out)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def forward_with_checkpointing(model: nn.Module, x: mx.array) -> mx.array:
    """Forward pass with gradient checkpointing on each block."""

    def block_fn(block, inp):
        return block(inp)

    out = x
    for block in model.layers:
        # Checkpoint each block's forward pass
        out = checkpoint(lambda inp: block_fn(block, inp), out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Gradient checkpointing example")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--dims", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    args = parser.parse_args()

    print("Gradient Checkpointing Example")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq length:  {args.seq_len}")
    print(f"  Dims:        {args.dims}")
    print(f"  Num heads:   {args.num_heads}")
    print(f"  Num layers:  {args.num_layers}")
    print()

    # Create a simple transformer model
    class SimpleTransformer(nn.Module):
        def __init__(self, dims, num_heads, num_layers):
            super().__init__()
            self.layers = [
                TransformerBlock(dims, num_heads) for _ in range(num_layers)
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleTransformer(args.dims, args.num_heads, args.num_layers)

    # Create input
    x = mx.random.normal((args.batch_size, args.seq_len, args.dims))

    # Standard forward
    print("Running standard forward pass...")
    out_standard = model(x)
    mx.eval(out_standard)

    # Forward with checkpointing
    print("Running forward with checkpointing...")
    out_checkpoint = forward_with_checkpointing(model, x)
    mx.eval(out_checkpoint)

    # Verify outputs match
    diff = mx.abs(out_standard - out_checkpoint).max().item()
    print(f"\nMax difference between outputs: {diff:.2e}")
    print("Outputs match!" if diff < 1e-5 else "Warning: outputs differ")

    print("\nNote: In MLX, gradient checkpointing is primarily for API")
    print("compatibility. For memory optimization, consider:")
    print("  - Smaller batch sizes with gradient accumulation")
    print("  - mx.eval() calls to control graph size")
    print("  - FlashAttention for O(n) memory attention")


if __name__ == "__main__":
    main()
