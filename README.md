# MLX Primitives

High-performance building blocks for [MLX](https://github.com/ml-explore/mlx).

## Installation

```bash
pip install mlx-primitives
```

For development:

```bash
git clone https://github.com/zakkeown/mlx-primitives
cd mlx-primitives
pip install -e ".[dev]"
```

## Features

### Attention Mechanisms

- **FlashAttention** - Memory-efficient attention with tiling
- **GroupedQueryAttention** - GQA with configurable head groups
- **MultiQueryAttention** - Single KV head shared across Q heads
- **SlidingWindowAttention** - Fixed window context (Mistral-style)
- **RoPE** - Optimized rotary position embeddings
- **ALiBi** - Attention with Linear Biases

### Custom Layers (coming in v0.2)

- Normalization: RMSNorm, GroupNorm, InstanceNorm
- Activations: SwiGLU, GeGLU, Mish
- Pooling: AdaptiveAvgPool, GeM

### Training Utilities (coming in v0.3)

- Configurable Trainer with callbacks
- Learning rate schedulers
- EMA, gradient clipping

### Data Pipeline (coming in v0.4)

- Efficient DataLoader
- Vision and text augmentations

## Quick Start

```python
import mlx.core as mx
from mlx_primitives.attention import FlashAttention, RoPE

# Create attention layer
attn = FlashAttention(
    dims=768,
    num_heads=12,
    causal=True,
)

# Apply rotary embeddings
rope = RoPE(dims=64, max_seq_len=8192)
q, k = rope(q, k)

# Forward pass
output = attn(q, k, v)
```

## Benchmarks

Coming soon - performance comparisons across M1/M2/M3/M4 chips.

## License

MIT
