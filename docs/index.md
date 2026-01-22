# MLX Primitives Documentation

High-performance building blocks for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

## Quick Links

- [Getting Started](getting_started.md) - Installation and first steps
- [API Reference](api/index.md) - Complete API documentation
- [Guides](guides/index.md) - Architecture and performance guides
- [Examples](https://github.com/zakkeown/mlx-primitives/tree/main/examples) - Runnable code examples

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

## Key Features

### Attention Mechanisms

- **FlashAttention** - Memory-efficient tiled attention with online softmax
- **Sliding Window Attention** - Fixed window context (Mistral style)
- **Chunked Cross Attention** - Memory-efficient cross attention for long KV sequences

### Position Encodings

- **RoPE** - Rotary Position Embeddings with NTK and YaRN extensions
- **ALiBi** - Attention with Linear Biases (no learned positional embeddings)
- **Fused RoPE + Attention** - Single kernel combining position encoding and attention

### Custom Layers

- **Normalization** - RMSNorm, GroupNorm, InstanceNorm, AdaLayerNorm
- **Activations** - SwiGLU, GeGLU, Mish, QuickGELU, and more
- **Pooling** - Adaptive pooling, GeM, Global Attention Pooling

### Training Utilities

- **Gradient Checkpointing** - Memory-efficient training via recomputation

### Advanced Primitives

- **MoE** - Mixture of Experts with Top-K and Expert Choice routing
- **SSM** - Mamba, S4, H3 state space models (experimental)
- **KV Cache** - Standard, sliding window, paged, compressed variants
- **Quantization** - INT8/INT4 basic quantization

### Planned Features (Not Yet Implemented)

- Grouped Query Attention, Multi-Query Attention
- Linear Attention, Sparse Attention (Longformer, BigBird)
- Training utilities (Trainer, Schedulers, SWA, SAM)
- Advanced quantization (QLoRA, GPTQ, AWQ)

## Quick Example

```python
import mlx.core as mx
from mlx_primitives.attention import FlashAttention, RoPE

# Create attention layer
attn = FlashAttention(
    dims=768,
    num_heads=12,
    causal=True,
)

# Input: (batch, seq, dims)
x = mx.random.normal((2, 512, 768))

# Forward pass with automatic precision selection
output = attn(x)  # (2, 512, 768)

# With incremental decoding
output, cache = attn(x, return_cache=True)
# Next step
new_token = mx.random.normal((2, 1, 768))
output, cache = attn(new_token, cache=cache, return_cache=True)
```

## Performance

MLX Primitives leverages several optimizations:

- **MLX SDPA** - Uses `mx.fast.scaled_dot_product_attention` for ~8-9x speedup
- **Custom Metal Kernels** - Fused operations for RMSNorm, SwiGLU, RoPE
- **Auto Precision** - Automatic fp16/fp32 selection based on numerical safety
- **Adaptive Block Sizing** - Runtime tuning for non-standard configurations

## License

MIT
