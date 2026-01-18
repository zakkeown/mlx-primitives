# MLX Primitives

Building blocks for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

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

## What This Library Provides

### Attention Mechanisms

- **SDPAAttention** - Convenience wrapper around `mx.fast.scaled_dot_product_attention`
- **GroupedQueryAttention** - GQA with configurable head groups
- **MultiQueryAttention** - Single KV head shared across Q heads
- **SlidingWindowAttention** - Fixed window context (Mistral-style)
- **FusedRoPEFlashAttention** - Fused RoPE + attention (genuine optimization, ~1.3-2x speedup)
- **RoPE** - Rotary position embeddings
- **ALiBi** - Attention with Linear Biases

> **Note**: `FlashAttention` is an alias for `SDPAAttention`. It wraps MLX's built-in
> SDPA and is NOT a custom flash attention implementation. Calling
> `mx.fast.scaled_dot_product_attention` directly gives equivalent performance.

### Custom Layers

- **Normalization**: RMSNorm, GroupNorm, InstanceNorm, AdaLayerNorm, QKNorm
- **Activations**: SwiGLU, GeGLU, ReGLU, Mish, GELUTanh, SquaredReLU, QuickGELU
- **Pooling**: AdaptiveAvgPool1d/2d, AdaptiveMaxPool1d/2d, GeM, GlobalAttentionPooling
- **Embeddings**: SinusoidalEmbedding, LearnedPositionalEmbedding, RotaryEmbedding

### Training Utilities

- **Trainer**: Training loop with callbacks (EarlyStopping, ModelCheckpoint, WandB)
- **Schedulers**: CosineAnnealing, WarmupCosine, OneCycle, PolynomialDecay
- **Optimization**: EMA, gradient clipping, gradient accumulation

### Data Pipeline

- **DataLoader**: Data loading with prefetching
- **Vision Transforms**: RandomCrop, ColorJitter, MixUp, CutMix
- **Text Transforms**: Padding, masking, sequence packing

### Advanced Primitives

- **MoE**: TopKRouter, ExpertChoiceRouter, MoELayer, SwitchMoE
- **KV Cache**: Standard, Sliding Window, Paged, Rotating, Compressed, Quantized
- **Quantization**: INT8/INT4, QLoRA, GPTQ, AWQ quantization schemes

### Experimental (Not Production-Ready)

- **SSM**: Mamba, S4, H3 state space models

  ⚠️ **Warning**: SSM implementations use sequential Python loops and are NOT
  efficient for long sequences (>512 tokens). They defeat the core value
  proposition of SSMs. Use attention-based models for production workloads.

## Honest Value Assessment

**What you get from this library:**
- Unified API for common MLX patterns
- Convenience wrappers with sensible defaults
- Fused RoPE+attention kernel (~1.3-2x speedup over separate ops)
- Auto-precision selection utilities
- Reference SSM implementations (for learning, not production)

**What you should NOT expect:**
- Novel performance improvements over MLX built-ins (most ops just wrap MLX)
- Efficient long-sequence SSM (the implementations are too slow)
- Published benchmarks (not yet integrated)

## Quick Start

```python
import mlx.core as mx
from mlx_primitives.attention import SDPAAttention, RoPE

# Create attention layer (wraps MLX's SDPA)
attn = SDPAAttention(
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

Not yet published. Run manually via `python -m benchmarks.runner`.

## License

MIT
