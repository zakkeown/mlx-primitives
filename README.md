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

- **FlashAttention** - Custom Flash Attention with O(n) memory via tiled online softmax
- **SlidingWindowAttention** - Fixed window context attention (Mistral-style)
- **ChunkedCrossAttention** - Memory-efficient cross-attention for long KV sequences

> **Note**: `FlashAttention` is a custom implementation of the Flash Attention algorithm
> (Dao et al., 2022) with O(n) memory complexity. It uses tiled online softmax and has
> both a Python fallback and an optimized Metal kernel. This is NOT a wrapper around
> `mx.fast.scaled_dot_product_attention`.

### Core Primitives

- **Parallel Scan**: `associative_scan`, `selective_scan` for SSM-style recurrences
- **MoE**: `SparseMoELayer`, `ExpertDispatch`, `build_expert_dispatch`, `compute_load_balancing_loss`
- **Gather/Scatter**: `selective_gather`, `selective_scatter_add`

### Training Utilities

- **Gradient Checkpointing**: `checkpoint`, `checkpoint_sequential` for memory-efficient training

### Additional Submodules

These are available via direct import from submodules:

| Submodule | Contents |
|-----------|----------|
| `mlx_primitives.cache` | Paged attention, KV cache management, eviction policies |
| `mlx_primitives.generation` | Batched generation engine, samplers, schedulers |
| `mlx_primitives.kernels` | Fused ops (SwiGLU, GeGLU, RMSNorm, INT8/INT4 quantization) |
| `mlx_primitives.dsl` | Metal-Triton DSL for writing Metal kernels in Python |
| `mlx_primitives.hardware` | Chip detection, auto-tuning |
| `mlx_primitives.memory` | Unified memory primitives, streaming |
| `mlx_primitives.ane` | Neural Engine offload (requires coremltools) |

## SSM Performance Notes

The `associative_scan` with `operator="ssm"` implements parallel prefix scan for SSM recurrences:

- **seq_len <= 8**: Uses vectorized MLX fallback (still GPU-accelerated)
- **8 < seq_len <= 1024**: Uses single-block parallel Metal kernel with O(log n) complexity
- **seq_len > 1024**: Uses multi-block parallel Metal kernel with O(log n) complexity

All sequence lengths now use parallel algorithms. For typical autoregressive inference
(seq_len=1 per step), the vectorized fallback is used.
Configure threshold via `MLX_PRIMITIVES_MIN_SEQ_FOR_METAL` environment variable.

## Quick Start

```python
import mlx.core as mx
from mlx_primitives import FlashAttention, flash_attention

# Create flash attention layer
attn = FlashAttention(num_heads=12, head_dim=64, causal=True)

q = mx.random.normal((2, 1024, 12, 64))
k = mx.random.normal((2, 1024, 12, 64))
v = mx.random.normal((2, 1024, 12, 64))

# Forward pass - O(n) memory
output = attn(q, k, v)

# Or use functional API
output = flash_attention(q, k, v, causal=True)
```

### Using Submodules

```python
# Fused kernels
from mlx_primitives.kernels import SwiGLU, quantize_int8

# KV Cache
from mlx_primitives.cache import KVCache, paged_attention

# Gradient checkpointing
from mlx_primitives import checkpoint, checkpoint_sequential

def transformer_block(x):
    # ... your block implementation
    return x

# Memory-efficient forward pass
output = checkpoint(transformer_block, x)
```

## Benchmarks

Not yet published. Run manually via `python -m benchmarks.runner`.

## License

MIT
