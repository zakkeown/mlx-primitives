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
- **seq_len > 1024**: Falls back to native MLX cumsum/cumprod ops (Metal multi-block overhead exceeds benefits)

For typical autoregressive inference (seq_len=1 per step), the vectorized fallback is used.
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

# For maximum performance with short sequences, use BHSD layout:
q_bhsd = mx.random.normal((2, 12, 512, 64))  # (batch, heads, seq, dim)
k_bhsd = mx.random.normal((2, 12, 512, 64))
v_bhsd = mx.random.normal((2, 12, 512, 64))
output = flash_attention(q_bhsd, k_bhsd, v_bhsd, causal=True, layout="BHSD")
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

Benchmarks run on Apple M4 Max (36GB) with MLX 0.30.3. Last verified: 2026-01-20 (RCA update).

### Attention (FlashAttention vs Naive O(n²))

**Default BSHD Layout** - (batch, seq, heads, dim):

| Sequence | Batch=1 | Batch=2 | Batch=4 |
|----------|---------|---------|---------|
| 128 | **1.75x** | 1.13x | 1.08x |
| 512 | **1.40x** | **1.57x** | **2.19x** |
| 1024 | **2.18x** | **3.30x** | **3.73x** |
| 2048 | **3.62x** | **4.20x** | **4.52x** |

**BHSD Layout** - (batch, heads, seq, dim) - no transpose overhead:

| Sequence | Batch=1 | Batch=2 | Batch=4 |
|----------|---------|---------|---------|
| 128 | **2.17x** | **1.41x** | **1.58x** |
| 512 | **1.96x** | **2.04x** | **2.93x** |
| 1024 | 1.06x | 0.99x | 1.00x |
| 2048 | 1.00x | 1.00x | 1.00x |

FlashAttention now provides speedups across all configurations. For short sequences (≤512), use `layout="BHSD"` if your data is already in (batch, heads, seq, dim) format for maximum performance. For longer sequences (1024+), both layouts perform similarly with significant speedups due to O(n) memory savings.

### Normalization (Fused vs Naive)

| Hidden Dim | LayerNorm (b=4, s=1024) | RMSNorm (b=4, s=1024) |
|------------|-------------------------|------------------------|
| 768 | 1.20x | 0.99x |
| 1024 | 1.31x | 1.07x |
| 4096 | **2.06x** | **1.36x** |

| Hidden Dim | LayerNorm (b=8, s=2048) | RMSNorm (b=8, s=2048) |
|------------|-------------------------|------------------------|
| 768 | **2.75x** | **1.71x** |
| 1024 | **2.87x** | **1.76x** |
| 4096 | **3.07x** | **1.91x** |

Fused kernels automatically fall back to MLX ops for small batch sizes (≤2) or short sequences (≤512) where kernel launch overhead exceeds benefits.

### Quantization (INT4/INT8 Linear)

Weight-only quantization now uses cached dequantization for inference efficiency:

| Weight Size | INT4 | INT8 | FP32 |
|-------------|------|------|------|
| 2048×2048 | 0.29ms | 0.30ms | 0.28ms |

INT4 quantization achieves near-parity with FP32 while using 8x less memory for weights.

### Sliding Window Attention

Uses SDPA with pre-computed window mask. Note: MLX SDPA doesn't have native sliding window support,
so performance is ~1.2-1.7x slower than flash attention for the same sequence length. The benefit
is semantic (local attention) rather than speed. Use this when you need bounded attention span.

| Config (batch=2) | Sliding Window | Flash Attention | Notes |
|------------------|----------------|-----------------|-------|
| seq=512, w=128 | 0.31ms | 0.25ms | 1.24x slower |
| seq=1024, w=128 | 0.77ms | 0.46ms | 1.68x slower |
| seq=2048, w=128 | 1.2ms | 1.25ms | ~parity |

### Fused RoPE + Attention

Composes optimized RoPE kernel with SDPA for maximum performance:

| Config | Time |
|--------|------|
| (2, 512, 8, 64) | **1.1ms** |
| (2, 1024, 8, 64) | **1.0ms** |
| (4, 2048, 8, 64) | **2.6ms** |

### Overall Performance

- **Average speedup**: 1.59x across all benchmarks
- **Max speedup**: 4.46x (FlashAttention at seq=2048, batch=4)
- **Min speedup**: 0.94x (near parity at smallest configs)
- **BHSD layout**: Eliminates transpose overhead for short sequences (up to 2.93x faster)

### Run Benchmarks

```bash
# Full suite
python -m benchmarks.runner --suite all

# Specific suite
python -m benchmarks.runner --suite attention

# With JSON output
python -m benchmarks.runner --suite all -o results.json
```

See [benchmark_results/rca_report.md](benchmark_results/rca_report.md) for detailed analysis.

## License

MIT
