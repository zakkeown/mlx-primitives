# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/attention/test_attention.py -v

# Run single test
pytest tests/attention/test_attention.py::test_flash_attention -v

# Skip slow/benchmark tests
pytest tests/ -m "not benchmark and not slow"

# Run with coverage
pytest tests/ --cov=mlx_primitives --cov-report=html

# Run benchmarks
pytest tests/ -m benchmark --benchmark-json=benchmark.json

# Linting and formatting
black .                                    # Format code
black --check .                            # Check formatting
ruff check .                               # Lint
ruff check . --fix                         # Lint with auto-fix
mypy mlx_primitives --ignore-missing-imports  # Type check
```

## Architecture Overview

### Module Hierarchy

```
mlx_primitives/
├── attention/     # Core attention mechanisms (FlashAttention, RoPE, GQA, sparse variants)
├── kernels/       # Metal kernel wrappers and optimized ops (fused RoPE, SIMD GQA)
├── layers/        # NN layers (normalization, activations, pooling, embeddings)
├── training/      # Training infrastructure (Trainer, schedulers, callbacks, EMA)
├── data/          # Data loading (DataLoader, samplers, transforms)
├── advanced/      # Advanced components (MoE, SSM/Mamba, KV cache, quantization)
└── config/        # Runtime configuration (precision modes)
```

### Key Technical Patterns

**MLX Fast SDPA Integration**: The primary attention path uses `mx.fast.scaled_dot_product_attention` which provides 8-9x speedup over Python implementations. All attention modules check for `_HAS_MLX_FAST_SDPA` and route to the optimized path:
```python
if use_mlx_sdpa and _HAS_MLX_FAST_SDPA:
    return _mlx_fast_sdpa(q, k, v, scale, causal)
```

**Metal Kernels**: Custom Metal shaders in `metal/` are accessed via `mx.fast.metal_kernel()`. The Python wrappers handle shape validation, parameter packing, and grid/threadgroup configuration. Key constraints:
- 32KB threadgroup memory limit
- Bank conflict avoidance uses `HEAD_DIM_PAD = head_dim + 4`
- BLOCK_SIZE=24 is standard for tiled kernels

**Tensor Layout Convention**: Attention tensors use `(batch, seq, heads, head_dim)` internally, transposed to `(batch, heads, seq, head_dim)` for MLX SDPA calls.

**Precision System**: Auto-precision selection (`mlx_primitives/config/precision.py`) detects when fp16 is safe based on sequence length and input magnitude, providing ~2x bandwidth improvement.

### Performance-Critical Modules

- `attention/flash.py`: FlashAttention with MLX SDPA, auto block sizing, precision selection
- `kernels/gqa_optimized.py`: GQA without K/V expansion (~6x faster via SDPA)
- `kernels/fused_rope_attention.py`: Fused RoPE + attention with bank-conflict-free tiling
- `attention/quantized_kv_cache.py`: INT8 KV cache (~4x memory reduction)
- `advanced/paged_attention/`: vLLM-style block-based KV cache with COW support

### Correctness Testing

Tests in `tests/correctness/` verify numerical accuracy against reference implementations:
```python
# Standard tolerances
rtol=1e-3, atol=1e-4  # For fp16
rtol=1e-5, atol=1e-6  # For fp32
```

## Metal Shader Guidelines

Shaders in `metal/` must:
- Declare shared memory with fixed sizes (no dynamic allocation)
- Use `threadgroup_barrier(mem_flags::mem_threadgroup)` between load and compute phases
- Pad shared memory indexing to avoid bank conflicts: `local_tid * (head_dim + 4) + d`
- Stay under 32KB total threadgroup memory

## Testing New Attention Variants

1. Implement in `mlx_primitives/attention/`
2. Add reference implementation using standard `mx` ops
3. Create correctness test comparing to reference
4. Benchmark against `mx.fast.scaled_dot_product_attention`
5. Only use custom Metal if measurably faster than MLX SDPA
