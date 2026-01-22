# Performance Guide

Strategies for optimizing MLX Primitives workloads on Apple Silicon.

## Quick Wins

### 1. Use MLX SDPA

The biggest performance gain comes from MLX's built-in SDPA:

```python
from mlx_primitives import flash_attention

# Automatically uses mx.fast.scaled_dot_product_attention
out = flash_attention(q, k, v, causal=True)
```

### 2. Choose the Right Layout

For short sequences (< 512), BHSD layout avoids transpose overhead:

```python
# BHSD: (batch, heads, seq, dim) - faster for short sequences
out = flash_attention(q, k, v, layout="BHSD")

# BSHD: (batch, seq, heads, dim) - standard, better for long sequences
out = flash_attention(q, k, v, layout="BSHD")
```

### 3. Batch Operations

Larger batches amortize kernel launch overhead:

```python
# Prefer batched operations
out = flash_attention(q_batched, k_batched, v_batched)  # Good

# Avoid looping over batch dimension
for i in range(batch_size):  # Slow
    out[i] = flash_attention(q[i], k[i], v[i])
```

## Optimal Configurations

### FlashAttention

| Configuration | Recommendation |
|---------------|----------------|
| Sequence length | >= 512 for best speedup |
| Head dimension | 64 or 128 (fits shared memory) |
| Batch size | >= 4 for good parallelism |
| Block size | 32 (default) - tune for specific workloads |

### Fused Operations

Fused kernels reduce memory bandwidth:

```python
from mlx_primitives.kernels import FusedRMSNormLinear

# Fused: one memory read, one write
fused = FusedRMSNormLinear(dims, out_dims)
out = fused(x)  # ~2x faster than separate ops

# Separate: two reads, two writes
norm = nn.RMSNorm(dims)
linear = nn.Linear(dims, out_dims)
out = linear(norm(x))  # Slower
```

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python -m benchmarks.runner --suite all -o results.json

# Run specific suite
python -m benchmarks.runner --suite attention

# Run pytest benchmarks
pytest tests/ -m benchmark --benchmark-json=benchmark.json
```

### Interpreting Results

The benchmark runner reports speedup vs naive implementations:

```
FlashAttention (batch=4, seq=2048)
  Naive:  45.2 ms
  Flash:  10.1 ms
  Speedup: 4.47x
```

### Writing Benchmarks

```python
import time
import mlx.core as mx

def benchmark(fn, *args, warmup=3, runs=10):
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }
```

## Memory Optimization

### FlashAttention Memory

Standard attention uses O(n^2) memory for attention weights:

```
Standard: batch * heads * seq^2 * 4 bytes
Flash:    batch * seq * heads * head_dim * 4 bytes
```

Example for batch=2, heads=32, seq=8192, head_dim=128:

- Standard: 32 GB (attention weights alone)
- Flash: 256 MB (output only)

### Gradient Checkpointing

For training with limited memory:

```python
from mlx_primitives import checkpoint

def forward_with_checkpoints(model, x):
    for block in model.blocks:
        x = checkpoint(block, x)
    return x
```

Note: MLX's lazy evaluation means traditional checkpointing benefits differ from PyTorch.

### KV Cache Strategies

For inference, choose cache type based on workload:

```python
from mlx_primitives.cache import KVCache, SlidingWindowCache, PagedKVCache

# Simple: grows with sequence
cache = KVCache(num_layers, max_seq_len)

# Sliding window: fixed memory for long sequences
cache = SlidingWindowCache(num_layers, window_size=4096)

# Paged: vLLM-style block allocation
cache = PagedKVCache(num_layers, block_size=16)
```

## Hardware-Specific Optimization

### Chip Detection

```python
from mlx_primitives.hardware import get_chip_info

info = get_chip_info()
print(f"Chip: {info.chip_family}")  # M1, M2, M3, M4
print(f"Memory bandwidth: {info.memory_bandwidth_gbps} GB/s")
```

### Auto-Tuning

The hardware module provides auto-tuning for kernels:

```python
from mlx_primitives.hardware import auto_tune_for_workload

config = auto_tune_for_workload(
    operation="attention",
    batch_size=4,
    seq_len=2048,
    num_heads=32,
    head_dim=128,
)
```

## Common Pitfalls

### 1. Not Evaluating Tensors

MLX uses lazy evaluation. Always `mx.eval()` when timing:

```python
# Wrong - doesn't include compute time
start = time.time()
out = model(x)
elapsed = time.time() - start  # Just graph construction time

# Correct - includes actual compute
start = time.time()
out = model(x)
mx.eval(out)
elapsed = time.time() - start  # Real execution time
```

### 2. Small Batch Sizes

Kernel launch overhead dominates for small batches:

```python
# Slow: kernel launch overhead per item
for x in dataset:
    out = model(x.unsqueeze(0))

# Fast: batched processing
out = model(batch)
```

### 3. Unnecessary Transposes

Check if your data is already in the right layout:

```python
# If your data is BHSD, use layout parameter
out = flash_attention(q, k, v, layout="BHSD")

# Don't transpose just to match default layout
q_bshd = q.transpose(0, 2, 1, 3)  # Unnecessary if BHSD works
```

## Performance Checklist

Before optimizing:

- [ ] Profile to identify bottleneck (compute vs memory vs IO)
- [ ] Check if MLX SDPA is being used
- [ ] Verify batch size is sufficient (>= 4)
- [ ] Ensure `mx.eval()` is called appropriately

After optimizing:

- [ ] Run benchmarks to verify improvement
- [ ] Check numerical accuracy vs reference
- [ ] Test on target hardware (M1/M2/M3/M4)
