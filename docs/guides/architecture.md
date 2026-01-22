# Architecture Overview

MLX Primitives provides optimized building blocks for ML on Apple Silicon. This guide explains the architecture, design decisions, and code organization.

## Module Hierarchy

```
mlx_primitives/
├── attention/     # Core attention mechanisms
├── kernels/       # Fused operations and Metal wrappers
├── primitives/    # Parallel primitives (scan, MoE dispatch)
├── cache/         # KV cache implementations
├── generation/    # Batched generation engine
├── training/      # Training utilities
├── hardware/      # Hardware detection and autotuning
├── memory/        # Memory primitives
├── dsl/           # Metal kernel DSL
├── ane/           # Apple Neural Engine dispatch
├── layers/        # NN layers
└── advanced/      # MoE, SSM, quantization
```

## Design Principles

### 1. SDPA-First Strategy

We always prefer MLX's built-in `mx.fast.scaled_dot_product_attention` when available:

```python
# In attention modules
if _HAS_MLX_FAST_SDPA:
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
else:
    return _reference_attention(q, k, v, scale=scale)
```

**Why?** MLX's SDPA is highly optimized (8-9x faster) and handles edge cases correctly. Custom Metal kernels are only used when they provide measurable benefits beyond SDPA.

### 2. Automatic Fallback Chains

Each module implements a priority chain:

1. **MLX built-ins** (fastest, most tested)
2. **Custom Metal kernels** (when beneficial)
3. **Reference Python implementation** (always correct)

```python
def flash_attention(q, k, v, ...):
    # Priority 1: MLX SDPA
    if use_mlx_sdpa and _HAS_MLX_FAST_SDPA:
        return _mlx_fast_sdpa(q, k, v, scale, causal)

    # Priority 2: Custom Metal kernel
    if use_metal and _HAS_METAL and _should_use_metal(q):
        return _flash_attention_metal(q, k, v, ...)

    # Priority 3: Reference implementation
    return _flash_attention_reference(q, k, v, ...)
```

### 3. Threshold-Based Dispatch

Operations have performance thresholds that determine which implementation to use:

```python
# Environment variable controls
MIN_SEQ_FOR_METAL = int(os.environ.get("MLX_PRIMITIVES_MIN_SEQ_FOR_METAL", "8"))

def _should_use_metal(q):
    seq_len = q.shape[1]
    return seq_len >= MIN_SEQ_FOR_METAL
```

## Tensor Layout Convention

### Standard Layout: BSHD

Most operations use `(batch, seq_len, num_heads, head_dim)`:

```python
q = mx.random.normal((batch, seq_len, num_heads, head_dim))
```

### MLX SDPA Layout: BHSD

MLX's SDPA expects `(batch, num_heads, seq_len, head_dim)`:

```python
# Internal transpose for SDPA
q_bhsd = q.transpose(0, 2, 1, 3)  # BSHD -> BHSD
out_bhsd = mx.fast.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, ...)
out = out_bhsd.transpose(0, 2, 1, 3)  # BHSD -> BSHD
```

### Layout Parameter

Some functions accept a `layout` parameter to avoid transpose overhead:

```python
# No transpose needed when input is already BHSD
out = flash_attention(q, k, v, layout="BHSD")
```

## Metal Kernel Integration

### Kernel Registration

Custom Metal kernels are registered via `mx.fast.metal_kernel()`:

```python
_kernel = mx.fast.metal_kernel(
    name="flash_attention",
    source=METAL_SOURCE,
    input_names=["q", "k", "v"],
    output_names=["out"],
)
```

### Memory Constraints

Metal shaders must respect hardware limits:

- **32KB threadgroup memory** - Maximum shared memory per threadgroup
- **Bank conflict avoidance** - Pad indices: `HEAD_DIM_PAD = head_dim + 4`
- **Standard block size** - `BLOCK_SIZE = 24` for most tiled kernels

```metal
// In Metal shader
threadgroup float shared_q[BLOCK_SIZE][HEAD_DIM + 4];  // +4 avoids bank conflicts
```

### Thread Synchronization

Always synchronize between load and compute phases:

```metal
// Load phase
shared_q[local_tid][d] = q[global_idx];
threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute phase
float sum = 0.0f;
for (int i = 0; i < BLOCK_SIZE; i++) {
    sum += shared_q[i][d] * shared_k[i][d];
}
```

## Precision System

### Auto-Precision Selection

The precision system (`mlx_primitives/config/precision.py`) detects when FP16 is safe:

```python
def should_use_fp16(x: mx.array) -> bool:
    # Check sequence length (longer = more accumulation = more precision needed)
    if x.shape[1] > FP16_MAX_SEQ_LEN:
        return False

    # Check magnitude (large values overflow in FP16)
    if x.abs().max() > FP16_MAX_VALUE:
        return False

    return True
```

### Benefits

- ~2x memory bandwidth improvement with FP16
- Automatic fallback to FP32 when needed

## Module Deep Dives

### attention/

Core attention mechanisms with MLX SDPA integration:

- `flash.py` - FlashAttention with O(n) memory
- `sliding_window.py` - Bounded context attention
- `gqa.py` - Grouped Query Attention
- `rope.py` - Rotary Position Embeddings

### kernels/

Fused operations that reduce memory bandwidth:

- `fused_rmsnorm_linear.py` - RMSNorm + Linear in one pass
- `swiglu.py` - Fused SwiGLU activation
- `gqa_optimized.py` - GQA without K/V expansion

### primitives/

Parallel algorithms:

- `scan.py` - Associative scan for prefix operations
- `dispatch.py` - Expert routing for MoE

### cache/

KV cache implementations:

- `kv_cache.py` - Simple growing cache
- `paged.py` - vLLM-style paged attention
- `speculative.py` - Speculative decoding support

## Testing Patterns

### Correctness Tests

Compare against reference implementations:

```python
def test_flash_attention_correctness():
    # Reference implementation
    expected = reference_attention(q, k, v)

    # Optimized implementation
    actual = flash_attention(q, k, v)

    # Compare with appropriate tolerance
    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-4)
```

### Standard Tolerances

- **FP16**: `rtol=1e-3, atol=1e-4`
- **FP32**: `rtol=1e-5, atol=1e-6`

### Benchmark Tests

Mark performance tests to skip in CI:

```python
@pytest.mark.benchmark
def test_flash_attention_performance():
    # Performance test code
    ...
```
