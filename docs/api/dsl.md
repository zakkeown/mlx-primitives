# mlx_primitives.dsl

Metal-Triton: Python DSL for writing Metal kernels.

## Quick Reference

| Decorator/Type | Description |
|----------------|-------------|
| `@metal_kernel` | Decorator to compile Python to Metal |
| `@autotune` | Auto-tune kernel parameters |
| `constexpr` | Compile-time constant |
| `float16`, `float32` | Scalar types |

## Usage Example

```python
from mlx_primitives.dsl import metal_kernel, load, store, program_id

@metal_kernel
def vector_add(a_ptr, b_ptr, c_ptr, N: constexpr):
    pid = program_id(0)
    offset = pid * 256

    for i in range(256):
        idx = offset + i
        if idx < N:
            a = load(a_ptr + idx)
            b = load(b_ptr + idx)
            store(c_ptr + idx, a + b)
```

## Primitives

### Indexing
- `program_id(axis)` - Get program/block ID
- `thread_id_in_threadgroup()` - Thread ID within threadgroup
- `simd_lane_id()` - SIMD lane ID

### Memory
- `load(ptr)` - Load from global memory
- `store(ptr, value)` - Store to global memory
- `load_shared(ptr)` - Load from threadgroup memory
- `store_shared(ptr, value)` - Store to threadgroup memory

### Math
- `dot(a, b)` - Dot product
- `exp(x)`, `sqrt(x)`, `tanh(x)` - Math functions
- `fma(a, b, c)` - Fused multiply-add

### Synchronization
- `threadgroup_barrier()` - Threadgroup synchronization
- `simd_barrier()` - SIMD synchronization

## Module Contents

::: mlx_primitives.dsl
    options:
      show_root_heading: false
      members_order: source
      show_source: true
