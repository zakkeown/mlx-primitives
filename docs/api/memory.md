# mlx_primitives.memory

Memory primitives for unified memory exploitation.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `UnifiedView` | Zero-copy memory view |
| `StreamingTensor` | Memory-efficient streaming |
| `PingPongBuffer` | Double buffering for overlap |
| `prefetch_to_gpu` | Explicit GPU prefetching |

## Unified Memory

Apple Silicon uses unified memory shared between CPU and GPU. This module provides primitives to exploit this architecture:

```python
from mlx_primitives.memory import create_unified_buffer, zero_copy_slice

# Create a unified buffer
buffer = create_unified_buffer(shape=(1024, 1024))

# Zero-copy slicing
view = zero_copy_slice(buffer, start=0, end=512)
```

## Module Contents

::: mlx_primitives.memory
    options:
      show_root_heading: false
      members_order: source
      show_source: true
