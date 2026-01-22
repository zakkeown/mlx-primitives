# API Reference

Complete API documentation for MLX Primitives modules.

## Module Overview

| Module | Description |
|--------|-------------|
| [attention](attention.md) | Memory-efficient attention mechanisms |
| [primitives](primitives.md) | Parallel primitives (scan, dispatch) |
| [kernels](kernels.md) | Fused operations and optimized kernels |
| [cache](cache.md) | KV cache implementations |
| [generation](generation.md) | Batched text generation |
| [training](training.md) | Training utilities |
| [hardware](hardware.md) | Hardware detection and tuning |
| [memory](memory.md) | Memory primitives |
| [dsl](dsl.md) | Metal kernel DSL |
| [advanced](advanced.md) | MoE, SSM, quantization |

## Quick Import Reference

```python
# Core attention
from mlx_primitives import (
    flash_attention,
    FlashAttention,
    SlidingWindowAttention,
    ChunkedCrossAttention,
)

# Primitives
from mlx_primitives import associative_scan, selective_scan

# Training
from mlx_primitives import checkpoint, checkpoint_sequential

# Submodule imports
from mlx_primitives.kernels import SwiGLU, FusedRMSNormLinear
from mlx_primitives.cache import KVCache, PagedKVCache
from mlx_primitives.generation import GenerationEngine
from mlx_primitives.hardware import get_chip_info
from mlx_primitives.advanced.moe import MoELayer
```
