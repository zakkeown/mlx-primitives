# mlx_primitives.cache

KV cache implementations for efficient inference.

## Quick Reference

| Class | Description |
|-------|-------------|
| `KVCache` | Simple growing KV cache |
| `SlidingWindowCache` | Fixed-size sliding window cache |
| `PagedKVCache` | vLLM-style paged attention cache |
| `SpeculativeCache` | Cache for speculative decoding |

## Choosing a Cache

- **KVCache**: Simple use cases, short sequences
- **SlidingWindowCache**: Long sequences with bounded context (Mistral-style)
- **PagedKVCache**: Production inference with memory constraints

## Module Contents

::: mlx_primitives.cache
    options:
      show_root_heading: false
      members_order: source
      show_source: true
