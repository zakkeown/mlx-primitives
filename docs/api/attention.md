# mlx_primitives.attention

Memory-efficient attention mechanisms optimized for Apple Silicon.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `flash_attention` | O(n) memory attention via tiled online softmax |
| `FlashAttention` | Module wrapper for flash_attention |
| `SlidingWindowAttention` | Bounded context attention |
| `ChunkedCrossAttention` | Memory-efficient cross-attention |
| `RoPE` | Rotary Position Embeddings |

## Performance Notes

FlashAttention provides 4-5x speedup for sequences >= 512 tokens when using MLX's built-in SDPA. Memory usage is O(n) instead of O(n^2).

## Module Contents

::: mlx_primitives.attention
    options:
      show_root_heading: false
      members_order: source
      show_source: true
