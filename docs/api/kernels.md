# mlx_primitives.kernels

Fused operations and optimized Metal kernels.

## Quick Reference

| Class | Description |
|-------|-------------|
| `FusedRMSNormLinear` | Fused RMSNorm + Linear |
| `SwiGLU` | Fused SwiGLU activation |
| `GeGLU` | Fused GeGLU activation |
| `QuantizedLinear` | INT4/INT8 quantized linear |
| `GQAOptimized` | Grouped Query Attention without K/V expansion |

## Performance Notes

Fused operations reduce memory bandwidth by combining multiple operations into a single kernel pass. Typical speedups:

- FusedRMSNormLinear: ~3x for large batches
- GQAOptimized: ~6x vs naive GQA

## Module Contents

::: mlx_primitives.kernels
    options:
      show_root_heading: false
      members_order: source
      show_source: true
