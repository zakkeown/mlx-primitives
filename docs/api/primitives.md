# mlx_primitives.primitives

Core parallel primitives for ML operations.

## Quick Reference

| Function | Description |
|----------|-------------|
| `associative_scan` | Parallel prefix scan (cumsum, cumprod, SSM) |
| `selective_scan` | SSM recurrence operator |
| `selective_gather` | Sparse gather operation |
| `selective_scatter_add` | Sparse scatter-add operation |

## Usage Example

```python
from mlx_primitives import associative_scan
import mlx.core as mx

# Parallel cumulative sum
x = mx.array([1, 2, 3, 4, 5])
cumsum = associative_scan(x, operator="add")
# [1, 3, 6, 10, 15]

# Parallel cumulative product
cumprod = associative_scan(x, operator="mul")
# [1, 2, 6, 24, 120]
```

## Module Contents

::: mlx_primitives.primitives
    options:
      show_root_heading: false
      members_order: source
      show_source: true
