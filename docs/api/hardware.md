# mlx_primitives.hardware

Hardware detection and optimization for Apple Silicon.

## Quick Reference

| Class/Function | Description |
|----------------|-------------|
| `get_chip_info` | Detect chip family and capabilities |
| `ChipFamily` | Enum of chip generations (M1, M2, M3, M4) |
| `TilingConfig` | Kernel tiling configuration |
| `AutoTuner` | Runtime kernel tuning |

## Usage Example

```python
from mlx_primitives.hardware import get_chip_info

info = get_chip_info()
print(f"Chip: {info.chip_family}")
print(f"Memory: {info.memory_gb} GB")
print(f"Bandwidth: {info.memory_bandwidth_gbps} GB/s")
print(f"ANE TOPS: {info.ane_tops}")
```

## Chip Families

| Family | GPU Cores | Memory Bandwidth |
|--------|-----------|------------------|
| M1 | 7-8 | 68 GB/s |
| M2 | 8-10 | 100 GB/s |
| M3 | 10-18 | 150 GB/s |
| M4 | 10-20 | 200 GB/s |

## Module Contents

::: mlx_primitives.hardware
    options:
      show_root_heading: false
      members_order: source
      show_source: true
