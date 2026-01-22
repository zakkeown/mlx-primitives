# MLX Primitives Examples

Runnable examples demonstrating key features of MLX Primitives.

## Basic Examples

| Example | Description |
|---------|-------------|
| [flash_attention.py](basic/flash_attention.py) | FlashAttention with O(n) memory usage |
| [sliding_window.py](basic/sliding_window.py) | Sliding window attention for bounded context |
| [gradient_checkpointing.py](basic/gradient_checkpointing.py) | Memory-efficient training with checkpointing |
| [moe_layer.py](basic/moe_layer.py) | Mixture of Experts layer |

## Running Examples

All examples are standalone and runnable:

```bash
# Activate your environment
source .venv313/bin/activate

# Run an example
python examples/basic/flash_attention.py

# Most examples support --help for options
python examples/basic/flash_attention.py --help
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- MLX >= 0.20.0
- mlx-primitives installed (`pip install -e .`)
