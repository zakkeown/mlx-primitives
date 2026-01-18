"""Training utilities for MLX.

This module provides memory-efficient training primitives including
gradient checkpointing for reducing memory usage during backpropagation.
"""

from mlx_primitives.training.checkpointing import (
    checkpoint,
    checkpoint_sequential,
)

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
]
