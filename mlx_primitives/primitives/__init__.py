"""Core parallel primitives for MLX.

This module provides fundamental parallel primitives that are missing from MLX,
including associative scan (parallel prefix sum) and selective gather/scatter.
"""

from mlx_primitives.primitives.gather_scatter import (
    ExpertDispatch,
    SparseMoELayer,
    build_expert_dispatch,
    compute_load_balancing_loss,
    selective_gather,
    selective_scatter_add,
)
from mlx_primitives.primitives.scan import associative_scan, selective_scan

__all__ = [
    "associative_scan",
    "selective_scan",
    "selective_gather",
    "selective_scatter_add",
    "build_expert_dispatch",
    "ExpertDispatch",
    "SparseMoELayer",
    "compute_load_balancing_loss",
]
