"""Training utilities for MLX.

This module provides training infrastructure:
- Trainer: Configurable training loop with callbacks
- Schedulers: Learning rate schedulers (cosine, warmup, one-cycle)
- Callbacks: EarlyStopping, ModelCheckpoint, logging integrations
- Utilities: EMA, gradient clipping, gradient accumulation
"""

# Training loop
# from mlx_primitives.training.trainer import Trainer, TrainingConfig

# Learning rate schedulers
# from mlx_primitives.training.schedulers import (
#     CosineAnnealingLR,
#     WarmupCosineScheduler,
#     OneCycleLR,
#     PolynomialDecayLR,
#     InverseSqrtScheduler,
# )

# Callbacks
# from mlx_primitives.training.callbacks import (
#     Callback,
#     EarlyStopping,
#     ModelCheckpoint,
#     LRMonitor,
#     GradientMonitor,
#     ProgressCallback,
# )

# Optimization utilities
# from mlx_primitives.training.utils import (
#     EMA,
#     GradientAccumulator,
#     GradientClipper,
# )

__all__: list[str] = []
