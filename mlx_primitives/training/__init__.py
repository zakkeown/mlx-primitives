"""Training utilities for MLX.

This module provides training infrastructure:
- Trainer: Configurable training loop with callbacks
- Schedulers: Learning rate schedulers (cosine, warmup, one-cycle)
- Callbacks: EarlyStopping, ModelCheckpoint, logging integrations
- Utilities: EMA, gradient clipping, gradient accumulation
"""

# Training loop
from mlx_primitives.training.trainer import Trainer, TrainingConfig

# Learning rate schedulers
from mlx_primitives.training.schedulers import (
    LRScheduler,
    CosineAnnealingLR,
    WarmupCosineScheduler,
    OneCycleLR,
    PolynomialDecayLR,
    InverseSqrtScheduler,
    LinearWarmupLR,
    ConstantLR,
    ExponentialDecayLR,
    MultiStepLR,
    ChainedScheduler,
)

# Callbacks
from mlx_primitives.training.callbacks import (
    Callback,
    CallbackList,
    TrainingState,
    EarlyStopping,
    ModelCheckpoint,
    LRMonitor,
    GradientMonitor,
    ProgressCallback,
    MetricLogger,
    LambdaCallback,
    WandbCallback,
    TensorBoardCallback,
)

# Optimization utilities
from mlx_primitives.training.utils import (
    EMA,
    EMAWithWarmup,
    GradientAccumulator,
    GradientClipper,
    MixedPrecisionManager,
    Checkpointer,
    SWA,
    Lookahead,
    SAM,
    GradientNoiseInjection,
    compute_gradient_norm,
    clip_grad_norm,
    clip_grad_value,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    # Schedulers
    "LRScheduler",
    "CosineAnnealingLR",
    "WarmupCosineScheduler",
    "OneCycleLR",
    "PolynomialDecayLR",
    "InverseSqrtScheduler",
    "LinearWarmupLR",
    "ConstantLR",
    "ExponentialDecayLR",
    "MultiStepLR",
    "ChainedScheduler",
    # Callbacks
    "Callback",
    "CallbackList",
    "TrainingState",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRMonitor",
    "GradientMonitor",
    "ProgressCallback",
    "MetricLogger",
    "LambdaCallback",
    "WandbCallback",
    "TensorBoardCallback",
    # Utilities
    "EMA",
    "EMAWithWarmup",
    "GradientAccumulator",
    "GradientClipper",
    "MixedPrecisionManager",
    "Checkpointer",
    "SWA",
    "Lookahead",
    "SAM",
    "GradientNoiseInjection",
    "compute_gradient_norm",
    "clip_grad_norm",
    "clip_grad_value",
]
