"""Training utilities for MLX.

This module provides memory-efficient training primitives including
gradient checkpointing for reducing memory usage during backpropagation.
"""

from mlx_primitives.training.checkpointing import (
    checkpoint,
    checkpoint_sequential,
)

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

from mlx_primitives.training.callbacks import (
    Callback,
    CallbackList,
    TrainingState,
    EarlyStopping,
    ModelCheckpoint,
    LRMonitor,
    GradientMonitor,
    ProgressCallback,
    LambdaCallback,
    WandbCallback,
    TensorBoardCallback,
)

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
    copy_params,
    flatten_dict,
    unflatten_dict,
)

from mlx_primitives.training.trainer import (
    Trainer,
    TrainingConfig,
)

__all__ = [
    # Checkpointing
    "checkpoint",
    "checkpoint_sequential",
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
    "LambdaCallback",
    "WandbCallback",
    "TensorBoardCallback",
    # Utils
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
    "copy_params",
    "flatten_dict",
    "unflatten_dict",
    # Trainer
    "Trainer",
    "TrainingConfig",
]
