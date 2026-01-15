"""Configuration module for MLX Primitives.

Provides runtime configuration for precision, hardware detection, and optimization settings.
"""

from mlx_primitives.config.precision import (
    PrecisionMode,
    PrecisionConfig,
    get_precision_config,
    set_precision_config,
    set_precision_mode,
    precision_context,
)

__all__ = [
    "PrecisionMode",
    "PrecisionConfig",
    "get_precision_config",
    "set_precision_config",
    "set_precision_mode",
    "precision_context",
]
