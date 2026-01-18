"""Logging utilities for MLXPrimitives.

Provides consistent logging for fallback behavior when optimized
kernels fail to execute.
"""

import logging
import os
from typing import Optional

_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the MLX Primitives logger.

    The log level can be controlled via the MLX_PRIMITIVES_LOG_LEVEL
    environment variable. Valid values: DEBUG, INFO, WARNING, ERROR.
    Default is WARNING.

    Returns:
        The configured logger instance.
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("mlx_primitives")
        level_name = os.environ.get("MLX_PRIMITIVES_LOG_LEVEL", "WARNING")
        level = getattr(logging, level_name.upper(), logging.WARNING)
        _logger.setLevel(level)

        # Add handler if none exists
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            )
            _logger.addHandler(handler)

    return _logger


def log_fallback(operation: str, exception: Exception) -> None:
    """Log a fallback from optimized to reference implementation.

    Args:
        operation: Name of the operation that fell back.
        exception: The exception that caused the fallback.
    """
    get_logger().debug(
        "%s: optimized kernel failed (%s: %s), using fallback",
        operation,
        type(exception).__name__,
        str(exception),
    )
