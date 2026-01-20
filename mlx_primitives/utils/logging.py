"""Logging utilities for MLXPrimitives.

Provides consistent logging for fallback behavior when optimized
kernels fail to execute, plus Metal availability checking.
"""

import logging
import os
import sys
import threading
import warnings
from typing import Optional

_logger: Optional[logging.Logger] = None
_logger_lock = threading.Lock()
_fallback_seen: set[str] = set()  # Track operations that have fallen back
_metal_available: Optional[bool] = None  # Cached Metal availability

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Environment variable to raise exceptions instead of falling back silently
RAISE_ON_METAL_FAILURE = os.environ.get("MLX_PRIMITIVES_RAISE_ON_METAL_FAILURE", "0") == "1"

# Minimum sequence length to use Metal kernels (below this, overhead dominates)
# This threshold was determined empirically - Metal kernel launch overhead
# makes fallback faster for very small sequences.
METAL_MIN_SEQ_LEN = 8


def get_logger() -> logging.Logger:
    """Get the MLX Primitives logger (thread-safe).

    The log level can be controlled via the MLX_PRIMITIVES_LOG_LEVEL
    environment variable. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Default is WARNING.

    Returns:
        The configured logger instance.
    """
    global _logger
    if _logger is None:
        with _logger_lock:
            # Double-check after acquiring lock
            if _logger is not None:
                return _logger

            _logger = logging.getLogger("mlx_primitives")
            level_name = os.environ.get("MLX_PRIMITIVES_LOG_LEVEL", "WARNING").upper()

            # Validate level
            if level_name not in _VALID_LEVELS:
                print(
                    f"Warning: Invalid MLX_PRIMITIVES_LOG_LEVEL='{level_name}'. "
                    f"Valid values: {', '.join(sorted(_VALID_LEVELS))}. "
                    f"Defaulting to WARNING.",
                    file=sys.stderr,
                )
                level_name = "WARNING"

            _logger.setLevel(getattr(logging, level_name))

            # Add handler if none exists
            if not _logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                )
                _logger.addHandler(handler)

    return _logger


def log_fallback(
    operation: str,
    exception: Exception,
    context: Optional[str] = None,
) -> None:
    """Log a fallback from optimized to reference implementation.

    First fallback per operation logs at WARNING level for visibility.
    Subsequent fallbacks log at DEBUG to reduce noise.

    Args:
        operation: Name of the operation that fell back.
        exception: The exception that caused the fallback.
        context: Optional context (e.g., input shapes) for debugging.
    """
    logger = get_logger()

    # Thread-safe check-and-add for first-time fallback tracking
    with _logger_lock:
        if operation not in _fallback_seen:
            _fallback_seen.add(operation)
            level = logging.WARNING
            first_time_note = " (first occurrence, further fallbacks logged at DEBUG)"
        else:
            level = logging.DEBUG
            first_time_note = ""

    msg = (
        f"{operation}: optimized kernel failed "
        f"({type(exception).__name__}: {exception}){first_time_note}"
    )
    if context:
        msg += f" | context: {context}"
    msg += ", using fallback"

    logger.log(level, msg)


def has_metal_kernels(force_recheck: bool = False) -> bool:
    """Check if MLX Metal kernel API is available.

    This is a centralized check for Metal kernel availability. The result
    is cached for performance, but can be force-rechecked if needed.

    Args:
        force_recheck: If True, bypass cache and recheck availability.
            Useful after dynamic MLX configuration changes.

    Returns:
        True if mx.fast.metal_kernel is available, False otherwise.
    """
    global _metal_available

    if _metal_available is None or force_recheck:
        try:
            import mlx.core as mx
            _metal_available = hasattr(mx.fast, "metal_kernel")
        except ImportError:
            _metal_available = False

    return _metal_available


def should_use_metal(seq_len: int, use_metal: bool = True) -> bool:
    """Check if Metal kernel should be used for the given operation.

    Combines user preference, Metal availability, and minimum sequence
    length threshold to determine if Metal kernel should be used.

    Args:
        seq_len: Sequence length of the operation.
        use_metal: User preference for Metal usage.

    Returns:
        True if Metal kernel should be used, False for fallback.
    """
    return use_metal and has_metal_kernels() and seq_len >= METAL_MIN_SEQ_LEN


# Valid dtypes for Metal kernel float32 computation
_FLOAT32_COMPATIBLE_DTYPES = {"float32", "float16", "bfloat16"}


def validate_dtype_for_metal(
    array_name: str,
    dtype,
    target_dtype: str = "float32",
    warn_on_precision_loss: bool = True,
) -> None:
    """Validate that a tensor's dtype is compatible with Metal kernel computation.

    Metal kernels typically compute in float32. This function warns when:
    - float64 inputs will lose precision when cast to float32
    - Integer inputs may produce unexpected results

    Args:
        array_name: Name of the array (for warning messages).
        dtype: The dtype of the input array (mlx dtype object).
        target_dtype: The dtype the kernel will use internally.
        warn_on_precision_loss: Whether to warn on precision loss.

    Raises:
        TypeError: If dtype is fundamentally incompatible.
    """
    dtype_str = str(dtype).replace("mlx.core.", "")

    if target_dtype == "float32":
        if dtype_str == "float64":
            if warn_on_precision_loss:
                warnings.warn(
                    f"Input '{array_name}' has dtype float64 which will be cast to float32, "
                    f"potentially losing precision. Consider converting to float32 beforehand.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        elif dtype_str not in _FLOAT32_COMPATIBLE_DTYPES and "int" not in dtype_str and "uint" not in dtype_str:
            warnings.warn(
                f"Input '{array_name}' has dtype {dtype_str} which may produce unexpected "
                f"results when cast to float32.",
                RuntimeWarning,
                stacklevel=3,
            )
