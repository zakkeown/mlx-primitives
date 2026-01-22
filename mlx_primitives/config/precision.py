"""Precision configuration for MLX Primitives.

Provides automatic precision selection for attention computations,
allowing transparent use of float16 kernels when safe for ~2x memory bandwidth improvement.
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


class PrecisionMode(Enum):
    """Precision mode for attention computations."""

    AUTO = "auto"  # Automatic selection based on heuristics
    FLOAT32 = "float32"  # Force full precision
    FLOAT16 = "float16"  # Force half precision
    MIXED = "mixed"  # fp16 compute, fp32 accumulation (default for half kernels)


@dataclass
class PrecisionConfig:
    """Configuration for automatic precision selection.

    Attributes:
        mode: Precision mode (AUTO, FLOAT32, FLOAT16, MIXED).
        max_seq_len_fp16: Maximum sequence length for safe fp16 (longer sequences
            accumulate more error). Default: 8192.
        min_seq_len_fp16: Minimum sequence length where fp16 overhead is worth it.
            Default: 64.
        check_input_range: Whether to verify input values are in fp16 safe range.
        max_safe_magnitude: Maximum fp16 representable value (~65504).
        max_safe_attention_score: Maximum pre-softmax score safe for fp16 exp()
            (exp(11) ~ 60000, near fp16 max).
        accumulate_fp32: Always accumulate in fp32 (recommended, already done in
            Metal kernels).
        fallback_on_overflow: Fall back to fp32 on detected overflow risk.
        warn_on_fallback: Emit warning when falling back to fp32.

    Example:
        >>> from mlx_primitives.config import PrecisionConfig, PrecisionMode
        >>> config = PrecisionConfig(mode=PrecisionMode.AUTO)
        >>> set_precision_config(config)
    """

    mode: PrecisionMode = PrecisionMode.AUTO
    max_seq_len_fp16: int = 8192
    min_seq_len_fp16: int = 64
    check_input_range: bool = True
    max_safe_magnitude: float = 65000.0  # Slightly below fp16 max for safety
    max_safe_attention_score: float = 11.0  # exp(11) ~ 60000
    accumulate_fp32: bool = True
    fallback_on_overflow: bool = True
    warn_on_fallback: bool = True


# Thread-local storage for precision configuration
# This ensures concurrent inference requests don't clobber each other's settings
_thread_local = threading.local()

# Global default configuration (used when thread-local is not set)
_global_default_config = PrecisionConfig()

# Lock for thread-safe access to global config
_global_config_lock = threading.Lock()


def get_precision_config() -> PrecisionConfig:
    """Get the current precision configuration for this thread.

    Returns thread-local config if set, otherwise global default.
    """
    return getattr(_thread_local, "config", _global_default_config)


def set_precision_config(config: PrecisionConfig) -> None:
    """Set the precision configuration for this thread.

    Note: This sets thread-local config. For global default, use
    set_global_precision_config().

    Args:
        config: New precision configuration to use.

    Example:
        >>> config = PrecisionConfig(mode=PrecisionMode.FLOAT16)
        >>> set_precision_config(config)
    """
    _thread_local.config = config


def set_global_precision_config(config: PrecisionConfig) -> None:
    """Set the global default precision configuration.

    This affects all threads that haven't set thread-local config.
    Thread-safe: protected by _global_config_lock.

    Args:
        config: New global default configuration.
    """
    global _global_default_config
    with _global_config_lock:
        _global_default_config = config


def clear_thread_precision_config() -> None:
    """Clear thread-local precision config, reverting to global default."""
    if hasattr(_thread_local, "config"):
        del _thread_local.config


def set_precision_mode(mode: PrecisionMode) -> None:
    """Convenience function to set just the precision mode.

    Thread-safe: protected by _global_config_lock.

    Args:
        mode: Precision mode to use.

    Example:
        >>> set_precision_mode(PrecisionMode.AUTO)
    """
    global _global_default_config
    with _global_config_lock:
        _global_default_config = replace(_global_default_config, mode=mode)


@contextmanager
def precision_context(
    mode: Optional[PrecisionMode] = None,
    **kwargs,
):
    """Context manager for temporarily changing precision settings.

    Args:
        mode: Precision mode to use within context.
        **kwargs: Additional PrecisionConfig parameters to override.

    Example:
        >>> with precision_context(mode=PrecisionMode.FLOAT16):
        ...     output = attention(q, k, v)

        >>> with precision_context(mode=PrecisionMode.AUTO, max_seq_len_fp16=4096):
        ...     output = attention(q, k, v)
    """
    old_config = get_precision_config()
    try:
        # Build new config from old config with overrides
        new_kwargs = {
            "mode": mode if mode is not None else old_config.mode,
            "max_seq_len_fp16": kwargs.get(
                "max_seq_len_fp16", old_config.max_seq_len_fp16
            ),
            "min_seq_len_fp16": kwargs.get(
                "min_seq_len_fp16", old_config.min_seq_len_fp16
            ),
            "check_input_range": kwargs.get(
                "check_input_range", old_config.check_input_range
            ),
            "max_safe_magnitude": kwargs.get(
                "max_safe_magnitude", old_config.max_safe_magnitude
            ),
            "max_safe_attention_score": kwargs.get(
                "max_safe_attention_score", old_config.max_safe_attention_score
            ),
            "accumulate_fp32": kwargs.get("accumulate_fp32", old_config.accumulate_fp32),
            "fallback_on_overflow": kwargs.get(
                "fallback_on_overflow", old_config.fallback_on_overflow
            ),
            "warn_on_fallback": kwargs.get(
                "warn_on_fallback", old_config.warn_on_fallback
            ),
        }
        new_config = PrecisionConfig(**new_kwargs)
        set_precision_config(new_config)
        yield new_config
    finally:
        set_precision_config(old_config)


def is_attention_safe_for_fp16(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    seq_len: int,
    scale: float,
    config: Optional[PrecisionConfig] = None,
) -> Tuple[bool, str]:
    """Determine if attention computation is safe for fp16.

    Checks:
    1. Sequence length bounds
    2. Input tensor magnitude (overflow risk)
    3. Expected attention score magnitude (exp overflow risk)

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        seq_len: Maximum sequence length involved.
        scale: Attention scale factor (1/sqrt(head_dim)).
        config: Precision config to use (default: global config).

    Returns:
        Tuple of (is_safe, reason_string).
    """
    import mlx.core as mx

    if config is None:
        config = get_precision_config()

    # Check sequence length bounds
    if seq_len > config.max_seq_len_fp16:
        return (
            False,
            f"Sequence length {seq_len} exceeds max safe length {config.max_seq_len_fp16}",
        )

    if seq_len < config.min_seq_len_fp16:
        return (
            False,
            f"Sequence length {seq_len} too short for fp16 overhead benefit",
        )

    # Check input magnitude
    if config.check_input_range:
        q_max = float(mx.max(mx.abs(q)))
        k_max = float(mx.max(mx.abs(k)))
        v_max = float(mx.max(mx.abs(v)))

        max_magnitude = max(q_max, k_max, v_max)
        if max_magnitude > config.max_safe_magnitude:
            return (
                False,
                f"Input magnitude {max_magnitude:.2e} exceeds fp16 safe range {config.max_safe_magnitude:.2e}",
            )

        # Check for inf/nan in inputs
        if bool(mx.any(mx.isinf(q))) or bool(mx.any(mx.isnan(q))):
            return False, "Q contains inf or nan values"
        if bool(mx.any(mx.isinf(k))) or bool(mx.any(mx.isnan(k))):
            return False, "K contains inf or nan values"
        if bool(mx.any(mx.isinf(v))) or bool(mx.any(mx.isnan(v))):
            return False, "V contains inf or nan values"

        # Estimate attention score magnitude
        # Rough estimate: max(Q) * max(K) * head_dim * scale
        # In practice, correlated Q/K can produce larger scores
        head_dim = q.shape[-1]
        estimated_max_score = q_max * k_max * head_dim * scale

        if estimated_max_score > config.max_safe_attention_score:
            return (
                False,
                f"Estimated attention score {estimated_max_score:.2f} may overflow in fp16 exp()",
            )

    return True, "Attention is safe for fp16"


def should_use_fp16(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    seq_len: int,
    scale: float,
    precision: Optional[PrecisionMode] = None,
    config: Optional[PrecisionConfig] = None,
) -> bool:
    """Determine if fp16 should be used for attention.

    This is the main decision function called by attention implementations.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        seq_len: Maximum sequence length involved.
        scale: Attention scale factor.
        precision: Override precision mode (default: from config).
        config: Precision config to use (default: global config).

    Returns:
        True if fp16 should be used.
    """
    import mlx.core as mx

    if config is None:
        config = get_precision_config()

    mode = precision if precision is not None else config.mode

    # Explicit mode selection
    if mode == PrecisionMode.FLOAT32:
        return False
    elif mode in (PrecisionMode.FLOAT16, PrecisionMode.MIXED):
        return True
    elif mode == PrecisionMode.AUTO:
        # If inputs are already fp16, use fp16 path
        if q.dtype == mx.float16:
            return True

        # Run safety checks
        safe, reason = is_attention_safe_for_fp16(q, k, v, seq_len, scale, config)

        if not safe:
            if config.warn_on_fallback and config.fallback_on_overflow:
                warnings.warn(f"Auto-precision falling back to fp32: {reason}")
            return False

        return True

    return False
