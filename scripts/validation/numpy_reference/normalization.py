"""NumPy reference implementations for normalization layers."""

import numpy as np


def rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor, shape (..., dims)
        weight: Scale parameter, shape (dims,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor, same shape as x
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def groupnorm(
    x: np.ndarray,
    num_groups: int,
    weight: np.ndarray = None,
    bias: np.ndarray = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Group Normalization.

    Normalizes over groups of channels.

    Args:
        x: Input tensor, shape (N, C, ...) where C is channels
        num_groups: Number of groups to divide channels into
        weight: Optional scale parameter, shape (C,)
        bias: Optional bias parameter, shape (C,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor, same shape as x
    """
    n, c = x.shape[:2]
    spatial = x.shape[2:]

    # Reshape to (N, num_groups, C // num_groups, ...)
    x_reshaped = x.reshape(n, num_groups, c // num_groups, *spatial)

    # Compute mean and variance over (C // num_groups, spatial dims)
    axes = tuple(range(2, x_reshaped.ndim))
    mean = np.mean(x_reshaped, axis=axes, keepdims=True)
    var = np.var(x_reshaped, axis=axes, keepdims=True)

    # Normalize
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)

    # Reshape back
    x_norm = x_norm.reshape(x.shape)

    # Apply scale and bias
    if weight is not None:
        shape = [1, c] + [1] * len(spatial)
        x_norm = x_norm * weight.reshape(shape)
    if bias is not None:
        shape = [1, c] + [1] * len(spatial)
        x_norm = x_norm + bias.reshape(shape)

    return x_norm


def instancenorm(
    x: np.ndarray,
    weight: np.ndarray = None,
    bias: np.ndarray = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Instance Normalization.

    Normalizes over spatial dimensions per instance per channel.

    Args:
        x: Input tensor, shape (N, C, ...) where ... is spatial dims
        weight: Optional scale parameter, shape (C,)
        bias: Optional bias parameter, shape (C,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor, same shape as x
    """
    n, c = x.shape[:2]
    spatial = x.shape[2:]

    # Compute mean and variance over spatial dimensions
    axes = tuple(range(2, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Apply scale and bias
    if weight is not None:
        shape = [1, c] + [1] * len(spatial)
        x_norm = x_norm * weight.reshape(shape)
    if bias is not None:
        shape = [1, c] + [1] * len(spatial)
        x_norm = x_norm + bias.reshape(shape)

    return x_norm


def ada_layernorm(
    x: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Adaptive Layer Normalization.

    AdaLN(x) = LayerNorm(x) * (1 + scale) + shift

    Args:
        x: Input tensor, shape (..., dims)
        scale: Scale modulation, shape (..., dims) or (dims,)
        shift: Shift modulation, shape (..., dims) or (dims,)
        eps: Small constant for numerical stability

    Returns:
        Modulated tensor, same shape as x
    """
    # Layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Adaptive modulation
    return x_norm * (1 + scale) + shift


def qknorm(
    q: np.ndarray,
    k: np.ndarray,
    eps: float = 1e-6,
) -> tuple:
    """Query-Key Normalization.

    Normalizes queries and keys independently for stable attention.

    Args:
        q: Query tensor, shape (..., head_dim)
        k: Key tensor, shape (..., head_dim)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normalized_q, normalized_k)
    """
    # L2 normalize along last dimension
    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)
    k_norm = k / (np.linalg.norm(k, axis=-1, keepdims=True) + eps)

    return q_norm, k_norm
