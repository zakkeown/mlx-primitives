"""NumPy reference implementations for pooling layers."""

import numpy as np


def adaptive_avg_pool1d(x: np.ndarray, output_size: int) -> np.ndarray:
    """Adaptive Average Pooling 1D.

    Pools input to specified output size by averaging regions.

    Args:
        x: Input tensor, shape (batch, channels, length)
        output_size: Target output length

    Returns:
        Pooled tensor, shape (batch, channels, output_size)
    """
    batch, channels, length = x.shape

    out = np.zeros((batch, channels, output_size), dtype=x.dtype)

    for i in range(output_size):
        start = int(np.floor(i * length / output_size))
        end = int(np.ceil((i + 1) * length / output_size))
        out[:, :, i] = np.mean(x[:, :, start:end], axis=-1)

    return out


def adaptive_avg_pool2d(
    x: np.ndarray,
    output_size: tuple,
) -> np.ndarray:
    """Adaptive Average Pooling 2D.

    Pools input to specified output size by averaging regions.

    Args:
        x: Input tensor, shape (batch, channels, height, width)
        output_size: Target (height, width)

    Returns:
        Pooled tensor, shape (batch, channels, out_h, out_w)
    """
    batch, channels, height, width = x.shape
    out_h, out_w = output_size

    out = np.zeros((batch, channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        start_h = int(np.floor(i * height / out_h))
        end_h = int(np.ceil((i + 1) * height / out_h))

        for j in range(out_w):
            start_w = int(np.floor(j * width / out_w))
            end_w = int(np.ceil((j + 1) * width / out_w))

            out[:, :, i, j] = np.mean(x[:, :, start_h:end_h, start_w:end_w], axis=(-2, -1))

    return out


def adaptive_max_pool1d(x: np.ndarray, output_size: int) -> np.ndarray:
    """Adaptive Max Pooling 1D.

    Pools input to specified output size by taking max of regions.

    Args:
        x: Input tensor, shape (batch, channels, length)
        output_size: Target output length

    Returns:
        Pooled tensor, shape (batch, channels, output_size)
    """
    batch, channels, length = x.shape

    out = np.zeros((batch, channels, output_size), dtype=x.dtype)

    for i in range(output_size):
        start = int(np.floor(i * length / output_size))
        end = int(np.ceil((i + 1) * length / output_size))
        out[:, :, i] = np.max(x[:, :, start:end], axis=-1)

    return out


def adaptive_max_pool2d(
    x: np.ndarray,
    output_size: tuple,
) -> np.ndarray:
    """Adaptive Max Pooling 2D.

    Pools input to specified output size by taking max of regions.

    Args:
        x: Input tensor, shape (batch, channels, height, width)
        output_size: Target (height, width)

    Returns:
        Pooled tensor, shape (batch, channels, out_h, out_w)
    """
    batch, channels, height, width = x.shape
    out_h, out_w = output_size

    out = np.zeros((batch, channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        start_h = int(np.floor(i * height / out_h))
        end_h = int(np.ceil((i + 1) * height / out_h))

        for j in range(out_w):
            start_w = int(np.floor(j * width / out_w))
            end_w = int(np.ceil((j + 1) * width / out_w))

            out[:, :, i, j] = np.max(x[:, :, start_h:end_h, start_w:end_w], axis=(-2, -1))

    return out


def gem_pool(
    x: np.ndarray,
    p: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generalized Mean (GeM) Pooling.

    GeM(x) = (mean(x^p))^(1/p)

    When p=1, equivalent to average pooling.
    When p->inf, approaches max pooling.

    Args:
        x: Input tensor, shape (batch, channels, *spatial)
        p: Power parameter
        eps: Small constant for numerical stability

    Returns:
        Pooled tensor, shape (batch, channels)
    """
    # Clamp to avoid numerical issues with negative values
    x_clamped = np.maximum(x, eps)

    # Raise to power p and average over spatial dimensions
    x_pow = x_clamped ** p

    # Average over all spatial dimensions
    axes = tuple(range(2, x.ndim))
    mean_pow = np.mean(x_pow, axis=axes)

    # Take p-th root
    return mean_pow ** (1.0 / p)
