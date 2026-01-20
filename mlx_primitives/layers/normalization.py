"""Normalization layers for MLX.

This module provides normalization layers not included in mlx.nn:
- RMSNorm: Root Mean Square Layer Normalization (Llama-style)
- GroupNorm: Group Normalization
- InstanceNorm: Instance Normalization
- AdaLayerNorm: Adaptive Layer Normalization (for diffusion models)
- QKNorm: Query-Key Normalization (stabilizes attention)
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs using only the root mean square, without
    centering (no mean subtraction). This is computationally simpler and
    works well in practice, especially for large language models.

    Formula: y = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dims: Number of features to normalize.
        eps: Small constant for numerical stability.

    Reference:
        "Root Mean Square Layer Normalization"
        https://arxiv.org/abs/1910.07467

    Example:
        >>> norm = RMSNorm(dims=768)
        >>> x = mx.random.normal((2, 16, 768))
        >>> y = norm(x)  # (2, 16, 768)
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS along last dimension
        # rms = sqrt(mean(x^2) + eps)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class GroupNorm(nn.Module):
    """Group Normalization.

    Divides channels into groups and normalizes within each group.
    This provides a middle ground between LayerNorm and InstanceNorm.

    Args:
        num_groups: Number of groups to divide channels into.
        num_channels: Total number of channels.
        eps: Small constant for numerical stability.
        affine: If True, apply learnable scale and shift.

    Reference:
        "Group Normalization"
        https://arxiv.org/abs/1803.08494

    Example:
        >>> norm = GroupNorm(num_groups=32, num_channels=256)
        >>> x = mx.random.normal((2, 256, 16, 16))  # NCHW
        >>> y = norm(x)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (N, C, *) where * is any number of spatial dimensions
        # or (N, *, C) for channels-last
        original_shape = x.shape

        # Assume channels-first for now: (N, C, H, W) or (N, C, L)
        batch_size = x.shape[0]
        num_channels = x.shape[1]

        # Reshape to (N, G, C//G, *)
        group_size = num_channels // self.num_groups
        x = x.reshape(batch_size, self.num_groups, group_size, -1)

        # Normalize within each group
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.var(x, axis=(2, 3), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(original_shape)

        # Apply affine transform
        if self.affine:
            # Reshape weight/bias for broadcasting
            weight = self.weight.reshape(1, -1, *([1] * (len(original_shape) - 2)))
            bias = self.bias.reshape(1, -1, *([1] * (len(original_shape) - 2)))
            x = x * weight + bias

        return x


class InstanceNorm(nn.Module):
    """Instance Normalization.

    Normalizes each instance (sample) independently across spatial dimensions.
    Commonly used in style transfer and image generation.

    Args:
        num_features: Number of features/channels.
        eps: Small constant for numerical stability.
        affine: If True, apply learnable scale and shift.

    Reference:
        "Instance Normalization: The Missing Ingredient for Fast Stylization"
        https://arxiv.org/abs/1607.08022

    Example:
        >>> norm = InstanceNorm(num_features=256)
        >>> x = mx.random.normal((2, 256, 16, 16))  # NCHW
        >>> y = norm(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (N, C, *) - normalize over spatial dimensions
        # Keep N and C, normalize over rest
        original_shape = x.shape
        batch_size = x.shape[0]
        num_channels = x.shape[1]

        # Reshape to (N, C, -1) to flatten spatial dims
        x = x.reshape(batch_size, num_channels, -1)

        # Normalize over spatial dimension
        mean = mx.mean(x, axis=2, keepdims=True)
        var = mx.var(x, axis=2, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(original_shape)

        # Apply affine transform
        if self.affine:
            weight = self.weight.reshape(1, -1, *([1] * (len(original_shape) - 2)))
            bias = self.bias.reshape(1, -1, *([1] * (len(original_shape) - 2)))
            x = x * weight + bias

        return x


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization.

    Applies layer normalization with scale and shift parameters
    conditioned on an external input (e.g., timestep embedding).
    Commonly used in diffusion models.

    Args:
        dims: Number of features to normalize.
        cond_dims: Dimension of conditioning input.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = AdaLayerNorm(dims=768, cond_dims=256)
        >>> x = mx.random.normal((2, 16, 768))
        >>> cond = mx.random.normal((2, 256))  # e.g., timestep embedding
        >>> y = norm(x, cond)
    """

    def __init__(
        self,
        dims: int,
        cond_dims: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps

        # Project conditioning to scale and shift
        self.proj = nn.Linear(cond_dims, dims * 2)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        # x: (batch, seq, dims) or (batch, dims)
        # cond: (batch, cond_dims)

        # Standard layer normalization (without affine)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Get scale and shift from conditioning
        scale_shift = self.proj(cond)  # (batch, dims * 2)
        scale, shift = mx.split(scale_shift, 2, axis=-1)

        # Reshape for broadcasting
        if x.ndim == 3:
            scale = scale[:, None, :]  # (batch, 1, dims)
            shift = shift[:, None, :]

        # Apply adaptive normalization
        return x_norm * (1 + scale) + shift


class QKNorm(nn.Module):
    """Query-Key Normalization.

    Applies RMSNorm to queries and keys separately before attention.
    This helps stabilize attention scores and improves training dynamics.

    Args:
        head_dim: Dimension of each attention head.
        eps: Small constant for numerical stability.

    Reference:
        Used in various modern architectures like SDXL, Stable Diffusion 3.

    Example:
        >>> qk_norm = QKNorm(head_dim=64)
        >>> q = mx.random.normal((2, 16, 12, 64))  # (batch, seq, heads, dim)
        >>> k = mx.random.normal((2, 16, 12, 64))
        >>> q_norm, k_norm = qk_norm(q, k)
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps

        # Separate scale parameters for Q and K
        self.q_scale = mx.ones((head_dim,))
        self.k_scale = mx.ones((head_dim,))

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # RMSNorm on last dimension
        def rms_norm(x: mx.array, scale: mx.array) -> mx.array:
            rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
            return (x / rms) * scale

        return rms_norm(q, self.q_scale), rms_norm(k, self.k_scale)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Functional RMSNorm.

    Args:
        x: Input tensor.
        weight: Scale parameter.
        eps: Numerical stability constant.

    Returns:
        Normalized tensor.
    """
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def group_norm(
    x: mx.array,
    num_groups: int,
    weight: Optional[mx.array] = None,
    bias: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    """Functional GroupNorm.

    Args:
        x: Input tensor of shape (N, C, *).
        num_groups: Number of groups.
        weight: Optional scale parameter.
        bias: Optional shift parameter.
        eps: Numerical stability constant.

    Returns:
        Normalized tensor.
    """
    original_shape = x.shape
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    group_size = num_channels // num_groups

    x = x.reshape(batch_size, num_groups, group_size, -1)
    mean = mx.mean(x, axis=(2, 3), keepdims=True)
    var = mx.var(x, axis=(2, 3), keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    x = x.reshape(original_shape)

    if weight is not None:
        weight = weight.reshape(1, -1, *([1] * (len(original_shape) - 2)))
        x = x * weight
    if bias is not None:
        bias = bias.reshape(1, -1, *([1] * (len(original_shape) - 2)))
        x = x + bias

    return x
