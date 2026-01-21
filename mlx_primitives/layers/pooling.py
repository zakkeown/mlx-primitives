"""Pooling layers for MLX.

This module provides pooling layers not included in mlx.nn:
- AdaptiveAvgPool1d/2d: Output size agnostic average pooling
- AdaptiveMaxPool1d/2d: Output size agnostic max pooling
- GlobalAttentionPooling: Learned attention-weighted pooling
- GeM: Generalized Mean Pooling (for image retrieval)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class AdaptiveAvgPool1d(nn.Module):
    """Adaptive 1D Average Pooling.

    Pools input to a specified output size regardless of input size.

    Args:
        output_size: Target output length.

    Example:
        >>> pool = AdaptiveAvgPool1d(output_size=8)
        >>> x = mx.random.normal((2, 64, 100))  # (batch, channels, length)
        >>> y = pool(x)  # (2, 64, 8)
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, length)
        input_size = x.shape[-1]

        if input_size == self.output_size:
            return x

        # Use fp32 accumulation for numerical stability (matches PyTorch behavior)
        orig_dtype = x.dtype
        if x.dtype in (mx.float16, mx.bfloat16):
            x = x.astype(mx.float32)

        # PyTorch adaptive pooling algorithm:
        # start[i] = floor(i * input_size / output_size)
        # end[i] = ceil((i + 1) * input_size / output_size)
        outputs = []
        for i in range(self.output_size):
            start = (i * input_size) // self.output_size
            end = ((i + 1) * input_size + self.output_size - 1) // self.output_size
            pooled = mx.mean(x[..., start:end], axis=-1, keepdims=True)
            outputs.append(pooled)

        result = mx.concatenate(outputs, axis=-1)
        return result.astype(orig_dtype) if orig_dtype != mx.float32 else result


class AdaptiveAvgPool2d(nn.Module):
    """Adaptive 2D Average Pooling.

    Pools input to a specified output size regardless of input size.

    Args:
        output_size: Target output size as (H, W) or single int for square.

    Example:
        >>> pool = AdaptiveAvgPool2d(output_size=(7, 7))
        >>> x = mx.random.normal((2, 256, 224, 224))  # NCHW
        >>> y = pool(x)  # (2, 256, 7, 7)
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, height, width)
        _, _, in_h, in_w = x.shape
        out_h, out_w = self.output_size

        if in_h == out_h and in_w == out_w:
            return x

        # Use fp32 accumulation for numerical stability (matches PyTorch behavior)
        orig_dtype = x.dtype
        if x.dtype in (mx.float16, mx.bfloat16):
            x = x.astype(mx.float32)

        # Global average pooling special case
        if out_h == 1 and out_w == 1:
            result = mx.mean(x, axis=(2, 3), keepdims=True)
            return result.astype(orig_dtype) if orig_dtype != mx.float32 else result

        # PyTorch adaptive pooling algorithm:
        # start[i] = floor(i * input_size / output_size)
        # end[i] = ceil((i + 1) * input_size / output_size)
        outputs = []
        for i in range(out_h):
            row_outputs = []
            start_h = (i * in_h) // out_h
            end_h = ((i + 1) * in_h + out_h - 1) // out_h
            for j in range(out_w):
                start_w = (j * in_w) // out_w
                end_w = ((j + 1) * in_w + out_w - 1) // out_w
                pooled = mx.mean(
                    x[:, :, start_h:end_h, start_w:end_w],
                    axis=(2, 3),
                    keepdims=True
                )
                row_outputs.append(pooled)
            outputs.append(mx.concatenate(row_outputs, axis=3))

        result = mx.concatenate(outputs, axis=2)
        return result.astype(orig_dtype) if orig_dtype != mx.float32 else result


class AdaptiveMaxPool1d(nn.Module):
    """Adaptive 1D Max Pooling.

    Pools input to a specified output size using max pooling.

    Args:
        output_size: Target output length.

    Example:
        >>> pool = AdaptiveMaxPool1d(output_size=8)
        >>> x = mx.random.normal((2, 64, 100))
        >>> y = pool(x)  # (2, 64, 8)
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        input_size = x.shape[-1]

        if input_size == self.output_size:
            return x

        # PyTorch adaptive pooling algorithm:
        # start[i] = floor(i * input_size / output_size)
        # end[i] = ceil((i + 1) * input_size / output_size)
        outputs = []
        for i in range(self.output_size):
            start = (i * input_size) // self.output_size
            end = ((i + 1) * input_size + self.output_size - 1) // self.output_size
            pooled = mx.max(x[..., start:end], axis=-1, keepdims=True)
            outputs.append(pooled)

        return mx.concatenate(outputs, axis=-1)


class AdaptiveMaxPool2d(nn.Module):
    """Adaptive 2D Max Pooling.

    Pools input to a specified output size using max pooling.

    Args:
        output_size: Target output size as (H, W) or single int for square.

    Example:
        >>> pool = AdaptiveMaxPool2d(output_size=(7, 7))
        >>> x = mx.random.normal((2, 256, 224, 224))
        >>> y = pool(x)  # (2, 256, 7, 7)
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        _, _, in_h, in_w = x.shape
        out_h, out_w = self.output_size

        if in_h == out_h and in_w == out_w:
            return x

        if out_h == 1 and out_w == 1:
            return mx.max(x, axis=(2, 3), keepdims=True)

        # PyTorch adaptive pooling algorithm:
        # start[i] = floor(i * input_size / output_size)
        # end[i] = ceil((i + 1) * input_size / output_size)
        outputs = []
        for i in range(out_h):
            row_outputs = []
            start_h = (i * in_h) // out_h
            end_h = ((i + 1) * in_h + out_h - 1) // out_h
            for j in range(out_w):
                start_w = (j * in_w) // out_w
                end_w = ((j + 1) * in_w + out_w - 1) // out_w
                pooled = mx.max(
                    x[:, :, start_h:end_h, start_w:end_w],
                    axis=(2, 3),
                    keepdims=True
                )
                row_outputs.append(pooled)
            outputs.append(mx.concatenate(row_outputs, axis=3))

        return mx.concatenate(outputs, axis=2)


class GlobalAttentionPooling(nn.Module):
    """Global Attention Pooling.

    Learns to weight different positions based on their importance,
    then computes a weighted average. Useful for variable-length sequences.

    Args:
        dims: Feature dimension.
        hidden_dims: Hidden dimension for attention MLP (default: dims // 4).

    Example:
        >>> pool = GlobalAttentionPooling(dims=768)
        >>> x = mx.random.normal((2, 100, 768))  # (batch, seq, dims)
        >>> y = pool(x)  # (2, 768)
    """

    def __init__(self, dims: int, hidden_dims: Optional[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or dims // 4

        self.attention = nn.Sequential(
            nn.Linear(dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, 1, bias=False),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # x: (batch, seq, dims)
        # Compute attention scores
        scores = self.attention(x)  # (batch, seq, 1)
        scores = scores.squeeze(-1)  # (batch, seq)

        # Apply mask if provided
        if mask is not None:
            scores = mx.where(mask, scores, float("-inf"))

        # Softmax to get weights
        weights = mx.softmax(scores, axis=-1)  # (batch, seq)

        # Weighted average
        return mx.sum(x * weights[:, :, None], axis=1)  # (batch, dims)


class GeM(nn.Module):
    """Generalized Mean Pooling.

    Computes the generalized mean (p-norm) over spatial dimensions.
    Commonly used in image retrieval where it often outperforms
    average and max pooling.

    Formula: GeM(x) = (mean(x^p))^(1/p)

    Args:
        p: Power parameter. p=1 is average pooling, p=inf is max pooling.
            Default is 3.0 which is commonly used for retrieval.
        eps: Small constant for numerical stability.
        learnable: If True, p is a learnable parameter.

    Reference:
        "Fine-tuning CNN Image Retrieval with No Human Annotation"
        https://arxiv.org/abs/1711.02512

    Example:
        >>> pool = GeM(p=3.0)
        >>> x = mx.random.normal((2, 512, 7, 7))  # NCHW
        >>> y = pool(x)  # (2, 512, 1, 1)
    """

    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learnable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.learnable = learnable

        if learnable:
            self.p = mx.array(p)
        else:
            self._p = p

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, height, width)
        p = self.p if self.learnable else self._p

        # Use fp32 accumulation for numerical stability (matches PyTorch behavior)
        orig_dtype = x.dtype
        if x.dtype in (mx.float16, mx.bfloat16):
            x = x.astype(mx.float32)

        # Clamp to avoid numerical issues
        x = mx.clip(x, a_min=self.eps, a_max=None)

        # Compute generalized mean
        x_pow = mx.power(x, p)
        mean_pow = mx.mean(x_pow, axis=(2, 3), keepdims=True)
        result = mx.power(mean_pow, 1.0 / p)

        return result.astype(orig_dtype) if orig_dtype != mx.float32 else result


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling.

    Pools input at multiple scales and concatenates the results,
    enabling fixed-size output regardless of input size.

    Args:
        output_sizes: List of output sizes for each pyramid level.

    Example:
        >>> spp = SpatialPyramidPooling(output_sizes=[1, 2, 4])
        >>> x = mx.random.normal((2, 256, 13, 13))
        >>> y = spp(x)  # (2, 256 * (1 + 4 + 16)) = (2, 5376)
    """

    def __init__(self, output_sizes: list[int] = [1, 2, 4]):
        super().__init__()
        self.output_sizes = output_sizes
        self.pools = [AdaptiveAvgPool2d(size) for size in output_sizes]

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, channels = x.shape[:2]

        pooled = []
        for pool in self.pools:
            p = pool(x)  # (batch, channels, size, size)
            p = p.reshape(batch_size, -1)  # Flatten spatial dims
            pooled.append(p)

        return mx.concatenate(pooled, axis=1)


class AvgPool1d(nn.Module):
    """1D Average Pooling.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Default: kernel_size.
        padding: Padding to apply. Default: 0.

    Example:
        >>> pool = AvgPool1d(kernel_size=3, stride=2)
        >>> x = mx.random.normal((2, 64, 100))
        >>> y = pool(x)
    """

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, length)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])

        batch, channels, length = x.shape

        # Calculate output length
        out_length = (length - self.kernel_size) // self.stride + 1

        outputs = []
        for i in range(out_length):
            start = i * self.stride
            end = start + self.kernel_size
            pooled = mx.mean(x[..., start:end], axis=-1, keepdims=True)
            outputs.append(pooled)

        return mx.concatenate(outputs, axis=-1)


class MaxPool1d(nn.Module):
    """1D Max Pooling.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Default: kernel_size.
        padding: Padding to apply. Default: 0.

    Example:
        >>> pool = MaxPool1d(kernel_size=3, stride=2)
        >>> x = mx.random.normal((2, 64, 100))
        >>> y = pool(x)
    """

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            x = mx.pad(
                x,
                [(0, 0), (0, 0), (self.padding, self.padding)],
                constant_values=float("-inf"),
            )

        batch, channels, length = x.shape
        out_length = (length - self.kernel_size) // self.stride + 1

        outputs = []
        for i in range(out_length):
            start = i * self.stride
            end = start + self.kernel_size
            pooled = mx.max(x[..., start:end], axis=-1, keepdims=True)
            outputs.append(pooled)

        return mx.concatenate(outputs, axis=-1)
