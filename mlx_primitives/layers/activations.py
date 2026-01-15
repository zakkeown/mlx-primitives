"""Activation functions for MLX.

This module provides activation functions not included in mlx.nn:
- SwiGLU: Swish-Gated Linear Unit (Llama/PaLM style)
- GeGLU: GELU-Gated Linear Unit
- ReGLU: ReLU-Gated Linear Unit
- Mish: Self-regularized non-monotonic activation
- GELU_tanh: Tanh approximation of GELU (faster)
- SquaredReLU: ReLU² (Primer paper)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit.

    Computes: SwiGLU(x) = (x @ W1) * SiLU(x @ W_gate) @ W2

    SwiGLU combines a linear projection with a swish-gated pathway,
    commonly used in modern LLMs like Llama, PaLM, and Mistral.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (before gating).
        out_features: Output dimension. If None, equals in_features.
        bias: Whether to use bias in linear layers.

    Reference:
        "GLU Variants Improve Transformer"
        https://arxiv.org/abs/2002.05202

    Example:
        >>> swiglu = SwiGLU(in_features=768, hidden_features=2048)
        >>> x = mx.random.normal((2, 16, 768))
        >>> y = swiglu(x)  # (2, 16, 768)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w_gate(x)) * self.w1(x))


class GeGLU(nn.Module):
    """GELU-Gated Linear Unit.

    Computes: GeGLU(x) = (x @ W1) * GELU(x @ W_gate) @ W2

    Like SwiGLU but uses GELU instead of SiLU for gating.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension. If None, equals in_features.
        bias: Whether to use bias in linear layers.

    Example:
        >>> geglu = GeGLU(in_features=768, hidden_features=2048)
        >>> x = mx.random.normal((2, 16, 768))
        >>> y = geglu(x)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.gelu(self.w_gate(x)) * self.w1(x))


class ReGLU(nn.Module):
    """ReLU-Gated Linear Unit.

    Computes: ReGLU(x) = (x @ W1) * ReLU(x @ W_gate) @ W2

    The simplest GLU variant, using ReLU for gating.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension. If None, equals in_features.
        bias: Whether to use bias in linear layers.

    Example:
        >>> reglu = ReGLU(in_features=768, hidden_features=2048)
        >>> x = mx.random.normal((2, 16, 768))
        >>> y = reglu(x)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.relu(self.w_gate(x)) * self.w1(x))


class FusedSwiGLU(nn.Module):
    """Memory-efficient SwiGLU with fused gate projection.

    Instead of separate W1 and W_gate, uses a single projection
    that outputs both, then splits. More memory efficient.

    Computes: SwiGLU(x) = chunk(x @ W_fused)[0] * SiLU(chunk(x @ W_fused)[1]) @ W2

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension. If None, equals in_features.
        bias: Whether to use bias in linear layers.

    Example:
        >>> swiglu = FusedSwiGLU(in_features=768, hidden_features=2048)
        >>> x = mx.random.normal((2, 16, 768))
        >>> y = swiglu(x)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features

        # Fused projection for both W1 and W_gate
        self.w_fused = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.w_fused(x)
        x1, gate = mx.split(fused, 2, axis=-1)
        return self.w2(nn.silu(gate) * x1)


def mish(x: mx.array) -> mx.array:
    """Mish activation function.

    Computes: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    A self-regularized non-monotonic activation function that
    tends to work well in deep networks.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.

    Reference:
        "Mish: A Self Regularized Non-Monotonic Activation Function"
        https://arxiv.org/abs/1908.08681
    """
    return x * mx.tanh(nn.softplus(x))


class Mish(nn.Module):
    """Mish activation as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return mish(x)


def gelu_tanh(x: mx.array) -> mx.array:
    """GELU with tanh approximation (faster than exact GELU).

    Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the approximation used by GPT-2 and many other models.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class GELUTanh(nn.Module):
    """GELU with tanh approximation as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return gelu_tanh(x)


def squared_relu(x: mx.array) -> mx.array:
    """Squared ReLU activation.

    Computes: ReLU(x)^2

    A simple but effective activation that provides smoother gradients
    than standard ReLU.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.

    Reference:
        "Primer: Searching for Efficient Transformers for Language Modeling"
        https://arxiv.org/abs/2109.08668
    """
    return mx.square(nn.relu(x))


class SquaredReLU(nn.Module):
    """Squared ReLU activation as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return squared_relu(x)


def quick_gelu(x: mx.array) -> mx.array:
    """Quick GELU approximation.

    Computes: x * sigmoid(1.702 * x)

    A faster approximation of GELU used in some models.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return x * mx.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    """Quick GELU as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return quick_gelu(x)


def swish(x: mx.array, beta: float = 1.0) -> mx.array:
    """Swish activation with configurable beta.

    Computes: x * sigmoid(beta * x)

    When beta=1, this is equivalent to SiLU.

    Args:
        x: Input tensor.
        beta: Scaling factor (default: 1.0).

    Returns:
        Activated tensor.
    """
    return x * mx.sigmoid(beta * x)


class Swish(nn.Module):
    """Swish activation with learnable or fixed beta."""

    def __init__(self, beta: float = 1.0, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.beta = mx.array(beta)
        else:
            self._beta = beta

    def __call__(self, x: mx.array) -> mx.array:
        beta = self.beta if self.learnable else self._beta
        return x * mx.sigmoid(beta * x)


def hard_swish(x: mx.array) -> mx.array:
    """Hard Swish activation.

    A computationally efficient approximation of Swish.

    Computes: x * ReLU6(x + 3) / 6

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return x * mx.clip(x + 3, 0, 6) / 6


class HardSwish(nn.Module):
    """Hard Swish activation as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return hard_swish(x)


def hard_sigmoid(x: mx.array) -> mx.array:
    """Hard Sigmoid activation.

    A computationally efficient approximation of sigmoid.

    Computes: clip(x + 3, 0, 6) / 6

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return mx.clip(x + 3, 0, 6) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation as a module."""

    def __call__(self, x: mx.array) -> mx.array:
        return hard_sigmoid(x)


# Activation registry for convenience
ACTIVATIONS = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_tanh": gelu_tanh,
    "silu": nn.silu,
    "swish": swish,
    "mish": mish,
    "squared_relu": squared_relu,
    "quick_gelu": quick_gelu,
    "hard_swish": hard_swish,
    "hard_sigmoid": hard_sigmoid,
    "tanh": mx.tanh,
    "sigmoid": mx.sigmoid,
}


def get_activation(name: str):
    """Get activation function by name.

    Args:
        name: Activation name (e.g., 'relu', 'gelu', 'swiglu').

    Returns:
        Activation function.

    Raises:
        ValueError: If activation name is not recognized.
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
