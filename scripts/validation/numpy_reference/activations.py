"""NumPy reference implementations for activation functions."""

import numpy as np
from scipy import special


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1 + special.erf(x / np.sqrt(2)))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU with tanh approximation.

    GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def quick_gelu(x: np.ndarray) -> np.ndarray:
    """QuickGELU activation (used in OpenAI CLIP).

    QuickGELU(x) = x * sigmoid(1.702 * x)
    """
    return x * sigmoid(1.702 * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation.

    SiLU(x) = x * sigmoid(x)
    """
    return x * sigmoid(x)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation.

    SwiGLU(gate, up) = SiLU(gate) * up
    """
    return silu(gate) * up


def geglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """GeGLU activation.

    GeGLU(gate, up) = GELU(gate) * up
    """
    return gelu(gate) * up


def reglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """ReGLU activation.

    ReGLU(gate, up) = ReLU(gate) * up
    """
    return np.maximum(0, gate) * up


def mish(x: np.ndarray) -> np.ndarray:
    """Mish activation.

    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    softplus = np.log1p(np.exp(np.clip(x, -20, 20)))  # Clip for numerical stability
    return x * np.tanh(softplus)


def squared_relu(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation.

    SquaredReLU(x) = ReLU(x)^2
    """
    return np.maximum(0, x) ** 2


def hard_swish(x: np.ndarray) -> np.ndarray:
    """Hard Swish activation.

    HardSwish(x) = x * clip(x + 3, 0, 6) / 6
    """
    return x * np.clip(x + 3, 0, 6) / 6


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """Hard Sigmoid activation.

    HardSigmoid(x) = clip(x + 3, 0, 6) / 6
    """
    return np.clip(x + 3, 0, 6) / 6
