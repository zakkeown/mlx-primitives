"""PyTorch reference implementations for cross-validation.

These are conditionally imported when torch is available.
Use these to validate MLX implementations against battle-tested PyTorch ops.
"""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def pytorch_available() -> bool:
    """Check if PyTorch is available for cross-validation."""
    return HAS_TORCH


def torch_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """PyTorch softmax for validation.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax output as numpy array.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32))
        result = F.softmax(t, dim=axis)
        return result.numpy()


def torch_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: Optional[float] = None,
    causal: bool = False,
) -> np.ndarray:
    """PyTorch scaled_dot_product_attention for validation.

    Args:
        q: Query tensor (batch, seq, heads, head_dim).
        k: Key tensor (batch, seq, heads, head_dim).
        v: Value tensor (batch, seq, heads, head_dim).
        scale: Optional scale factor. Defaults to 1/sqrt(head_dim).
        causal: Whether to apply causal masking.

    Returns:
        Attention output as numpy array (batch, seq, heads, head_dim).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    batch, seq, heads, dim = q.shape

    with torch.no_grad():
        # Convert to torch and transpose to (batch, heads, seq, dim)
        q_t = torch.from_numpy(q.astype(np.float32)).transpose(1, 2)
        k_t = torch.from_numpy(k.astype(np.float32)).transpose(1, 2)
        v_t = torch.from_numpy(v.astype(np.float32)).transpose(1, 2)

        if scale is None:
            scale = 1.0 / (dim ** 0.5)

        result = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            scale=scale,
            is_causal=causal,
        )

        # Transpose back to (batch, seq, heads, dim)
        return result.transpose(1, 2).numpy()


def torch_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """PyTorch RMSNorm for validation.

    Args:
        x: Input tensor.
        weight: Weight tensor for scaling.
        eps: Small value for numerical stability.

    Returns:
        Normalized output as numpy array.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32))
        w = torch.from_numpy(weight.astype(np.float32))

        variance = t.pow(2).mean(-1, keepdim=True)
        normalized = t * torch.rsqrt(variance + eps)
        result = normalized * w

        return result.numpy()


def torch_gelu(x: np.ndarray, approximate: str = "none") -> np.ndarray:
    """PyTorch GELU for validation.

    Args:
        x: Input tensor.
        approximate: Approximation method ("none" for exact, "tanh" for tanh approx).

    Returns:
        GELU output as numpy array.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32))
        result = F.gelu(t, approximate=approximate)
        return result.numpy()


def torch_silu(x: np.ndarray) -> np.ndarray:
    """PyTorch SiLU (Swish) for validation.

    Args:
        x: Input tensor.

    Returns:
        SiLU output as numpy array.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32))
        result = F.silu(t)
        return result.numpy()


def torch_cumsum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """PyTorch cumsum for validation.

    Args:
        x: Input tensor.
        axis: Axis along which to compute cumsum.

    Returns:
        Cumulative sum as numpy array.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not available")

    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32))
        result = torch.cumsum(t, dim=axis)
        return result.numpy()
