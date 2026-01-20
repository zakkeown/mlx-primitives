"""JAX reference implementations for cross-validation.

These are conditionally imported when JAX is available.
Use these to validate MLX implementations against JAX's functional API,
which is closer in paradigm to MLX than PyTorch.
"""

from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def jax_available() -> bool:
    """Check if JAX is available for cross-validation."""
    return HAS_JAX


def jax_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """JAX softmax for validation.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax output as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    result = jax.nn.softmax(jnp.array(x, dtype=jnp.float32), axis=axis)
    return np.array(result)


def jax_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: Optional[float] = None,
    causal: bool = False,
) -> np.ndarray:
    """JAX scaled dot-product attention for validation.

    Args:
        q: Query tensor (batch, seq_q, heads, head_dim).
        k: Key tensor (batch, seq_kv, heads, head_dim).
        v: Value tensor (batch, seq_kv, heads, head_dim).
        scale: Optional scale factor. Defaults to 1/sqrt(head_dim).
        causal: Whether to apply causal masking.

    Returns:
        Attention output as numpy array (batch, seq_q, heads, head_dim).
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    q_j = jnp.array(q, dtype=jnp.float32)
    k_j = jnp.array(k, dtype=jnp.float32)
    v_j = jnp.array(v, dtype=jnp.float32)

    batch, seq_q, heads, dim = q_j.shape
    seq_kv = k_j.shape[1]

    if scale is None:
        scale = 1.0 / jnp.sqrt(dim).astype(jnp.float32)

    # Compute attention scores: (batch, heads, seq_q, seq_kv)
    # First transpose to (batch, heads, seq, dim)
    q_t = jnp.transpose(q_j, (0, 2, 1, 3))
    k_t = jnp.transpose(k_j, (0, 2, 1, 3))
    v_t = jnp.transpose(v_j, (0, 2, 1, 3))

    scores = jnp.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

    if causal:
        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_q, seq_kv), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_t)

    # Transpose back to (batch, seq, heads, dim)
    result = jnp.transpose(output, (0, 2, 1, 3))
    return np.array(result)


def jax_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """JAX RMSNorm for validation.

    Args:
        x: Input tensor.
        weight: Weight tensor for scaling.
        eps: Small value for numerical stability.

    Returns:
        Normalized output as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)
    w_j = jnp.array(weight, dtype=jnp.float32)

    variance = jnp.mean(x_j ** 2, axis=-1, keepdims=True)
    normalized = x_j * jax.lax.rsqrt(variance + eps)
    result = normalized * w_j

    return np.array(result)


def jax_gelu(x: np.ndarray, approximate: bool = False) -> np.ndarray:
    """JAX GELU for validation.

    Args:
        x: Input tensor.
        approximate: Whether to use the tanh approximation.

    Returns:
        GELU output as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)
    result = jax.nn.gelu(x_j, approximate=approximate)
    return np.array(result)


def jax_silu(x: np.ndarray) -> np.ndarray:
    """JAX SiLU (Swish) for validation.

    Args:
        x: Input tensor.

    Returns:
        SiLU output as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)
    result = jax.nn.silu(x_j)
    return np.array(result)


def jax_cumsum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """JAX cumsum for validation.

    Args:
        x: Input tensor.
        axis: Axis along which to compute cumsum.

    Returns:
        Cumulative sum as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)
    result = jnp.cumsum(x_j, axis=axis)
    return np.array(result)


def jax_associative_scan(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """JAX associative scan (cumsum) using lax.associative_scan.

    This validates that MLX's associative_scan produces the same result
    as JAX's lax.associative_scan, which is the canonical functional implementation.

    Args:
        x: Input tensor.
        axis: Axis along which to scan.

    Returns:
        Scanned output as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)

    # Use lax.associative_scan with addition
    # This is the canonical functional parallel scan implementation
    result = lax.associative_scan(jnp.add, x_j, axis=axis)
    return np.array(result)


def jax_ssm_scan(
    A: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """JAX SSM scan using lax.associative_scan.

    Computes h[t] = A[t] * h[t-1] + x[t] using parallel associative scan.

    The SSM recurrence is associative with the binary operation:
    (A1, h1) ⊕ (A2, h2) = (A1*A2, A2*h1 + h2)

    Args:
        A: Decay coefficients (batch, seq, state).
        x: Input values (batch, seq, state).

    Returns:
        Hidden states h of shape (batch, seq, state).
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    A_j = jnp.array(A, dtype=jnp.float32)
    x_j = jnp.array(x, dtype=jnp.float32)

    # Define the SSM associative binary operation
    def ssm_combine(left, right):
        A_left, h_left = left
        A_right, h_right = right
        return (A_left * A_right, A_right * h_left + h_right)

    # Stack (A, x) as the initial elements
    # Each element is a tuple (A[t], x[t])
    init_elements = (A_j, x_j)

    # Apply associative scan along sequence axis (axis=1)
    A_cumulative, h_result = lax.associative_scan(
        ssm_combine, init_elements, axis=1
    )

    return np.array(h_result)


def jax_cumprod(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """JAX cumulative product for validation.

    Args:
        x: Input tensor.
        axis: Axis along which to compute cumprod.

    Returns:
        Cumulative product as numpy array.
    """
    if not HAS_JAX:
        raise ImportError("JAX not available")

    x_j = jnp.array(x, dtype=jnp.float32)
    result = jnp.cumprod(x_j, axis=axis)
    return np.array(result)
