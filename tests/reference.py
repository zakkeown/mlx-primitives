"""NumPy reference implementations for validation.

These implementations prioritize correctness and clarity over performance.
They serve as ground truth for validating MLX implementations.
"""

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Activation Functions
# =============================================================================


def silu(x: NDArray) -> NDArray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu_approximate(x: NDArray) -> NDArray:
    """GELU with tanh approximation."""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def gelu_exact(x: NDArray) -> NDArray:
    """Exact GELU using erf."""
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


# =============================================================================
# Normalization
# =============================================================================


def rmsnorm(
    x: NDArray,
    weight: NDArray,
    eps: float = 1e-5,
) -> NDArray:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def layernorm(
    x: NDArray,
    weight: NDArray,
    bias: Optional[NDArray] = None,
    eps: float = 1e-5,
) -> NDArray:
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    out = norm * weight
    if bias is not None:
        out = out + bias
    return out


# =============================================================================
# Fused Operations
# =============================================================================


def fused_rmsnorm_linear(
    x: NDArray,
    norm_weight: NDArray,
    linear_weight: NDArray,
    linear_bias: Optional[NDArray] = None,
    eps: float = 1e-5,
) -> NDArray:
    """Fused RMSNorm + Linear: Linear(RMSNorm(x))."""
    norm_x = rmsnorm(x, norm_weight, eps)
    out = np.einsum("...d,od->...o", norm_x, linear_weight)
    if linear_bias is not None:
        out = out + linear_bias
    return out


def swiglu(
    x: NDArray,
    W_gate: NDArray,
    W_up: NDArray,
) -> NDArray:
    """SwiGLU: silu(x @ W_gate.T) * (x @ W_up.T)."""
    gate = np.einsum("...d,od->...o", x, W_gate)
    up = np.einsum("...d,od->...o", x, W_up)
    return silu(gate) * up


def geglu(
    x: NDArray,
    W_gate: NDArray,
    W_up: NDArray,
) -> NDArray:
    """GeGLU: gelu(x @ W_gate.T) * (x @ W_up.T)."""
    gate = np.einsum("...d,od->...o", x, W_gate)
    up = np.einsum("...d,od->...o", x, W_up)
    return gelu_approximate(gate) * up


# =============================================================================
# Scan Operations
# =============================================================================


def cumsum(x: NDArray, axis: int = -1) -> NDArray:
    """Cumulative sum (associative scan with add)."""
    return np.cumsum(x, axis=axis)


def cumprod(x: NDArray, axis: int = -1) -> NDArray:
    """Cumulative product (associative scan with mul)."""
    return np.cumprod(x, axis=axis)


def ssm_scan_sequential(
    A: NDArray,
    h: NDArray,
) -> NDArray:
    """Sequential SSM scan for validation.

    Computes: y[t] = A[t] * y[t-1] + h[t]

    This is the classic recurrence that associative scan parallelizes.

    Args:
        A: Decay factors of shape (..., seq_len, state_dim)
        h: Input contributions of shape (..., seq_len, state_dim)

    Returns:
        Output states of shape (..., seq_len, state_dim)
    """
    *batch_dims, seq_len, state_dim = A.shape

    # Flatten batch dimensions
    A_flat = A.reshape(-1, seq_len, state_dim)
    h_flat = h.reshape(-1, seq_len, state_dim)

    batch_size = A_flat.shape[0]
    output = np.zeros_like(A_flat)

    for b in range(batch_size):
        y_prev = np.zeros(state_dim)
        for t in range(seq_len):
            y_curr = A_flat[b, t] * y_prev + h_flat[b, t]
            output[b, t] = y_curr
            y_prev = y_curr

    return output.reshape(*batch_dims, seq_len, state_dim)


# =============================================================================
# Attention
# =============================================================================


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(
    q: NDArray,
    k: NDArray,
    v: NDArray,
    scale: Optional[float] = None,
    mask: Optional[NDArray] = None,
) -> NDArray:
    """Standard attention: softmax(Q @ K.T / scale) @ V.

    Args:
        q: Query of shape (batch, seq, heads, dim)
        k: Key of shape (batch, seq, heads, dim)
        v: Value of shape (batch, seq, heads, dim)
        scale: Attention scale (default: 1/sqrt(dim))
        mask: Boolean mask of shape (seq, seq), True = attend

    Returns:
        Output of shape (batch, seq, heads, dim)
    """
    batch, seq, heads, dim = q.shape

    if scale is None:
        scale = 1.0 / np.sqrt(dim)

    # Transpose to (batch, heads, seq, dim)
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Compute attention scores: (batch, heads, seq_q, seq_k)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale

    # Apply mask
    if mask is not None:
        scores = np.where(mask[None, None, :, :], scores, -1e9)

    # Softmax
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    out = np.einsum("bhqk,bhkd->bhqd", weights, v)

    # Transpose back to (batch, seq, heads, dim)
    return np.transpose(out, (0, 2, 1, 3))


def sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> NDArray:
    """Create sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: One-sided window size
        causal: If True, apply causal masking

    Returns:
        Boolean mask of shape (seq_len, seq_len)
    """
    positions = np.arange(seq_len)
    row_pos = positions[:, None]
    col_pos = positions[None, :]

    # Window constraint: |i - j| <= window_size
    mask = np.abs(row_pos - col_pos) <= window_size

    if causal:
        # Causal constraint: j <= i
        causal_mask = col_pos <= row_pos
        mask = mask & causal_mask

    return mask


def sliding_window_attention(
    q: NDArray,
    k: NDArray,
    v: NDArray,
    window_size: int,
    scale: Optional[float] = None,
    causal: bool = True,
) -> NDArray:
    """Sliding window attention using dense mask."""
    seq_len = q.shape[1]
    mask = sliding_window_mask(seq_len, window_size, causal)
    return attention(q, k, v, scale, mask)


# =============================================================================
# Quantization
# =============================================================================


def quantize_int8_per_channel(
    weights: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Quantize to INT8 with per-channel (row) scale/zero_point.

    Returns:
        (quantized_weights, scales, zero_points)
    """
    w_min = np.min(weights, axis=1)
    w_max = np.max(weights, axis=1)

    scale = (w_max - w_min) / 255.0
    scale = np.maximum(scale, 1e-8)

    zero_point = -np.round(w_min / scale)
    zero_point = np.clip(zero_point, 0, 255)

    # Quantize
    quantized = np.round(weights / scale[:, None]) + zero_point[:, None]
    quantized = np.clip(quantized, 0, 255)

    # Convert to signed int8
    quantized = (quantized - 128).astype(np.int8)
    zero_point = zero_point - 128

    return quantized, scale.astype(np.float32), zero_point.astype(np.float32)


def quantize_int8_per_tensor(
    weights: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Quantize to INT8 with per-tensor scale/zero_point."""
    w_min = np.min(weights)
    w_max = np.max(weights)

    scale = (w_max - w_min) / 255.0
    scale = max(scale, 1e-8)

    zero_point = -round(w_min / scale)
    zero_point = np.clip(zero_point, 0, 255)

    quantized = np.round(weights / scale) + zero_point
    quantized = np.clip(quantized, 0, 255)

    quantized = (quantized - 128).astype(np.int8)
    zero_point = zero_point - 128

    return quantized, np.array([scale], dtype=np.float32), np.array([zero_point], dtype=np.float32)


def dequantize_int8(
    W_quant: NDArray,
    scale: NDArray,
    zero_point: NDArray,
) -> NDArray:
    """Dequantize INT8 weights."""
    W_float = W_quant.astype(np.float32)
    if scale.size > 1:
        return scale[:, None] * (W_float - zero_point[:, None])
    else:
        return scale * (W_float - zero_point)


def int8_linear(
    x: NDArray,
    W_quant: NDArray,
    scale: NDArray,
    zero_point: NDArray,
    bias: Optional[NDArray] = None,
) -> NDArray:
    """INT8 linear with dequantization."""
    W_dequant = dequantize_int8(W_quant, scale, zero_point)
    out = np.einsum("...d,od->...o", x, W_dequant)
    if bias is not None:
        out = out + bias
    return out


# =============================================================================
# Gather/Scatter for MoE
# =============================================================================


def selective_gather(
    x: NDArray,
    indices: NDArray,
) -> NDArray:
    """Gather elements by indices.

    Args:
        x: Input of shape (batch, features)
        indices: Indices of shape (num_selected,)

    Returns:
        Gathered of shape (num_selected, features)
    """
    return x[indices]


def selective_scatter_add(
    output: NDArray,
    values: NDArray,
    indices: NDArray,
    weights: Optional[NDArray] = None,
) -> NDArray:
    """Scatter-add values into output.

    Args:
        output: Target array of shape (batch, features)
        values: Values to scatter of shape (num_selected, features)
        indices: Indices of shape (num_selected,)
        weights: Optional weights of shape (num_selected,)

    Returns:
        Updated output array
    """
    result = output.copy()

    if weights is not None:
        values = values * weights[:, None]

    for i, idx in enumerate(indices):
        result[idx] += values[i]

    return result


# =============================================================================
# Analytical Test Values
# =============================================================================


class AnalyticalTests:
    """Known input-output pairs for validation."""

    @staticmethod
    def silu_known_values() -> Tuple[NDArray, NDArray]:
        """SiLU at specific points."""
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        # silu(x) = x * sigmoid(x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        return x, expected

    @staticmethod
    def gelu_known_values() -> Tuple[NDArray, NDArray]:
        """GELU at specific points."""
        x = np.array([0.0, 1.0, -1.0])
        # GELU(0) = 0 exactly
        # GELU(1) ≈ 0.8413 (exact via erf)
        # GELU(-1) ≈ -0.1587 (exact via erf)
        expected = np.array([0.0, 0.8413447, -0.1586553])
        return x, expected

    @staticmethod
    def rmsnorm_known_values() -> Tuple[NDArray, NDArray, NDArray]:
        """RMSNorm with simple inputs."""
        x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
        weight = np.ones(4)
        # RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        rms = np.sqrt(30.0 / 4)
        expected = x / rms
        return x, weight, expected

    @staticmethod
    def cumsum_known_values() -> Tuple[NDArray, NDArray]:
        """Cumulative sum of simple sequence."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        return x, expected

    @staticmethod
    def ssm_scan_known_values() -> Tuple[NDArray, NDArray, NDArray]:
        """SSM scan with simple inputs.

        A = [0.9, 0.9, 0.9, 0.9]
        h = [1.0, 1.0, 1.0, 1.0]

        y[0] = 0 * 0.9 + 1.0 = 1.0
        y[1] = 1.0 * 0.9 + 1.0 = 1.9
        y[2] = 1.9 * 0.9 + 1.0 = 2.71
        y[3] = 2.71 * 0.9 + 1.0 = 3.439
        """
        A = np.array([[0.9, 0.9, 0.9, 0.9]]).reshape(1, 4, 1)
        h = np.array([[1.0, 1.0, 1.0, 1.0]]).reshape(1, 4, 1)
        expected = np.array([[1.0, 1.9, 2.71, 3.439]]).reshape(1, 4, 1)
        return A, h, expected

    @staticmethod
    def attention_identity() -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Attention where Q=K gives uniform weights over valid positions."""
        # Single head, seq_len=4, dim=1
        # All queries and keys are identical -> uniform attention
        q = np.ones((1, 4, 1, 1))
        k = np.ones((1, 4, 1, 1))
        v = np.array([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])

        # With causal mask, position i attends to 0..i with uniform weights
        # pos 0: v[0] = 1.0
        # pos 1: (v[0] + v[1]) / 2 = 1.5
        # pos 2: (v[0] + v[1] + v[2]) / 3 = 2.0
        # pos 3: (v[0] + v[1] + v[2] + v[3]) / 4 = 2.5
        expected = np.array([[[[1.0]], [[1.5]], [[2.0]], [[2.5]]]])
        return q, k, v, expected
