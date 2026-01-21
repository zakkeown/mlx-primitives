"""Fused activation functions for MLX.

This module provides fused versions of common gated activation functions
used in modern LLMs:

- SwiGLU: silu(x @ W_gate) * (x @ W_up) - Used in LLaMA, Mistral
- GeGLU: gelu(x @ W_gate) * (x @ W_up)
- ReGLU: relu(x @ W_gate) * (x @ W_up)

Fusion reduces memory bandwidth by computing gate and up projections
together without writing intermediate results.
"""

import math
from typing import Literal, Optional

import mlx.core as mx

from mlx_primitives.constants import (
    GELU_SQRT_2_OVER_PI,
    GELU_TANH_COEFF,
)

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_fused_swiglu_kernel: Optional[mx.fast.metal_kernel] = None
_fused_geglu_kernel: Optional[mx.fast.metal_kernel] = None


def _get_fused_swiglu_kernel() -> mx.fast.metal_kernel:
    """Get or create the fused SwiGLU kernel."""
    global _fused_swiglu_kernel
    if _fused_swiglu_kernel is None:
        source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint out_idx = thread_position_in_grid.x;

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _in_features = in_features[0];
        uint _out_features = out_features[0];

        if (batch_idx >= _batch_size || seq_idx >= _seq_len || out_idx >= _out_features) return;

        uint x_offset = batch_idx * _seq_len * _in_features + seq_idx * _in_features;

        float gate = 0.0f;
        float up = 0.0f;

        for (uint d = 0; d < _in_features; d++) {
            float x_d = x[x_offset + d];
            gate += x_d * W_gate[out_idx * _in_features + d];
            up += x_d * W_up[out_idx * _in_features + d];
        }

        // SiLU: x * sigmoid(x)
        float silu_gate = gate / (1.0f + exp(-gate));
        float result = silu_gate * up;

        uint out_offset = batch_idx * _seq_len * _out_features + seq_idx * _out_features + out_idx;
        output[out_offset] = result;
        """
        _fused_swiglu_kernel = mx.fast.metal_kernel(
            name="fused_swiglu",
            input_names=[
                "x", "W_gate", "W_up",
                "batch_size", "seq_len", "in_features", "out_features"
            ],
            output_names=["output"],
            source=source,
        )
    return _fused_swiglu_kernel


def _get_fused_geglu_kernel() -> mx.fast.metal_kernel:
    """Get or create the fused GeGLU kernel."""
    global _fused_geglu_kernel
    if _fused_geglu_kernel is None:
        source = """
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint out_idx = thread_position_in_grid.x;

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _in_features = in_features[0];
        uint _out_features = out_features[0];

        if (batch_idx >= _batch_size || seq_idx >= _seq_len || out_idx >= _out_features) return;

        uint x_offset = batch_idx * _seq_len * _in_features + seq_idx * _in_features;

        float gate = 0.0f;
        float up = 0.0f;

        for (uint d = 0; d < _in_features; d++) {
            float x_d = x[x_offset + d];
            gate += x_d * W_gate[out_idx * _in_features + d];
            up += x_d * W_up[out_idx * _in_features + d];
        }

        // GELU (tanh approximation)
        float sqrt_2_over_pi = 0.7978845608f;
        float coeff = 0.044715f;
        float gate3 = gate * gate * gate;
        float inner = sqrt_2_over_pi * (gate + coeff * gate3);
        float gelu_gate = 0.5f * gate * (1.0f + tanh(inner));

        float result = gelu_gate * up;

        uint out_offset = batch_idx * _seq_len * _out_features + seq_idx * _out_features + out_idx;
        output[out_offset] = result;
        """
        _fused_geglu_kernel = mx.fast.metal_kernel(
            name="fused_geglu",
            input_names=[
                "x", "W_gate", "W_up",
                "batch_size", "seq_len", "in_features", "out_features"
            ],
            output_names=["output"],
            source=source,
        )
    return _fused_geglu_kernel


@mx.custom_function
def _fused_swiglu_with_vjp(x, W_gate, W_up):
    """Fused SwiGLU with custom VJP for gradient support."""
    return _metal_fused_swiglu(x, W_gate, W_up)


@_fused_swiglu_with_vjp.vjp
def _fused_swiglu_vjp(primals, cotangent, output):
    """VJP for fused SwiGLU - uses reference implementation for gradients."""
    x, W_gate, W_up = primals

    # Recompute forward using reference (autodiff-compatible)
    gate = x @ W_gate.T
    up = x @ W_up.T
    # silu(gate) = gate * sigmoid(gate)
    sigmoid_gate = mx.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # Gradient of silu_gate * up w.r.t. inputs
    # d/d(silu_gate) = up * cotangent
    # d/d(up) = silu_gate * cotangent
    d_silu_gate = up * cotangent
    d_up = silu_gate * cotangent

    # Gradient through SiLU: d(silu)/d(gate)
    # silu(x) = x * sigmoid(x)
    # d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    d_silu_d_gate = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))
    d_gate = d_silu_gate * d_silu_d_gate

    # Gradient w.r.t. x: d_gate @ W_gate + d_up @ W_up
    dx = d_gate @ W_gate + d_up @ W_up

    # Gradient w.r.t. W_gate: d_gate.T @ x (summed over batch/seq)
    # Shape: (out_features, in_features)
    d_gate_flat = d_gate.reshape(-1, d_gate.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    dW_gate = d_gate_flat.T @ x_flat

    # Gradient w.r.t. W_up: d_up.T @ x
    d_up_flat = d_up.reshape(-1, d_up.shape[-1])
    dW_up = d_up_flat.T @ x_flat

    return dx, dW_gate, dW_up


def fused_swiglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
    use_metal: bool = True,
) -> mx.array:
    """Fused SwiGLU activation with projections.

    Computes: silu(x @ W_gate.T) * (x @ W_up.T)

    Where silu(x) = x * sigmoid(x)

    This is used in LLaMA, Mistral, and other modern LLMs.
    Supports gradients via custom VJP.

    Args:
        x: Input tensor of shape (batch, seq_len, in_features).
        W_gate: Gate projection weight (out_features, in_features).
        W_up: Up projection weight (out_features, in_features).
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, out_features).

    Example:
        >>> x = mx.random.normal((2, 128, 768))
        >>> W_gate = mx.random.normal((2048, 768))
        >>> W_up = mx.random.normal((2048, 768))
        >>> out = fused_swiglu(x, W_gate, W_up)
        >>> out.shape
        (2, 128, 2048)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input (batch, seq, features), got {x.ndim}D")

    batch_size, seq_len, in_features = x.shape
    out_features = W_gate.shape[0]

    if W_up.shape != W_gate.shape:
        raise ValueError("W_gate and W_up must have same shape")

    # For small tensors or when Metal not available, use separate ops
    if not use_metal or not _HAS_METAL or seq_len < 8:
        return _reference_swiglu(x, W_gate, W_up)

    try:
        return _fused_swiglu_with_vjp(x, W_gate, W_up)
    except RuntimeError as e:
        # Catch Metal kernel errors, but let programming bugs propagate
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("fused_swiglu", e)
        return _reference_swiglu(x, W_gate, W_up)


def _metal_fused_swiglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
) -> mx.array:
    """Metal kernel implementation of fused SwiGLU."""
    batch_size, seq_len, in_features = x.shape
    out_features = W_gate.shape[0]

    kernel = _get_fused_swiglu_kernel()

    # Ensure contiguous float32
    x = mx.contiguous(x.astype(mx.float32))
    W_gate = mx.contiguous(W_gate.astype(mx.float32))
    W_up = mx.contiguous(W_up.astype(mx.float32))

    # Scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    in_arr = mx.array([in_features], dtype=mx.uint32)
    out_arr = mx.array([out_features], dtype=mx.uint32)

    outputs = kernel(
        inputs=[x, W_gate, W_up, batch_arr, seq_arr, in_arr, out_arr],
        grid=(out_features, seq_len, batch_size),
        threadgroup=(min(out_features, 64), 1, 1),
        output_shapes=[(batch_size, seq_len, out_features)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_swiglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
) -> mx.array:
    """Reference implementation using separate operations."""
    gate = x @ W_gate.T
    up = x @ W_up.T
    return mx.sigmoid(gate) * gate * up  # silu(gate) * up


def fused_geglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
    use_metal: bool = True,
) -> mx.array:
    """Fused GeGLU activation with projections.

    Computes: gelu(x @ W_gate.T) * (x @ W_up.T)

    Args:
        x: Input tensor of shape (batch, seq_len, in_features).
        W_gate: Gate projection weight (out_features, in_features).
        W_up: Up projection weight (out_features, in_features).
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, out_features).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input (batch, seq, features), got {x.ndim}D")

    batch_size, seq_len, in_features = x.shape
    out_features = W_gate.shape[0]

    if not use_metal or not _HAS_METAL or seq_len < 8:
        return _reference_geglu(x, W_gate, W_up)

    try:
        return _fused_geglu_with_vjp(x, W_gate, W_up)
    except RuntimeError as e:
        # Catch Metal kernel errors, but let programming bugs propagate
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("fused_geglu", e)
        return _reference_geglu(x, W_gate, W_up)


@mx.custom_function
def _fused_geglu_with_vjp(x, W_gate, W_up):
    """Fused GeGLU with custom VJP for gradient support."""
    return _metal_fused_geglu(x, W_gate, W_up)


@_fused_geglu_with_vjp.vjp
def _fused_geglu_vjp(primals, cotangent, output):
    """VJP for fused GeGLU - uses reference implementation for gradients."""
    x, W_gate, W_up = primals

    # Recompute forward using reference (autodiff-compatible)
    gate = x @ W_gate.T
    up = x @ W_up.T
    gelu_gate = 0.5 * gate * (1 + mx.tanh(GELU_SQRT_2_OVER_PI * (gate + GELU_TANH_COEFF * gate ** 3)))

    # Gradient of gelu_gate * up w.r.t. inputs
    # d/d(gelu_gate) = up * cotangent
    # d/d(up) = gelu_gate * cotangent
    d_gelu_gate = up * cotangent
    d_up = gelu_gate * cotangent

    # Gradient through GELU: d(gelu)/d(gate)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inner = GELU_SQRT_2_OVER_PI * (gate + GELU_TANH_COEFF * gate ** 3)
    tanh_inner = mx.tanh(inner)
    sech2_inner = 1 - tanh_inner ** 2
    d_inner = GELU_SQRT_2_OVER_PI * (1 + 3 * GELU_TANH_COEFF * gate ** 2)
    d_gelu_d_gate = 0.5 * (1 + tanh_inner) + 0.5 * gate * sech2_inner * d_inner
    d_gate = d_gelu_gate * d_gelu_d_gate

    # Gradient w.r.t. x: d_gate @ W_gate + d_up @ W_up
    dx = d_gate @ W_gate + d_up @ W_up

    # Gradient w.r.t. W_gate: d_gate.T @ x (summed over batch/seq)
    # Shape: (out_features, in_features)
    d_gate_flat = d_gate.reshape(-1, d_gate.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    dW_gate = d_gate_flat.T @ x_flat

    # Gradient w.r.t. W_up: d_up.T @ x
    d_up_flat = d_up.reshape(-1, d_up.shape[-1])
    dW_up = d_up_flat.T @ x_flat

    return dx, dW_gate, dW_up


def _metal_fused_geglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
) -> mx.array:
    """Metal kernel implementation of fused GeGLU."""
    batch_size, seq_len, in_features = x.shape
    out_features = W_gate.shape[0]

    kernel = _get_fused_geglu_kernel()

    x = mx.contiguous(x.astype(mx.float32))
    W_gate = mx.contiguous(W_gate.astype(mx.float32))
    W_up = mx.contiguous(W_up.astype(mx.float32))

    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    in_arr = mx.array([in_features], dtype=mx.uint32)
    out_arr = mx.array([out_features], dtype=mx.uint32)

    outputs = kernel(
        inputs=[x, W_gate, W_up, batch_arr, seq_arr, in_arr, out_arr],
        grid=(out_features, seq_len, batch_size),
        threadgroup=(min(out_features, 64), 1, 1),
        output_shapes=[(batch_size, seq_len, out_features)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_geglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
) -> mx.array:
    """Reference implementation using separate operations."""
    gate = x @ W_gate.T
    up = x @ W_up.T
    # GELU approximation using centralized constants
    gelu_gate = 0.5 * gate * (1 + mx.tanh(GELU_SQRT_2_OVER_PI * (gate + GELU_TANH_COEFF * gate ** 3)))
    return gelu_gate * up


def silu(x: mx.array) -> mx.array:
    """SiLU (Swish) activation function.

    Computes: x * sigmoid(x)

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return x * mx.sigmoid(x)


def gelu(x: mx.array, approximate: bool = True) -> mx.array:
    """GELU activation function.

    Args:
        x: Input tensor.
        approximate: Use tanh approximation (faster).

    Returns:
        Activated tensor.
    """
    if approximate:
        return 0.5 * x * (1 + mx.tanh(GELU_SQRT_2_OVER_PI * (x + GELU_TANH_COEFF * x ** 3)))
    else:
        # Exact GELU using erf
        return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


class SwiGLU:
    """SwiGLU feed-forward layer.

    Implements the SwiGLU activation used in LLaMA and Mistral:
        output = silu(x @ W_gate) * (x @ W_up) @ W_down

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (gate/up projection size).
        out_features: Output dimension. Defaults to in_features.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        scale = 1.0 / math.sqrt(in_features)
        self.W_gate = mx.random.normal((hidden_features, in_features)) * scale
        self.W_up = mx.random.normal((hidden_features, in_features)) * scale
        self.W_down = mx.random.normal((out_features, hidden_features)) * scale

        if bias:
            self.b_gate = mx.zeros((hidden_features,))
            self.b_up = mx.zeros((hidden_features,))
            self.b_down = mx.zeros((out_features,))
        else:
            self.b_gate = None
            self.b_up = None
            self.b_down = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, in_features).

        Returns:
            Output tensor (batch, seq, out_features).
        """
        # Fused gate and up projections with SwiGLU activation
        hidden = fused_swiglu(x, self.W_gate, self.W_up)

        # Down projection
        out = hidden @ self.W_down.T
        if self.b_down is not None:
            out = out + self.b_down

        return out


class GeGLU:
    """GeGLU feed-forward layer.

    Similar to SwiGLU but uses GELU instead of SiLU activation.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension. Defaults to in_features.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        scale = 1.0 / math.sqrt(in_features)
        self.W_gate = mx.random.normal((hidden_features, in_features)) * scale
        self.W_up = mx.random.normal((hidden_features, in_features)) * scale
        self.W_down = mx.random.normal((out_features, hidden_features)) * scale

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        hidden = fused_geglu(x, self.W_gate, self.W_up)
        return hidden @ self.W_down.T
