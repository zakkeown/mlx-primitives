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

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;

        float gate = 0.0f;
        float up = 0.0f;

        for (uint d = 0; d < in_features; d++) {
            float x_d = x[x_offset + d];
            gate += x_d * W_gate[out_idx * in_features + d];
            up += x_d * W_up[out_idx * in_features + d];
        }

        // SiLU: x * sigmoid(x)
        float silu_gate = gate / (1.0f + exp(-gate));
        float result = silu_gate * up;

        uint out_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
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

        if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= out_features) return;

        uint x_offset = batch_idx * seq_len * in_features + seq_idx * in_features;

        float gate = 0.0f;
        float up = 0.0f;

        for (uint d = 0; d < in_features; d++) {
            float x_d = x[x_offset + d];
            gate += x_d * W_gate[out_idx * in_features + d];
            up += x_d * W_up[out_idx * in_features + d];
        }

        // GELU (tanh approximation)
        float sqrt_2_over_pi = 0.7978845608f;
        float coeff = 0.044715f;
        float gate3 = gate * gate * gate;
        float inner = sqrt_2_over_pi * (gate + coeff * gate3);
        float gelu_gate = 0.5f * gate * (1.0f + tanh(inner));

        float result = gelu_gate * up;

        uint out_offset = batch_idx * seq_len * out_features + seq_idx * out_features + out_idx;
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
        return _metal_fused_swiglu(x, W_gate, W_up)
    except Exception:
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
    x = mx.ascontiguousarray(x.astype(mx.float32))
    W_gate = mx.ascontiguousarray(W_gate.astype(mx.float32))
    W_up = mx.ascontiguousarray(W_up.astype(mx.float32))

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
        return _metal_fused_geglu(x, W_gate, W_up)
    except Exception:
        return _reference_geglu(x, W_gate, W_up)


def _metal_fused_geglu(
    x: mx.array,
    W_gate: mx.array,
    W_up: mx.array,
) -> mx.array:
    """Metal kernel implementation of fused GeGLU."""
    batch_size, seq_len, in_features = x.shape
    out_features = W_gate.shape[0]

    kernel = _get_fused_geglu_kernel()

    x = mx.ascontiguousarray(x.astype(mx.float32))
    W_gate = mx.ascontiguousarray(W_gate.astype(mx.float32))
    W_up = mx.ascontiguousarray(W_up.astype(mx.float32))

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
    # GELU approximation
    gelu_gate = 0.5 * gate * (1 + mx.tanh(0.7978845608 * (gate + 0.044715 * gate ** 3)))
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
        return 0.5 * x * (1 + mx.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
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
