"""Quantization utilities for MLX.

This module provides quantization tools:
- Dynamic quantization (per-tensor and per-channel)
- Quantized linear layers
- Weight-only quantization
- Calibration utilities
- Gradient checkpointing support for memory-efficient training
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn


# Check if mx.checkpoint is available (requires MLX >= 0.1.0)
_HAS_CHECKPOINT = hasattr(mx, "checkpoint")

# Dtype-aware epsilon values to prevent division by zero and underflow
# These should be practical values that avoid numerical issues, not just machine epsilon
_EPSILON = {
    mx.float16: 6.104e-5,   # 2^-14 (smallest normal for fp16)
    mx.bfloat16: 1e-7,       # BF16 has 7-bit mantissa, use practical epsilon
    mx.float32: 1.175e-38,   # 2^-126 (smallest normal for fp32)
}

# Default epsilon for unknown dtypes
_DEFAULT_EPSILON = 1e-8

# Valid bit widths for quantization
_VALID_BITS = {2, 4, 8, 16}


def _get_epsilon(dtype: mx.Dtype) -> float:
    """Get appropriate epsilon for a given dtype."""
    return _EPSILON.get(dtype, _DEFAULT_EPSILON)


def quantize_tensor(
    x: mx.array,
    num_bits: int = 8,
    per_channel: bool = False,
    symmetric: bool = True,
) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """Quantize a tensor to lower precision.

    Args:
        x: Input tensor to quantize.
        num_bits: Number of bits (2, 4, 8, or 16).
        per_channel: If True, compute scale per output channel.
        symmetric: If True, use symmetric quantization (no zero point).

    Returns:
        Tuple of (quantized_tensor, scale, zero_point).
        zero_point is None for symmetric quantization.

    Raises:
        ValueError: If num_bits is not a valid bit width (2, 4, 8, or 16).
    """
    if num_bits not in _VALID_BITS:
        raise ValueError(
            f"num_bits must be one of {sorted(_VALID_BITS)}, got {num_bits}"
        )

    if num_bits == 8:
        qmin, qmax = -128, 127
    elif num_bits == 4:
        qmin, qmax = -8, 7
    else:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

    if per_channel:
        # Quantize per output channel (axis 0)
        axis = tuple(range(1, x.ndim))
        x_min = mx.min(x, axis=axis, keepdims=True)
        x_max = mx.max(x, axis=axis, keepdims=True)
    else:
        x_min = mx.min(x)
        x_max = mx.max(x)

    # Use dtype-aware epsilon to avoid underflow
    eps = _get_epsilon(x.dtype)

    if symmetric:
        # Symmetric: scale = max(|min|, |max|) / qmax
        x_absmax = mx.maximum(mx.abs(x_min), mx.abs(x_max))
        scale = x_absmax / qmax
        scale = mx.maximum(scale, eps)  # Avoid division by zero

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, qmin, qmax).astype(mx.int8)

        return x_q, scale, None
    else:
        # Asymmetric: compute zero point
        scale = (x_max - x_min) / (qmax - qmin)
        scale = mx.maximum(scale, eps)

        zero_point = qmin - mx.round(x_min / scale)
        zero_point = mx.clip(zero_point, qmin, qmax)

        x_q = mx.round(x / scale + zero_point)
        x_q = mx.clip(x_q, qmin, qmax).astype(mx.int8)

        return x_q, scale, zero_point


def dequantize_tensor(
    x_q: mx.array,
    scale: mx.array,
    zero_point: Optional[mx.array] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize a tensor back to floating point.

    Args:
        x_q: Quantized tensor.
        scale: Quantization scale.
        zero_point: Zero point (None for symmetric).
        dtype: Output dtype.

    Returns:
        Dequantized tensor.
    """
    x = x_q.astype(dtype)
    if zero_point is not None:
        x = (x - zero_point) * scale
    else:
        x = x * scale
    return x


class QuantizedLinear(nn.Module):
    """Quantized linear layer with weight quantization.

    Stores weights in low precision and dequantizes during forward pass.
    This reduces memory usage but maintains computation in float.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        num_bits: Number of bits for weight quantization.
        per_channel: Per-channel or per-tensor quantization.

    Example:
        >>> layer = QuantizedLinear(768, 3072, num_bits=4)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = layer(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
        per_channel: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.per_channel = per_channel

        # Initialize with float weights (will quantize on first forward or manually)
        self._weight_float = mx.random.normal((out_features, in_features)) * 0.02
        self._quantized = False

        # Quantized weight storage
        self.weight_q: Optional[mx.array] = None
        self.weight_scale: Optional[mx.array] = None
        self.weight_zero_point: Optional[mx.array] = None

        # Cache for dequantized weights (populated on first forward pass)
        self._weight_cache: Optional[mx.array] = None

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def quantize_weights(self) -> None:
        """Quantize the weights."""
        # Clear cache since weights are changing
        self._weight_cache = None

        self.weight_q, self.weight_scale, self.weight_zero_point = quantize_tensor(
            self._weight_float,
            num_bits=self.num_bits,
            per_channel=self.per_channel,
        )
        self._quantized = True

    def clear_cache(self) -> None:
        """Clear the dequantized weight cache."""
        self._weight_cache = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor (*, in_features).

        Returns:
            Output tensor (*, out_features).
        """
        import warnings

        if not self._quantized:
            # Warn user that weights haven't been quantized
            warnings.warn(
                "QuantizedLinear.quantize_weights() has not been called. "
                "Using float weights instead. Call quantize_weights() or use "
                "QuantizedLinear.from_linear() to enable quantization.",
                stacklevel=2,
            )
            weight = self._weight_float
        else:
            # Use cached dequantized weights for performance
            if self._weight_cache is None:
                self._weight_cache = dequantize_tensor(
                    self.weight_q,
                    self.weight_scale,
                    self.weight_zero_point,
                )
            weight = self._weight_cache

        # Matrix multiply
        y = x @ weight.T

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_bits: int = 8,
        per_channel: bool = True,
    ) -> "QuantizedLinear":
        """Create quantized layer from existing linear layer.

        Args:
            linear: Source linear layer.
            num_bits: Number of bits for quantization.
            per_channel: Per-channel quantization.

        Returns:
            Quantized linear layer.
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, 'bias') and linear.bias is not None

        q_linear = cls(
            in_features,
            out_features,
            bias=has_bias,
            num_bits=num_bits,
            per_channel=per_channel,
        )

        # Copy weights
        q_linear._weight_float = linear.weight

        if has_bias:
            q_linear.bias = linear.bias

        # Quantize
        q_linear.quantize_weights()

        return q_linear


class DynamicQuantizer:
    """Dynamic quantization manager.

    Quantizes activations dynamically during inference.

    Args:
        num_bits: Number of bits for activation quantization.
        symmetric: Whether to use symmetric quantization.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
    ):
        self.num_bits = num_bits
        self.symmetric = symmetric

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Dynamically quantize input.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (quantized, scale, zero_point).
        """
        return quantize_tensor(
            x,
            num_bits=self.num_bits,
            per_channel=False,
            symmetric=self.symmetric,
        )

    def quantize_dequantize(self, x: mx.array) -> mx.array:
        """Quantize and immediately dequantize (for simulated quantization).

        Args:
            x: Input tensor.

        Returns:
            Tensor after quantization noise.
        """
        x_q, scale, zero_point = self.quantize(x)
        return dequantize_tensor(x_q, scale, zero_point, dtype=x.dtype)


class CalibrationCollector:
    """Collect activation statistics for calibration.

    Used to determine optimal quantization parameters.

    Args:
        method: Calibration method ('minmax', 'percentile', 'entropy').
        percentile: Percentile for percentile method (default: 99.99).
        max_samples: Maximum number of samples to keep for percentile method.
            Uses reservoir sampling to maintain bounded memory. Default: 100000.
    """

    def __init__(
        self,
        method: str = "minmax",
        percentile: float = 99.99,
        max_samples: int = 100000,
    ):
        self.method = method
        self.percentile = percentile
        self.max_samples = max_samples

        # For minmax: just track running min/max
        self._global_min: Optional[float] = None
        self._global_max: Optional[float] = None

        # For percentile: use reservoir sampling with bounded memory
        self._reservoir: Optional[mx.array] = None
        self._samples_seen: int = 0

    def observe(self, x: mx.array) -> None:
        """Record activation statistics.

        Args:
            x: Activation tensor to observe.
        """
        import random

        # Update running min/max (used by minmax method)
        batch_min = float(mx.min(x))
        batch_max = float(mx.max(x))

        if self._global_min is None:
            self._global_min = batch_min
            self._global_max = batch_max
        else:
            self._global_min = min(self._global_min, batch_min)
            self._global_max = max(self._global_max, batch_max)

        # For percentile method: reservoir sampling
        if self.method == "percentile":
            abs_flat = mx.abs(x).flatten()
            n_new = abs_flat.shape[0]

            if self._reservoir is None:
                # First batch - take up to max_samples
                if n_new <= self.max_samples:
                    self._reservoir = abs_flat
                else:
                    # Random sample from first batch
                    indices = mx.array(random.sample(range(n_new), self.max_samples))
                    self._reservoir = abs_flat[indices]
                self._samples_seen = n_new
            else:
                # Reservoir sampling: for each new element, include with probability
                # max_samples / samples_seen. We do this approximately in batches.
                current_size = self._reservoir.shape[0]

                if current_size < self.max_samples:
                    # Still filling reservoir
                    space_left = self.max_samples - current_size
                    if n_new <= space_left:
                        self._reservoir = mx.concatenate([self._reservoir, abs_flat])
                    else:
                        # Take random sample from new batch
                        indices = mx.array(random.sample(range(n_new), space_left))
                        self._reservoir = mx.concatenate([self._reservoir, abs_flat[indices]])
                else:
                    # Reservoir is full - use sampling to potentially replace elements
                    # Probability of including each new element: max_samples / (samples_seen + i)
                    # We approximate by including each new element with avg probability
                    avg_prob = self.max_samples / (self._samples_seen + n_new / 2)
                    n_to_include = int(n_new * avg_prob)

                    if n_to_include > 0:
                        # Select random new elements to include
                        new_indices = mx.array(random.sample(range(n_new), min(n_to_include, n_new)))
                        new_samples = abs_flat[new_indices]

                        # Replace random existing elements
                        replace_indices = mx.array(random.sample(range(self.max_samples), len(new_indices)))
                        self._reservoir = self._reservoir.at[replace_indices].add(
                            new_samples - self._reservoir[replace_indices]
                        )

                self._samples_seen += n_new

    def compute_scale(self, num_bits: int = 8) -> Tuple[float, float]:
        """Compute quantization scale from collected statistics.

        Args:
            num_bits: Number of bits.

        Returns:
            Tuple of (scale, zero_point).
        """
        qmax = 2 ** (num_bits - 1) - 1

        if self._global_min is None:
            raise ValueError("No data observed. Call observe() before compute_scale().")

        if self.method == "minmax":
            x_absmax = max(abs(self._global_min), abs(self._global_max))
            scale = x_absmax / qmax
            return scale, 0.0

        elif self.method == "percentile":
            if self._reservoir is None:
                raise ValueError("No data observed for percentile method.")
            # Sort reservoir and take percentile
            sorted_vals = mx.sort(self._reservoir)
            idx = int(len(sorted_vals) * self.percentile / 100)
            idx = min(idx, len(sorted_vals) - 1)  # Clamp to valid range
            x_absmax = float(sorted_vals[idx])
            scale = x_absmax / qmax
            return scale, 0.0

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def reset(self) -> None:
        """Reset collected statistics."""
        self._global_min = None
        self._global_max = None
        self._reservoir = None
        self._samples_seen = 0


def quantize_model_weights(
    model: nn.Module,
    num_bits: int = 8,
    per_channel: bool = True,
    skip_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Quantize all linear layer weights in a model.

    Args:
        model: Model to quantize.
        num_bits: Number of bits.
        per_channel: Per-channel quantization.
        skip_layers: List of layer names to skip.

    Returns:
        Dictionary mapping layer names to quantization info.
    """
    skip_layers = skip_layers or []
    quantization_info = {}

    def quantize_recursive(
        module: Any, prefix: str = ""
    ) -> None:
        if isinstance(module, nn.Linear):
            if prefix not in skip_layers:
                weight = module.weight
                w_q, scale, zp = quantize_tensor(
                    weight,
                    num_bits=num_bits,
                    per_channel=per_channel,
                )
                quantization_info[prefix] = {
                    "original_shape": weight.shape,
                    "num_bits": num_bits,
                    "scale_shape": scale.shape,
                    "compression_ratio": 32 / num_bits,
                }

        # Recurse into submodules
        if hasattr(module, "__dict__"):
            for name, child in module.__dict__.items():
                if isinstance(child, nn.Module):
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    quantize_recursive(child, child_prefix)
                elif isinstance(child, list):
                    for i, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            child_prefix = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                            quantize_recursive(item, child_prefix)

    quantize_recursive(model)
    return quantization_info


class Int4Linear(nn.Module):
    """4-bit quantized linear layer.

    Packs two 4-bit values into each int8 for memory efficiency.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        group_size: Group size for grouped quantization (default: 128).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Number of groups
        self.num_groups = (in_features + group_size - 1) // group_size

        # Packed weights: 2 int4 values per int8
        packed_size = (in_features + 1) // 2
        self.weight_packed = mx.zeros((out_features, packed_size), dtype=mx.uint8)

        # Scales per group
        self.scales = mx.ones((out_features, self.num_groups))

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Cache for dequantized weights (populated on first forward pass)
        # This avoids expensive dequantization on every call
        self._weight_cache: Optional[mx.array] = None

    def pack_weights(self, weight: mx.array) -> None:
        """Pack float weights into int4 (vectorized).

        Args:
            weight: Float weight tensor (out_features, in_features).
        """
        # Clear the cache since weights are changing
        self._weight_cache = None

        out_features, in_features = weight.shape

        # Pad in_features to be divisible by group_size
        padded_in = self.num_groups * self.group_size
        if in_features < padded_in:
            padding = mx.zeros((out_features, padded_in - in_features))
            weight_padded = mx.concatenate([weight, padding], axis=1)
        else:
            weight_padded = weight

        # Reshape to (out_features, num_groups, group_size)
        grouped = weight_padded.reshape(out_features, self.num_groups, self.group_size)

        # Compute absmax per group: (out_features, num_groups)
        # Use dtype-aware epsilon to prevent underflow
        eps = _get_epsilon(weight.dtype)
        absmax = mx.max(mx.abs(grouped), axis=-1)
        self.scales = mx.maximum(absmax / 7.0, eps)

        # Quantize all groups at once using broadcasting
        # scales: (out_features, num_groups, 1)
        q_grouped = mx.round(grouped / self.scales[:, :, None])
        q_grouped = mx.clip(q_grouped, -8, 7).astype(mx.int8)

        # Flatten back to (out_features, padded_in) and trim
        q_flat = q_grouped.reshape(out_features, padded_in)[:, :in_features]

        # Pad to even length for packing pairs
        if in_features % 2 == 1:
            q_flat = mx.concatenate([q_flat, mx.zeros((out_features, 1), dtype=mx.int8)], axis=1)

        # Pack pairs of int4 into uint8 using vectorized bit operations
        # low nibble: q_flat[:, 0::2], high nibble: q_flat[:, 1::2]
        low = (q_flat[:, 0::2].astype(mx.uint8)) & 0x0F
        high = (q_flat[:, 1::2].astype(mx.uint8)) & 0x0F
        self.weight_packed = (high << 4) | low

    def unpack_weights(self) -> mx.array:
        """Unpack int4 weights to float (vectorized).

        Returns:
            Float weight tensor (out_features, in_features).
        """
        out_features, packed_cols = self.weight_packed.shape

        # Extract low and high nibbles using vectorized bit operations
        low = self.weight_packed & 0x0F  # (out_features, packed_cols)
        high = (self.weight_packed >> 4) & 0x0F

        # Convert from unsigned [0-15] to signed [-8 to 7]
        # Values > 7 become negative: val - 16
        low = mx.where(low > 7, low.astype(mx.int16) - 16, low.astype(mx.int16))
        high = mx.where(high > 7, high.astype(mx.int16) - 16, high.astype(mx.int16))

        # Interleave low and high to get original order
        # Shape: (out_features, packed_cols * 2)
        unpacked = mx.zeros((out_features, packed_cols * 2), dtype=mx.float32)
        unpacked = unpacked.at[:, 0::2].add(low.astype(mx.float32))
        unpacked = unpacked.at[:, 1::2].add(high.astype(mx.float32))

        # Trim to actual in_features
        unpacked = unpacked[:, :self.in_features]

        # Dequantize using scales via broadcasting
        # Need to expand each scale to cover its group_size elements
        # scales: (out_features, num_groups) -> (out_features, in_features)
        scale_expanded = mx.repeat(self.scales, self.group_size, axis=1)[:, :self.in_features]

        return unpacked * scale_expanded

    def clear_cache(self) -> None:
        """Clear the dequantized weight cache.

        Call this if you need to force re-dequantization (e.g., after modifying
        packed weights or scales). Usually not needed in normal usage.
        """
        self._weight_cache = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).

        Raises:
            ValueError: If input's last dimension doesn't match in_features.
        """
        if x.ndim < 1:
            raise ValueError(f"Int4Linear expects at least 1D input, got {x.ndim}D")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Int4Linear expects input with last dimension {self.in_features}, "
                f"got {x.shape[-1]} (shape: {x.shape})"
            )

        # Use cached dequantized weights for performance
        # Dequantize only on first call, then reuse
        if self._weight_cache is None:
            self._weight_cache = self.unpack_weights()

        # Matrix multiply
        y = x @ self._weight_cache.T

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
    ) -> "Int4Linear":
        """Create from existing linear layer.

        Args:
            linear: Source linear layer.
            group_size: Group size for quantization.

        Returns:
            Int4 quantized layer.
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, 'bias') and linear.bias is not None

        q_linear = cls(
            in_features,
            out_features,
            bias=has_bias,
            group_size=group_size,
        )

        q_linear.pack_weights(linear.weight)

        if has_bias:
            q_linear.bias = linear.bias

        return q_linear


class QLoRALinear(nn.Module):
    """QLoRA: Quantized Low-Rank Adaptation layer.

    Combines 4-bit quantization of base weights with trainable low-rank
    adapters for efficient fine-tuning of large models.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank (default: 8).
        alpha: LoRA scaling factor (default: 16).
        dropout: Dropout rate for LoRA (default: 0.0).
        bias: Whether to include bias.
        group_size: Group size for NF4 quantization (default: 64).

    Reference:
        "QLoRA: Efficient Finetuning of Quantized LLMs"
        https://arxiv.org/abs/2305.14314

    Example:
        >>> layer = QLoRALinear(768, 3072, rank=8)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = layer(x)  # Forward with quantized base + LoRA adapters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.group_size = group_size
        self.bits = bits

        # Quantized base weights (frozen during training)
        self.num_groups = (in_features + group_size - 1) // group_size
        packed_size = (in_features + 1) // 2
        self.base_weight_packed = mx.zeros((out_features, packed_size), dtype=mx.uint8)
        self.base_scales = mx.ones((out_features, self.num_groups))

        # LoRA adapters (trainable)
        self.lora_A = mx.random.normal((in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))  # Initialize to zero

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        self._quantized = False

        # Cache for dequantized base weights (populated on first forward)
        self._base_weight_cache: Optional[mx.array] = None

    def quantize_base(self, weight: mx.array) -> None:
        """Quantize base weights to NF4 (vectorized).

        Args:
            weight: Float weight tensor (out_features, in_features).
        """
        # Clear cache since weights are changing
        self._base_weight_cache = None

        out_features, in_features = weight.shape

        # Pad in_features to be divisible by group_size
        padded_in = self.num_groups * self.group_size
        if in_features < padded_in:
            padding = mx.zeros((out_features, padded_in - in_features))
            weight_padded = mx.concatenate([weight, padding], axis=1)
        else:
            weight_padded = weight

        # Reshape to (out_features, num_groups, group_size)
        grouped = weight_padded.reshape(out_features, self.num_groups, self.group_size)

        # Compute absmax per group
        # Use dtype-aware epsilon
        eps = _get_epsilon(weight.dtype)
        absmax = mx.max(mx.abs(grouped), axis=-1)
        self.base_scales = mx.maximum(absmax / 7.0, eps)

        # Quantize all groups at once
        q_grouped = mx.round(grouped / self.base_scales[:, :, None])
        q_grouped = mx.clip(q_grouped, -8, 7).astype(mx.int8)

        # Flatten and trim
        q_flat = q_grouped.reshape(out_features, padded_in)[:, :in_features]

        # Pad to even length
        if in_features % 2 == 1:
            q_flat = mx.concatenate([q_flat, mx.zeros((out_features, 1), dtype=mx.int8)], axis=1)

        # Pack pairs of int4 into uint8
        low = (q_flat[:, 0::2].astype(mx.uint8)) & 0x0F
        high = (q_flat[:, 1::2].astype(mx.uint8)) & 0x0F
        self.base_weight_packed = (high << 4) | low
        self._quantized = True

    def _dequantize_base(self) -> mx.array:
        """Dequantize base weights to float (vectorized)."""
        out_features, packed_cols = self.base_weight_packed.shape

        # Extract nibbles
        low = self.base_weight_packed & 0x0F
        high = (self.base_weight_packed >> 4) & 0x0F

        # Convert to signed
        low = mx.where(low > 7, low.astype(mx.int16) - 16, low.astype(mx.int16))
        high = mx.where(high > 7, high.astype(mx.int16) - 16, high.astype(mx.int16))

        # Interleave
        unpacked = mx.zeros((out_features, packed_cols * 2), dtype=mx.float32)
        unpacked = unpacked.at[:, 0::2].add(low.astype(mx.float32))
        unpacked = unpacked.at[:, 1::2].add(high.astype(mx.float32))

        # Trim and dequantize
        unpacked = unpacked[:, :self.in_features]
        scale_expanded = mx.repeat(self.base_scales, self.group_size, axis=1)[:, :self.in_features]

        return unpacked * scale_expanded

    def clear_cache(self) -> None:
        """Clear the dequantized base weight cache."""
        self._base_weight_cache = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized base and LoRA adapters.

        Args:
            x: Input tensor (*, in_features).

        Returns:
            Output tensor (*, out_features).

        Raises:
            ValueError: If input's last dimension doesn't match in_features.
        """
        if x.ndim < 1:
            raise ValueError(f"QLoRALinear expects at least 1D input, got {x.ndim}D")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QLoRALinear expects input with last dimension {self.in_features}, "
                f"got {x.shape[-1]} (shape: {x.shape})"
            )

        # Use cached dequantized base weights for performance
        if self._quantized:
            if self._base_weight_cache is None:
                self._base_weight_cache = self._dequantize_base()
            base_weight = self._base_weight_cache
        else:
            base_weight = mx.zeros((self.out_features, self.in_features))

        # Base forward pass
        y = x @ base_weight.T

        # LoRA forward pass
        lora_x = x
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)
        lora_out = (lora_x @ self.lora_A) @ self.lora_B
        y = y + lora_out * self.scaling

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        group_size: int = 64,
    ) -> "QLoRALinear":
        """Create QLoRA layer from existing linear layer.

        Args:
            linear: Source linear layer.
            rank: LoRA rank.
            alpha: LoRA scaling.
            dropout: Dropout rate.
            group_size: Quantization group size.

        Returns:
            QLoRA layer with quantized base weights.
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, 'bias') and linear.bias is not None

        qlora = cls(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
            group_size=group_size,
        )

        qlora.quantize_base(linear.weight)

        if has_bias:
            qlora.bias = linear.bias

        return qlora


class GPTQLinear(nn.Module):
    """GPTQ: Post-training quantization using optimal brain quantization.

    GPTQ quantizes weights one column at a time, using the inverse Hessian
    to optimally distribute quantization error.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Number of bits (default: 4).
        group_size: Group size for grouped quantization (default: 128).
        bias: Whether to include bias.

    Reference:
        "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
        https://arxiv.org/abs/2210.17323

    Example:
        >>> layer = GPTQLinear.from_linear(linear_layer, calibration_data)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Quantized weights storage
        self.num_groups = (in_features + group_size - 1) // group_size
        self.qweight = mx.zeros((out_features, in_features), dtype=mx.int8)
        self.scales = mx.ones((out_features, self.num_groups))
        self.zeros = mx.zeros((out_features, self.num_groups))

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def quantize(
        self,
        weight: mx.array,
        H: Optional[mx.array] = None,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        """Quantize weights using GPTQ algorithm.

        GPTQ quantizes weights column-by-column (or in blocks), using the
        Hessian inverse to optimally distribute quantization error to
        not-yet-quantized columns. This minimizes the overall output error.

        Args:
            weight: Float weights (out_features, in_features).
            H: Hessian matrix from calibration (in_features, in_features).
                Computed as H = X^T @ X / n where X is calibration data.
            blocksize: Block size for column processing (default: 128).
            percdamp: Damping percentage for Hessian diagonal (default: 0.01).
        """
        W = mx.array(weight)
        out_features, in_features = W.shape

        if H is None:
            # Simple quantization without Hessian (fallback)
            self._simple_quantize(W)
            return

        # Add damping to Hessian diagonal for numerical stability
        damp = percdamp * mx.mean(mx.diag(H))
        H_damped = H + damp * mx.eye(in_features)

        # Compute Cholesky decomposition for efficient inverse operations
        # We need H_inv for error propagation
        try:
            # Use Cholesky: H = L @ L^T, then H^{-1} = L^{-T} @ L^{-1}
            L = mx.linalg.cholesky(H_damped)
            # We'll compute H_inv columns as needed, but for block processing
            # we need the full inverse for the block
            Hinv = mx.linalg.inv(H_damped)
        except Exception:
            # Fallback to simple quantization if Hessian is singular
            self._simple_quantize(W)
            return

        # GPTQ algorithm: process columns in blocks
        # For each block, quantize and propagate error to remaining columns
        qmax = 2 ** self.bits - 1
        eps = _get_epsilon(W.dtype)

        # Initialize quantized output
        Q = mx.zeros_like(W)
        scales_per_col = mx.zeros((out_features, in_features))
        zeros_per_col = mx.zeros((out_features, in_features))

        # Process in blocks for efficiency
        for block_start in range(0, in_features, blocksize):
            block_end = min(block_start + blocksize, in_features)
            block_size = block_end - block_start

            # Get the block of weights to quantize
            W_block = W[:, block_start:block_end]

            # Get relevant portion of Hessian inverse for this block
            Hinv_block = Hinv[block_start:block_end, block_start:block_end]

            # Process each column in the block
            for i in range(block_size):
                col_idx = block_start + i
                w_col = W[:, col_idx]

                # Determine which group this column belongs to
                group_idx = col_idx // self.group_size

                # Get or compute scale/zero for this group
                group_start = group_idx * self.group_size
                group_end = min(group_start + self.group_size, in_features)

                # Compute scale from current (possibly error-adjusted) weights
                w_group = W[:, group_start:group_end]
                w_min = mx.min(w_group, axis=1, keepdims=True)
                w_max = mx.max(w_group, axis=1, keepdims=True)
                scale = (w_max - w_min) / qmax
                scale = mx.maximum(scale, eps)
                zero = -mx.round(w_min / scale)

                # Quantize this column
                q_col = mx.round(w_col / scale.squeeze() + zero.squeeze())
                q_col = mx.clip(q_col, 0, qmax)
                Q[:, col_idx] = q_col

                # Store scale and zero (they may be recomputed per-group later)
                scales_per_col[:, col_idx] = scale.squeeze()
                zeros_per_col[:, col_idx] = zero.squeeze()

                # Dequantize to get actual quantized value
                w_q = (q_col - zero.squeeze()) * scale.squeeze()

                # Compute quantization error
                err = w_col - w_q

                # Propagate error to remaining columns in the block using Hessian
                # This is the key GPTQ insight: distribute error optimally
                if i < block_size - 1:
                    # Error propagation: W[:, j] += err * H_inv[i, j] / H_inv[i, i]
                    # for all j > i in the current block
                    diag = Hinv_block[i, i]
                    if mx.abs(diag) > 1e-10:  # Avoid division by near-zero
                        for j in range(i + 1, block_size):
                            update = err * (Hinv_block[i, j] / diag)
                            W[:, block_start + j] = W[:, block_start + j] + update

            # After processing block, propagate remaining error to future blocks
            if block_end < in_features:
                # Compute total block error and propagate
                W_block_q = (Q[:, block_start:block_end] - zeros_per_col[:, block_start:block_end]) * scales_per_col[:, block_start:block_end]
                block_err = W_block - W_block_q

                # Propagate to remaining columns: W[:, block_end:] += block_err @ Hinv[block, remaining] / diag(Hinv[block, block])
                Hinv_block_diag = mx.diag(Hinv_block)
                # Avoid division by zero
                Hinv_block_diag = mx.where(mx.abs(Hinv_block_diag) > 1e-10, Hinv_block_diag, 1.0)
                normalized_err = block_err / Hinv_block_diag  # (out_features, block_size)

                Hinv_cross = Hinv[block_start:block_end, block_end:]  # (block_size, remaining)
                W[:, block_end:] = W[:, block_end:] + normalized_err @ Hinv_cross

        # Now collect into grouped scales and zeros
        scales_list = []
        zeros_list = []
        qweight_list = []

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, in_features)

            # Use the first column's scale/zero as representative for the group
            # (they should be similar within a group)
            group_scale = scales_per_col[:, start]
            group_zero = zeros_per_col[:, start]

            scales_list.append(group_scale)
            zeros_list.append(group_zero)
            qweight_list.append(Q[:, start:end].astype(mx.int8))

        self.scales = mx.stack(scales_list, axis=1)
        self.zeros = mx.stack(zeros_list, axis=1)
        self.qweight = mx.concatenate(qweight_list, axis=1)

    def _simple_quantize(self, W: mx.array) -> None:
        """Simple grouped quantization without Hessian."""
        scales_list = []
        zeros_list = []
        qweight_list = []

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            group_w = W[:, start:end]

            w_min = mx.min(group_w, axis=1, keepdims=True)
            w_max = mx.max(group_w, axis=1, keepdims=True)

            qmax = 2 ** self.bits - 1
            eps = _get_epsilon(W.dtype)
            scale = (w_max - w_min) / qmax
            scale = mx.maximum(scale, eps)
            zero = -mx.round(w_min / scale)

            Q = mx.round(group_w / scale + zero)
            Q = mx.clip(Q, 0, qmax)

            scales_list.append(scale.squeeze(1))
            zeros_list.append(zero.squeeze(1))
            qweight_list.append(Q.astype(mx.int8))

        self.scales = mx.stack(scales_list, axis=1)
        self.zeros = mx.stack(zeros_list, axis=1)
        self.qweight = mx.concatenate(qweight_list, axis=1)

    def _dequantize(self) -> mx.array:
        """Dequantize weights to float (vectorized)."""
        # Expand scales and zeros to match weight dimensions
        # scales/zeros: (out_features, num_groups) -> (out_features, in_features)
        scale_expanded = mx.repeat(self.scales, self.group_size, axis=1)[:, :self.in_features]
        zero_expanded = mx.repeat(self.zeros, self.group_size, axis=1)[:, :self.in_features]

        # Dequantize: W = (Q - zero) * scale
        return (self.qweight.astype(mx.float32) - zero_expanded) * scale_expanded

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with dequantized weights.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).

        Raises:
            ValueError: If input's last dimension doesn't match in_features.
        """
        if x.ndim < 1:
            raise ValueError(f"GPTQLinear expects at least 1D input, got {x.ndim}D")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"GPTQLinear expects input with last dimension {self.in_features}, "
                f"got {x.shape[-1]} (shape: {x.shape})"
            )

        weight = self._dequantize()
        y = x @ weight.T

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        calibration_data: Optional[mx.array] = None,
        bits: int = 4,
        group_size: int = 128,
    ) -> "GPTQLinear":
        """Create GPTQ layer from linear with optional calibration.

        Args:
            linear: Source linear layer.
            calibration_data: Calibration inputs (batch, seq, in_features).
            bits: Number of bits.
            group_size: Group size.

        Returns:
            GPTQ quantized layer.
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, 'bias') and linear.bias is not None

        gptq = cls(
            in_features,
            out_features,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        # Compute Hessian if calibration data provided
        H = None
        if calibration_data is not None:
            # Flatten calibration data
            X = calibration_data.reshape(-1, in_features)
            # Compute H = X^T X / n
            H = (X.T @ X) / X.shape[0]

        gptq.quantize(linear.weight, H=H)

        if has_bias:
            gptq.bias = linear.bias

        return gptq


class AWQLinear(nn.Module):
    """AWQ: Activation-aware Weight Quantization.

    AWQ protects salient weights based on activation magnitudes,
    achieving better quality than simple round-to-nearest.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Number of bits (default: 4).
        group_size: Group size (default: 128).
        bias: Whether to include bias.

    Reference:
        "AWQ: Activation-aware Weight Quantization for LLM Compression"
        https://arxiv.org/abs/2306.00978

    Example:
        >>> layer = AWQLinear.from_linear(linear_layer, calibration_data)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.num_groups = (in_features + group_size - 1) // group_size
        self.qweight = mx.zeros((out_features, in_features), dtype=mx.int8)
        self.scales = mx.ones((out_features, self.num_groups))
        self.zeros = mx.zeros((out_features, self.num_groups))

        # Per-channel activation scale (learned from calibration)
        self.act_scale = mx.ones((in_features,))

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def quantize(
        self,
        weight: mx.array,
        act_scales: Optional[mx.array] = None,
        w_bit: int = 4,
        auto_scale: bool = True,
        mse_range: int = 100,
    ) -> None:
        """Quantize weights with activation-aware scaling.

        Args:
            weight: Float weights (out_features, in_features).
            act_scales: Per-channel activation scales from calibration.
            w_bit: Number of bits.
            auto_scale: Whether to search for optimal scales.
            mse_range: Range for scale search.
        """
        W = mx.array(weight)

        if act_scales is not None:
            self.act_scale = act_scales

        if auto_scale and act_scales is not None:
            # Search for optimal per-channel scale
            best_scale = self._search_scale(W, act_scales, w_bit, mse_range)
            self.act_scale = best_scale

        # Apply activation scaling to weights
        W_scaled = W * self.act_scale[None, :]

        # Quantize with grouped quantization
        self._quantize_weights(W_scaled)

    def _search_scale(
        self,
        W: mx.array,
        act_scales: mx.array,
        w_bit: int,
        mse_range: int,
    ) -> mx.array:
        """Search for optimal activation scale."""
        best_error = float('inf')
        best_scale = act_scales

        for ratio in range(1, mse_range + 1):
            r = ratio / mse_range
            scale = act_scales ** r

            W_scaled = W * scale[None, :]
            W_quant = self._quantize_dequantize(W_scaled)
            W_restored = W_quant / scale[None, :]

            error = float(mx.mean((W - W_restored) ** 2))
            if error < best_error:
                best_error = error
                best_scale = scale

        return best_scale

    def _quantize_dequantize(self, W: mx.array) -> mx.array:
        """Quantize and dequantize for error measurement."""
        qmax = 2 ** self.bits - 1
        eps = _get_epsilon(W.dtype)
        w_min = mx.min(W)
        w_max = mx.max(W)
        scale = (w_max - w_min) / qmax
        scale = mx.maximum(scale, eps)
        zero = -mx.round(w_min / scale)

        Q = mx.round(W / scale + zero)
        Q = mx.clip(Q, 0, qmax)

        return (Q - zero) * scale

    def _quantize_weights(self, W: mx.array) -> None:
        """Quantize weights with grouped quantization."""
        scales_list = []
        zeros_list = []
        qweight_list = []

        qmax = 2 ** self.bits - 1
        eps = _get_epsilon(W.dtype)

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            group_w = W[:, start:end]

            w_min = mx.min(group_w, axis=1, keepdims=True)
            w_max = mx.max(group_w, axis=1, keepdims=True)

            scale = (w_max - w_min) / qmax
            scale = mx.maximum(scale, eps)
            zero = -mx.round(w_min / scale)

            Q = mx.round(group_w / scale + zero)
            Q = mx.clip(Q, 0, qmax)

            scales_list.append(scale.squeeze(1))
            zeros_list.append(zero.squeeze(1))
            qweight_list.append(Q.astype(mx.int8))

        self.scales = mx.stack(scales_list, axis=1)
        self.zeros = mx.stack(zeros_list, axis=1)
        self.qweight = mx.concatenate(qweight_list, axis=1)

    def _dequantize(self) -> mx.array:
        """Dequantize weights and remove activation scaling (vectorized)."""
        # Expand scales and zeros to match weight dimensions
        scale_expanded = mx.repeat(self.scales, self.group_size, axis=1)[:, :self.in_features]
        zero_expanded = mx.repeat(self.zeros, self.group_size, axis=1)[:, :self.in_features]

        # Dequantize: W = (Q - zero) * scale
        W = (self.qweight.astype(mx.float32) - zero_expanded) * scale_expanded

        # Remove activation scaling
        W = W / self.act_scale[None, :]

        return W

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).

        Raises:
            ValueError: If input's last dimension doesn't match in_features.
        """
        if x.ndim < 1:
            raise ValueError(f"AWQLinear expects at least 1D input, got {x.ndim}D")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"AWQLinear expects input with last dimension {self.in_features}, "
                f"got {x.shape[-1]} (shape: {x.shape})"
            )

        weight = self._dequantize()
        y = x @ weight.T

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        calibration_data: Optional[mx.array] = None,
        bits: int = 4,
        group_size: int = 128,
    ) -> "AWQLinear":
        """Create AWQ layer from linear with calibration.

        Args:
            linear: Source linear layer.
            calibration_data: Calibration inputs for activation scales.
            bits: Number of bits.
            group_size: Group size.

        Returns:
            AWQ quantized layer.
        """
        out_features, in_features = linear.weight.shape
        has_bias = hasattr(linear, 'bias') and linear.bias is not None

        awq = cls(
            in_features,
            out_features,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        # Compute activation scales from calibration data
        act_scales = None
        if calibration_data is not None:
            X = calibration_data.reshape(-1, in_features)
            act_scales = mx.mean(mx.abs(X), axis=0)
            act_scales = mx.maximum(act_scales, mx.array(1e-8))

        awq.quantize(linear.weight, act_scales=act_scales)

        if has_bias:
            awq.bias = linear.bias

        return awq

# =============================================================================
# Gradient Checkpointing Utilities
# =============================================================================


def enable_gradient_checkpointing(
    layer: nn.Module,
    layer_types: Optional[Tuple[Type[nn.Module], ...]] = None,
) -> None:
    """Enable gradient checkpointing on a quantized layer or model.

    Gradient checkpointing trades compute for memory by recomputing activations
    during the backward pass instead of storing them. This is particularly useful
    for training with quantized models where memory is constrained.

    This function modifies the layer's __call__ method in-place to use
    mx.checkpoint for gradient computation.

    Args:
        layer: The layer or model to enable checkpointing on.
        layer_types: Optional tuple of layer types to apply checkpointing to.
            If None, defaults to (QLoRALinear,) which are the trainable
            quantized layers. Pass (Int4Linear, QLoRALinear, GPTQLinear,
            AWQLinear) to checkpoint all quantized linear layers.

    Raises:
        RuntimeError: If mx.checkpoint is not available (MLX version too old).

    Example:
        >>> # Enable checkpointing on a single QLoRA layer
        >>> qlora_layer = QLoRALinear.from_linear(linear, rank=8)
        >>> enable_gradient_checkpointing(qlora_layer)

        >>> # Enable checkpointing on all QLoRA layers in a model
        >>> for name, module in model.named_modules():
        ...     if isinstance(module, QLoRALinear):
        ...         enable_gradient_checkpointing(module)

        >>> # Using the convenience function for entire models
        >>> enable_model_gradient_checkpointing(model)

    Note:
        - Checkpointing is most beneficial for QLoRALinear during fine-tuning
        - For inference-only layers (Int4Linear, GPTQLinear, AWQLinear),
          checkpointing has no effect since no gradients are computed
        - Checkpointing adds ~30% compute overhead but can reduce memory
          significantly for large batch sizes or sequence lengths
    """
    if not _HAS_CHECKPOINT:
        raise RuntimeError(
            "mx.checkpoint is not available. Gradient checkpointing requires "
            "MLX >= 0.1.0. Please upgrade: pip install -U mlx"
        )

    if layer_types is None:
        layer_types = (QLoRALinear,)

    # Get the original __call__ method for this layer's type
    layer_type = type(layer)
    original_call = layer_type.__call__

    def checkpointed_call(self, *args, **kwargs):
        """Checkpointed forward pass that recomputes activations during backprop."""

        def inner_fn(params, *args, **kwargs):
            # Temporarily update trainable parameters
            self.update(params)
            return original_call(self, *args, **kwargs)

        # Use mx.checkpoint to wrap the forward pass
        # This saves inputs and recomputes outputs during backward pass
        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    # Only apply checkpointing if this layer is of a target type
    if isinstance(layer, layer_types):
        # Bind the checkpointed call to this specific instance
        import types
        layer.__call__ = types.MethodType(
            lambda self, *args, **kwargs: checkpointed_call(self, *args, **kwargs),
            layer,
        )


def enable_model_gradient_checkpointing(
    model: nn.Module,
    layer_types: Optional[Tuple[Type[nn.Module], ...]] = None,
) -> int:
    """Enable gradient checkpointing on all quantized layers in a model.

    This is a convenience function that walks through a model and enables
    gradient checkpointing on all layers of the specified types.

    Args:
        model: The model to enable checkpointing on.
        layer_types: Optional tuple of layer types to checkpoint.
            Defaults to (QLoRALinear,) for fine-tuning scenarios.

    Returns:
        Number of layers that had checkpointing enabled.

    Example:
        >>> model = load_quantized_model(...)
        >>> num_checkpointed = enable_model_gradient_checkpointing(model)
        >>> print(f"Enabled checkpointing on {num_checkpointed} layers")

    Note:
        For models with many QLoRA layers, this can significantly reduce
        peak memory usage during training, allowing larger batch sizes.
    """
    if not _HAS_CHECKPOINT:
        raise RuntimeError(
            "mx.checkpoint is not available. Gradient checkpointing requires "
            "MLX >= 0.1.0. Please upgrade: pip install -U mlx"
        )

    if layer_types is None:
        layer_types = (QLoRALinear,)

    count = 0

    def apply_checkpointing(module: nn.Module, prefix: str = "") -> None:
        nonlocal count

        # Check if this module is a target type
        if isinstance(module, layer_types):
            enable_gradient_checkpointing(module, layer_types)
            count += 1

        # Recursively process children
        if hasattr(module, "__dict__"):
            for name, child in module.__dict__.items():
                if isinstance(child, nn.Module):
                    apply_checkpointing(child, f"{prefix}.{name}" if prefix else name)
                elif isinstance(child, list):
                    for i, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            apply_checkpointing(
                                item, f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                            )

    apply_checkpointing(model)
    return count


def disable_gradient_checkpointing(layer: nn.Module) -> None:
    """Disable gradient checkpointing on a layer.

    Restores the original __call__ method if checkpointing was enabled.

    Args:
        layer: The layer to disable checkpointing on.
    """
    # Remove instance-level __call__ override, falling back to class method
    if hasattr(layer, "__call__") and "__call__" in layer.__dict__:
        delattr(layer, "__call__")
