"""Quantization utilities for MLX.

This module provides quantization tools:
- Dynamic quantization (per-tensor and per-channel)
- Quantized linear layers
- Weight-only quantization
- Calibration utilities
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


def quantize_tensor(
    x: mx.array,
    num_bits: int = 8,
    per_channel: bool = False,
    symmetric: bool = True,
) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """Quantize a tensor to lower precision.

    Args:
        x: Input tensor to quantize.
        num_bits: Number of bits (4, 8, etc.).
        per_channel: If True, compute scale per output channel.
        symmetric: If True, use symmetric quantization (no zero point).

    Returns:
        Tuple of (quantized_tensor, scale, zero_point).
        zero_point is None for symmetric quantization.
    """
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

    if symmetric:
        # Symmetric: scale = max(|min|, |max|) / qmax
        x_absmax = mx.maximum(mx.abs(x_min), mx.abs(x_max))
        scale = x_absmax / qmax
        scale = mx.maximum(scale, mx.array(1e-8))  # Avoid division by zero

        x_q = mx.round(x / scale)
        x_q = mx.clip(x_q, qmin, qmax).astype(mx.int8)

        return x_q, scale, None
    else:
        # Asymmetric: compute zero point
        scale = (x_max - x_min) / (qmax - qmin)
        scale = mx.maximum(scale, mx.array(1e-8))

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

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def quantize_weights(self) -> None:
        """Quantize the weights."""
        self.weight_q, self.weight_scale, self.weight_zero_point = quantize_tensor(
            self._weight_float,
            num_bits=self.num_bits,
            per_channel=self.per_channel,
        )
        self._quantized = True

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor (*, in_features).

        Returns:
            Output tensor (*, out_features).
        """
        if not self._quantized:
            # Use float weights if not quantized yet
            weight = self._weight_float
        else:
            # Dequantize weights
            weight = dequantize_tensor(
                self.weight_q,
                self.weight_scale,
                self.weight_zero_point,
            )

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
    """

    def __init__(
        self,
        method: str = "minmax",
        percentile: float = 99.99,
    ):
        self.method = method
        self.percentile = percentile

        self.min_vals: List[float] = []
        self.max_vals: List[float] = []
        self.all_vals: List[mx.array] = []

    def observe(self, x: mx.array) -> None:
        """Record activation statistics.

        Args:
            x: Activation tensor to observe.
        """
        self.min_vals.append(float(mx.min(x)))
        self.max_vals.append(float(mx.max(x)))

        if self.method == "percentile":
            self.all_vals.append(mx.abs(x).flatten())

    def compute_scale(self, num_bits: int = 8) -> Tuple[float, float]:
        """Compute quantization scale from collected statistics.

        Args:
            num_bits: Number of bits.

        Returns:
            Tuple of (scale, zero_point).
        """
        qmax = 2 ** (num_bits - 1) - 1

        if self.method == "minmax":
            x_min = min(self.min_vals)
            x_max = max(self.max_vals)
            x_absmax = max(abs(x_min), abs(x_max))
            scale = x_absmax / qmax
            return scale, 0.0

        elif self.method == "percentile":
            # Concatenate all observed values
            all_vals = mx.concatenate(self.all_vals)
            # Sort and take percentile
            sorted_vals = mx.sort(all_vals)
            idx = int(len(sorted_vals) * self.percentile / 100)
            x_absmax = float(sorted_vals[idx])
            scale = x_absmax / qmax
            return scale, 0.0

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def reset(self) -> None:
        """Reset collected statistics."""
        self.min_vals = []
        self.max_vals = []
        self.all_vals = []


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

    def pack_weights(self, weight: mx.array) -> None:
        """Pack float weights into int4.

        Args:
            weight: Float weight tensor (out_features, in_features).
        """
        out_features, in_features = weight.shape

        # Quantize per group
        scales = []
        packed_rows = []

        for i in range(out_features):
            row = weight[i]
            packed_row = []

            for g in range(self.num_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                group = row[start:end]

                # Compute scale for this group
                absmax = mx.max(mx.abs(group))
                scale = absmax / 7.0  # int4 range is -8 to 7
                scale = mx.maximum(scale, mx.array(1e-8))
                scales.append(float(scale))

                # Quantize group
                q_group = mx.round(group / scale)
                q_group = mx.clip(q_group, -8, 7).astype(mx.int8)

                packed_row.extend(q_group.tolist())

            packed_rows.append(packed_row)

        # Reshape scales
        self.scales = mx.array(scales).reshape(out_features, self.num_groups)

        # Pack pairs of int4 into uint8
        packed = []
        for row in packed_rows:
            packed_row = []
            for j in range(0, len(row), 2):
                low = row[j] & 0xF
                high = (row[j + 1] & 0xF) if j + 1 < len(row) else 0
                packed_row.append((high << 4) | low)
            packed.append(packed_row)

        self.weight_packed = mx.array(packed, dtype=mx.uint8)

    def unpack_weights(self) -> mx.array:
        """Unpack int4 weights to float.

        Returns:
            Float weight tensor.
        """
        out_features = self.weight_packed.shape[0]
        packed_list = self.weight_packed.tolist()

        unpacked_rows = []
        for i, packed_row in enumerate(packed_list):
            unpacked = []
            for byte in packed_row:
                low = byte & 0xF
                high = (byte >> 4) & 0xF
                # Convert from unsigned to signed
                low = low - 16 if low > 7 else low
                high = high - 16 if high > 7 else high
                unpacked.extend([low, high])

            # Trim to in_features
            unpacked = unpacked[: self.in_features]

            # Dequantize using scales
            row = []
            for g in range(self.num_groups):
                start = g * self.group_size
                end = min(start + self.group_size, self.in_features)
                scale = float(self.scales[i, g])

                for j in range(start, end):
                    row.append(unpacked[j] * scale)

            unpacked_rows.append(row)

        return mx.array(unpacked_rows)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Unpack and dequantize weights
        weight = self.unpack_weights()

        # Matrix multiply
        y = x @ weight.T

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

    def quantize_base(self, weight: mx.array) -> None:
        """Quantize base weights to NF4.

        Args:
            weight: Float weight tensor (out_features, in_features).
        """
        out_features, in_features = weight.shape
        scales = []
        packed_rows = []

        for i in range(out_features):
            row = weight[i]
            packed_row = []

            for g in range(self.num_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                group = row[start:end]

                # NF4 quantization with absmax scaling
                absmax = mx.max(mx.abs(group))
                scale = absmax / 7.0
                scale = mx.maximum(scale, mx.array(1e-8))
                scales.append(float(scale))

                # Quantize to int4
                q_group = mx.round(group / scale)
                q_group = mx.clip(q_group, -8, 7).astype(mx.int8)
                packed_row.extend(q_group.tolist())

            packed_rows.append(packed_row)

        self.base_scales = mx.array(scales).reshape(out_features, self.num_groups)

        # Pack int4 pairs into uint8
        packed = []
        for row in packed_rows:
            packed_row = []
            for j in range(0, len(row), 2):
                low = row[j] & 0xF
                high = (row[j + 1] & 0xF) if j + 1 < len(row) else 0
                packed_row.append((high << 4) | low)
            packed.append(packed_row)

        self.base_weight_packed = mx.array(packed, dtype=mx.uint8)
        self._quantized = True

    def _dequantize_base(self) -> mx.array:
        """Dequantize base weights to float."""
        out_features = self.base_weight_packed.shape[0]
        packed_list = self.base_weight_packed.tolist()

        unpacked_rows = []
        for i, packed_row in enumerate(packed_list):
            unpacked = []
            for byte in packed_row:
                low = byte & 0xF
                high = (byte >> 4) & 0xF
                low = low - 16 if low > 7 else low
                high = high - 16 if high > 7 else high
                unpacked.extend([low, high])

            unpacked = unpacked[:self.in_features]

            row = []
            for g in range(self.num_groups):
                start = g * self.group_size
                end = min(start + self.group_size, self.in_features)
                scale = float(self.base_scales[i, g])

                for j in range(start, end):
                    row.append(unpacked[j] * scale)

            unpacked_rows.append(row)

        return mx.array(unpacked_rows)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized base and LoRA adapters.

        Args:
            x: Input tensor (*, in_features).

        Returns:
            Output tensor (*, out_features).
        """
        # Dequantize base weights
        if self._quantized:
            base_weight = self._dequantize_base()
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

        Args:
            weight: Float weights (out_features, in_features).
            H: Hessian matrix from calibration (in_features, in_features).
            blocksize: Block size for processing.
            percdamp: Damping percentage for Hessian.
        """
        W = mx.array(weight)
        out_features, in_features = W.shape

        if H is None:
            # Simple quantization without Hessian (fallback)
            self._simple_quantize(W)
            return

        # Add damping to Hessian
        damp = percdamp * mx.mean(mx.diag(H))
        H = H + damp * mx.eye(in_features)

        # Compute Hessian inverse
        try:
            Hinv = mx.linalg.inv(H)
        except Exception:
            # Fallback to simple quantization
            self._simple_quantize(W)
            return

        scales_list = []
        zeros_list = []
        qweight_list = []

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, in_features)
            group_w = W[:, start:end]

            # Compute scale and zero point for this group
            w_min = mx.min(group_w, axis=1, keepdims=True)
            w_max = mx.max(group_w, axis=1, keepdims=True)

            qmax = 2 ** self.bits - 1
            scale = (w_max - w_min) / qmax
            scale = mx.maximum(scale, mx.array(1e-8))
            zero = -mx.round(w_min / scale)

            # Quantize with Hessian-guided error compensation
            Q = mx.round(group_w / scale + zero)
            Q = mx.clip(Q, 0, qmax)

            scales_list.append(scale.squeeze(1))
            zeros_list.append(zero.squeeze(1))
            qweight_list.append(Q.astype(mx.int8))

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
            scale = (w_max - w_min) / qmax
            scale = mx.maximum(scale, mx.array(1e-8))
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
        """Dequantize weights to float."""
        W = mx.zeros((self.out_features, self.in_features))

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            scale = self.scales[:, g:g+1]
            zero = self.zeros[:, g:g+1]
            q = self.qweight[:, start:end].astype(mx.float32)

            W = mx.concatenate([
                W[:, :start],
                (q - zero) * scale,
                W[:, end:]
            ], axis=1) if start > 0 else mx.concatenate([
                (q - zero) * scale,
                W[:, end:]
            ], axis=1) if end < self.in_features else (q - zero) * scale

        return W

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with dequantized weights.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
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
        w_min = mx.min(W)
        w_max = mx.max(W)
        scale = (w_max - w_min) / qmax
        scale = mx.maximum(scale, mx.array(1e-8))
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

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            group_w = W[:, start:end]

            w_min = mx.min(group_w, axis=1, keepdims=True)
            w_max = mx.max(group_w, axis=1, keepdims=True)

            scale = (w_max - w_min) / qmax
            scale = mx.maximum(scale, mx.array(1e-8))
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
        """Dequantize weights and remove activation scaling."""
        rows = []
        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            scale = self.scales[:, g:g+1]
            zero = self.zeros[:, g:g+1]
            q = self.qweight[:, start:end].astype(mx.float32)
            rows.append((q - zero) * scale)

        W = mx.concatenate(rows, axis=1)

        # Remove activation scaling
        W = W / self.act_scale[None, :]

        return W

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
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