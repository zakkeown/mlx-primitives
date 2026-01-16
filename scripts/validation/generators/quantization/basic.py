"""Basic quantization generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class QuantizeDequantizeGenerator(GoldenGenerator):
    """Generate golden files for basic quantize/dequantize operations.

    Symmetric quantization: q = round(x / scale), x_approx = q * scale
    where scale = max(|x|) / (2^(bits-1) - 1)
    """

    @property
    def name(self) -> str:
        return "quantize_dequantize"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["quantization_int8"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "quant_int8_small", "shape": (64, 128), "bits": 8},
            {"name": "quant_int8_medium", "shape": (256, 512), "bits": 8},
            {"name": "quant_int8_large", "shape": (1024, 2048), "bits": 8},
            {"name": "quant_int4_small", "shape": (64, 128), "bits": 4},
            {"name": "quant_int4_medium", "shape": (256, 512), "bits": 4},
            {"name": "quant_uniform", "shape": (256, 256), "bits": 8, "distribution": "uniform"},
            {"name": "quant_sparse", "shape": (256, 256), "bits": 8, "distribution": "sparse"},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = tuple(config["shape"])
        bits = config["bits"]
        distribution = config.get("distribution", "normal")

        # Generate input tensor with specified distribution
        if distribution == "uniform":
            x = torch.rand(shape) * 2 - 1  # [-1, 1]
        elif distribution == "sparse":
            x = torch.randn(shape)
            mask = torch.rand(shape) > 0.8
            x = x * mask.float()
        else:
            x = torch.randn(shape)

        # Compute quantization scale (per-tensor symmetric)
        qmax = 2 ** (bits - 1) - 1
        scale = x.abs().max() / qmax

        # Quantize
        q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)

        # Dequantize
        x_dequant = q * scale

        # Compute quantization error
        quant_error = (x - x_dequant).abs()

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"bits": bits},
            expected_outputs={
                "quantized": q.numpy().astype(np.int8 if bits == 8 else np.int8),
                "dequantized": x_dequant.numpy(),
                "scale": scale.numpy(),
            },
            metadata={
                "shape": shape,
                "bits": bits,
                "max_error": quant_error.max().item(),
                "mean_error": quant_error.mean().item(),
            },
        )


class Int8LinearGenerator(GoldenGenerator):
    """Generate golden files for INT8 quantized linear layer.

    W8A8: 8-bit weights and 8-bit activations.
    """

    @property
    def name(self) -> str:
        return "int8_linear"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["quantization_int8"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "int8_linear_small", "batch": 2, "seq": 64, "in_features": 256, "out_features": 512},
            {"name": "int8_linear_medium", "batch": 2, "seq": 128, "in_features": 512, "out_features": 1024},
            {"name": "int8_linear_large", "batch": 4, "seq": 256, "in_features": 1024, "out_features": 2048},
            {"name": "int8_linear_square", "batch": 2, "seq": 64, "in_features": 512, "out_features": 512},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        in_features = config["in_features"]
        out_features = config["out_features"]

        # Generate input and weights
        x = torch.randn(batch, seq, in_features)
        weight = torch.randn(out_features, in_features) * 0.02

        # Full precision output (reference)
        fp_out = x @ weight.T

        # Quantize weight (per-output-channel)
        w_scales = weight.abs().amax(dim=1, keepdim=True) / 127
        w_quant = torch.clamp(torch.round(weight / w_scales), -128, 127)
        w_dequant = w_quant * w_scales

        # Quantize activation (per-token)
        x_flat = x.view(-1, in_features)
        x_scales = x_flat.abs().amax(dim=1, keepdim=True) / 127
        x_quant = torch.clamp(torch.round(x_flat / x_scales), -128, 127)
        x_dequant = x_quant * x_scales
        x_dequant = x_dequant.view(batch, seq, in_features)

        # INT8 output (approximate)
        int8_out = x_dequant @ w_dequant.T

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "weight": weight.numpy(),
            },
            params={},
            expected_outputs={
                "fp_out": fp_out.numpy(),
                "int8_out": int8_out.numpy(),
                "w_quant": w_quant.numpy().astype(np.int8),
                "w_scales": w_scales.numpy(),
                "x_scales": x_scales.view(batch, seq, 1).numpy(),
            },
            metadata={"batch": batch, "seq": seq, "in_features": in_features, "out_features": out_features},
        )
