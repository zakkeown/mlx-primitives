"""INT4 quantization generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class Int4LinearGenerator(GoldenGenerator):
    """Generate golden files for INT4 quantized linear layer.

    Uses group-wise quantization with specified group size.
    Common in GPTQ, AWQ, and other 4-bit quantization methods.
    """

    @property
    def name(self) -> str:
        return "int4_linear"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["quantization_int4"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "int4_linear_g128", "batch": 2, "seq": 64, "in_features": 256, "out_features": 512, "group_size": 128},
            {"name": "int4_linear_g64", "batch": 2, "seq": 64, "in_features": 256, "out_features": 512, "group_size": 64},
            {"name": "int4_linear_g32", "batch": 2, "seq": 128, "in_features": 512, "out_features": 1024, "group_size": 32},
            {"name": "int4_linear_large", "batch": 4, "seq": 256, "in_features": 1024, "out_features": 2048, "group_size": 128},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        in_features = config["in_features"]
        out_features = config["out_features"]
        group_size = config["group_size"]

        # Generate input and weights
        x = torch.randn(batch, seq, in_features)
        weight = torch.randn(out_features, in_features) * 0.02

        # Full precision output (reference)
        fp_out = x @ weight.T

        # Group-wise INT4 quantization
        num_groups = in_features // group_size
        weight_grouped = weight.view(out_features, num_groups, group_size)

        # Compute scales and zeros per group (symmetric quantization)
        # For INT4: range is [-8, 7]
        qmax = 7
        scales = weight_grouped.abs().amax(dim=2, keepdim=True) / qmax

        # Quantize
        w_quant = torch.clamp(torch.round(weight_grouped / scales), -8, 7)

        # Dequantize for computation
        w_dequant = (w_quant * scales).view(out_features, in_features)

        # INT4 output (approximate)
        int4_out = x @ w_dequant.T

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "weight": weight.numpy(),
            },
            params={"group_size": group_size},
            expected_outputs={
                "fp_out": fp_out.numpy(),
                "int4_out": int4_out.numpy(),
                "w_quant": w_quant.numpy().astype(np.int8),
                "scales": scales.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "in_features": in_features,
                "out_features": out_features,
                "num_groups": num_groups,
            },
        )
