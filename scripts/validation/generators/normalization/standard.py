"""Standard normalization generators: RMSNorm, GroupNorm."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class RMSNormGenerator(GoldenGenerator):
    """Generate golden files for RMS Normalization.

    RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    """

    @property
    def name(self) -> str:
        return "rmsnorm"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"rmsnorm_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "dims": shape["dims"],
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "rmsnorm_zeros",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "zeros",
                },
                {
                    "name": "rmsnorm_small_values",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "small",
                },
                {
                    "name": "rmsnorm_large_values",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "large",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        input_type = config.get("input_type", "random")
        eps = 1e-6

        shape = (batch, seq, dims)

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "small":
            x = torch.full(shape, 1e-7)
        elif input_type == "large":
            x = torch.full(shape, 1e4)
        else:
            x = torch.randn(shape)

        weight = torch.ones(dims)

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        out = x / rms * weight

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy(), "weight": weight.numpy()},
            params={"eps": eps},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "seq": seq, "dims": dims, "input_type": input_type},
        )


class GroupNormGenerator(GoldenGenerator):
    """Generate golden files for Group Normalization."""

    @property
    def name(self) -> str:
        return "groupnorm"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different num_groups configurations
        gn_configs = [
            {"channels": 32, "num_groups": 8, "name": "c32_g8"},
            {"channels": 64, "num_groups": 8, "name": "c64_g8"},
            {"channels": 128, "num_groups": 16, "name": "c128_g16"},
            {"channels": 256, "num_groups": 32, "name": "c256_g32"},
        ]

        for gn_cfg in gn_configs:
            # 2D input (images)
            configs.append(
                {
                    "name": f"groupnorm_{gn_cfg['name']}_2d",
                    "batch": 2,
                    "height": 16,
                    "width": 16,
                    "channels": gn_cfg["channels"],
                    "num_groups": gn_cfg["num_groups"],
                    "input_format": "2d",
                }
            )
            # 1D input (sequences)
            configs.append(
                {
                    "name": f"groupnorm_{gn_cfg['name']}_1d",
                    "batch": 2,
                    "seq": 64,
                    "channels": gn_cfg["channels"],
                    "num_groups": gn_cfg["num_groups"],
                    "input_format": "1d",
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        num_groups = config["num_groups"]
        input_format = config["input_format"]
        eps = 1e-5

        if input_format == "2d":
            height = config["height"]
            width = config["width"]
            x = torch.randn(batch, channels, height, width)
        else:
            seq = config["seq"]
            # For 1D, PyTorch GroupNorm expects (N, C, L)
            x = torch.randn(batch, channels, seq)

        # Create GroupNorm with default weight=1, bias=0
        gn = nn.GroupNorm(num_groups, channels, eps=eps)
        with torch.no_grad():
            gn.weight.fill_(1.0)
            gn.bias.fill_(0.0)

        out = gn(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"num_groups": num_groups, "eps": eps},
            expected_outputs={"out": out.detach().numpy()},
            metadata={
                "batch": batch,
                "channels": channels,
                "input_format": input_format,
            },
        )
