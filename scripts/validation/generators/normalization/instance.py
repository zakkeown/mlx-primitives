"""Instance normalization generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class InstanceNormGenerator(GoldenGenerator):
    """Generate golden files for Instance Normalization.

    InstanceNorm normalizes each sample across spatial dimensions independently.
    """

    @property
    def name(self) -> str:
        return "instancenorm"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # 2D configurations (images)
        in_configs = [
            {"channels": 32, "height": 16, "width": 16, "name": "small"},
            {"channels": 64, "height": 32, "width": 32, "name": "medium"},
            {"channels": 128, "height": 64, "width": 64, "name": "large"},
        ]

        for cfg in in_configs:
            # Without affine
            configs.append(
                {
                    "name": f"instancenorm_{cfg['name']}_no_affine",
                    "batch": 2,
                    **cfg,
                    "affine": False,
                }
            )
            # With affine
            configs.append(
                {
                    "name": f"instancenorm_{cfg['name']}_affine",
                    "batch": 2,
                    **cfg,
                    "affine": True,
                }
            )

        # 1D configurations (sequences)
        configs.extend(
            [
                {
                    "name": "instancenorm_1d_small",
                    "batch": 2,
                    "channels": 64,
                    "length": 128,
                    "input_dim": 1,
                    "affine": True,
                },
                {
                    "name": "instancenorm_1d_medium",
                    "batch": 2,
                    "channels": 128,
                    "length": 256,
                    "input_dim": 1,
                    "affine": True,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        affine = config["affine"]
        input_dim = config.get("input_dim", 2)
        eps = 1e-5

        if input_dim == 1:
            length = config["length"]
            x = torch.randn(batch, channels, length)
            norm = nn.InstanceNorm1d(channels, eps=eps, affine=affine)
        else:
            height = config["height"]
            width = config["width"]
            x = torch.randn(batch, channels, height, width)
            norm = nn.InstanceNorm2d(channels, eps=eps, affine=affine)

        if affine:
            with torch.no_grad():
                norm.weight.fill_(1.0)
                norm.bias.fill_(0.0)

        out = norm(x)

        inputs = {"x": x.numpy()}
        if affine:
            inputs["weight"] = norm.weight.detach().numpy()
            inputs["bias"] = norm.bias.detach().numpy()

        return TestConfig(
            name=config["name"],
            inputs=inputs,
            params={"eps": eps, "affine": affine},
            expected_outputs={"out": out.detach().numpy()},
            metadata={
                "batch": batch,
                "channels": channels,
                "input_dim": input_dim,
            },
        )
