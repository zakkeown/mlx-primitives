"""Adaptive Layer Normalization generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class AdaLayerNormGenerator(GoldenGenerator):
    """Generate golden files for Adaptive Layer Normalization.

    AdaLayerNorm applies conditioning-dependent scale and shift:
    out = norm(x) * (1 + scale) + shift
    where scale and shift are derived from conditioning input.

    Used in diffusion models like DiT and SDXL.
    """

    @property
    def name(self) -> str:
        return "adalayernorm"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization_adaptive"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        ada_configs = [
            {"dims": 256, "cond_dims": 128, "name": "small"},
            {"dims": 512, "cond_dims": 256, "name": "medium"},
            {"dims": 1024, "cond_dims": 512, "name": "large"},
        ]

        for cfg in ada_configs:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"adalayernorm_{cfg['name']}_{size_name}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        **cfg,
                    }
                )

        # Test with 2D input (no sequence dimension)
        configs.append(
            {
                "name": "adalayernorm_2d",
                "batch": 4,
                "dims": 256,
                "cond_dims": 128,
                "input_2d": True,
            }
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        dims = config["dims"]
        cond_dims = config["cond_dims"]
        input_2d = config.get("input_2d", False)
        eps = 1e-6

        if input_2d:
            x = torch.randn(batch, dims)
        else:
            seq = config["seq"]
            x = torch.randn(batch, seq, dims)

        # Conditioning input
        cond = torch.randn(batch, cond_dims)

        # AdaLN projection: cond -> scale, shift
        proj_weight = torch.randn(dims * 2, cond_dims) * 0.02
        proj_bias = torch.zeros(dims * 2)

        # LayerNorm parameters
        ln_weight = torch.ones(dims)
        ln_bias = torch.zeros(dims)

        # Forward pass
        # 1. Apply LayerNorm (without affine, then apply our own)
        x_norm = F.layer_norm(x, [dims], eps=eps)
        x_norm = x_norm * ln_weight + ln_bias

        # 2. Project conditioning to get scale and shift
        cond_proj = cond @ proj_weight.T + proj_bias
        scale, shift = cond_proj.chunk(2, dim=-1)

        # 3. Apply adaptive modulation
        if input_2d:
            out = x_norm * (1 + scale) + shift
        else:
            # scale/shift are (batch, dims), need to broadcast over seq
            out = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "cond": cond.numpy(),
                "proj_weight": proj_weight.numpy(),
                "proj_bias": proj_bias.numpy(),
                "ln_weight": ln_weight.numpy(),
                "ln_bias": ln_bias.numpy(),
            },
            params={"eps": eps},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "dims": dims,
                "cond_dims": cond_dims,
                "input_2d": input_2d,
            },
        )
