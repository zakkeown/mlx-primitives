"""Specialized pooling generators: GeM, SPP, GlobalAttentionPooling."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class GeMGenerator(GoldenGenerator):
    """Generate golden files for Generalized Mean (GeM) Pooling.

    GeM: (mean(x^p))^(1/p)
    When p=1, equivalent to average pooling.
    When p->inf, approaches max pooling.
    """

    @property
    def name(self) -> str:
        return "gem"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different p values
        for p in [1.0, 2.0, 3.0, 4.0]:
            configs.append({
                "name": f"gem_p{p:.0f}",
                "batch": 2,
                "channels": 64,
                "height": 32,
                "width": 32,
                "p": p,
            })

        # Learnable p (test with common values)
        configs.extend([
            {"name": "gem_learnable_p3", "batch": 2, "channels": 128, "height": 16, "width": 16, "p": 3.0},
            {"name": "gem_large", "batch": 4, "channels": 256, "height": 64, "width": 64, "p": 3.0},
        ])

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        p = config["p"]
        eps = 1e-6

        # GeM expects positive inputs, use abs to ensure positivity
        x = torch.randn(batch, channels, height, width).abs() + eps

        # GeM formula: (mean(x^p))^(1/p)
        x_pow = x.pow(p)
        pooled = F.adaptive_avg_pool2d(x_pow, 1)
        out = pooled.pow(1.0 / p)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"p": p, "eps": eps},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "height": height, "width": width},
        )


class SpatialPyramidPoolingGenerator(GoldenGenerator):
    """Generate golden files for Spatial Pyramid Pooling (SPP).

    SPP pools at multiple scales and concatenates results.
    Common in detection models for handling variable input sizes.
    """

    @property
    def name(self) -> str:
        return "spp"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "spp_standard", "batch": 2, "channels": 64, "height": 32, "width": 32, "levels": [1, 2, 4]},
            {"name": "spp_fine", "batch": 2, "channels": 64, "height": 64, "width": 64, "levels": [1, 2, 4, 8]},
            {"name": "spp_coarse", "batch": 2, "channels": 128, "height": 16, "width": 16, "levels": [1, 2]},
            {"name": "spp_asymmetric", "batch": 2, "channels": 64, "height": 64, "width": 32, "levels": [1, 2, 4]},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        levels = config["levels"]

        x = torch.randn(batch, channels, height, width)

        # SPP: Pool at each level and concatenate
        pyramid = []
        for level in levels:
            pooled = F.adaptive_avg_pool2d(x, level)  # (B, C, level, level)
            pooled = pooled.view(batch, channels, -1)  # (B, C, level*level)
            pyramid.append(pooled)

        out = torch.cat(pyramid, dim=-1)  # (B, C, sum(level^2))

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"levels": levels},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "height": height, "width": width},
        )


class GlobalAttentionPoolingGenerator(GoldenGenerator):
    """Generate golden files for Global Attention Pooling.

    Learns attention weights over spatial/sequence positions:
    out = sum(softmax(attention_scores) * features)
    """

    @property
    def name(self) -> str:
        return "global_attention_pooling"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "global_attn_pool_small", "batch": 2, "seq": 64, "dims": 128},
            {"name": "global_attn_pool_medium", "batch": 2, "seq": 256, "dims": 256},
            {"name": "global_attn_pool_large", "batch": 4, "seq": 512, "dims": 512},
            {"name": "global_attn_pool_long", "batch": 2, "seq": 1024, "dims": 256},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]

        x = torch.randn(batch, seq, dims)

        # Attention weights: project to scalar per position
        attn_weight = torch.randn(dims, 1) * 0.02
        attn_bias = torch.zeros(1)

        # Compute attention scores
        scores = x @ attn_weight + attn_bias  # (B, seq, 1)
        attn = F.softmax(scores, dim=1)  # (B, seq, 1)

        # Weighted sum
        out = (attn * x).sum(dim=1)  # (B, dims)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "attn_weight": attn_weight.numpy(),
                "attn_bias": attn_bias.numpy(),
            },
            params={},
            expected_outputs={"out": out.numpy(), "attention": attn.numpy()},
            metadata={"batch": batch, "seq": seq, "dims": dims},
        )
