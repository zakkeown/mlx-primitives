"""QK Normalization generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class QKNormGenerator(GoldenGenerator):
    """Generate golden files for QK Normalization.

    QKNorm applies RMSNorm to Q and K tensors before attention,
    with separate learnable scales for each.

    Used in Stable Diffusion 3 and other diffusion models.
    """

    @property
    def name(self) -> str:
        return "qknorm"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"qknorm_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "heads": shape["heads"],
                    "head_dim": shape["head_dim"],
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "qknorm_single_head",
                    "batch": 2,
                    "seq": 64,
                    "heads": 1,
                    "head_dim": 64,
                },
                {
                    "name": "qknorm_many_heads",
                    "batch": 2,
                    "seq": 64,
                    "heads": 32,
                    "head_dim": 64,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        eps = 1e-6

        # Q and K tensors in attention format
        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)

        # Learnable scales for Q and K
        q_scale = torch.ones(head_dim)
        k_scale = torch.ones(head_dim)

        # QKNorm: RMSNorm on each head
        def rms_norm(x, scale, eps):
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
            return (x / rms) * scale

        q_norm = rms_norm(q, q_scale, eps)
        k_norm = rms_norm(k, k_scale, eps)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "q_scale": q_scale.numpy(),
                "k_scale": k_scale.numpy(),
            },
            params={"eps": eps},
            expected_outputs={
                "q_norm": q_norm.numpy(),
                "k_norm": k_norm.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
