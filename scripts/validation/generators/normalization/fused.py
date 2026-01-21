"""Fused normalization generators: FusedRMSNormLinear."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class FusedRMSNormLinearGenerator(GoldenGenerator):
    """Generate golden files for fused RMSNorm + Linear.

    FusedRMSNormLinear: Linear(RMSNorm(x))
    Computes: (x / sqrt(mean(x^2) + eps) * norm_weight) @ linear_weight.T + bias

    This fused operation avoids materializing the intermediate normalized tensor,
    reducing memory bandwidth for large models.
    """

    @property
    def name(self) -> str:
        return "fused_rmsnorm_linear"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["normalization"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            # Without bias
            configs.append(
                {
                    "name": f"fused_rmsnorm_linear_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "hidden": shape["dims"],
                    "out_features": shape["hidden"],
                    "with_bias": False,
                }
            )
            # With bias
            configs.append(
                {
                    "name": f"fused_rmsnorm_linear_{size_name}_bias",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "hidden": shape["dims"],
                    "out_features": shape["hidden"],
                    "with_bias": True,
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        hidden = config["hidden"]
        out_features = config["out_features"]
        with_bias = config["with_bias"]
        eps = 1e-5

        # Input
        x = torch.randn(batch, seq, hidden)

        # RMSNorm weight (scale)
        norm_weight = torch.ones(hidden)

        # Linear weight and optional bias
        linear_weight = torch.randn(out_features, hidden) * 0.02
        linear_bias = torch.randn(out_features) * 0.02 if with_bias else None

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        norm_x = x / rms * norm_weight

        # Linear: norm_x @ weight.T + bias
        out = F.linear(norm_x, linear_weight, linear_bias)

        inputs = {
            "x": x.numpy(),
            "norm_weight": norm_weight.numpy(),
            "linear_weight": linear_weight.numpy(),
        }
        if with_bias:
            inputs["linear_bias"] = linear_bias.numpy()

        return TestConfig(
            name=config["name"],
            inputs=inputs,
            params={"eps": eps, "with_bias": with_bias},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "hidden": hidden,
                "out_features": out_features,
            },
        )

    def generate_numpy_reference(self, config: Dict[str, Any]) -> TestConfig:
        np.random.seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        hidden = config["hidden"]
        out_features = config["out_features"]
        with_bias = config["with_bias"]
        eps = 1e-5

        # Input
        x = np.random.randn(batch, seq, hidden).astype(np.float32)

        # RMSNorm weight (scale)
        norm_weight = np.ones(hidden, dtype=np.float32)

        # Linear weight and optional bias
        linear_weight = np.random.randn(out_features, hidden).astype(np.float32) * 0.02
        linear_bias = np.random.randn(out_features).astype(np.float32) * 0.02 if with_bias else None

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        norm_x = x / rms * norm_weight

        # Linear: norm_x @ weight.T + bias
        out = norm_x @ linear_weight.T
        if with_bias:
            out = out + linear_bias

        inputs = {
            "x": x,
            "norm_weight": norm_weight,
            "linear_weight": linear_weight,
        }
        if with_bias:
            inputs["linear_bias"] = linear_bias

        return TestConfig(
            name=config["name"],
            inputs=inputs,
            params={"eps": eps, "with_bias": with_bias},
            expected_outputs={"out": out},
            metadata={
                "batch": batch,
                "seq": seq,
                "hidden": hidden,
                "out_features": out_features,
            },
        )
