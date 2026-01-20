"""QLoRA generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class QLoRALinearGenerator(GoldenGenerator):
    """Generate golden files for QLoRA (Quantized LoRA).

    Combines 4-bit quantized base weights with trainable low-rank adapters:
    out = (quantized_W + alpha * B @ A) @ x

    Where B and A are the LoRA matrices.
    """

    @property
    def name(self) -> str:
        return "qlora_linear"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["quantization_int4"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "qlora_r8",
                "batch": 2,
                "seq": 64,
                "in_features": 256,
                "out_features": 512,
                "rank": 8,
                "alpha": 16,
                "group_size": 64,
            },
            {
                "name": "qlora_r16",
                "batch": 2,
                "seq": 64,
                "in_features": 512,
                "out_features": 1024,
                "rank": 16,
                "alpha": 32,
                "group_size": 64,
            },
            {
                "name": "qlora_r32",
                "batch": 4,
                "seq": 128,
                "in_features": 1024,
                "out_features": 2048,
                "rank": 32,
                "alpha": 64,
                "group_size": 128,
            },
            {
                "name": "qlora_small_rank",
                "batch": 2,
                "seq": 64,
                "in_features": 256,
                "out_features": 256,
                "rank": 4,
                "alpha": 8,
                "group_size": 64,
            },
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        in_features = config["in_features"]
        out_features = config["out_features"]
        rank = config["rank"]
        alpha = config["alpha"]
        group_size = config["group_size"]

        # Generate input
        x = torch.randn(batch, seq, in_features)

        # Base weights (to be quantized)
        base_weight = torch.randn(out_features, in_features) * 0.02

        # LoRA weights
        lora_A = torch.randn(rank, in_features) * 0.01
        lora_B = torch.randn(out_features, rank) * 0.01

        # Scaling factor
        scaling = alpha / rank

        # Quantize base weight (group-wise INT4)
        num_groups = in_features // group_size
        weight_grouped = base_weight.view(out_features, num_groups, group_size)
        scales = weight_grouped.abs().amax(dim=2, keepdim=True) / 7
        w_quant = torch.clamp(torch.round(weight_grouped / scales), -8, 7)
        w_dequant = (w_quant * scales).view(out_features, in_features)

        # Full precision output (reference, no quantization)
        fp_out = x @ (base_weight + scaling * lora_B @ lora_A).T

        # QLoRA output: quantized base + LoRA
        qlora_out = x @ w_dequant.T + scaling * (x @ lora_A.T @ lora_B.T)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "base_weight": base_weight.numpy(),
                "lora_A": lora_A.numpy(),
                "lora_B": lora_B.numpy(),
            },
            params={"rank": rank, "alpha": alpha, "group_size": group_size, "scaling": scaling},
            expected_outputs={
                "fp_out": fp_out.numpy(),
                "qlora_out": qlora_out.numpy(),
                "w_quant": w_quant.numpy().astype(np.int8),
                "scales": scales.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "in_features": in_features,
                "out_features": out_features,
            },
        )
