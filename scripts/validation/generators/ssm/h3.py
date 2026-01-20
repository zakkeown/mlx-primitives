"""H3 (Hungry Hungry Hippos) generator.

Reference: "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
https://arxiv.org/abs/2212.14052
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class H3Generator(GoldenGenerator):
    """Generate golden files for H3 layer.

    H3 combines:
    1. SSM (like S4) for sequence mixing
    2. Multiplicative gating
    3. Short convolutions for local patterns

    It's a hybrid architecture between attention and SSM.
    """

    @property
    def name(self) -> str:
        return "h3"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["ssm"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        h3_configs = [
            {"d_model": 64, "d_state": 32, "head_dim": 32, "name": "tiny"},
            {"d_model": 256, "d_state": 64, "head_dim": 64, "name": "small"},
            {"d_model": 512, "d_state": 64, "head_dim": 64, "name": "medium"},
        ]

        for cfg in h3_configs:
            # Use "small" shape for all H3 tests (test expects h3_tiny, h3_small, h3_medium)
            shape = STANDARD_SHAPES["small"]
            configs.append(
                {
                    **cfg,
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "name": f"h3_{cfg['name']}",  # Test expects simple names like h3_tiny
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        d_model = config["d_model"]
        d_state = config["d_state"]
        head_dim = config["head_dim"]

        num_heads = d_model // head_dim

        # Input
        x = torch.randn(batch, seq, d_model)

        # H3 layer weights
        # Input projections (Q, K, V like attention but processed differently)
        q_proj = torch.randn(d_model, d_model) * 0.02
        k_proj = torch.randn(d_model, d_model) * 0.02
        v_proj = torch.randn(d_model, d_model) * 0.02

        # Short convolution for local patterns
        conv_weight = torch.randn(d_model, 1, 3) * 0.02

        # SSM parameters (simplified diagonal version)
        Lambda = -torch.exp(torch.randn(d_model, d_state))
        B_ssm = torch.randn(d_model, d_state) * 0.02
        C_ssm = torch.randn(d_model, d_state) * 0.02

        # Output projection
        out_proj = torch.randn(d_model, d_model) * 0.02

        # Forward pass
        # 1. Project inputs
        q = x @ q_proj.T
        k = x @ k_proj.T
        v = x @ v_proj.T

        # 2. Short convolution on K (for local context)
        k_conv = k.transpose(1, 2)  # (batch, d_model, seq)
        k_conv = F.pad(k_conv, (2, 0))  # causal padding
        k_conv = F.conv1d(k_conv, conv_weight, groups=d_model)
        k_conv = k_conv.transpose(1, 2)  # (batch, seq, d_model)

        # 3. Multiplicative interaction: Q * K_conv
        qk = q * k_conv

        # 4. SSM on QK (simplified version)
        step = 1.0 / seq
        ssm_out = torch.zeros_like(qk)

        for b_idx in range(batch):
            for d in range(d_model):
                h = torch.zeros(d_state)
                A_discrete = torch.exp(step * Lambda[d])

                for t in range(seq):
                    h = A_discrete * h + step * B_ssm[d] * qk[b_idx, t, d]
                    ssm_out[b_idx, t, d] = (C_ssm[d] * h).sum()

        # 5. Gate with V
        gated = ssm_out * v

        # 6. Output projection
        out = gated @ out_proj.T

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "q_proj": q_proj.numpy(),
                "k_proj": k_proj.numpy(),
                "v_proj": v_proj.numpy(),
                "conv_weight": conv_weight.numpy(),
                "Lambda": Lambda.numpy(),
                "B_ssm": B_ssm.numpy(),
                "C_ssm": C_ssm.numpy(),
                "out_proj": out_proj.numpy(),
            },
            params={
                "d_state": d_state,
                "head_dim": head_dim,
                "num_heads": num_heads,
            },
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "d_model": d_model,
            },
        )
