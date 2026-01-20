"""Standard Scaled Dot-Product Attention (SDPA) generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class SDPAGenerator(GoldenGenerator):
    """Generate golden files for Scaled Dot-Product Attention.

    SDPA: softmax(QK^T / sqrt(d_k)) @ V
    """

    @property
    def name(self) -> str:
        return "sdpa"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Standard shapes with causal and non-causal
        for size_name, shape in STANDARD_SHAPES.items():
            for causal in [False, True]:
                causal_str = "causal" if causal else "noncausal"
                configs.append(
                    {
                        "name": f"sdpa_{size_name}_{causal_str}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "causal": causal,
                    }
                )

        # Cross-attention (different KV sequence length)
        configs.extend(
            [
                {
                    "name": "sdpa_cross_attention_small",
                    "batch": 2,
                    "seq_q": 64,
                    "seq_kv": 128,
                    "heads": 8,
                    "head_dim": 64,
                    "causal": False,
                },
                {
                    "name": "sdpa_cross_attention_medium",
                    "batch": 4,
                    "seq_q": 128,
                    "seq_kv": 256,
                    "heads": 16,
                    "head_dim": 64,
                    "causal": False,
                },
            ]
        )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "sdpa_single_token",
                    "batch": 1,
                    "seq": 1,
                    "heads": 8,
                    "head_dim": 64,
                    "causal": False,
                },
                {
                    "name": "sdpa_long_sequence",
                    "batch": 1,
                    "seq": 2048,
                    "heads": 8,
                    "head_dim": 64,
                    "causal": True,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq_q = config.get("seq_q", config.get("seq"))
        seq_kv = config.get("seq_kv", seq_q)
        heads = config["heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]

        # Generate Q, K, V tensors
        # Shape: (batch, seq, heads, head_dim) - MLX convention
        q = torch.randn(batch, seq_q, heads, head_dim)
        k = torch.randn(batch, seq_kv, heads, head_dim)
        v = torch.randn(batch, seq_kv, heads, head_dim)

        # Transpose to PyTorch convention: (batch, heads, seq, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Use PyTorch SDPA
        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)

        # Transpose back to MLX convention
        out = out_t.transpose(1, 2)

        scale = 1.0 / (head_dim**0.5)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={
                "causal": causal,
                "scale": scale,
            },
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
