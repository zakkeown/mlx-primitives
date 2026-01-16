"""Rotary embedding generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class RotaryEmbeddingGenerator(GoldenGenerator):
    """Generate golden files for Rotary Positional Embeddings (RoPE).

    RoPE applies rotation to pairs of dimensions based on position.
    x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)
    """

    @property
    def name(self) -> str:
        return "rotary_embedding"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["elementwise"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Standard configurations
        rope_configs = [
            {"dims": 64, "seq": 128, "base": 10000, "name": "small"},
            {"dims": 128, "seq": 512, "base": 10000, "name": "medium"},
            {"dims": 128, "seq": 2048, "base": 10000, "name": "long"},
        ]

        for cfg in rope_configs:
            configs.append({
                "name": f"rope_{cfg['name']}",
                "batch": 2,
                "seq": cfg["seq"],
                "heads": 8,
                "dims": cfg["dims"],
                "base": cfg["base"],
            })

        # NTK-aware scaling (for extended context)
        configs.append({
            "name": "rope_ntk_scaled",
            "batch": 2,
            "seq": 4096,
            "heads": 8,
            "dims": 128,
            "base": 10000,
            "scale": 2.0,
        })

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        dims = config["dims"]
        base = config["base"]
        scale = config.get("scale", 1.0)

        # Scale base for NTK-aware scaling
        scaled_base = base * scale

        # Create query and key tensors
        q = torch.randn(batch, seq, heads, dims)
        k = torch.randn(batch, seq, heads, dims)

        # Compute rotation frequencies
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dims, 2).float() / dims))
        positions = torch.arange(seq).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        # Create rotation embeddings
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dims//2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        # Apply rotary embedding
        def apply_rope(x, cos, sin):
            x1, x2 = x[..., ::2], x[..., 1::2]
            # Rotate pairs of dimensions
            x_rotated = torch.stack([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ], dim=-1)
            return x_rotated.flatten(-2)

        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
            },
            params={"base": base, "scale": scale},
            expected_outputs={
                "q_rope": q_rope.numpy(),
                "k_rope": k_rope.numpy(),
            },
            metadata={"batch": batch, "seq": seq, "heads": heads, "dims": dims},
        )
