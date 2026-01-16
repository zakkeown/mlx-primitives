"""Sinusoidal and learned positional embedding generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class SinusoidalEmbeddingGenerator(GoldenGenerator):
    """Generate golden files for Sinusoidal Positional Embeddings.

    The original transformer positional encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    @property
    def name(self) -> str:
        return "sinusoidal_embedding"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["embeddings_sinusoidal"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "sinusoidal_emb_small", "max_len": 128, "dims": 256},
            {"name": "sinusoidal_emb_medium", "max_len": 512, "dims": 512},
            {"name": "sinusoidal_emb_large", "max_len": 2048, "dims": 1024},
            {"name": "sinusoidal_emb_bert", "max_len": 512, "dims": 768},
            {"name": "sinusoidal_emb_gpt2", "max_len": 1024, "dims": 768},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        max_len = config["max_len"]
        dims = config["dims"]

        # Create sinusoidal position encodings
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dims, 2).float() * (-np.log(10000.0) / dims)
        )

        pe = torch.zeros(max_len, dims)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Mathematically justified tolerance:
        # exp() ULP error (~6e-8) × max_len × 1.2 safety margin
        computed_atol = 6e-8 * max_len * 1.2

        return TestConfig(
            name=config["name"],
            inputs={"positions": torch.arange(max_len).numpy()},
            params={"dims": dims, "max_len": max_len},
            expected_outputs={"embeddings": pe.numpy()},
            metadata={"max_len": max_len, "dims": dims},
            tolerance=ToleranceConfig(
                rtol_fp32=1e-5,
                atol_fp32=computed_atol,
                rtol_fp16=1e-3,
                atol_fp16=max(1e-4, computed_atol * 10),
            ),
        )


class LearnedPositionalEmbeddingGenerator(GoldenGenerator):
    """Generate golden files for Learned Positional Embeddings.

    Standard embedding lookup for positions.
    """

    @property
    def name(self) -> str:
        return "learned_positional_embedding"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["embeddings_sinusoidal"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "learned_pos_emb_small", "batch": 2, "seq": 64, "max_len": 128, "dims": 256},
            {"name": "learned_pos_emb_medium", "batch": 2, "seq": 256, "max_len": 512, "dims": 512},
            {"name": "learned_pos_emb_large", "batch": 4, "seq": 512, "max_len": 1024, "dims": 768},
            {"name": "learned_pos_emb_offset", "batch": 2, "seq": 64, "max_len": 512, "dims": 256, "offset": 100},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        max_len = config["max_len"]
        dims = config["dims"]
        offset = config.get("offset", 0)

        # Initialize embedding weights
        embedding = nn.Embedding(max_len, dims)
        with torch.no_grad():
            embedding.weight.normal_(mean=0.0, std=0.02)

        # Create position indices with offset
        positions = torch.arange(seq) + offset
        positions = positions.unsqueeze(0).expand(batch, -1)

        # Look up embeddings
        out = embedding(positions)

        return TestConfig(
            name=config["name"],
            inputs={
                "positions": positions.numpy(),
                "weight": embedding.weight.detach().numpy(),
            },
            params={"max_len": max_len, "dims": dims},
            expected_outputs={"embeddings": out.detach().numpy()},
            metadata={"batch": batch, "seq": seq, "max_len": max_len, "dims": dims, "offset": offset},
        )
