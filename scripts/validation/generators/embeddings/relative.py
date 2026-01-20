"""Relative positional embedding generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class RelativePositionalEmbeddingGenerator(GoldenGenerator):
    """Generate golden files for Relative Positional Embeddings.

    Used in T5 and similar models. Maps relative position distances
    to learned embedding vectors that are added to attention scores.
    """

    @property
    def name(self) -> str:
        return "relative_positional_embedding"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "rel_pos_small", "seq": 64, "heads": 8, "num_buckets": 32, "max_distance": 128},
            {"name": "rel_pos_medium", "seq": 256, "heads": 8, "num_buckets": 32, "max_distance": 128},
            {"name": "rel_pos_large", "seq": 512, "heads": 12, "num_buckets": 64, "max_distance": 256},
            {"name": "rel_pos_bidirectional", "seq": 128, "heads": 8, "num_buckets": 32, "max_distance": 128, "bidirectional": True},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        seq = config["seq"]
        heads = config["heads"]
        num_buckets = config["num_buckets"]
        max_distance = config["max_distance"]
        bidirectional = config.get("bidirectional", False)

        def relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
            """Translate relative position to bucket number (T5-style).

            Maps relative positions to buckets:
            - First half of buckets for exact positions (0, 1, 2, ...)
            - Second half for logarithmically increasing distances
            """
            ret = 0
            n = -relative_position

            if bidirectional:
                num_buckets //= 2
                ret += (n < 0).long() * num_buckets
                n = torch.abs(n)
            else:
                n = torch.max(n, torch.zeros_like(n))

            # Half of buckets for exact positions
            max_exact = num_buckets // 2
            is_small = n < max_exact

            # The other half for log-spaced buckets
            val_if_large = max_exact + (
                torch.log(n.float() / max_exact)
                / np.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()

            val_if_large = torch.min(
                val_if_large, torch.full_like(val_if_large, num_buckets - 1)
            )

            ret += torch.where(is_small, n, val_if_large)
            return ret

        # Create relative position matrix
        query_positions = torch.arange(seq).unsqueeze(1)
        key_positions = torch.arange(seq).unsqueeze(0)
        relative_positions = key_positions - query_positions

        # Convert to bucket indices
        bucket_indices = relative_position_bucket(
            relative_positions,
            bidirectional=bidirectional,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )

        # Create embedding table
        embedding = nn.Embedding(num_buckets, heads)
        with torch.no_grad():
            embedding.weight.normal_(mean=0.0, std=0.02)

        # Look up embeddings
        rel_pos_bias = embedding(bucket_indices)  # (seq, seq, heads)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # (heads, seq, seq)

        return TestConfig(
            name=config["name"],
            inputs={
                "bucket_indices": bucket_indices.numpy(),
                "weight": embedding.weight.detach().numpy(),
            },
            params={
                "num_buckets": num_buckets,
                "max_distance": max_distance,
                "bidirectional": bidirectional,
            },
            expected_outputs={"rel_pos_bias": rel_pos_bias.detach().numpy()},
            metadata={"seq": seq, "heads": heads},
        )
