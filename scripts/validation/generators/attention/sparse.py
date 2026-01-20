"""Sparse attention variant generators: BlockSparse, Longformer, BigBird."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class BlockSparseGenerator(GoldenGenerator):
    """Generate golden files for Block Sparse Attention.

    Attention is computed only for specified block patterns.
    """

    @property
    def name(self) -> str:
        return "block_sparse"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_sparse"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different block sizes
        for block_size in [16, 32, 64]:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                # Ensure sequence length is divisible by block size
                seq = (shape["seq"] // block_size) * block_size
                if seq > 0:
                    configs.append(
                        {
                            "name": f"block_sparse_b{block_size}_{size_name}",
                            "batch": shape["batch"],
                            "seq": seq,
                            "heads": shape["heads"],
                            "head_dim": shape["head_dim"],
                            "block_size": block_size,
                        }
                    )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        block_size = config["block_size"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        num_blocks = seq // block_size

        # Create block-sparse pattern: diagonal blocks + some off-diagonal
        # Each block attends to itself and adjacent blocks (banded pattern)
        block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)
        for i in range(num_blocks):
            # Diagonal
            block_mask[i, i] = True
            # Adjacent blocks
            if i > 0:
                block_mask[i, i - 1] = True
            if i < num_blocks - 1:
                block_mask[i, i + 1] = True

        # Expand block mask to full attention mask
        full_mask = torch.zeros(seq, seq, dtype=torch.bool)
        for bi in range(num_blocks):
            for bj in range(num_blocks):
                if block_mask[bi, bj]:
                    row_start, row_end = bi * block_size, (bi + 1) * block_size
                    col_start, col_end = bj * block_size, (bj + 1) * block_size
                    full_mask[row_start:row_end, col_start:col_end] = True

        # Apply causal constraint
        causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        full_mask = full_mask & causal_mask

        # Convert to attention bias
        attn_mask = torch.where(full_mask, 0.0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Transpose and compute attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={"block_size": block_size},
            expected_outputs={
                "out": out.numpy(),
                "block_mask": block_mask.numpy(),
                "full_mask": full_mask.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
                "num_blocks": num_blocks,
            },
        )


class LongformerGenerator(GoldenGenerator):
    """Generate golden files for Longformer attention.

    Combines sliding window attention with global attention for select tokens.
    """

    @property
    def name(self) -> str:
        return "longformer"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_sparse"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different window sizes and global token configurations
        window_configs = [
            {"window_size": 64, "num_global": 4},
            {"window_size": 128, "num_global": 8},
            {"window_size": 256, "num_global": 16},
        ]

        for win_cfg in window_configs:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                if win_cfg["window_size"] < shape["seq"]:
                    configs.append(
                        {
                            "name": f"longformer_w{win_cfg['window_size']}_g{win_cfg['num_global']}_{size_name}",
                            "batch": shape["batch"],
                            "seq": shape["seq"],
                            "heads": shape["heads"],
                            "head_dim": shape["head_dim"],
                            "window_size": win_cfg["window_size"],
                            "num_global": win_cfg["num_global"],
                        }
                    )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        window_size = config["window_size"]
        num_global = config["num_global"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Global token indices (first num_global tokens)
        global_indices = torch.arange(num_global)

        # Create Longformer attention mask
        rows = torch.arange(seq).unsqueeze(1)
        cols = torch.arange(seq).unsqueeze(0)

        # Sliding window component
        window_mask = (cols >= rows - window_size // 2) & (cols <= rows + window_size // 2)

        # Global component: global tokens can attend to/be attended by all
        global_row_mask = rows < num_global  # Global tokens attend to all
        global_col_mask = cols < num_global  # All attend to global tokens

        # Combine: window OR global
        full_mask = window_mask | global_row_mask | global_col_mask

        # Make causal (optional, depends on use case - we'll make it causal)
        causal_mask = cols <= rows
        full_mask = full_mask & causal_mask

        # Convert to attention bias
        attn_mask = torch.where(full_mask, 0.0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={
                "window_size": window_size,
                "num_global": num_global,
            },
            expected_outputs={
                "out": out.numpy(),
                "mask": full_mask.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class BigBirdGenerator(GoldenGenerator):
    """Generate golden files for BigBird attention.

    Combines random attention, window attention, and global attention.
    """

    @property
    def name(self) -> str:
        return "bigbird"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_sparse"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        bigbird_configs = [
            {"window_size": 32, "num_global": 4, "num_random": 8},
            {"window_size": 64, "num_global": 8, "num_random": 16},
        ]

        for bb_cfg in bigbird_configs:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"bigbird_w{bb_cfg['window_size']}_g{bb_cfg['num_global']}_r{bb_cfg['num_random']}_{size_name}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        **bb_cfg,
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        window_size = config["window_size"]
        num_global = config["num_global"]
        num_random = config["num_random"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        rows = torch.arange(seq).unsqueeze(1)
        cols = torch.arange(seq).unsqueeze(0)

        # Window attention
        window_mask = (cols >= rows - window_size // 2) & (cols <= rows + window_size // 2)

        # Global attention (first num_global tokens)
        global_row_mask = rows < num_global
        global_col_mask = cols < num_global
        global_mask = global_row_mask | global_col_mask

        # Random attention: each token randomly attends to num_random other tokens
        random_mask = torch.zeros(seq, seq, dtype=torch.bool)
        for i in range(seq):
            # Pick random tokens to attend to
            available = torch.arange(seq)
            available = available[available != i]  # Exclude self
            random_indices = available[torch.randperm(len(available))[:num_random]]
            random_mask[i, random_indices] = True

        # Combine all patterns
        full_mask = window_mask | global_mask | random_mask

        # Make causal
        causal_mask = cols <= rows
        full_mask = full_mask & causal_mask

        # Convert to attention bias
        attn_mask = torch.where(full_mask, 0.0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={
                "window_size": window_size,
                "num_global": num_global,
                "num_random": num_random,
            },
            expected_outputs={
                "out": out.numpy(),
                "mask": full_mask.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
