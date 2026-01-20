"""Efficient attention variant generators: GQA, MQA, SlidingWindowAttention."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class GQAGenerator(GoldenGenerator):
    """Generate golden files for Grouped Query Attention.

    GQA uses fewer KV heads than Q heads, with each KV head serving multiple Q heads.
    """

    @property
    def name(self) -> str:
        return "gqa"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Standard GQA configurations
        gqa_configs = [
            # (num_heads, num_kv_heads) - typical ratios
            {"heads": 8, "kv_heads": 2},  # 4:1 ratio
            {"heads": 16, "kv_heads": 4},  # 4:1 ratio
            {"heads": 32, "kv_heads": 8},  # 4:1 ratio
            {"heads": 8, "kv_heads": 1},  # MQA-like
            {"heads": 16, "kv_heads": 2},  # 8:1 ratio
        ]

        for gqa_cfg in gqa_configs:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                for causal in [False, True]:
                    causal_str = "causal" if causal else "noncausal"
                    configs.append(
                        {
                            "name": f"gqa_h{gqa_cfg['heads']}_kv{gqa_cfg['kv_heads']}_{size_name}_{causal_str}",
                            "batch": shape["batch"],
                            "seq": shape["seq"],
                            "heads": gqa_cfg["heads"],
                            "kv_heads": gqa_cfg["kv_heads"],
                            "head_dim": shape["head_dim"],
                            "causal": causal,
                        }
                    )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        kv_heads = config["kv_heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]

        # Q has full heads, K/V have fewer heads
        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, kv_heads, head_dim)
        v = torch.randn(batch, seq, kv_heads, head_dim)

        # Expand K/V to match Q heads by repeating
        # Each KV head serves (heads // kv_heads) Q heads
        num_groups = heads // kv_heads

        # Expand K/V: (batch, seq, kv_heads, head_dim) -> (batch, seq, heads, head_dim)
        k_expanded = k.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
        k_expanded = k_expanded.reshape(batch, seq, heads, head_dim)

        v_expanded = v.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
        v_expanded = v_expanded.reshape(batch, seq, heads, head_dim)

        # Transpose to PyTorch convention
        q_t = q.transpose(1, 2)
        k_t = k_expanded.transpose(1, 2)
        v_t = v_expanded.transpose(1, 2)

        # Use SDPA
        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)

        # Transpose back
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
                "num_kv_heads": kv_heads,
            },
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
            },
        )


class MQAGenerator(GoldenGenerator):
    """Generate golden files for Multi-Query Attention.

    MQA uses a single KV head shared across all Q heads.
    This is a special case of GQA with num_kv_heads=1.
    """

    @property
    def name(self) -> str:
        return "mqa"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            for causal in [False, True]:
                causal_str = "causal" if causal else "noncausal"
                configs.append(
                    {
                        "name": f"mqa_{size_name}_{causal_str}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "causal": causal,
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]

        # Q has full heads, K/V have single head
        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, 1, head_dim)
        v = torch.randn(batch, seq, 1, head_dim)

        # Expand K/V to all heads
        k_expanded = k.expand(-1, -1, heads, -1)
        v_expanded = v.expand(-1, -1, heads, -1)

        # Transpose to PyTorch convention
        q_t = q.transpose(1, 2)
        k_t = k_expanded.transpose(1, 2)
        v_t = v_expanded.transpose(1, 2)

        # Use SDPA
        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)

        # Transpose back
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
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class SlidingWindowGenerator(GoldenGenerator):
    """Generate golden files for Sliding Window Attention.

    Each token attends only to tokens within a fixed window size.
    Used in Mistral and similar models.
    """

    @property
    def name(self) -> str:
        return "sliding_window"

    def get_tolerance_config(self) -> ToleranceConfig:
        # Slightly higher tolerance for softmax boundary effects
        return ToleranceConfig(
            rtol_fp32=1e-5,
            atol_fp32=1.5e-6,  # Increased from 1e-6 for softmax boundary effects
            rtol_fp16=1e-3,
            atol_fp16=1e-4,
            max_diff_fp32=1e-4,
            max_diff_fp16=5e-3,
        )

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different window sizes
        window_sizes = [64, 128, 256, 512]

        for window_size in window_sizes:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                # Only test if window size is smaller than sequence length
                if window_size < shape["seq"]:
                    configs.append(
                        {
                            "name": f"sliding_window_w{window_size}_{size_name}",
                            "batch": shape["batch"],
                            "seq": shape["seq"],
                            "heads": shape["heads"],
                            "head_dim": shape["head_dim"],
                            "window_size": window_size,
                        }
                    )

        # Window size equals sequence length (full attention)
        configs.append(
            {
                "name": "sliding_window_full",
                "batch": 2,
                "seq": 64,
                "heads": 8,
                "head_dim": 64,
                "window_size": 64,
            }
        )

        # Very small window
        configs.append(
            {
                "name": "sliding_window_tiny",
                "batch": 2,
                "seq": 128,
                "heads": 8,
                "head_dim": 64,
                "window_size": 16,
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

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Create sliding window mask
        # Token i can attend to tokens [max(0, i - window_size + 1), i]
        rows = torch.arange(seq).unsqueeze(1)
        cols = torch.arange(seq).unsqueeze(0)

        # Mask: True where attention is allowed
        # For causal sliding window: i can attend to j if j <= i and j >= i - window_size + 1
        causal_mask = cols <= rows
        window_mask = cols >= (rows - window_size + 1)
        mask = causal_mask & window_mask

        # Convert to attention bias (0 for allowed, -inf for masked)
        attn_mask = torch.where(mask, 0.0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        # Transpose to PyTorch convention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Use SDPA with mask
        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)

        # Transpose back
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
                "window_size": window_size,
                "scale": scale,
            },
            expected_outputs={
                "out": out.numpy(),
                "mask": mask.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
                "window_size": window_size,
            },
        )
