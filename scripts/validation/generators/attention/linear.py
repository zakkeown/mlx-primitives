"""Linear attention variant generators: LinearAttention, Performer, CosFormer."""

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class LinearAttentionGenerator(GoldenGenerator):
    """Generate golden files for Linear Attention.

    Linear attention approximates softmax attention using feature maps:
    output = phi(Q) @ (phi(K).T @ V) / (phi(Q) @ phi(K).T @ 1)

    Default feature map: elu(x) + 1
    """

    @property
    def name(self) -> str:
        return "linear_attention"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_linear"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"linear_attention_{size_name}",
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
                    "name": "linear_attention_long_seq",
                    "batch": 1,
                    "seq": 2048,
                    "heads": 8,
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

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Feature map: elu(x) + 1
        def elu_feature_map(x):
            return F.elu(x) + 1

        q_prime = elu_feature_map(q)
        k_prime = elu_feature_map(k)

        # Transpose to (batch, heads, seq, dim)
        q_t = q_prime.transpose(1, 2)
        k_t = k_prime.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Linear attention: O(n) computation
        # KV = K^T @ V: (batch, heads, dim, dim)
        kv = torch.einsum("bhsd,bhse->bhde", k_t, v_t)

        # QKV = Q @ KV: (batch, heads, seq, dim)
        qkv = torch.einsum("bhqd,bhde->bhqe", q_t, kv)

        # Normalizer: Q @ K^T @ 1 = Q @ (sum of K rows)
        k_sum = k_t.sum(dim=2, keepdim=True)  # (batch, heads, 1, dim)
        normalizer = torch.einsum("bhqd,bhkd->bhqk", q_t, k_sum).squeeze(-1)  # (batch, heads, seq)
        normalizer = normalizer.unsqueeze(-1) + eps  # (batch, heads, seq, 1)

        out_t = qkv / normalizer

        # Transpose back
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={"eps": eps},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class PerformerGenerator(GoldenGenerator):
    """Generate golden files for Performer (FAVOR+) attention.

    Uses random feature maps to approximate softmax attention.
    phi(x) = exp(x @ W - ||x||^2/2) / sqrt(m)
    """

    @property
    def name(self) -> str:
        return "performer"

    def get_tolerance_config(self) -> ToleranceConfig:
        # Performer is a stochastic approximation, higher tolerance needed
        return ToleranceConfig(
            rtol_fp32=5e-2,
            atol_fp32=1e-2,
            rtol_fp16=0.1,
            atol_fp16=5e-2,
        )

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Test with different numbers of random features
        for num_features in [64, 128, 256]:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"performer_f{num_features}_{size_name}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "num_features": num_features,
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        num_features = config["num_features"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Generate orthogonal random features
        # For reproducibility, use the same seed for random projection
        torch.manual_seed(self.seed + 1000)
        omega = torch.randn(head_dim, num_features)
        # Orthogonalize
        omega, _ = torch.linalg.qr(omega)

        # FAVOR+ kernel approximation
        def favor_plus_kernel(x, omega):
            # x: (batch, seq, heads, head_dim)
            # omega: (head_dim, num_features)
            x_norm_sq = (x**2).sum(dim=-1, keepdim=True) / 2

            # Project: (batch, seq, heads, num_features)
            x_proj = torch.einsum("bshd,df->bshf", x, omega)

            # Softmax approximation
            phi = torch.exp(x_proj - x_norm_sq) / math.sqrt(num_features)
            return phi

        q_prime = favor_plus_kernel(q, omega)
        k_prime = favor_plus_kernel(k, omega)

        # Transpose to (batch, heads, seq, features)
        q_t = q_prime.transpose(1, 2)
        k_t = k_prime.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Linear attention with random features
        kv = torch.einsum("bhsf,bhsd->bhfd", k_t, v_t)
        qkv = torch.einsum("bhqf,bhfd->bhqd", q_t, kv)

        # Normalizer
        k_sum = k_t.sum(dim=2)  # (batch, heads, features)
        normalizer = torch.einsum("bhqf,bhf->bhq", q_t, k_sum)
        normalizer = normalizer.unsqueeze(-1) + 1e-6

        out_t = qkv / normalizer
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
                "omega": omega.numpy(),
            },
            params={"num_features": num_features},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class CosFormerGenerator(GoldenGenerator):
    """Generate golden files for CosFormer attention.

    Uses cosine-based reweighting with ReLU feature map:
    cos(pi * i / (2 * M)) * relu(x)
    """

    @property
    def name(self) -> str:
        return "cosformer"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_linear"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"cosformer_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "heads": shape["heads"],
                    "head_dim": shape["head_dim"],
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        eps = 1e-6

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Positional reweighting: cos(pi * i / (2 * M))
        positions = torch.arange(seq, dtype=torch.float32)
        cos_weights = torch.cos(math.pi * positions / (2 * seq))
        cos_weights = cos_weights.view(1, seq, 1, 1)

        # Feature map: relu(x) * cos_weight
        q_prime = F.relu(q) * cos_weights
        k_prime = F.relu(k) * cos_weights

        # Transpose to (batch, heads, seq, dim)
        q_t = q_prime.transpose(1, 2)
        k_t = k_prime.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Linear attention
        kv = torch.einsum("bhsd,bhse->bhde", k_t, v_t)
        qkv = torch.einsum("bhqd,bhde->bhqe", q_t, kv)

        k_sum = k_t.sum(dim=2, keepdim=True)
        normalizer = torch.einsum("bhqd,bhkd->bhqk", q_t, k_sum).squeeze(-1)
        normalizer = normalizer.unsqueeze(-1) + eps

        out_t = qkv / normalizer
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={"eps": eps},
            expected_outputs={
                "out": out.numpy(),
                "cos_weights": cos_weights.squeeze().numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
