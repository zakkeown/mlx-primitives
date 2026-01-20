"""MoE router generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class TopKRouterGenerator(GoldenGenerator):
    """Generate golden files for Top-K Router.

    Standard MoE routing: select top-k experts per token.
    """

    @property
    def name(self) -> str:
        return "topk_router"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["moe"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "topk_router_top1", "batch": 2, "seq": 64, "dims": 256, "num_experts": 8, "top_k": 1},
            {"name": "topk_router_top2", "batch": 2, "seq": 64, "dims": 256, "num_experts": 8, "top_k": 2},
            {"name": "topk_router_top4", "batch": 2, "seq": 128, "dims": 512, "num_experts": 16, "top_k": 4},
            {"name": "topk_router_large", "batch": 4, "seq": 256, "dims": 512, "num_experts": 32, "top_k": 2},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        num_experts = config["num_experts"]
        top_k = config["top_k"]

        x = torch.randn(batch, seq, dims)

        # Router weights
        router_weight = torch.randn(num_experts, dims) * 0.02

        # Compute router logits
        router_logits = x @ router_weight.T  # (batch, seq, num_experts)

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, top_k, dim=-1)

        # Compute routing weights (softmax over top-k)
        routing_weights = F.softmax(top_k_logits, dim=-1)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "router_weight": router_weight.numpy(),
            },
            params={"num_experts": num_experts, "top_k": top_k},
            expected_outputs={
                "router_logits": router_logits.numpy(),
                "top_k_indices": top_k_indices.numpy(),
                "routing_weights": routing_weights.numpy(),
            },
            metadata={"batch": batch, "seq": seq, "dims": dims},
        )


class ExpertChoiceRouterGenerator(GoldenGenerator):
    """Generate golden files for Expert Choice Router.

    Experts choose their top-k tokens instead of tokens choosing experts.
    More balanced load distribution.
    """

    @property
    def name(self) -> str:
        return "expert_choice_router"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["moe"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "expert_choice_small", "batch": 2, "seq": 64, "dims": 256, "num_experts": 8, "capacity_factor": 1.0},
            {"name": "expert_choice_medium", "batch": 2, "seq": 128, "dims": 512, "num_experts": 8, "capacity_factor": 1.25},
            {"name": "expert_choice_large", "batch": 4, "seq": 256, "dims": 512, "num_experts": 16, "capacity_factor": 1.0},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        num_experts = config["num_experts"]
        capacity_factor = config["capacity_factor"]

        x = torch.randn(batch, seq, dims)

        # Router weights
        router_weight = torch.randn(num_experts, dims) * 0.02

        # Flatten batch and sequence
        x_flat = x.view(-1, dims)  # (batch*seq, dims)
        num_tokens = x_flat.shape[0]

        # Compute router logits
        router_logits = x_flat @ router_weight.T  # (batch*seq, num_experts)

        # Expert capacity: how many tokens each expert can handle
        expert_capacity = int(capacity_factor * num_tokens / num_experts)

        # Each expert chooses top-k tokens
        # Transpose to get (num_experts, num_tokens)
        expert_logits = router_logits.T

        # Get top tokens for each expert
        top_k_logits, top_k_indices = torch.topk(expert_logits, expert_capacity, dim=-1)

        # Compute dispatch weights (softmax over selected tokens)
        dispatch_weights = F.softmax(top_k_logits, dim=-1)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "router_weight": router_weight.numpy(),
            },
            params={
                "num_experts": num_experts,
                "capacity_factor": capacity_factor,
                "expert_capacity": expert_capacity,
            },
            expected_outputs={
                "router_logits": router_logits.numpy(),
                "top_k_indices": top_k_indices.numpy(),
                "dispatch_weights": dispatch_weights.numpy(),
            },
            metadata={"batch": batch, "seq": seq, "dims": dims, "num_tokens": num_tokens},
        )
