"""MoE layer generator."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class MoELayerGenerator(GoldenGenerator):
    """Generate golden files for MoE Layer.

    Full MoE layer with routing and expert computation.
    """

    @property
    def name(self) -> str:
        return "moe_layer"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["moe"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "moe_layer_small",
                "batch": 2,
                "seq": 32,
                "dims": 128,
                "hidden_dims": 256,
                "num_experts": 4,
                "top_k": 1,
            },
            {
                "name": "moe_layer_medium",
                "batch": 2,
                "seq": 64,
                "dims": 256,
                "hidden_dims": 512,
                "num_experts": 8,
                "top_k": 2,
            },
            {
                "name": "moe_layer_large",
                "batch": 4,
                "seq": 128,
                "dims": 512,
                "hidden_dims": 1024,
                "num_experts": 8,
                "top_k": 2,
            },
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        hidden_dims = config["hidden_dims"]
        num_experts = config["num_experts"]
        top_k = config["top_k"]

        x = torch.randn(batch, seq, dims)

        # Router weights
        router_weight = torch.randn(num_experts, dims) * 0.02

        # Expert weights (each expert is a 2-layer FFN)
        expert_w1 = torch.randn(num_experts, dims, hidden_dims) * 0.02
        expert_w2 = torch.randn(num_experts, hidden_dims, dims) * 0.02

        # Flatten input
        x_flat = x.view(-1, dims)  # (batch*seq, dims)
        num_tokens = x_flat.shape[0]

        # Compute router logits and get top-k
        router_logits = x_flat @ router_weight.T  # (num_tokens, num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
        routing_weights = F.softmax(top_k_logits, dim=-1)  # (num_tokens, top_k)

        # Compute expert outputs
        # For simplicity, compute all experts and select (not efficient but correct)
        out = torch.zeros(num_tokens, dims)

        for k in range(top_k):
            expert_idx = top_k_indices[:, k]  # (num_tokens,)
            weight = routing_weights[:, k]  # (num_tokens,)

            for e in range(num_experts):
                mask = (expert_idx == e)
                if mask.sum() == 0:
                    continue

                # Get tokens for this expert
                tokens = x_flat[mask]  # (num_selected, dims)

                # Expert forward: x -> relu(x @ w1) @ w2
                hidden = F.relu(tokens @ expert_w1[e])
                expert_out = hidden @ expert_w2[e]

                # Add weighted output
                out[mask] += weight[mask].unsqueeze(1) * expert_out

        out = out.view(batch, seq, dims)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "router_weight": router_weight.numpy(),
                "expert_w1": expert_w1.numpy(),
                "expert_w2": expert_w2.numpy(),
            },
            params={"num_experts": num_experts, "top_k": top_k, "hidden_dims": hidden_dims},
            expected_outputs={
                "out": out.numpy(),
                "router_logits": router_logits.view(batch, seq, -1).numpy(),
                "top_k_indices": top_k_indices.view(batch, seq, -1).numpy(),
                "routing_weights": routing_weights.view(batch, seq, -1).numpy(),
            },
            metadata={"batch": batch, "seq": seq, "dims": dims},
        )
