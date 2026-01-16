"""MoE auxiliary loss generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class LoadBalancingLossGenerator(GoldenGenerator):
    """Generate golden files for Load Balancing Loss.

    Encourages equal expert utilization:
    loss = num_experts * sum(fraction_tokens * fraction_routing)

    Where:
    - fraction_tokens = (tokens assigned to expert) / total_tokens
    - fraction_routing = mean(routing_prob for tokens to expert)
    """

    @property
    def name(self) -> str:
        return "load_balancing_loss"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["moe"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "lb_loss_balanced", "batch": 2, "seq": 64, "num_experts": 8, "distribution": "balanced"},
            {"name": "lb_loss_imbalanced", "batch": 2, "seq": 64, "num_experts": 8, "distribution": "imbalanced"},
            {"name": "lb_loss_extreme", "batch": 2, "seq": 64, "num_experts": 8, "distribution": "extreme"},
            {"name": "lb_loss_large", "batch": 4, "seq": 256, "num_experts": 16, "distribution": "balanced"},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        num_experts = config["num_experts"]
        distribution = config["distribution"]
        num_tokens = batch * seq

        # Create router logits with different distributions
        if distribution == "balanced":
            # Uniform random logits
            router_logits = torch.randn(num_tokens, num_experts)
        elif distribution == "imbalanced":
            # Bias toward first few experts
            router_logits = torch.randn(num_tokens, num_experts)
            router_logits[:, :num_experts//4] += 2.0
        else:  # extreme
            # Almost all tokens go to one expert
            router_logits = torch.randn(num_tokens, num_experts) * 0.1
            router_logits[:, 0] += 5.0

        # Get routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Get selected expert (top-1)
        selected_experts = torch.argmax(router_logits, dim=-1)

        # Compute load balancing loss
        # fraction_tokens: what fraction of tokens go to each expert
        # fraction_routing: mean routing probability for each expert
        fraction_tokens = torch.zeros(num_experts)
        fraction_routing = routing_probs.mean(dim=0)

        for e in range(num_experts):
            fraction_tokens[e] = (selected_experts == e).float().mean()

        loss = num_experts * (fraction_tokens * fraction_routing).sum()

        return TestConfig(
            name=config["name"],
            inputs={
                "router_logits": router_logits.numpy(),
                "selected_experts": selected_experts.numpy(),
            },
            params={"num_experts": num_experts},
            expected_outputs={
                "loss": loss.numpy(),
                "routing_probs": routing_probs.numpy(),
                "fraction_tokens": fraction_tokens.numpy(),
                "fraction_routing": fraction_routing.numpy(),
            },
            metadata={"batch": batch, "seq": seq, "distribution": distribution},
        )


class RouterZLossGenerator(GoldenGenerator):
    """Generate golden files for Router Z-Loss.

    Penalizes large router logits to improve training stability:
    loss = mean(logsumexp(router_logits, dim=-1)^2)
    """

    @property
    def name(self) -> str:
        return "router_z_loss"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["moe"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "z_loss_small_logits", "batch": 2, "seq": 64, "num_experts": 8, "logit_scale": 1.0},
            {"name": "z_loss_large_logits", "batch": 2, "seq": 64, "num_experts": 8, "logit_scale": 5.0},
            {"name": "z_loss_medium", "batch": 4, "seq": 128, "num_experts": 16, "logit_scale": 2.0},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        num_experts = config["num_experts"]
        logit_scale = config["logit_scale"]
        num_tokens = batch * seq

        # Create router logits with specified scale
        router_logits = torch.randn(num_tokens, num_experts) * logit_scale

        # Z-loss: penalize large logsumexp values
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()

        return TestConfig(
            name=config["name"],
            inputs={"router_logits": router_logits.numpy()},
            params={"num_experts": num_experts},
            expected_outputs={
                "loss": z_loss.numpy(),
                "log_z": log_z.numpy(),
            },
            metadata={"batch": batch, "seq": seq, "logit_scale": logit_scale},
        )
