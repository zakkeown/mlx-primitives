"""Golden file tests for Mixture of Experts (MoE) layers.

These tests compare MLX MoE implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category moe

To run tests:
    pytest tests/correctness/test_moe_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# Top-K Router
# =============================================================================


class TestTopKRouterGolden:
    """Test Top-K router against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "topk_router_top1",
            "topk_router_top2",
            "topk_router_top4",
            "topk_router_large",
        ],
    )
    def test_topk_router(self, config):
        """Top-K router matches PyTorch."""
        if not golden_exists("moe", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("moe", config)

        x = mx.array(golden["x"])
        router_weight = mx.array(golden["router_weight"])
        top_k = golden["__metadata__"]["params"]["top_k"]

        # Compute router logits
        router_logits = x @ router_weight.T
        mx.eval(router_logits)

        assert_close_golden(router_logits, golden, "router_logits")

        # Get top-k experts
        # MLX topk returns only values, use argsort for indices
        top_k_indices = mx.argsort(router_logits, axis=-1)[..., -top_k:]
        # Note: Adding + 0 forces a copy, which fixes an MLX slice view bug
        # where [::-1] creates a view with incorrect column indexing behavior
        top_k_indices = top_k_indices[..., ::-1] + 0
        top_k_logits = mx.take_along_axis(router_logits, top_k_indices, axis=-1)

        # Indices might not match exactly due to ties, so we verify the logits
        routing_weights = mx.softmax(top_k_logits, axis=-1)
        mx.eval(routing_weights)

        assert_close_golden(routing_weights, golden, "routing_weights")


# =============================================================================
# Expert Choice Router
# =============================================================================


class TestExpertChoiceRouterGolden:
    """Test Expert Choice router against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "expert_choice_small",
            "expert_choice_medium",
            "expert_choice_large",
        ],
    )
    def test_expert_choice_router(self, config):
        """Expert Choice router matches PyTorch."""
        if not golden_exists("moe", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("moe", config)

        x = mx.array(golden["x"])
        router_weight = mx.array(golden["router_weight"])
        num_experts = golden["__metadata__"]["params"]["num_experts"]
        expert_capacity = golden["__metadata__"]["params"]["expert_capacity"]

        # Flatten batch and sequence
        batch, seq, dims = x.shape
        x_flat = x.reshape(-1, dims)

        # Compute router logits
        router_logits = x_flat @ router_weight.T
        mx.eval(router_logits)

        assert_close_golden(router_logits, golden, "router_logits")


# =============================================================================
# MoE Layer
# =============================================================================


class TestMoELayerGolden:
    """Test MoE layer against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "moe_layer_small",
            "moe_layer_medium",
            "moe_layer_large",
        ],
    )
    def test_moe_layer(self, config):
        """MoE layer output matches PyTorch."""
        if not golden_exists("moe", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("moe", config)

        x = mx.array(golden["x"])
        router_weight = mx.array(golden["router_weight"])
        expert_w1 = mx.array(golden["expert_w1"])
        expert_w2 = mx.array(golden["expert_w2"])
        num_experts = golden["__metadata__"]["params"]["num_experts"]
        top_k = golden["__metadata__"]["params"]["top_k"]

        batch, seq, dims = x.shape
        x_flat = x.reshape(-1, dims)
        num_tokens = x_flat.shape[0]

        # Router
        router_logits = x_flat @ router_weight.T
        # MLX topk returns only values, use argsort for indices
        top_k_indices = mx.argsort(router_logits, axis=-1)[..., -top_k:]
        # Note: Adding + 0 forces a copy, which fixes an MLX slice view bug
        # where [::-1] creates a view with incorrect column indexing behavior
        top_k_indices = top_k_indices[..., ::-1] + 0
        top_k_logits = mx.take_along_axis(router_logits, top_k_indices, axis=-1)
        routing_weights = mx.softmax(top_k_logits, axis=-1)

        # Compute expert outputs - compute all experts and mask
        out = mx.zeros((num_tokens, dims))

        for k in range(top_k):
            expert_idx = top_k_indices[:, k]  # (num_tokens,)
            weight = routing_weights[:, k]    # (num_tokens,)

            for e in range(num_experts):
                # Compute output for this expert on all tokens
                hidden = nn.relu(x_flat @ expert_w1[e])
                expert_out = hidden @ expert_w2[e]

                # Create mask and apply
                mask = (expert_idx == e).astype(mx.float32)  # (num_tokens,)
                mask_3d = mx.expand_dims(mask, -1)  # (num_tokens, 1)
                weight_3d = mx.expand_dims(weight, -1)  # (num_tokens, 1)

                # Weighted sum with mask
                out = out + mask_3d * weight_3d * expert_out

        out = out.reshape(batch, seq, dims)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Load Balancing Loss
# =============================================================================


class TestLoadBalancingLossGolden:
    """Test load balancing loss against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "lb_loss_balanced",
            "lb_loss_imbalanced",
            "lb_loss_extreme",
            "lb_loss_large",
        ],
    )
    def test_load_balancing_loss(self, config):
        """Load balancing loss matches PyTorch."""
        if not golden_exists("moe", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("moe", config)

        router_logits = mx.array(golden["router_logits"])
        selected_experts = mx.array(golden["selected_experts"])
        num_experts = golden["__metadata__"]["params"]["num_experts"]

        num_tokens = router_logits.shape[0]

        # Routing probabilities
        routing_probs = mx.softmax(router_logits, axis=-1)

        # Fraction of tokens per expert
        fraction_tokens = mx.zeros(num_experts)
        for e in range(num_experts):
            fraction_tokens = mx.where(
                mx.arange(num_experts) == e,
                mx.mean((selected_experts == e).astype(mx.float32)),
                fraction_tokens,
            )

        # Fraction of routing probability per expert
        fraction_routing = mx.mean(routing_probs, axis=0)

        # Loss
        loss = num_experts * mx.sum(fraction_tokens * fraction_routing)
        mx.eval(loss)

        assert_close_golden(loss, golden, "loss")


# =============================================================================
# Router Z-Loss
# =============================================================================


class TestRouterZLossGolden:
    """Test router Z-loss against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "z_loss_small_logits",
            "z_loss_large_logits",
            "z_loss_medium",
        ],
    )
    def test_router_z_loss(self, config):
        """Router Z-loss matches PyTorch."""
        if not golden_exists("moe", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("moe", config)

        router_logits = mx.array(golden["router_logits"])

        # Z-loss: mean(logsumexp^2)
        log_z = mx.logsumexp(router_logits, axis=-1)
        z_loss = mx.mean(log_z ** 2)
        mx.eval(z_loss)

        assert_close_golden(z_loss, golden, "loss")
