"""Mixture of Experts layers for MLX.

This module provides MoE components:
- MoELayer: Sparse mixture of experts layer
- TopKRouter: Top-k expert routing
- ExpertChoiceRouter: Expert-choice routing
- LoadBalancingLoss: Auxiliary load balancing loss
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class TopKRouter(nn.Module):
    """Top-K routing for Mixture of Experts.

    Routes each token to the top-k experts based on learned gating scores.

    Args:
        dims: Input dimension.
        num_experts: Number of experts.
        top_k: Number of experts to route to per token (default: 2).
        jitter_noise: Noise to add during training for load balancing (default: 0.0).

    Example:
        >>> router = TopKRouter(dims=768, num_experts=8, top_k=2)
        >>> x = mx.random.normal((2, 100, 768))
        >>> gates, indices = router(x)
        >>> # gates: (2, 100, 2) - weights for top-2 experts
        >>> # indices: (2, 100, 2) - which experts were selected
    """

    def __init__(
        self,
        dims: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.dims = dims
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        # Gating network
        self.gate = nn.Linear(dims, num_experts, bias=False)

    def __call__(
        self, x: mx.array, training: bool = False
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode (adds jitter noise).

        Returns:
            Tuple of (gate_weights, expert_indices, router_logits)
            - gate_weights: (batch, seq_len, top_k) normalized weights
            - expert_indices: (batch, seq_len, top_k) selected expert indices
            - router_logits: (batch, seq_len, num_experts) raw logits for aux loss
        """
        # Compute router logits
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Add jitter noise during training
        if training and self.jitter_noise > 0:
            noise = mx.random.uniform(
                shape=router_logits.shape,
                low=1.0 - self.jitter_noise,
                high=1.0 + self.jitter_noise,
            )
            router_logits = router_logits * noise

        # Get top-k experts
        # Note: MLX doesn't have topk, so we use argsort
        sorted_indices = mx.argsort(-router_logits, axis=-1)
        expert_indices = sorted_indices[..., : self.top_k]

        # Gather the logits for selected experts
        batch_size, seq_len, _ = router_logits.shape

        # Use take_along_axis for gathering
        # Flatten and gather
        flat_logits = router_logits.reshape(-1, self.num_experts)
        flat_indices = expert_indices.reshape(-1, self.top_k)

        gathered = []
        for k in range(self.top_k):
            idx = flat_indices[:, k]
            vals = mx.take_along_axis(flat_logits, idx[:, None], axis=1).squeeze(-1)
            gathered.append(vals)

        selected_logits = mx.stack(gathered, axis=-1).reshape(
            batch_size, seq_len, self.top_k
        )

        # Softmax over selected experts
        gate_weights = mx.softmax(selected_logits, axis=-1)

        return gate_weights, expert_indices, router_logits


class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing for Mixture of Experts.

    Instead of tokens choosing experts, experts choose tokens.
    This provides better load balancing.

    Args:
        dims: Input dimension.
        num_experts: Number of experts.
        capacity_factor: Capacity per expert as fraction of sequence length.

    Reference:
        "Mixture-of-Experts with Expert Choice Routing"
        https://arxiv.org/abs/2202.09368
    """

    def __init__(
        self,
        dims: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(dims, num_experts, bias=False)

    def __call__(
        self, x: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Route using expert choice.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).

        Returns:
            Tuple of (dispatch_mask, combine_weights, router_logits)
        """
        batch_size, seq_len, _ = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Transpose to get expert view: (batch, num_experts, seq_len)
        expert_logits = mx.transpose(router_logits, (0, 2, 1))

        # Each expert selects top-k tokens
        capacity = int(seq_len * self.capacity_factor / self.num_experts)
        capacity = max(1, capacity)

        # Get top-k tokens per expert
        sorted_indices = mx.argsort(-expert_logits, axis=-1)
        selected_tokens = sorted_indices[..., :capacity]

        # Create dispatch mask using broadcasting comparison
        # selected_tokens: (batch, num_experts, capacity)
        # We want: dispatch_mask: (batch, num_experts, seq_len)
        # For each position, check if it's in the selected set for that expert
        seq_indices = mx.arange(seq_len)[None, None, None, :]  # (1, 1, 1, seq_len)
        selected_expanded = selected_tokens[..., :, None]  # (batch, num_experts, capacity, 1)

        # Check if any capacity slot matches each sequence position
        matches = (selected_expanded == seq_indices)  # (batch, num_experts, capacity, seq_len)
        dispatch_mask = mx.any(matches, axis=2).astype(mx.float32)  # (batch, num_experts, seq_len)

        # Compute combine weights (softmax over experts for each token)
        combine_weights = mx.softmax(router_logits, axis=-1)

        return dispatch_mask, combine_weights, router_logits


class Expert(nn.Module):
    """Single expert network (typically an MLP).

    Args:
        dims: Input/output dimension.
        hidden_dims: Hidden dimension.
        activation: Activation function.
    """

    def __init__(
        self,
        dims: int,
        hidden_dims: int,
        activation: Callable = nn.silu,
    ):
        super().__init__()
        self.w1 = nn.Linear(dims, hidden_dims, bias=False)
        self.w2 = nn.Linear(hidden_dims, dims, bias=False)
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(self.activation(self.w1(x)))


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Routes tokens to multiple experts and combines their outputs.
    Uses sparse computation - only selected experts process each token.

    Args:
        dims: Input/output dimension.
        hidden_dims: Hidden dimension for expert MLPs.
        num_experts: Number of expert networks.
        top_k: Number of experts per token (default: 2).
        activation: Activation function for experts.
        jitter_noise: Router jitter noise (default: 0.0).

    Example:
        >>> moe = MoELayer(dims=768, hidden_dims=3072, num_experts=8, top_k=2)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output, aux_loss = moe(x)
    """

    def __init__(
        self,
        dims: int,
        hidden_dims: int,
        num_experts: int,
        top_k: int = 2,
        activation: Callable = nn.silu,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.dims = dims
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(
            dims=dims,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )

        # Experts
        self.experts = [
            Expert(dims, hidden_dims, activation) for _ in range(num_experts)
        ]

    def __call__(
        self, x: mx.array, training: bool = False
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode.

        Returns:
            Tuple of (output, auxiliary_loss)
        """
        batch_size, seq_len, dims = x.shape

        # Get routing decisions
        gate_weights, expert_indices, router_logits = self.router(x, training)

        # Initialize output
        output = mx.zeros_like(x)

        # Process each expert (sparse computation)
        for expert_idx in range(self.num_experts):
            # Find which tokens go to this expert
            expert_mask = expert_indices == expert_idx  # (batch, seq, top_k)
            expert_weights = mx.where(expert_mask, gate_weights, 0.0)
            expert_weights = mx.sum(expert_weights, axis=-1, keepdims=True)  # (batch, seq, 1)

            # Only process if this expert has tokens
            if mx.any(expert_weights > 0):
                expert_out = self.experts[expert_idx](x)
                output = output + expert_out * expert_weights

        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)

        return output, aux_loss

    def _compute_aux_loss(
        self, router_logits: mx.array, expert_indices: mx.array
    ) -> mx.array:
        """Compute auxiliary load balancing loss.

        Encourages balanced expert usage.
        """
        batch_size, seq_len, _ = router_logits.shape

        # Compute routing probabilities
        router_probs = mx.softmax(router_logits, axis=-1)

        # Compute fraction of tokens routed to each expert
        total_tokens = batch_size * seq_len * self.top_k

        # Count tokens for each expert using list comprehension
        expert_counts = []
        for e in range(self.num_experts):
            count = mx.sum(mx.sum(expert_indices == e, axis=-1))  # Sum over top_k, then all
            expert_counts.append(count)

        expert_usage = mx.stack(expert_counts)
        expert_fraction = expert_usage.astype(mx.float32) / total_tokens

        # Compute mean routing probability to each expert
        mean_router_prob = mx.mean(router_probs, axis=(0, 1))

        # Load balancing loss: encourage uniform distribution
        aux_loss = self.num_experts * mx.sum(expert_fraction * mean_router_prob)

        return aux_loss


class SwitchMoE(nn.Module):
    """Switch Transformer style MoE (top-1 routing).

    Simpler MoE that routes each token to exactly one expert.

    Args:
        dims: Input/output dimension.
        hidden_dims: Hidden dimension for expert MLPs.
        num_experts: Number of expert networks.
        capacity_factor: Expert capacity as fraction of batch size.
        activation: Activation function.

    Reference:
        "Switch Transformers: Scaling to Trillion Parameter Models"
        https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        dims: int,
        hidden_dims: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        activation: Callable = nn.silu,
    ):
        super().__init__()
        self.dims = dims
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(dims, num_experts, bias=False)
        self.experts = [
            Expert(dims, hidden_dims, activation) for _ in range(num_experts)
        ]

    def __call__(
        self, x: mx.array, training: bool = False
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass through Switch MoE.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode.

        Returns:
            Tuple of (output, auxiliary_loss)
        """
        batch_size, seq_len, dims = x.shape

        # Compute gating scores
        router_logits = self.gate(x)
        router_probs = mx.softmax(router_logits, axis=-1)

        # Top-1 selection
        expert_indices = mx.argmax(router_logits, axis=-1)  # (batch, seq)
        expert_weights = mx.max(router_probs, axis=-1, keepdims=True)  # (batch, seq, 1)

        # Initialize output
        output = mx.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx)[..., None]  # (batch, seq, 1)
            if mx.any(mask):
                expert_out = self.experts[expert_idx](x)
                output = output + mx.where(mask, expert_out * expert_weights, 0.0)

        # Auxiliary loss
        # Fraction of tokens to each expert
        counts_list = []
        for e in range(self.num_experts):
            counts_list.append(mx.sum(expert_indices == e).astype(mx.float32))
        expert_counts = mx.stack(counts_list)
        expert_fraction = expert_counts / (batch_size * seq_len)

        # Mean routing probability
        mean_router_prob = mx.mean(router_probs, axis=(0, 1))

        aux_loss = self.num_experts * mx.sum(expert_fraction * mean_router_prob)

        return output, aux_loss


def load_balancing_loss(
    router_logits: mx.array,
    expert_indices: mx.array,
    num_experts: int,
) -> mx.array:
    """Compute load balancing auxiliary loss.

    Encourages balanced routing across experts.

    Args:
        router_logits: Router logits (batch, seq, num_experts).
        expert_indices: Selected expert indices (batch, seq, top_k).
        num_experts: Total number of experts.

    Returns:
        Auxiliary loss value.

    Reference:
        GShard: Scaling Giant Models with Conditional Computation
    """
    batch_size, seq_len, _ = router_logits.shape
    top_k = expert_indices.shape[-1]

    # Routing probabilities
    router_probs = mx.softmax(router_logits, axis=-1)

    # Expert usage counts
    total_tokens = batch_size * seq_len * top_k

    # Count tokens for each expert
    expert_counts = []
    for e in range(num_experts):
        count = mx.sum(expert_indices == e)  # Count across all positions and top_k
        expert_counts.append(count)

    expert_fraction = mx.stack(expert_counts).astype(mx.float32) / total_tokens

    # Mean routing probability
    mean_prob = mx.mean(router_probs, axis=(0, 1))

    # Loss
    return num_experts * mx.sum(expert_fraction * mean_prob)


def router_z_loss(router_logits: mx.array) -> mx.array:
    """Router z-loss for training stability.

    Penalizes large router logits to prevent router collapse.

    Args:
        router_logits: Router logits (batch, seq, num_experts).

    Returns:
        Z-loss value.

    Reference:
        ST-MoE: Designing Stable and Transferable Sparse Expert Models
    """
    # Log-sum-exp of router logits
    log_z = mx.logsumexp(router_logits, axis=-1)
    return mx.mean(log_z ** 2)
