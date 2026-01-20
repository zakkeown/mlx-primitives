"""Mixture of Experts layers for MLX.

This module provides MoE components:
- MoELayer: Sparse mixture of experts layer with gather/scatter
- BatchedMoELayer: Performance-optimized MoE with batched expert computation
- BatchedExperts: Batched expert weights for parallel execution
- TopKRouter: Top-k expert routing
- ExpertChoiceRouter: Expert-choice routing
- LoadBalancingLoss: Auxiliary load balancing loss
- MoEOutput: Standardized output dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.primitives.gather_scatter import selective_scatter_add
from mlx_primitives.utils.array_ops import nonzero


@dataclass
class MoEOutput:
    """Standardized output from MoE layers.

    Attributes:
        output: The layer output tensor (batch, seq_len, dims).
        aux_loss: Auxiliary load balancing loss for training.
        router_logits: Raw router logits (batch, seq_len, num_experts).
    """

    output: mx.array
    aux_loss: mx.array
    router_logits: mx.array

    def __iter__(self):
        """Allow unpacking as (output, aux_loss, router_logits)."""
        return iter((self.output, self.aux_loss, self.router_logits))


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
        selected_tokens = sorted_indices[..., :capacity]  # (batch, num_experts, capacity)

        # Create dispatch mask efficiently using scatter operations
        # Memory: O(batch * experts * seq_len) instead of O(batch * experts * capacity * seq_len)
        #
        # Instead of broadcasting comparison, we build the mask incrementally
        # by scattering 1s to the selected positions for each expert
        dispatch_mask = mx.zeros((batch_size, self.num_experts, seq_len), dtype=mx.float32)

        # Process each expert separately to avoid memory explosion
        for e in range(self.num_experts):
            # selected_tokens[:, e, :] has shape (batch, capacity) - token positions for expert e
            expert_selections = selected_tokens[:, e, :]  # (batch, capacity)

            # For each batch, scatter 1s to selected positions
            # Use indicator matrix approach per expert (much smaller than full broadcast)
            for b in range(batch_size):
                # Get selected positions for this batch and expert
                positions = expert_selections[b]  # (capacity,)

                # Create indicator via comparison (capacity, seq_len)
                # This is O(capacity * seq_len) per expert per batch
                # Much smaller than O(batch * experts * capacity * seq_len) all at once
                pos_expanded = positions[:, None]  # (capacity, 1)
                seq_positions = mx.arange(seq_len)[None, :]  # (1, seq_len)
                matches = (pos_expanded == seq_positions).astype(mx.float32)  # (capacity, seq_len)

                # Sum over capacity to get mask for this expert
                expert_mask = mx.sum(matches, axis=0)  # (seq_len,)
                dispatch_mask = dispatch_mask.at[b, e, :].add(expert_mask)

        # Clamp to [0, 1] in case of duplicate selections
        dispatch_mask = mx.minimum(dispatch_mask, 1.0)

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


class BatchedExperts(nn.Module):
    """Batched expert computation for efficient parallel execution.

    Instead of storing experts as separate modules and looping over them,
    this class stores all expert weights in batched tensors and processes
    all experts in a single batched matrix multiplication.

    This eliminates Python loops in the forward pass, providing significant
    speedups for MoE layers.

    Args:
        dims: Input/output dimension.
        hidden_dims: Hidden dimension per expert.
        num_experts: Number of experts.
        activation: Activation function.
    """

    def __init__(
        self,
        dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: Callable = nn.silu,
    ):
        super().__init__()
        self.dims = dims
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts
        self.activation = activation

        # Batched weight matrices: (num_experts, out_dim, in_dim)
        # w1: projects from dims to hidden_dims
        # w2: projects from hidden_dims back to dims
        scale1 = (1.0 / dims) ** 0.5
        scale2 = (1.0 / hidden_dims) ** 0.5
        self.w1 = mx.random.uniform(
            low=-scale1,
            high=scale1,
            shape=(num_experts, hidden_dims, dims),
        )
        self.w2 = mx.random.uniform(
            low=-scale2,
            high=scale2,
            shape=(num_experts, dims, hidden_dims),
        )

    def __call__(
        self,
        x: mx.array,
        expert_indices: mx.array,
        expert_weights: mx.array,
        capacity: int,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Process all experts in parallel using batched operations.

        Args:
            x: Flattened input tensor (n_tokens, dims).
            expert_indices: Expert index per token (n_tokens, top_k).
            expert_weights: Routing weights per token (n_tokens, top_k).
            capacity: Maximum tokens per expert.

        Returns:
            Tuple of (expert_outputs, routed_indices, routed_weights) where:
            - expert_outputs: (num_experts, capacity, dims)
            - routed_indices: (num_experts, capacity) - token indices
            - routed_weights: (num_experts, capacity) - routing weights
        """
        n_tokens = x.shape[0]

        # Pre-compute routing info for all experts at once
        # Create indicator of which tokens go to which expert
        # expert_indices: (n_tokens, top_k)
        # We need: for each expert, which token indices are routed to it

        # Gather tokens for each expert using vectorized operations
        # Build (num_experts, capacity) index matrices and weight matrices
        routed_indices_list = []
        routed_weights_list = []

        for expert_idx in range(self.num_experts):
            # Mask where this expert is selected
            expert_mask = expert_indices == expert_idx  # (n_tokens, top_k)
            # Weights for tokens routed to this expert
            expert_weights_per_topk = mx.where(expert_mask, expert_weights, 0.0)
            weights_sum = mx.sum(expert_weights_per_topk, axis=-1)  # (n_tokens,)

            # Sort by weight to get top 'capacity' tokens
            sorted_order = mx.argsort(-weights_sum)
            top_indices = sorted_order[:capacity]
            top_weights = weights_sum[top_indices]

            routed_indices_list.append(top_indices)
            routed_weights_list.append(top_weights)

        # Stack into batched tensors
        routed_indices = mx.stack(routed_indices_list, axis=0)  # (num_experts, capacity)
        routed_weights = mx.stack(routed_weights_list, axis=0)  # (num_experts, capacity)

        # Gather tokens for all experts at once
        # x: (n_tokens, dims)
        # routed_indices: (num_experts, capacity)
        # We want: x_gathered[e, c, :] = x[routed_indices[e, c], :]
        x_gathered = x[routed_indices]  # (num_experts, capacity, dims)

        # Batched expert computation: all experts in parallel
        # First projection: (num_experts, capacity, dims) @ (num_experts, dims, hidden_dims)
        # Transpose w1 from (num_experts, hidden_dims, dims) to (num_experts, dims, hidden_dims)
        w1_t = mx.transpose(self.w1, (0, 2, 1))
        hidden = mx.matmul(x_gathered, w1_t)  # (num_experts, capacity, hidden_dims)

        # Activation
        hidden = self.activation(hidden)

        # Second projection: (num_experts, capacity, hidden_dims) @ (num_experts, hidden_dims, dims)
        # Transpose w2 from (num_experts, dims, hidden_dims) to (num_experts, hidden_dims, dims)
        w2_t = mx.transpose(self.w2, (0, 2, 1))
        expert_outputs = mx.matmul(hidden, w2_t)  # (num_experts, capacity, dims)

        return expert_outputs, routed_indices, routed_weights


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing.

    Routes tokens to top-k experts and combines their weighted outputs.
    Uses truly sparse gather/scatter computation for efficiency.

    This implementation uses selective_gather to extract only tokens routed
    to each expert, computes expert outputs, then uses scatter_add to
    accumulate results back. This provides:
    - Compute savings: O(n_tokens * top_k) instead of O(n_tokens * num_experts)
    - Memory savings: Only routed tokens are processed per expert

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
    ) -> MoEOutput:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode.

        Returns:
            MoEOutput with output, aux_loss, and router_logits.
            Can be unpacked as: output, aux_loss, router_logits = layer(x)

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.ndim != 3:
            raise ValueError(
                f"MoELayer expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        batch_size, seq_len, dims = x.shape

        # Get routing decisions
        gate_weights, expert_indices, router_logits = self.router(x, training)

        # Flatten to (batch * seq_len, dims) for sparse processing
        x_flat = x.reshape(-1, dims)  # (n_tokens, dims)
        n_tokens = batch_size * seq_len

        # Flatten routing info
        gate_weights_flat = gate_weights.reshape(n_tokens, self.top_k)  # (n_tokens, top_k)
        expert_indices_flat = expert_indices.reshape(n_tokens, self.top_k)  # (n_tokens, top_k)

        # Fixed capacity per expert to avoid GPU sync
        # This is an upper bound - we process capacity tokens but weights=0 for non-routed
        capacity = (n_tokens * self.top_k + self.num_experts - 1) // self.num_experts
        capacity = max(1, capacity)

        # Pre-compute all expert routing info WITHOUT GPU sync
        expert_dispatch = []
        for expert_idx in range(self.num_experts):
            expert_mask = expert_indices_flat == expert_idx  # (n_tokens, top_k)
            expert_weights_per_topk = mx.where(expert_mask, gate_weights_flat, 0.0)
            expert_weights = mx.sum(expert_weights_per_topk, axis=-1)  # (n_tokens,)

            # Use argsort trick to get indices without nonzero
            # Sort by (-weight) so highest weights (routed tokens) come first
            # Non-routed tokens have weight=0 and will be at the end
            sort_key = -expert_weights
            sorted_order = mx.argsort(sort_key)

            expert_dispatch.append((sorted_order, expert_weights))

        # Initialize output
        output_flat = mx.zeros((n_tokens, dims), dtype=x.dtype)

        # Process each expert with fixed capacity (NO GPU SYNC)
        # Non-routed tokens have weight=0, so they contribute nothing to output
        for expert_idx in range(self.num_experts):
            sorted_order, expert_weights = expert_dispatch[expert_idx]

            # Take first 'capacity' tokens (includes both routed and padding)
            routed_indices = sorted_order[:capacity]
            x_expert = x_flat[routed_indices]  # (capacity, dims)

            # Get weights for these tokens (0 for non-routed)
            weights_expert = expert_weights[routed_indices]  # (capacity,)

            # Process through expert (all capacity tokens)
            expert_out = self.experts[expert_idx](x_expert)  # (capacity, dims)

            # Scatter-add back to output with weights (using Metal kernel)
            # Non-routed tokens have weight=0, contributing nothing
            output_flat = selective_scatter_add(
                output_flat,
                expert_out,
                routed_indices.astype(mx.uint32),
                weights_expert,
            )

        # Reshape back to (batch, seq_len, dims)
        output = output_flat.reshape(batch_size, seq_len, dims)

        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_logits, expert_indices)

        return MoEOutput(output=output, aux_loss=aux_loss, router_logits=router_logits)

    def _compute_aux_loss(
        self, router_logits: mx.array, expert_indices: mx.array
    ) -> mx.array:
        """Compute auxiliary load balancing loss.

        Encourages balanced expert usage. Delegates to the standalone
        load_balancing_loss function to avoid code duplication.
        """
        return load_balancing_loss(router_logits, expert_indices, self.num_experts)


class BatchedMoELayer(nn.Module):
    """Mixture of Experts layer with batched expert computation.

    This is a performance-optimized version of MoELayer that eliminates Python
    loops in the forward pass by using batched matrix multiplications across
    all experts simultaneously.

    Performance benefits:
    - Expert computation is parallelized via batched matmul
    - Single gather/scatter operations instead of per-expert loops
    - Better GPU utilization due to larger batch sizes

    Args:
        dims: Input/output dimension.
        hidden_dims: Hidden dimension for expert MLPs.
        num_experts: Number of expert networks.
        top_k: Number of experts per token (default: 2).
        activation: Activation function for experts.
        jitter_noise: Router jitter noise (default: 0.0).

    Example:
        >>> moe = BatchedMoELayer(dims=768, hidden_dims=3072, num_experts=8, top_k=2)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output, aux_loss, router_logits = moe(x)
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
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(
            dims=dims,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )

        # Batched experts for parallel computation
        self.experts = BatchedExperts(
            dims=dims,
            hidden_dims=hidden_dims,
            num_experts=num_experts,
            activation=activation,
        )

    def __call__(
        self, x: mx.array, training: bool = False
    ) -> MoEOutput:
        """Forward pass through MoE layer with batched expert computation.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode.

        Returns:
            MoEOutput with output, aux_loss, and router_logits.
        """
        if x.ndim != 3:
            raise ValueError(
                f"BatchedMoELayer expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        batch_size, seq_len, dims = x.shape

        # Get routing decisions
        gate_weights, expert_indices, router_logits = self.router(x, training)

        # Flatten to (batch * seq_len, dims) for sparse processing
        x_flat = x.reshape(-1, dims)  # (n_tokens, dims)
        n_tokens = batch_size * seq_len

        # Flatten routing info
        gate_weights_flat = gate_weights.reshape(n_tokens, self.top_k)  # (n_tokens, top_k)
        expert_indices_flat = expert_indices.reshape(n_tokens, self.top_k)  # (n_tokens, top_k)

        # Fixed capacity per expert to avoid GPU sync
        capacity = (n_tokens * self.top_k + self.num_experts - 1) // self.num_experts
        capacity = max(1, capacity)

        # Process all experts in parallel using batched computation
        expert_outputs, routed_indices, routed_weights = self.experts(
            x_flat, expert_indices_flat, gate_weights_flat, capacity
        )
        # expert_outputs: (num_experts, capacity, dims)
        # routed_indices: (num_experts, capacity)
        # routed_weights: (num_experts, capacity)

        # Initialize output
        output_flat = mx.zeros((n_tokens, dims), dtype=x.dtype)

        # Scatter-add all expert outputs back in a single loop
        # Note: We still need a loop for scatter since each expert scatters to different positions
        # But the expensive computation (expert forward) is now batched
        for expert_idx in range(self.num_experts):
            output_flat = selective_scatter_add(
                output_flat,
                expert_outputs[expert_idx],  # (capacity, dims)
                routed_indices[expert_idx].astype(mx.uint32),  # (capacity,)
                routed_weights[expert_idx],  # (capacity,)
            )

        # Reshape back to (batch, seq_len, dims)
        output = output_flat.reshape(batch_size, seq_len, dims)

        # Compute auxiliary load balancing loss
        aux_loss = load_balancing_loss(router_logits, expert_indices, self.num_experts)

        return MoEOutput(output=output, aux_loss=aux_loss, router_logits=router_logits)


class SwitchMoE(nn.Module):
    """Switch Transformer style MoE (top-1 routing).

    Simpler MoE that routes each token to exactly one expert.
    Uses masked computation to apply weights per-token.

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
    ) -> MoEOutput:
        """Forward pass through Switch MoE.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            training: Whether in training mode.

        Returns:
            MoEOutput with output, aux_loss, and router_logits.
            Can be unpacked as: output, aux_loss, router_logits = layer(x)

        Raises:
            ValueError: If input tensor has incorrect dimensions.
        """
        if x.ndim != 3:
            raise ValueError(
                f"SwitchMoE expects 3D input (batch, seq_len, dims), "
                f"got {x.ndim}D with shape {x.shape}"
            )

        batch_size, seq_len, dims = x.shape

        # Compute gating scores
        router_logits = self.gate(x)
        router_probs = mx.softmax(router_logits, axis=-1)

        # Top-1 selection
        expert_indices = mx.argmax(router_logits, axis=-1)  # (batch, seq)
        expert_weights = mx.max(router_probs, axis=-1)  # (batch, seq)

        # Flatten for sparse processing
        x_flat = x.reshape(-1, dims)  # (n_tokens, dims)
        n_tokens = batch_size * seq_len
        expert_indices_flat = expert_indices.reshape(-1)  # (n_tokens,)
        expert_weights_flat = expert_weights.reshape(-1)  # (n_tokens,)

        # Fixed capacity per expert to avoid GPU sync
        # For top-1 routing, capacity = ceil(n_tokens / num_experts)
        capacity = (n_tokens + self.num_experts - 1) // self.num_experts
        capacity = max(1, capacity)

        # Pre-compute routing for all experts WITHOUT GPU sync
        expert_dispatch = []
        for expert_idx in range(self.num_experts):
            routed_mask = expert_indices_flat == expert_idx
            # Sort by (1-mask, -weight) so routed tokens with highest weight come first
            # Non-routed tokens will be at the end with weight 0
            effective_weight = mx.where(routed_mask, expert_weights_flat, 0.0)
            sort_key = -effective_weight  # Descending by weight
            sorted_order = mx.argsort(sort_key)
            expert_dispatch.append((sorted_order, effective_weight))

        # Initialize output
        output_flat = mx.zeros((n_tokens, dims), dtype=x.dtype)

        # Process each expert with fixed capacity (NO GPU SYNC)
        for expert_idx in range(self.num_experts):
            sorted_order, effective_weight = expert_dispatch[expert_idx]

            # Take first 'capacity' tokens
            routed_indices = sorted_order[:capacity]
            x_expert = x_flat[routed_indices]  # (capacity, dims)
            weights_expert = effective_weight[routed_indices]  # (capacity,)

            # Process through expert
            expert_out = self.experts[expert_idx](x_expert)  # (capacity, dims)

            # Scatter-add back to output with weights
            # Non-routed tokens have weight=0, contributing nothing
            output_flat = selective_scatter_add(
                output_flat,
                expert_out,
                routed_indices.astype(mx.uint32),
                weights_expert,
            )

        # Reshape back
        output = output_flat.reshape(batch_size, seq_len, dims)

        # Auxiliary loss - compute counts without sync using one-hot encoding
        # expert_indices_flat: (n_tokens,) with values in [0, num_experts)
        one_hot = mx.zeros((n_tokens, self.num_experts))
        # Create one-hot encoding via comparison
        for e in range(self.num_experts):
            one_hot = one_hot.at[:, e].add((expert_indices_flat == e).astype(mx.float32))
        expert_counts = mx.sum(one_hot, axis=0)  # (num_experts,)
        expert_fraction = expert_counts / n_tokens

        # Mean routing probability
        mean_router_prob = mx.mean(router_probs, axis=(0, 1))

        aux_loss = self.num_experts * mx.sum(expert_fraction * mean_router_prob)

        return MoEOutput(output=output, aux_loss=aux_loss, router_logits=router_logits)


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
