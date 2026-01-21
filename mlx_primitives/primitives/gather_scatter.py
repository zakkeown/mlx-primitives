"""Selective gather/scatter operations for Mixture of Experts.

These operations enable efficient sparse MoE routing without materializing
full tensors. Instead of running all tokens through all experts and masking,
we gather only the tokens that route to each expert, compute, and scatter back.

This provides significant memory and compute savings for MoE models:
- Memory: O(tokens * top_k * d) instead of O(tokens * experts * d)
- Compute: O(tokens * top_k) forward passes instead of O(tokens * experts)
"""

import threading
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_primitives.advanced.moe import MoEOutput

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Thread-safe kernel cache
_kernel_lock = threading.Lock()
_gather_kernel: Optional[mx.fast.metal_kernel] = None
_scatter_add_kernel: Optional[mx.fast.metal_kernel] = None


def _get_gather_kernel() -> mx.fast.metal_kernel:
    """Get or create the selective gather kernel (thread-safe)."""
    global _gather_kernel
    if _gather_kernel is not None:
        return _gather_kernel

    with _kernel_lock:
        # Double-check after acquiring lock
        if _gather_kernel is None:
            source = """
            uint idx = thread_position_in_grid.x;
            uint d = thread_position_in_grid.y;

            // Dereference scalar parameters (passed as single-element arrays)
            uint _capacity = capacity[0];
            uint _dim = dim[0];

            if (idx >= _capacity || d >= _dim) return;

            uint src_idx = indices[idx];
            output[idx * _dim + d] = input[src_idx * _dim + d];
            """
            _gather_kernel = mx.fast.metal_kernel(
                name="selective_gather",
                input_names=["input", "indices", "capacity", "dim"],
                output_names=["output"],
                source=source,
            )
    return _gather_kernel


def _get_scatter_add_kernel() -> mx.fast.metal_kernel:
    """Get or create the selective scatter-add kernel (thread-safe).

    Uses atomic operations for thread-safe accumulation.
    Takes accumulator as input and atomically adds values to it.
    """
    global _scatter_add_kernel
    if _scatter_add_kernel is not None:
        return _scatter_add_kernel

    with _kernel_lock:
        # Double-check after acquiring lock
        if _scatter_add_kernel is None:
            source = """
            uint idx = thread_position_in_grid.x;
            uint d = thread_position_in_grid.y;

            // Dereference scalar parameters (passed as single-element arrays)
            uint _capacity = capacity[0];
            uint _dim = dim[0];
            uint _n_tokens = n_tokens[0];

            if (idx >= _capacity || d >= _dim) return;

            uint dst_idx = indices[idx];
            float weight = weights[idx];
            float value = values[idx * _dim + d] * weight;

            // Copy accumulator to output (first thread for each output position)
            // Then atomically add the scatter value
            // Note: This relies on accumulator being pre-copied to output
            atomic_fetch_add_explicit(
                (device atomic_float*)&output[dst_idx * _dim + d],
                value,
                memory_order_relaxed
            );
            """
            _scatter_add_kernel = mx.fast.metal_kernel(
                name="selective_scatter_add",
                input_names=["values", "indices", "weights", "capacity", "dim", "n_tokens"],
                output_names=["output"],
                source=source,
            )
    return _scatter_add_kernel


def selective_gather(
    x: mx.array,
    indices: mx.array,
    use_metal: bool = True,
    differentiable: bool = False,
) -> mx.array:
    """Gather selected rows from input tensor.

    Extracts rows from x at the positions specified by indices.
    This is equivalent to x[indices] but optimized for MoE routing
    where we gather tokens for each expert.

    Args:
        x: Input tensor of shape (n_tokens, dim) or (n_tokens, ...).
        indices: 1D array of indices to gather, shape (capacity,).
        use_metal: Use Metal kernel if available.
        differentiable: If True, use pure MLX operations that support gradients.

    Returns:
        Gathered tensor of shape (capacity, dim) or (capacity, ...).

    Example:
        >>> x = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> indices = mx.array([0, 2, 3])
        >>> selective_gather(x, indices)
        array([[1, 2], [5, 6], [7, 8]])
    """
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices.ndim}D")

    # For simple cases, when Metal not available, or when differentiable, use indexing
    if not use_metal or not _HAS_METAL or x.ndim > 2 or differentiable:
        return x[indices]

    # Ensure 2D input
    original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_tokens, dim = x.shape
    capacity = indices.shape[0]

    # Use Metal kernel for 2D case
    try:
        kernel = _get_gather_kernel()

        x = mx.contiguous(x.astype(mx.float32))
        indices = mx.contiguous(indices.astype(mx.uint32))
        capacity_arr = mx.array([capacity], dtype=mx.uint32)
        dim_arr = mx.array([dim], dtype=mx.uint32)

        outputs = kernel(
            inputs=[x, indices, capacity_arr, dim_arr],
            grid=(capacity, dim, 1),
            threadgroup=(min(capacity, 32), min(dim, 32), 1),
            output_shapes=[(capacity, dim)],
            output_dtypes=[mx.float32],
            stream=mx.default_stream(mx.default_device()),
        )

        result = outputs[0]

        # Restore original dimensionality
        if len(original_shape) == 1:
            result = result.squeeze(-1)

        return result
    except (ImportError, RuntimeError, ValueError, TypeError) as e:
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("selective_gather", e)
        # Fallback to simple indexing
        return x[indices]


def selective_scatter_add(
    output: mx.array,
    values: mx.array,
    indices: mx.array,
    weights: mx.array,
    use_metal: bool = True,
    differentiable: bool = False,
) -> mx.array:
    """Scatter-add values into output tensor with routing weights.

    For each (value, index, weight) triple, adds value * weight to
    output[index]. This is the reverse of gather, used to accumulate
    expert outputs back into the original token positions.

    Note: This operation modifies output in-place when using the Metal
    kernel. The returned array is the modified output.

    Args:
        output: Accumulator tensor of shape (n_tokens, dim). Will be modified.
        values: Values to scatter, shape (capacity, dim).
        indices: Where to scatter each value, shape (capacity,).
        weights: Routing weights for each value, shape (capacity,).
        use_metal: Use Metal kernel if available.
        differentiable: If True, use pure MLX operations that support gradients.

    Returns:
        The modified output tensor.

    Example:
        >>> output = mx.zeros((4, 2))
        >>> values = mx.array([[1, 2], [3, 4]])
        >>> indices = mx.array([0, 2])
        >>> weights = mx.array([0.5, 0.5])
        >>> selective_scatter_add(output, values, indices, weights)
        array([[0.5, 1], [0, 0], [1.5, 2], [0, 0]])
    """
    if indices.ndim != 1 or weights.ndim != 1:
        raise ValueError("indices and weights must be 1D")
    if values.shape[0] != indices.shape[0] or values.shape[0] != weights.shape[0]:
        raise ValueError("values, indices, and weights must have same first dimension")

    capacity = indices.shape[0]
    n_tokens, dim = output.shape

    # For simple cases, fallback, or when differentiable (Metal kernels don't support VJP)
    if not use_metal or not _HAS_METAL or values.ndim > 2 or differentiable:
        # Fully vectorized MLX implementation - no GPU sync
        # Use broadcasting to scatter-add without explicit loops
        #
        # We want: output[indices[i]] += values[i] * weights[i] for all i
        # Vectorized: for each output position j, sum all (values[i] * weights[i])
        # where indices[i] == j
        #
        # indices: (capacity,)
        # values: (capacity, dim)
        # weights: (capacity,)
        weighted_values = values * weights[:, None]  # (capacity, dim)

        # Create indicator matrix: indicator[i, j] = 1 if indices[i] == j
        # Shape: (capacity, n_tokens)
        position_indices = mx.arange(n_tokens)[None, :]  # (1, n_tokens)
        indicator = (indices[:, None] == position_indices).astype(output.dtype)  # (capacity, n_tokens)

        # Scatter-add via matrix multiply: updates = indicator.T @ weighted_values
        # This sums weighted_values[i] for all i where indices[i] == j
        updates = indicator.T @ weighted_values  # (n_tokens, dim)

        return output + updates

    try:
        kernel = _get_scatter_add_kernel()

        # Prepare inputs
        accumulator = mx.contiguous(output.astype(mx.float32))
        values = mx.contiguous(values.astype(mx.float32))
        indices = mx.contiguous(indices.astype(mx.uint32))
        weights = mx.contiguous(weights.astype(mx.float32))
        capacity_arr = mx.array([capacity], dtype=mx.uint32)
        dim_arr = mx.array([dim], dtype=mx.uint32)
        n_tokens_arr = mx.array([n_tokens], dtype=mx.uint32)

        # Initialize output with accumulator values (scalar init_value 0.0)
        # Then atomically add scatter values
        outputs = kernel(
            inputs=[values, indices, weights, capacity_arr, dim_arr, n_tokens_arr],
            grid=(capacity, dim, 1),
            threadgroup=(min(capacity, 32), min(dim, 32), 1),
            output_shapes=[(n_tokens, dim)],
            output_dtypes=[mx.float32],
            init_value=0.0,  # Initialize with zeros, add accumulator separately
            stream=mx.default_stream(mx.default_device()),
        )

        # Add accumulator to the scattered output
        return outputs[0] + accumulator
    except (ImportError, RuntimeError, ValueError, TypeError) as e:
        from mlx_primitives.utils.logging import log_fallback
        log_fallback("selective_scatter_add", e)
        # Fallback to pure MLX - vectorized via indicator matrix (no GPU sync)
        weighted_values = values * weights[:, None]
        position_indices = mx.arange(n_tokens)[None, :]
        indicator = (indices[:, None] == position_indices).astype(output.dtype)
        updates = indicator.T @ weighted_values
        return output + updates


@dataclass
class ExpertDispatch:
    """Dispatch information for routing tokens to experts.

    Attributes:
        expert_indices: Dict mapping expert_idx -> array of token indices.
        expert_weights: Dict mapping expert_idx -> array of routing weights.
        expert_counts: Array of token counts per expert.
    """

    expert_indices: dict[int, mx.array]
    expert_weights: dict[int, mx.array]
    expert_counts: mx.array


def build_expert_dispatch(
    gate_logits: mx.array,
    num_experts: int,
    top_k: int = 2,
    capacity_factor: float = 1.25,
) -> Tuple[ExpertDispatch, mx.array]:
    """Build dispatch indices from gate logits.

    Given routing logits, compute which tokens go to which experts
    and build the index arrays needed for selective gather/scatter.

    Args:
        gate_logits: Router logits of shape (n_tokens, num_experts).
        num_experts: Number of experts.
        top_k: Number of experts per token.
        capacity_factor: Multiplier for expert capacity.
            capacity = capacity_factor * n_tokens * top_k / num_experts

    Returns:
        Tuple of (ExpertDispatch, router_probs).

    Example:
        >>> logits = mx.random.normal((100, 8))  # 100 tokens, 8 experts
        >>> dispatch, probs = build_expert_dispatch(logits, num_experts=8, top_k=2)
        >>> # dispatch.expert_indices[0] contains indices of tokens for expert 0
    """
    n_tokens = gate_logits.shape[0]

    # Compute routing probabilities
    router_probs = mx.softmax(gate_logits, axis=-1)

    # Select top-k experts per token
    top_k_indices = mx.argpartition(-router_probs, top_k, axis=-1)[:, :top_k]
    top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)

    # Normalize weights to sum to 1
    top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)

    # Build per-expert dispatch lists
    capacity = int(capacity_factor * n_tokens * top_k / num_experts)

    expert_indices = {}
    expert_weights = {}
    expert_counts = mx.zeros((num_experts,), dtype=mx.int32)

    # Create token indices array
    all_token_indices = mx.arange(n_tokens, dtype=mx.uint32)

    for expert_idx in range(num_experts):
        # Find tokens assigned to this expert
        mask = mx.any(top_k_indices == expert_idx, axis=-1)

        # Get weights for tokens going to this expert
        expert_mask = top_k_indices == expert_idx
        weights_per_token = mx.where(expert_mask, top_k_weights, 0.0)
        weights_per_token = mx.sum(weights_per_token, axis=-1)

        # Use mask to filter token indices and weights
        # Count how many tokens are assigned to this expert
        count = int(mx.sum(mask).item())

        if count > 0:
            # Get indices where mask is True using sorting trick:
            # Sort indices by (1-mask) so True values come first
            sort_keys = 1.0 - mask.astype(mx.float32)
            sorted_order = mx.argsort(sort_keys)
            token_indices = all_token_indices[sorted_order[:count]]
            token_weights = weights_per_token[sorted_order[:count]]

            # Enforce capacity
            if count > capacity:
                # Keep tokens with highest weights
                keep_indices = mx.argpartition(-token_weights, capacity)[:capacity]
                token_indices = token_indices[keep_indices]
                token_weights = token_weights[keep_indices]
                count = capacity

            expert_indices[expert_idx] = token_indices.astype(mx.uint32)
            expert_weights[expert_idx] = token_weights
            expert_counts = expert_counts.at[expert_idx].add(count)
        else:
            expert_indices[expert_idx] = mx.array([], dtype=mx.uint32)
            expert_weights[expert_idx] = mx.array([], dtype=mx.float32)

    dispatch = ExpertDispatch(
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        expert_counts=expert_counts,
    )

    return dispatch, router_probs


class SparseMoELayer(nn.Module):
    """Mixture of Experts layer with sparse routing.

    Uses selective gather/scatter for efficient computation where only
    tokens assigned to each expert are processed, rather than running
    all tokens through all experts and masking.

    Inherits from nn.Module for proper weight serialization and gradient tracking.

    Args:
        num_experts: Number of expert networks.
        d_model: Model dimension.
        d_hidden: Hidden dimension in expert MLPs.
        top_k: Number of experts per token.
        capacity_factor: Capacity multiplier for load balancing.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router - uses nn.Linear for proper weight management
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Expert MLPs (SwiGLU) - store as lists of nn.Linear
        # Each expert: gate_proj, up_proj -> down_proj
        self.gate_proj = [nn.Linear(d_model, d_hidden, bias=False) for _ in range(num_experts)]
        self.up_proj = [nn.Linear(d_model, d_hidden, bias=False) for _ in range(num_experts)]
        self.down_proj = [nn.Linear(d_hidden, d_model, bias=False) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> "MoEOutput":
        """Forward pass with sparse routing.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model) or (n_tokens, d_model).

        Returns:
            MoEOutput with output, aux_loss, and router_logits.
            Can be unpacked as: output, aux_loss, router_logits = layer(x)
        """
        from mlx_primitives.advanced.moe import MoEOutput

        # Flatten to (n_tokens, d_model)
        original_shape = x.shape
        if x.ndim == 3:
            batch, seq_len, d_model = x.shape
            x = x.reshape(-1, d_model)
            router_logits_shape = (batch, seq_len, self.num_experts)
        else:
            router_logits_shape = None
        n_tokens = x.shape[0]

        # Compute router logits
        router_logits = self.gate(x)  # (n_tokens, num_experts)

        # Build dispatch indices
        dispatch, router_probs = build_expert_dispatch(
            router_logits,
            self.num_experts,
            self.top_k,
            self.capacity_factor,
        )

        # Process each expert
        output = mx.zeros_like(x)

        for expert_idx in range(self.num_experts):
            indices = dispatch.expert_indices[expert_idx]
            weights = dispatch.expert_weights[expert_idx]

            if indices.size == 0:
                continue

            # Gather tokens for this expert
            x_expert = selective_gather(x, indices)

            # Expert forward (SwiGLU MLP) using nn.Linear layers
            gate_out = self.gate_proj[expert_idx](x_expert)
            up_out = self.up_proj[expert_idx](x_expert)
            # SwiGLU activation: SiLU(gate) * up = (sigmoid(gate) * gate) * up
            hidden = mx.sigmoid(gate_out) * gate_out * up_out
            y_expert = self.down_proj[expert_idx](hidden)

            # Scatter back with weights
            output = selective_scatter_add(output, y_expert, indices, weights)

        # Compute auxiliary load balancing loss
        aux_loss = compute_load_balancing_loss(
            router_probs, dispatch.expert_counts, self.num_experts
        )

        # Reshape back
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
            router_logits = router_logits.reshape(router_logits_shape)

        return MoEOutput(output=output, aux_loss=aux_loss, router_logits=router_logits)


def compute_load_balancing_loss(
    router_probs: mx.array,
    expert_counts: mx.array,
    num_experts: int,
) -> mx.array:
    """Compute load balancing auxiliary loss for MoE.

    Encourages uniform distribution of tokens across experts.

    Args:
        router_probs: Softmax router probabilities (n_tokens, num_experts).
        expert_counts: Number of tokens assigned to each expert.
        num_experts: Number of experts.

    Returns:
        Scalar load balancing loss.
    """
    n_tokens = router_probs.shape[0]

    # Fraction of tokens assigned to each expert
    f = expert_counts.astype(mx.float32) / n_tokens

    # Average router probability for each expert
    p = mx.mean(router_probs, axis=0)

    # Loss = num_experts * sum(f * p)
    # Minimized when both f and p are uniform (1/num_experts)
    loss = num_experts * mx.sum(f * p)

    return loss
