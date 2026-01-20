"""NumPy reference implementations for Mixture of Experts (MoE)."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def topk_routing(
    router_logits: np.ndarray,
    k: int,
    normalize: bool = True,
) -> tuple:
    """Top-K routing for Mixture of Experts.

    Selects top-k experts for each token based on router logits.

    Args:
        router_logits: Router output, shape (batch * seq_len, num_experts)
        k: Number of experts to route to per token
        normalize: Whether to normalize routing weights

    Returns:
        Tuple of:
        - routing_weights: Weights for selected experts, shape (batch * seq_len, k)
        - selected_experts: Expert indices, shape (batch * seq_len, k)
    """
    num_tokens, num_experts = router_logits.shape

    # Get top-k expert indices
    selected_experts = np.argsort(router_logits, axis=-1)[:, -k:][:, ::-1]

    # Get weights for selected experts
    routing_weights = np.take_along_axis(router_logits, selected_experts, axis=-1)

    if normalize:
        # Softmax over selected experts only
        routing_weights = softmax(routing_weights, axis=-1)
    else:
        # Apply softmax to all experts then select
        all_weights = softmax(router_logits, axis=-1)
        routing_weights = np.take_along_axis(all_weights, selected_experts, axis=-1)

    return routing_weights, selected_experts


def load_balancing_loss(
    router_logits: np.ndarray,
    expert_mask: np.ndarray = None,
    num_experts: int = None,
) -> float:
    """Load Balancing Loss for MoE training.

    Encourages equal expert utilization:
    L = num_experts * sum_i (f_i * P_i)

    where:
    - f_i = fraction of tokens routed to expert i
    - P_i = mean probability of routing to expert i

    Args:
        router_logits: Router output, shape (batch * seq_len, num_experts)
        expert_mask: One-hot mask of selected experts, shape (batch * seq_len, num_experts)
                    If None, will use argmax of router_logits
        num_experts: Number of experts (inferred from logits if not provided)

    Returns:
        Load balancing loss scalar
    """
    if num_experts is None:
        num_experts = router_logits.shape[-1]

    num_tokens = router_logits.shape[0]

    # Get routing probabilities
    router_probs = softmax(router_logits, axis=-1)

    # If no expert mask provided, use argmax
    if expert_mask is None:
        expert_indices = np.argmax(router_logits, axis=-1)
        expert_mask = np.zeros_like(router_logits)
        np.put_along_axis(expert_mask, expert_indices[:, None], 1.0, axis=-1)

    # Fraction of tokens per expert: (num_experts,)
    tokens_per_expert = np.sum(expert_mask, axis=0) / num_tokens

    # Mean probability per expert: (num_experts,)
    mean_prob_per_expert = np.mean(router_probs, axis=0)

    # Load balancing loss
    loss = num_experts * np.sum(tokens_per_expert * mean_prob_per_expert)

    return float(loss)


def router_z_loss(router_logits: np.ndarray) -> float:
    """Router Z-Loss for training stability.

    Penalizes large logits to improve training stability:
    L = mean(log(sum(exp(logits))))^2

    Args:
        router_logits: Router output, shape (batch * seq_len, num_experts)

    Returns:
        Router z-loss scalar
    """
    # Log-sum-exp over experts
    log_z = np.log(np.sum(np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True)), axis=-1))
    log_z = log_z + np.max(router_logits, axis=-1)  # Add back the max for numerical stability

    # Mean squared
    loss = np.mean(log_z ** 2)

    return float(loss)
