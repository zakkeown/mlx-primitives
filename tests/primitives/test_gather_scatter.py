"""Tests for selective gather/scatter operations."""

import mlx.core as mx
import pytest

from mlx_primitives.primitives import (
    ExpertDispatch,
    SparseMoELayer,
    build_expert_dispatch,
    compute_load_balancing_loss,
    selective_gather,
    selective_scatter_add,
)


class TestSelectiveGather:
    """Tests for selective gather operation."""

    def test_simple_gather(self) -> None:
        """Test basic gather operation."""
        x = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        indices = mx.array([0, 2, 3], dtype=mx.uint32)
        result = selective_gather(x, indices)
        expected = mx.array([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]])
        assert mx.allclose(result, expected).item()

    def test_gather_all(self) -> None:
        """Test gathering all elements."""
        x = mx.array([[1.0], [2.0], [3.0]])
        indices = mx.array([0, 1, 2], dtype=mx.uint32)
        result = selective_gather(x, indices)
        assert mx.allclose(result, x).item()

    def test_gather_single(self) -> None:
        """Test gathering single element."""
        x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indices = mx.array([1], dtype=mx.uint32)
        result = selective_gather(x, indices)
        expected = mx.array([[4.0, 5.0, 6.0]])
        assert mx.allclose(result, expected).item()

    def test_gather_repeated_indices(self) -> None:
        """Test gathering with repeated indices."""
        x = mx.array([[1.0], [2.0], [3.0]])
        indices = mx.array([0, 0, 1, 1], dtype=mx.uint32)
        result = selective_gather(x, indices)
        expected = mx.array([[1.0], [1.0], [2.0], [2.0]])
        assert mx.allclose(result, expected).item()

    def test_gather_large_batch(self) -> None:
        """Test gathering from large tensor."""
        n_tokens = 1000
        dim = 64
        capacity = 100

        x = mx.random.normal((n_tokens, dim))
        indices = mx.random.randint(0, n_tokens, (capacity,)).astype(mx.uint32)

        result = selective_gather(x, indices, use_metal=True)
        expected = x[indices]

        assert mx.allclose(result, expected, rtol=1e-4, atol=1e-4).item()


class TestSelectiveScatterAdd:
    """Tests for selective scatter-add operation."""

    def test_simple_scatter(self) -> None:
        """Test basic scatter-add operation."""
        output = mx.zeros((4, 2))
        values = mx.array([[1.0, 2.0], [3.0, 4.0]])
        indices = mx.array([0, 2], dtype=mx.uint32)
        weights = mx.array([1.0, 1.0])

        result = selective_scatter_add(output, values, indices, weights, use_metal=False)

        expected = mx.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        assert mx.allclose(result, expected).item()

    def test_scatter_with_weights(self) -> None:
        """Test scatter-add with non-unit weights."""
        output = mx.zeros((3, 2))
        values = mx.array([[2.0, 4.0], [6.0, 8.0]])
        indices = mx.array([0, 1], dtype=mx.uint32)
        weights = mx.array([0.5, 0.25])

        result = selective_scatter_add(output, values, indices, weights, use_metal=False)

        expected = mx.array([[1.0, 2.0], [1.5, 2.0], [0.0, 0.0]])
        assert mx.allclose(result, expected).item()

    def test_scatter_accumulate(self) -> None:
        """Test that scatter-add accumulates to same index."""
        output = mx.zeros((2, 1))
        values = mx.array([[1.0], [2.0], [3.0]])
        indices = mx.array([0, 0, 1], dtype=mx.uint32)
        weights = mx.array([1.0, 1.0, 1.0])

        result = selective_scatter_add(output, values, indices, weights, use_metal=False)

        # indices 0 gets 1.0 + 2.0 = 3.0
        # indices 1 gets 3.0
        expected = mx.array([[3.0], [3.0]])
        assert mx.allclose(result, expected).item()


class TestBuildExpertDispatch:
    """Tests for expert dispatch building."""

    def test_dispatch_shape(self) -> None:
        """Test that dispatch has correct structure."""
        n_tokens = 100
        num_experts = 8
        top_k = 2

        gate_logits = mx.random.normal((n_tokens, num_experts))
        dispatch, router_probs = build_expert_dispatch(
            gate_logits, num_experts, top_k
        )

        assert isinstance(dispatch, ExpertDispatch)
        assert len(dispatch.expert_indices) == num_experts
        assert len(dispatch.expert_weights) == num_experts
        assert dispatch.expert_counts.shape == (num_experts,)
        assert router_probs.shape == (n_tokens, num_experts)

    def test_dispatch_total_assignments(self) -> None:
        """Test that total assignments is approximately n_tokens * top_k."""
        n_tokens = 100
        num_experts = 8
        top_k = 2

        gate_logits = mx.random.normal((n_tokens, num_experts))
        dispatch, _ = build_expert_dispatch(gate_logits, num_experts, top_k)

        total_assigned = mx.sum(dispatch.expert_counts).item()
        # Should be approximately n_tokens * top_k (some may be dropped due to capacity)
        assert total_assigned <= n_tokens * top_k
        assert total_assigned > n_tokens  # At least some tokens routed to >1 expert

    def test_dispatch_weights_normalized(self) -> None:
        """Test that weights are normalized."""
        n_tokens = 50
        num_experts = 4
        top_k = 2

        gate_logits = mx.random.normal((n_tokens, num_experts))
        dispatch, _ = build_expert_dispatch(gate_logits, num_experts, top_k)

        # All weights should be positive
        for expert_idx in range(num_experts):
            weights = dispatch.expert_weights[expert_idx]
            if weights.size > 0:
                assert mx.all(weights > 0).item()
                assert mx.all(weights <= 1).item()


class TestSparseMoELayer:
    """Tests for sparse MoE layer."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input."""
        num_experts = 4
        d_model = 32
        d_hidden = 64
        batch_size = 2
        seq_len = 16

        moe = SparseMoELayer(num_experts, d_model, d_hidden, top_k=2)
        x = mx.random.normal((batch_size, seq_len, d_model))

        output, router_logits = moe(x)

        assert output.shape == x.shape
        assert router_logits.shape == (batch_size * seq_len, num_experts)

    def test_2d_input(self) -> None:
        """Test with 2D input (n_tokens, d_model)."""
        num_experts = 4
        d_model = 32
        d_hidden = 64
        n_tokens = 50

        moe = SparseMoELayer(num_experts, d_model, d_hidden, top_k=1)
        x = mx.random.normal((n_tokens, d_model))

        output, router_logits = moe(x)

        assert output.shape == x.shape
        assert router_logits.shape == (n_tokens, num_experts)

    def test_forward_deterministic(self) -> None:
        """Test that forward pass is deterministic."""
        num_experts = 4
        d_model = 16
        d_hidden = 32

        moe = SparseMoELayer(num_experts, d_model, d_hidden, top_k=2)
        x = mx.random.normal((10, d_model))

        output1, _ = moe(x)
        output2, _ = moe(x)

        assert mx.allclose(output1, output2).item()


class TestLoadBalancingLoss:
    """Tests for load balancing loss."""

    def test_uniform_distribution(self) -> None:
        """Test that uniform distribution minimizes loss."""
        num_experts = 4
        n_tokens = 100

        # Uniform router probs
        router_probs = mx.ones((n_tokens, num_experts)) / num_experts

        # Uniform counts
        expert_counts = mx.full((num_experts,), n_tokens // num_experts)

        loss = compute_load_balancing_loss(router_probs, expert_counts, num_experts)

        # Loss should be approximately num_experts * (1/num_experts)^2 * num_experts = 1
        assert abs(loss.item() - 1.0) < 0.1

    def test_imbalanced_increases_loss(self) -> None:
        """Test that imbalanced distribution increases loss."""
        num_experts = 4
        n_tokens = 100

        # Uniform baseline
        uniform_probs = mx.ones((n_tokens, num_experts)) / num_experts
        uniform_counts = mx.full((num_experts,), n_tokens // num_experts)
        uniform_loss = compute_load_balancing_loss(
            uniform_probs, uniform_counts, num_experts
        )

        # Imbalanced: all tokens to expert 0
        imbalanced_probs = mx.zeros((n_tokens, num_experts))
        imbalanced_probs = imbalanced_probs.at[:, 0].add(1.0)
        imbalanced_counts = mx.array([n_tokens, 0, 0, 0])
        imbalanced_loss = compute_load_balancing_loss(
            imbalanced_probs, imbalanced_counts, num_experts
        )

        assert imbalanced_loss.item() > uniform_loss.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
