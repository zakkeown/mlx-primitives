"""JAX Metal parity tests for MoE operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn
    from tests.reference_jax_extended import (
        jax_topk_routing,
        jax_expert_dispatch,
        jax_load_balancing_loss,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestTopKRoutingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("top_k", [1, 2, 4])
    def test_forward_parity(self, size, top_k, skip_without_jax):
        """Test top-k routing forward pass parity with JAX."""
        from mlx_primitives.advanced.moe import TopKRouter

        config = SIZE_CONFIGS[size]["moe"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        experts = config["experts"]

        # Skip if top_k > experts
        if top_k > experts:
            pytest.skip(f"top_k={top_k} > experts={experts}")

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX TopKRouter
        router_mlx = TopKRouter(dims=dim, num_experts=experts, top_k=top_k)
        mx.eval(router_mlx.parameters())

        # Get gate weights for JAX reference
        gate_weight_np = np.array(router_mlx.gate.weight)

        x_mlx = mx.array(x_np)
        mlx_weights, mlx_indices, mlx_logits = router_mlx(x_mlx)
        mx.eval(mlx_weights, mlx_indices, mlx_logits)

        # JAX reference: compute router logits
        x_j = jnp.array(x_np, dtype=jnp.float32)
        w_j = jnp.array(gate_weight_np, dtype=jnp.float32)
        router_logits_np = np.array(x_j @ w_j.T)  # (batch, seq, num_experts)

        # Use JAX reference for top-k routing
        jax_weights, jax_indices, jax_logits = jax_topk_routing(router_logits_np, top_k)

        # Compare router logits
        rtol, atol = get_tolerance("moe", "routing", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_logits), jax_logits,
            rtol=rtol, atol=atol,
            err_msg=f"TopK router logits mismatch (JAX) [{size}, top_k={top_k}]"
        )

        # Compare gate weights (softmaxed)
        np.testing.assert_allclose(
            _to_numpy(mlx_weights), jax_weights,
            rtol=rtol, atol=atol,
            err_msg=f"TopK gate weights mismatch (JAX) [{size}, top_k={top_k}]"
        )

        # Compare expert indices (should be exact)
        np.testing.assert_array_equal(
            _to_numpy(mlx_indices).astype(np.int32), jax_indices.astype(np.int32),
            err_msg=f"TopK expert indices mismatch (JAX) [{size}, top_k={top_k}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        """Test top-k routing backward pass parity with JAX."""
        from mlx_primitives.advanced.moe import TopKRouter

        config = SIZE_CONFIGS["small"]["moe"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        experts = config["experts"]
        top_k = 2

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX backward
        router_mlx = TopKRouter(dims=dim, num_experts=experts, top_k=top_k)
        mx.eval(router_mlx.parameters())
        gate_weight_np = np.array(router_mlx.gate.weight)

        def mlx_fn(x):
            weights, _, _ = router_mlx(x)
            return weights.sum()

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(x):
            # Compute router logits
            logits = x @ jnp.array(gate_weight_np).T
            # Top-k selection
            sorted_indices = jnp.argsort(-logits, axis=-1)
            expert_indices = sorted_indices[..., :top_k]
            selected_logits = jnp.take_along_axis(logits, expert_indices, axis=-1)
            gate_weights = jnn.softmax(selected_logits, axis=-1)
            return gate_weights.sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("moe", "routing", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg="TopK routing backward mismatch (JAX)"
        )


class TestExpertDispatchParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test expert dispatch forward pass parity with JAX.

        Note: This test uses the SAME routing for both MLX and JAX to isolate
        the expert computation parity from routing numerical differences.
        """
        from mlx_primitives.advanced.moe import MoELayer

        config = SIZE_CONFIGS[size]["moe"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["dim"]
        experts = config["experts"]
        top_k = min(config["top_k"], experts)
        hidden_dims = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # Create MLX MoE layer
        moe_mlx = MoELayer(
            dims=dim,
            hidden_dims=hidden_dims,
            num_experts=experts,
            top_k=top_k,
        )
        mx.eval(moe_mlx.parameters())

        # Get expert weights for JAX reference
        expert_weights_list = []
        for expert in moe_mlx.experts:
            w1_np = np.array(expert.w1.weight)
            w2_np = np.array(expert.w2.weight)
            expert_weights_list.append((w1_np, w2_np))

        # MLX forward - get output AND routing info
        x_mlx = mx.array(x_np)
        moe_out = moe_mlx(x_mlx)
        mx.eval(moe_out.output, moe_out.router_logits)

        # Extract routing from MLX to ensure consistency
        # Recompute routing from MLX's router logits to get same indices/weights
        mlx_logits_np = _to_numpy(moe_out.router_logits)

        # Use the same routing computation for JAX (based on MLX logits)
        # This ensures we're comparing expert dispatch, not routing differences
        mlx_weights, mlx_indices, _ = jax_topk_routing(mlx_logits_np, top_k)

        # Flatten for dispatch
        x_flat = x_np.reshape(-1, dim)
        weights_flat = mlx_weights.reshape(-1, top_k)
        indices_flat = mlx_indices.reshape(-1, top_k)

        # Expert dispatch with JAX using same routing as MLX
        jax_out_flat = jax_expert_dispatch(
            x_flat, indices_flat, weights_flat, expert_weights_list, experts
        )
        jax_out = jax_out_flat.reshape(batch, seq, dim)

        rtol, atol = get_tolerance("moe", "dispatch", "fp32")
        np.testing.assert_allclose(
            _to_numpy(moe_out.output), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Expert dispatch mismatch (JAX) [{size}]"
        )


class TestLoadBalancingLossParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test load balancing loss forward pass parity with JAX."""
        from mlx_primitives.advanced.moe import load_balancing_loss

        config = SIZE_CONFIGS[size]["moe"]
        batch = config["batch"]
        seq = config["seq"]
        experts = config["experts"]
        top_k = min(config["top_k"], experts)

        np.random.seed(42)
        # Generate random router logits
        router_logits_np = np.random.randn(batch, seq, experts).astype(np.float32)

        # Generate random expert indices (each token selects top_k experts)
        expert_indices_np = np.zeros((batch, seq, top_k), dtype=np.int32)
        for b in range(batch):
            for s in range(seq):
                expert_indices_np[b, s] = np.random.choice(experts, top_k, replace=False)

        # MLX load balancing loss
        router_logits_mlx = mx.array(router_logits_np)
        expert_indices_mlx = mx.array(expert_indices_np)
        mlx_loss = load_balancing_loss(router_logits_mlx, expert_indices_mlx, experts)
        mx.eval(mlx_loss)

        # JAX reference
        jax_loss = jax_load_balancing_loss(router_logits_np, expert_indices_np, experts)

        rtol, atol = get_tolerance("moe", "load_balancing_loss", "fp32")
        np.testing.assert_allclose(
            float(mlx_loss), jax_loss,
            rtol=rtol, atol=atol,
            err_msg=f"Load balancing loss mismatch (JAX) [{size}]"
        )
