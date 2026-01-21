"""JAX Metal parity tests for MoE operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestTopKRoutingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("top_k", [1, 2, 4])
    def test_forward_parity(self, size, top_k, skip_without_jax):
        raise NotImplementedError("Stub: topk_routing forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: topk_routing backward parity (JAX)")


class TestExpertDispatchParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: expert_dispatch forward parity (JAX)")


class TestLoadBalancingLossParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: load_balancing_loss forward parity (JAX)")
