"""JAX Metal parity tests for cache operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestPagedAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: paged_attention forward parity (JAX)")


class TestBlockAllocationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: block_allocation forward parity (JAX)")


class TestEvictionPoliciesParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_lru_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: lru_eviction forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_fifo_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: fifo_eviction forward parity (JAX)")


class TestSpeculativeVerificationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: speculative_verification forward parity (JAX)")
