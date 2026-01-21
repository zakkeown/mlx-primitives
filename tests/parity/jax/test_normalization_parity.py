"""JAX Metal parity tests for normalization operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import normalization_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestRMSNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: rmsnorm forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: rmsnorm backward parity (JAX)")


class TestLayerNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: layernorm forward parity (JAX)")


class TestGroupNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("num_groups", [1, 4, 8, 32])
    def test_forward_parity(self, size, num_groups, skip_without_jax):
        raise NotImplementedError("Stub: groupnorm forward parity (JAX)")


class TestInstanceNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: instancenorm forward parity (JAX)")


class TestAdaLayerNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: adalayernorm forward parity (JAX)")
