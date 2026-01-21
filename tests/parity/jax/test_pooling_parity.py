"""JAX Metal parity tests for pooling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestAdaptiveAvgPool1dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [1, 4, 16])
    def test_forward_parity(self, size, output_size, skip_without_jax):
        raise NotImplementedError("Stub: adaptive_avg_pool1d forward parity (JAX)")


class TestAdaptiveAvgPool2dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4), (7, 7)])
    def test_forward_parity(self, size, output_size, skip_without_jax):
        raise NotImplementedError("Stub: adaptive_avg_pool2d forward parity (JAX)")


class TestAdaptiveMaxPool1dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: adaptive_max_pool1d forward parity (JAX)")


class TestAdaptiveMaxPool2dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: adaptive_max_pool2d forward parity (JAX)")


class TestGlobalAttentionPoolingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: global_attention_pooling forward parity (JAX)")


class TestGeMParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_forward_parity(self, size, p, skip_without_jax):
        raise NotImplementedError("Stub: gem forward parity (JAX)")


class TestSpatialPyramidPoolingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: spp forward parity (JAX)")
