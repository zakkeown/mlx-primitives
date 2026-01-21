"""JAX Metal parity tests for generation/sampling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestTemperatureSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_forward_parity(self, size, temperature, skip_without_jax):
        raise NotImplementedError("Stub: temperature_sampling forward parity (JAX)")


class TestTopKSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("k", [1, 10, 50, 100])
    def test_forward_parity(self, size, k, skip_without_jax):
        raise NotImplementedError("Stub: top_k_sampling forward parity (JAX)")


class TestTopPSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.95])
    def test_forward_parity(self, size, p, skip_without_jax):
        raise NotImplementedError("Stub: top_p_sampling forward parity (JAX)")
