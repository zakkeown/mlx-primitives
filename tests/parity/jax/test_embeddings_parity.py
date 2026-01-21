"""JAX Metal parity tests for embedding operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestSinusoidalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: sinusoidal_embedding forward parity (JAX)")


class TestLearnedPositionalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: learned_positional forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: learned_positional backward parity (JAX)")


class TestRotaryEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: rotary_embedding forward parity (JAX)")


class TestAlibiEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: alibi_embedding forward parity (JAX)")


class TestRelativePositionalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: relative_positional forward parity (JAX)")
