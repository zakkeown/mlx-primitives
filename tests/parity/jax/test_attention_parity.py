"""JAX Metal parity tests for attention operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import attention_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestFlashAttentionParity:
    """Flash attention parity tests vs JAX implementation."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: flash_attention forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: flash_attention backward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_causal_masking_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: flash_attention causal parity (JAX)")


class TestSlidingWindowAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: sliding_window forward parity (JAX)")


class TestChunkedCrossAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: chunked_cross forward parity (JAX)")


class TestGroupedQueryAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: gqa forward parity (JAX)")


class TestMultiQueryAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: mqa forward parity (JAX)")


class TestSparseAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: sparse_attention forward parity (JAX)")


class TestLinearAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: linear_attention forward parity (JAX)")


class TestALiBiAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: alibi_attention forward parity (JAX)")


class TestQuantizedKVCacheAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: quantized_kv_attention forward parity (JAX)")


class TestRoPEVariantsParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: rope forward parity (JAX)")


class TestLayoutVariantsParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("layout", ["bhsd", "bshd"])
    def test_forward_parity(self, layout, skip_without_jax):
        raise NotImplementedError("Stub: layout variants forward parity (JAX)")
