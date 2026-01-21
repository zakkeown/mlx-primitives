"""JAX Metal parity tests for fused operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestFusedRMSNormLinearParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: fused_rmsnorm_linear forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: fused_rmsnorm_linear backward parity (JAX)")


class TestFusedSwiGLUParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: fused_swiglu forward parity (JAX)")


class TestFusedGeGLUParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: fused_geglu forward parity (JAX)")


class TestFusedRoPEAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: fused_rope_attention forward parity (JAX)")
