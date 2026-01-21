"""JAX Metal parity tests for parallel primitives."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.tolerance_config import get_tolerance, assert_close


class TestAssociativeScanAddParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        raise NotImplementedError("Stub: associative_scan_add forward parity (JAX)")

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        raise NotImplementedError("Stub: associative_scan_add backward parity (JAX)")


class TestAssociativeScanMulParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: associative_scan_mul forward parity (JAX)")


class TestAssociativeScanSSMParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: associative_scan_ssm forward parity (JAX)")


class TestSelectiveScanParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: selective_scan forward parity (JAX)")


class TestSelectiveGatherParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: selective_gather forward parity (JAX)")


class TestSelectiveScatterAddParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        raise NotImplementedError("Stub: selective_scatter_add forward parity (JAX)")
