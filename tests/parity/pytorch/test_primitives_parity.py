"""PyTorch parity tests for parallel primitives (scan, gather, scatter)."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import scan_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close


# =============================================================================
# Associative Scan (Add) Parity Tests
# =============================================================================

class TestAssociativeScanAddParity:
    """Associative scan with add operator (cumsum) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan add forward pass parity (vs torch.cumsum)."""
        raise NotImplementedError("Stub: associative_scan_add forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan add backward pass parity."""
        raise NotImplementedError("Stub: associative_scan_add backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_different_axes(self, axis, skip_without_pytorch):
        """Test associative scan add on different axes."""
        raise NotImplementedError("Stub: associative_scan_add axis parity")


# =============================================================================
# Associative Scan (Mul) Parity Tests
# =============================================================================

class TestAssociativeScanMulParity:
    """Associative scan with multiply operator (cumprod) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan mul forward pass parity (vs torch.cumprod)."""
        raise NotImplementedError("Stub: associative_scan_mul forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan mul backward pass parity."""
        raise NotImplementedError("Stub: associative_scan_mul backward parity")


# =============================================================================
# Associative Scan (SSM) Parity Tests
# =============================================================================

class TestAssociativeScanSSMParity:
    """Associative scan with SSM operator parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test associative scan SSM forward pass parity."""
        raise NotImplementedError("Stub: associative_scan_ssm forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test associative scan SSM backward pass parity."""
        raise NotImplementedError("Stub: associative_scan_ssm backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_mamba_style(self, skip_without_pytorch):
        """Test Mamba-style selective scan."""
        raise NotImplementedError("Stub: associative_scan_ssm mamba parity")


# =============================================================================
# Selective Scan Parity Tests
# =============================================================================

class TestSelectiveScanParity:
    """Selective scan (Mamba-style) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective scan forward pass parity."""
        raise NotImplementedError("Stub: selective_scan forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective scan backward pass parity."""
        raise NotImplementedError("Stub: selective_scan backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_d_parameter(self, skip_without_pytorch):
        """Test selective scan with D (skip connection) parameter."""
        raise NotImplementedError("Stub: selective_scan with D parity")


# =============================================================================
# Selective Gather Parity Tests
# =============================================================================

class TestSelectiveGatherParity:
    """Selective gather operation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective gather forward pass parity."""
        raise NotImplementedError("Stub: selective_gather forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective gather backward pass parity."""
        raise NotImplementedError("Stub: selective_gather backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_torch_gather(self, skip_without_pytorch):
        """Test selective gather vs torch.gather."""
        raise NotImplementedError("Stub: selective_gather vs torch.gather parity")


# =============================================================================
# Selective Scatter Add Parity Tests
# =============================================================================

class TestSelectiveScatterAddParity:
    """Selective scatter-add operation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test selective scatter-add forward pass parity."""
        raise NotImplementedError("Stub: selective_scatter_add forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test selective scatter-add backward pass parity."""
        raise NotImplementedError("Stub: selective_scatter_add backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_torch_scatter_add(self, skip_without_pytorch):
        """Test selective scatter-add vs torch.scatter_add."""
        raise NotImplementedError("Stub: selective_scatter_add vs torch.scatter_add parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_weights(self, skip_without_pytorch):
        """Test selective scatter-add with weighted values."""
        raise NotImplementedError("Stub: selective_scatter_add with weights parity")
