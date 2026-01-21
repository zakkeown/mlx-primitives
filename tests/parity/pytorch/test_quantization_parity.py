"""PyTorch parity tests for quantization operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import quantization_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close


# =============================================================================
# INT8 Quantization Parity Tests
# =============================================================================

class TestINT8QuantizationParity:
    """INT8 quantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 quantization forward pass parity."""
        raise NotImplementedError("Stub: int8_quantize forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_scale_computation(self, skip_without_pytorch):
        """Test INT8 scale computation matches PyTorch."""
        raise NotImplementedError("Stub: int8_quantize scale parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_zero_point_computation(self, skip_without_pytorch):
        """Test INT8 zero point computation matches PyTorch."""
        raise NotImplementedError("Stub: int8_quantize zero_point parity")


# =============================================================================
# INT8 Dequantization Parity Tests
# =============================================================================

class TestINT8DequantizationParity:
    """INT8 dequantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 dequantization forward pass parity."""
        raise NotImplementedError("Stub: int8_dequantize forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_pytorch):
        """Test quantize -> dequantize roundtrip error matches PyTorch."""
        raise NotImplementedError("Stub: int8 roundtrip parity")


# =============================================================================
# INT4 Quantization Parity Tests
# =============================================================================

class TestINT4QuantizationParity:
    """INT4 quantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 quantization forward pass parity."""
        raise NotImplementedError("Stub: int4_quantize forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_group_quantization(self, skip_without_pytorch):
        """Test INT4 group-wise quantization."""
        raise NotImplementedError("Stub: int4_quantize group parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_different_group_sizes(self, group_size, skip_without_pytorch):
        """Test INT4 quantization with different group sizes."""
        raise NotImplementedError("Stub: int4_quantize group_size parity")


# =============================================================================
# INT4 Dequantization Parity Tests
# =============================================================================

class TestINT4DequantizationParity:
    """INT4 dequantization parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 dequantization forward pass parity."""
        raise NotImplementedError("Stub: int4_dequantize forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_roundtrip(self, skip_without_pytorch):
        """Test quantize -> dequantize roundtrip error matches PyTorch."""
        raise NotImplementedError("Stub: int4 roundtrip parity")


# =============================================================================
# INT8 Linear Parity Tests
# =============================================================================

class TestINT8LinearParity:
    """INT8 quantized linear layer parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT8 linear forward pass parity."""
        raise NotImplementedError("Stub: int8_linear forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test INT8 linear backward pass parity (STE gradient)."""
        raise NotImplementedError("Stub: int8_linear backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_pytorch):
        """Test INT8 linear vs FP32 linear error bounds."""
        raise NotImplementedError("Stub: int8_linear vs fp32 parity")


# =============================================================================
# INT4 Linear Parity Tests
# =============================================================================

class TestINT4LinearParity:
    """INT4 quantized linear layer parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test INT4 linear forward pass parity."""
        raise NotImplementedError("Stub: int4_linear forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test INT4 linear backward pass parity (STE gradient)."""
        raise NotImplementedError("Stub: int4_linear backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_fp32_linear(self, skip_without_pytorch):
        """Test INT4 linear vs FP32 linear error bounds."""
        raise NotImplementedError("Stub: int4_linear vs fp32 parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_group_wise_quantization(self, group_size, skip_without_pytorch):
        """Test INT4 linear with group-wise quantization."""
        raise NotImplementedError("Stub: int4_linear group_size parity")
