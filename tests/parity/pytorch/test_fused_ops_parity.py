"""PyTorch parity tests for fused operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close


# =============================================================================
# Fused RMSNorm + Linear Parity Tests
# =============================================================================

class TestFusedRMSNormLinearParity:
    """Fused RMSNorm + Linear parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused RMSNorm+Linear forward pass parity."""
        raise NotImplementedError("Stub: fused_rmsnorm_linear forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused RMSNorm+Linear backward pass parity."""
        raise NotImplementedError("Stub: fused_rmsnorm_linear backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_pytorch):
        """Test that fused op matches separate RMSNorm and Linear."""
        raise NotImplementedError("Stub: fused_rmsnorm_linear vs separate parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_with_bias(self, skip_without_pytorch):
        """Test fused op with bias parameter."""
        raise NotImplementedError("Stub: fused_rmsnorm_linear with bias parity")


# =============================================================================
# Fused SwiGLU Parity Tests
# =============================================================================

class TestFusedSwiGLUParity:
    """Fused SwiGLU parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused SwiGLU forward pass parity."""
        raise NotImplementedError("Stub: fused_swiglu forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused SwiGLU backward pass parity."""
        raise NotImplementedError("Stub: fused_swiglu backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_pytorch):
        """Test that fused SwiGLU matches separate ops."""
        raise NotImplementedError("Stub: fused_swiglu vs separate parity")


# =============================================================================
# Fused GeGLU Parity Tests
# =============================================================================

class TestFusedGeGLUParity:
    """Fused GeGLU parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused GeGLU forward pass parity."""
        raise NotImplementedError("Stub: fused_geglu forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused GeGLU backward pass parity."""
        raise NotImplementedError("Stub: fused_geglu backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_pytorch):
        """Test that fused GeGLU matches separate ops."""
        raise NotImplementedError("Stub: fused_geglu vs separate parity")


# =============================================================================
# Fused RoPE + Attention Parity Tests
# =============================================================================

class TestFusedRoPEAttentionParity:
    """Fused RoPE + Attention parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test fused RoPE+Attention forward pass parity."""
        raise NotImplementedError("Stub: fused_rope_attention forward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test fused RoPE+Attention backward pass parity."""
        raise NotImplementedError("Stub: fused_rope_attention backward parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_separate_ops(self, skip_without_pytorch):
        """Test that fused op matches separate RoPE and Attention."""
        raise NotImplementedError("Stub: fused_rope_attention vs separate parity")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_causal_masking(self, skip_without_pytorch):
        """Test fused RoPE+Attention with causal masking."""
        raise NotImplementedError("Stub: fused_rope_attention causal parity")
