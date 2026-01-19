"""Cross-validation tests for attention against PyTorch."""

import numpy as np
import pytest

import mlx.core as mx

from mlx_primitives import flash_attention


class TestAttentionCrossValidation:
    """Cross-validate attention implementations against PyTorch."""

    @pytest.mark.cross_validation
    def test_flash_attention_vs_pytorch_basic(self, skip_without_pytorch) -> None:
        """Compare flash attention against PyTorch SDPA for basic case."""
        from tests.reference_pytorch import torch_attention

        np.random.seed(42)
        batch, seq, heads, dim = 2, 64, 8, 64

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        # MLX flash attention (Python fallback for reliable comparison)
        mlx_out = np.array(flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=False, use_metal=False
        ))

        # PyTorch reference
        torch_out = torch_attention(q_np, k_np, v_np, causal=False)

        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-3, atol=1e-4)

    @pytest.mark.cross_validation
    def test_flash_attention_vs_pytorch_causal(self, skip_without_pytorch) -> None:
        """Compare causal flash attention against PyTorch SDPA."""
        from tests.reference_pytorch import torch_attention

        np.random.seed(123)
        batch, seq, heads, dim = 2, 128, 12, 64

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        # MLX flash attention with causal mask
        mlx_out = np.array(flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=True, use_metal=False
        ))

        # PyTorch reference with causal mask
        torch_out = torch_attention(q_np, k_np, v_np, causal=True)

        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-3, atol=1e-4)

    @pytest.mark.cross_validation
    def test_flash_attention_vs_pytorch_large_seq(self, skip_without_pytorch) -> None:
        """Compare flash attention for longer sequences."""
        from tests.reference_pytorch import torch_attention

        np.random.seed(456)
        batch, seq, heads, dim = 1, 512, 4, 32

        q_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, dim).astype(np.float32)

        # MLX
        mlx_out = np.array(flash_attention(
            mx.array(q_np), mx.array(k_np), mx.array(v_np),
            causal=True, use_metal=False
        ))

        # PyTorch
        torch_out = torch_attention(q_np, k_np, v_np, causal=True)

        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-3, atol=1e-4)


class TestScanCrossValidation:
    """Cross-validate scan operations against PyTorch."""

    @pytest.mark.cross_validation
    def test_cumsum_vs_pytorch(self, skip_without_pytorch) -> None:
        """Compare associative_scan add against PyTorch cumsum."""
        from tests.reference_pytorch import torch_cumsum
        from mlx_primitives import associative_scan

        np.random.seed(789)
        batch, seq, dim = 4, 128, 64

        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        mlx_out = np.array(associative_scan(mx.array(x_np), operator="add", axis=1))

        # PyTorch
        torch_out = torch_cumsum(x_np, axis=1)

        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-5, atol=1e-5)
