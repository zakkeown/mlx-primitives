"""Cross-validation tests for attention and scan operations against JAX.

JAX is arguably closer to MLX's functional paradigm than PyTorch, so validating
against JAX provides an additional reference point for numerical correctness.
"""

import numpy as np
import pytest

import mlx.core as mx

from mlx_primitives import flash_attention
from mlx_primitives import associative_scan


class TestAttentionJAXCrossValidation:
    """Cross-validate attention implementations against JAX."""

    @pytest.mark.cross_validation_jax
    def test_flash_attention_vs_jax_basic(self, skip_without_jax) -> None:
        """Compare flash attention against JAX reference for basic case."""
        from tests.reference_jax import jax_attention

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

        # JAX reference
        jax_out = jax_attention(q_np, k_np, v_np, causal=False)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.cross_validation_jax
    def test_flash_attention_vs_jax_causal(self, skip_without_jax) -> None:
        """Compare causal flash attention against JAX reference."""
        from tests.reference_jax import jax_attention

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

        # JAX reference with causal mask
        jax_out = jax_attention(q_np, k_np, v_np, causal=True)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.cross_validation_jax
    def test_flash_attention_vs_jax_large_seq(self, skip_without_jax) -> None:
        """Compare flash attention for longer sequences against JAX."""
        from tests.reference_jax import jax_attention

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

        # JAX
        jax_out = jax_attention(q_np, k_np, v_np, causal=True)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-3, atol=1e-4)


class TestScanJAXCrossValidation:
    """Cross-validate scan operations against JAX's lax.associative_scan."""

    @pytest.mark.cross_validation_jax
    def test_cumsum_vs_jax(self, skip_without_jax) -> None:
        """Compare associative_scan add against JAX cumsum."""
        from tests.reference_jax import jax_cumsum

        np.random.seed(789)
        batch, seq, dim = 4, 128, 64

        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        mlx_out = np.array(associative_scan(mx.array(x_np), operator="add", axis=1))

        # JAX
        jax_out = jax_cumsum(x_np, axis=1)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.cross_validation_jax
    def test_associative_scan_vs_jax_lax(self, skip_without_jax) -> None:
        """Compare MLX associative_scan against JAX lax.associative_scan.

        This is the most important validation since lax.associative_scan is
        the canonical functional implementation of parallel prefix scan.
        """
        from tests.reference_jax import jax_associative_scan

        np.random.seed(101)
        batch, seq = 8, 256

        x_np = np.random.randn(batch, seq).astype(np.float32)

        # MLX associative_scan
        mlx_out = np.array(associative_scan(mx.array(x_np), operator="add", axis=-1))

        # JAX lax.associative_scan
        jax_out = jax_associative_scan(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.cross_validation_jax
    def test_ssm_scan_vs_jax(self, skip_without_jax) -> None:
        """Compare MLX SSM scan against JAX lax.associative_scan with SSM operator.

        This validates the parallel SSM recurrence h[t] = A[t] * h[t-1] + x[t]
        against JAX's canonical implementation.
        """
        from tests.reference_jax import jax_ssm_scan

        np.random.seed(202)
        batch, seq, state = 4, 64, 32

        A_np = np.random.uniform(0.8, 0.99, (batch, seq, state)).astype(np.float32)
        x_np = np.random.randn(batch, seq, state).astype(np.float32)

        # MLX SSM scan
        mlx_out = np.array(associative_scan(
            mx.array(x_np), operator="ssm", A=mx.array(A_np), axis=1
        ))

        # JAX SSM scan using lax.associative_scan
        jax_out = jax_ssm_scan(A_np, x_np)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.cross_validation_jax
    def test_ssm_scan_vs_jax_long_sequence(self, skip_without_jax) -> None:
        """Compare MLX SSM scan against JAX for longer sequences (multi-block)."""
        from tests.reference_jax import jax_ssm_scan

        np.random.seed(303)
        batch, seq, state = 2, 2048, 16

        A_np = np.random.uniform(0.85, 0.99, (batch, seq, state)).astype(np.float32)
        x_np = np.random.randn(batch, seq, state).astype(np.float32)

        # MLX SSM scan (will use multi-block for seq > 1024)
        mlx_out = np.array(associative_scan(
            mx.array(x_np), operator="ssm", A=mx.array(A_np), axis=1
        ))

        # JAX SSM scan
        jax_out = jax_ssm_scan(A_np, x_np)

        # Slightly relaxed tolerance for long sequences
        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-2, atol=1e-2)

    @pytest.mark.cross_validation_jax
    def test_cumprod_vs_jax(self, skip_without_jax) -> None:
        """Compare MLX cumprod against JAX."""
        from tests.reference_jax import jax_cumprod

        np.random.seed(404)
        # Use small positive values to avoid overflow
        x_np = np.random.uniform(0.5, 1.5, (4, 32)).astype(np.float32)

        # MLX
        mlx_out = np.array(associative_scan(mx.array(x_np), operator="mul", axis=-1))

        # JAX
        jax_out = jax_cumprod(x_np, axis=-1)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-5, atol=1e-5)


class TestActivationsJAXCrossValidation:
    """Cross-validate activation functions against JAX."""

    @pytest.mark.cross_validation_jax
    def test_gelu_vs_jax(self, skip_without_jax) -> None:
        """Compare GELU activation against JAX."""
        from tests.reference_jax import jax_gelu
        from mlx_primitives.kernels import gelu_approximate

        np.random.seed(505)
        x_np = np.random.randn(8, 64, 256).astype(np.float32)

        # MLX approximate GELU
        mlx_out = np.array(gelu_approximate(mx.array(x_np)))

        # JAX approximate GELU
        jax_out = jax_gelu(x_np, approximate=True)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-4, atol=1e-4)

    @pytest.mark.cross_validation_jax
    def test_silu_vs_jax(self, skip_without_jax) -> None:
        """Compare SiLU (Swish) activation against JAX."""
        from tests.reference_jax import jax_silu
        from mlx_primitives.kernels import silu

        np.random.seed(606)
        x_np = np.random.randn(8, 64, 256).astype(np.float32)

        # MLX SiLU
        mlx_out = np.array(silu(mx.array(x_np)))

        # JAX SiLU
        jax_out = jax_silu(x_np)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-5, atol=1e-5)


class TestNormalizationJAXCrossValidation:
    """Cross-validate normalization operations against JAX."""

    @pytest.mark.cross_validation_jax
    def test_rmsnorm_vs_jax(self, skip_without_jax) -> None:
        """Compare RMSNorm against JAX."""
        from tests.reference_jax import jax_rmsnorm
        from mlx_primitives.kernels import rmsnorm

        np.random.seed(707)
        batch, seq, dim = 4, 128, 512
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        weight_np = np.random.randn(dim).astype(np.float32)

        # MLX RMSNorm
        mlx_out = np.array(rmsnorm(mx.array(x_np), mx.array(weight_np)))

        # JAX RMSNorm
        jax_out = jax_rmsnorm(x_np, weight_np)

        np.testing.assert_allclose(mlx_out, jax_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
