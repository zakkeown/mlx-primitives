"""Numerical gradient tests for attention implementations.

Tests that analytical gradients from MLX autodiff match numerical approximations.
This validates the correctness of backward passes.
"""

import math

import pytest
import mlx.core as mx

from mlx_primitives.attention.flash import flash_attention, _reference_flash_attention
from mlx_primitives.attention.chunked import chunked_cross_attention
from mlx_primitives.attention.sliding_window import sliding_window_attention
from tests.utils.gradient_check import (
    gradient_check_attention,
    check_gradient,
    numerical_gradient_fast,
)


class TestFlashAttentionGradients:
    """Test gradients for flash attention."""

    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    @pytest.mark.parametrize("causal", [True, False])
    def test_flash_attention_qkv_gradients(self, seq_len: int, causal: bool):
        """Test that Q, K, V gradients are numerically correct."""
        mx.random.seed(42)
        batch, heads, dim = 1, 2, 16

        q = mx.random.normal((batch, seq_len, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq_len, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq_len, heads, dim)) * 0.1

        passed, details = gradient_check_attention(
            flash_attention,
            q, k, v,
            causal=causal,
            use_metal=False,  # Use Python reference for gradient checking
            rtol=5e-2,  # Relaxed tolerance for attention numerics
            atol=5e-2,
            sample_ratio=0.1,
        )

        assert passed, (
            f"Flash attention gradient check failed:\n"
            f"  Q gradient: max_abs_diff={details['q_gradient']['max_abs_diff']:.6f}\n"
            f"  K gradient: max_abs_diff={details['k_gradient']['max_abs_diff']:.6f}\n"
            f"  V gradient: max_abs_diff={details['v_gradient']['max_abs_diff']:.6f}"
        )

    def test_flash_attention_gradient_with_scale(self):
        """Test gradients with custom scale factor."""
        mx.random.seed(123)
        batch, seq, heads, dim = 1, 12, 2, 8

        q = mx.random.normal((batch, seq, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq, heads, dim)) * 0.1

        custom_scale = 0.5

        passed, details = gradient_check_attention(
            flash_attention,
            q, k, v,
            scale=custom_scale,
            causal=True,
            use_metal=False,
            rtol=5e-2,
            atol=5e-2,
        )

        assert passed, f"Gradient check with custom scale failed: {details}"


class TestReferenceAttentionGradients:
    """Test gradients for reference attention implementation."""

    def test_reference_attention_gradients(self):
        """Test reference attention has correct gradients."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 8, 2, 8

        q = mx.random.normal((batch, seq, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq, heads, dim)) * 0.1

        scale = 1.0 / math.sqrt(dim)

        passed, details = gradient_check_attention(
            _reference_flash_attention,
            q, k, v,
            scale=scale,
            causal=True,
            rtol=5e-2,
            atol=5e-2,
        )

        assert passed, f"Reference attention gradient check failed: {details}"


class TestChunkedAttentionGradients:
    """Test gradients for chunked cross-attention."""

    @pytest.mark.parametrize("seq_q,seq_kv", [(8, 16), (16, 8), (12, 12)])
    def test_chunked_attention_qkv_gradients(self, seq_q: int, seq_kv: int):
        """Test Q, K, V gradients for chunked attention."""
        mx.random.seed(42)
        batch, heads, dim = 1, 2, 16

        q = mx.random.normal((batch, seq_q, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq_kv, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq_kv, heads, dim)) * 0.1

        passed, details = gradient_check_attention(
            chunked_cross_attention,
            q, k, v,
            causal=False,  # Cross-attention typically non-causal
            use_metal=False,
            rtol=5e-2,
            atol=5e-2,
        )

        assert passed, f"Chunked attention gradient check failed: {details}"


class TestSlidingWindowAttentionGradients:
    """Test gradients for sliding window attention."""

    @pytest.mark.parametrize("window_size", [4, 8])
    def test_sliding_window_qkv_gradients(self, window_size: int):
        """Test Q, K, V gradients for sliding window attention."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 16, 2, 8

        q = mx.random.normal((batch, seq, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq, heads, dim)) * 0.1

        passed, details = gradient_check_attention(
            sliding_window_attention,
            q, k, v,
            window_size=window_size,
            causal=True,
            use_metal=False,
            rtol=5e-2,
            atol=5e-2,
        )

        assert passed, f"Sliding window attention gradient check failed: {details}"


class TestOnlineSoftmaxMergeGradients:
    """Test gradients for online softmax merge operation."""

    def test_online_softmax_merge_gradient(self):
        """Test that online softmax merge has correct gradients."""
        from mlx_primitives.attention._online_softmax import online_softmax_merge

        mx.random.seed(42)
        batch, seq, heads, dim = 1, 4, 2, 8

        # Create inputs for merge
        acc_output = mx.random.normal((batch, seq, heads, dim)) * 0.1
        acc_max = mx.random.normal((batch, seq, heads))
        acc_sum = mx.abs(mx.random.normal((batch, seq, heads))) + 0.1

        new_output = mx.random.normal((batch, seq, heads, dim)) * 0.1
        new_max = mx.random.normal((batch, seq, heads))
        new_sum = mx.abs(mx.random.normal((batch, seq, heads))) + 0.1

        # Function that returns scalar loss from merge
        def merge_loss(acc_out, new_out):
            merged, _, _ = online_softmax_merge(
                acc_out, acc_max, acc_sum,
                new_out, new_max, new_sum,
            )
            return mx.sum(merged)

        # Check gradients
        inputs = [acc_output, new_output]
        passed, errors = check_gradient(
            merge_loss, inputs, rtol=5e-2, atol=5e-2, sample_ratio=0.2
        )

        assert passed, (
            f"Online softmax merge gradient check failed:\n"
            f"  acc_output: max_diff={errors[0][1]:.6f}\n"
            f"  new_output: max_diff={errors[1][1]:.6f}"
        )


class TestGradientNumericalStability:
    """Test gradient numerical stability edge cases."""

    def test_gradient_with_large_values(self):
        """Test gradients don't explode with larger input magnitudes."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 8, 2, 8

        # Larger magnitude inputs (but not extreme)
        q = mx.random.normal((batch, seq, heads, dim)) * 1.0
        k = mx.random.normal((batch, seq, heads, dim)) * 1.0
        v = mx.random.normal((batch, seq, heads, dim)) * 1.0

        passed, details = gradient_check_attention(
            flash_attention,
            q, k, v,
            causal=True,
            use_metal=False,
            rtol=5e-2,  # Slightly relaxed tolerance for larger values
            atol=1e-2,
        )

        # Check gradients are finite
        def get_grad(fn, inputs, idx):
            grad_fn = mx.grad(lambda *args: mx.sum(fn(*args)), argnums=idx)
            return grad_fn(*inputs)

        q_grad = get_grad(lambda q, k, v: flash_attention(q, k, v, causal=True, use_metal=False), [q, k, v], 0)
        k_grad = get_grad(lambda q, k, v: flash_attention(q, k, v, causal=True, use_metal=False), [q, k, v], 1)
        v_grad = get_grad(lambda q, k, v: flash_attention(q, k, v, causal=True, use_metal=False), [q, k, v], 2)

        assert mx.all(mx.isfinite(q_grad)), "Q gradient contains non-finite values"
        assert mx.all(mx.isfinite(k_grad)), "K gradient contains non-finite values"
        assert mx.all(mx.isfinite(v_grad)), "V gradient contains non-finite values"

    def test_gradient_with_uniform_attention(self):
        """Test gradients when attention is uniformly distributed."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 8, 2, 8

        # Zero queries lead to uniform attention (all scores equal)
        q = mx.zeros((batch, seq, heads, dim))
        k = mx.random.normal((batch, seq, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq, heads, dim)) * 0.1

        # Should still have valid gradients for k and v
        def fn(k, v):
            return flash_attention(q, k, v, causal=True, use_metal=False)

        k_grad = mx.grad(lambda k, v: mx.sum(fn(k, v)), argnums=0)(k, v)
        v_grad = mx.grad(lambda k, v: mx.sum(fn(k, v)), argnums=1)(k, v)

        assert mx.all(mx.isfinite(k_grad)), "K gradient contains non-finite values"
        assert mx.all(mx.isfinite(v_grad)), "V gradient contains non-finite values"


class TestGradientConsistency:
    """Test gradient consistency across implementations."""

    def test_flash_vs_reference_gradients_match(self):
        """Test that flash and reference attention have same gradients."""
        mx.random.seed(42)
        batch, seq, heads, dim = 1, 16, 2, 16

        q = mx.random.normal((batch, seq, heads, dim)) * 0.1
        k = mx.random.normal((batch, seq, heads, dim)) * 0.1
        v = mx.random.normal((batch, seq, heads, dim)) * 0.1

        # Get flash attention gradients
        flash_q_grad = mx.grad(
            lambda q: mx.sum(flash_attention(q, k, v, causal=True, use_metal=False))
        )(q)

        # Get reference attention gradients
        ref_q_grad = mx.grad(
            lambda q: mx.sum(_reference_flash_attention(q, k, v, scale=None, causal=True))
        )(q)

        # Should match closely
        diff = mx.max(mx.abs(flash_q_grad - ref_q_grad))
        assert diff < 1e-4, f"Flash vs reference Q gradient diff: {diff}"
