"""JAX Metal parity tests for generation/sampling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from tests.reference_jax_extended import (
        jax_temperature_sampling,
        jax_top_k_sampling,
        jax_top_p_sampling,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestTemperatureSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_forward_parity(self, size, temperature, skip_without_jax):
        """Test temperature sampling forward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_temperature

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX temperature sampling
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_temperature(logits_mlx, temperature)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_temperature_sampling(logits_np, temperature)

        rtol, atol = get_tolerance("sampling", "temperature", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Temperature sampling mismatch (JAX) [{size}, temp={temperature}]"
        )


class TestTopKSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("k", [1, 10, 50, 100])
    def test_forward_parity(self, size, k, skip_without_jax):
        """Test top-k sampling forward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_top_k

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        # Skip if k > vocab_size
        if k > vocab_size:
            pytest.skip(f"k={k} > vocab_size={vocab_size}")

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX top-k sampling
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_top_k_sampling(logits_np, k)

        # For top-k, we compare the masked logits
        # Note: -inf values should match
        rtol, atol = get_tolerance("sampling", "top_k", "fp32")

        mlx_np = _to_numpy(mlx_out)

        # Compare finite values
        mlx_finite = np.where(np.isfinite(mlx_np), mlx_np, 0)
        jax_finite = np.where(np.isfinite(jax_out), jax_out, 0)

        # Check that the same positions are masked (-inf)
        mlx_masked = ~np.isfinite(mlx_np)
        jax_masked = ~np.isfinite(jax_out)

        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg=f"Top-K mask mismatch (JAX) [{size}, k={k}]"
        )

        # Check finite values match
        np.testing.assert_allclose(
            mlx_finite, jax_finite,
            rtol=rtol, atol=atol,
            err_msg=f"Top-K logits mismatch (JAX) [{size}, k={k}]"
        )


class TestTopPSamplingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.95])
    def test_forward_parity(self, size, p, skip_without_jax):
        """Test top-p (nucleus) sampling forward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_top_p

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX top-p sampling
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_top_p_sampling(logits_np, p)

        # For top-p, we compare the masked logits
        rtol, atol = get_tolerance("sampling", "top_p", "fp32")

        mlx_np = _to_numpy(mlx_out)

        # Compare finite values
        mlx_finite = np.where(np.isfinite(mlx_np), mlx_np, 0)
        jax_finite = np.where(np.isfinite(jax_out), jax_out, 0)

        # Check that the same positions are masked (-inf)
        mlx_masked = ~np.isfinite(mlx_np)
        jax_masked = ~np.isfinite(jax_out)

        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg=f"Top-P mask mismatch (JAX) [{size}, p={p}]"
        )

        # Check finite values match
        np.testing.assert_allclose(
            mlx_finite, jax_finite,
            rtol=rtol, atol=atol,
            err_msg=f"Top-P logits mismatch (JAX) [{size}, p={p}]"
        )


class TestRepetitionPenaltyParity:
    """Repetition penalty parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("penalty", [1.0, 1.1, 1.5, 2.0])
    def test_forward_parity(self, size, penalty, skip_without_jax):
        """Test repetition penalty forward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_repetition_penalty_batch

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # Generate some previous tokens to penalize (as list of lists for batch API)
        # Use Python ints (not numpy int64) for MLX compatibility
        num_prev_tokens = min(20, vocab_size // 10)
        prev_tokens = [
            [int(x) for x in np.random.randint(0, vocab_size, (num_prev_tokens,))]
            for _ in range(batch)
        ]

        # MLX repetition penalty
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_repetition_penalty_batch(logits_mlx, prev_tokens, penalty)
        mx.eval(mlx_out)

        # JAX reference: apply repetition penalty manually
        jax_out = logits_np.copy()
        for b in range(batch):
            for tok in prev_tokens[b]:
                if jax_out[b, tok] > 0:
                    jax_out[b, tok] = jax_out[b, tok] / penalty
                else:
                    jax_out[b, tok] = jax_out[b, tok] * penalty

        rtol, atol = get_tolerance("sampling", "temperature", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Repetition penalty mismatch (JAX) [{size}, penalty={penalty}]"
        )


class TestMinPSamplingParity:
    """Min-P sampling parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("min_p", [0.01, 0.05, 0.1])
    def test_forward_parity(self, size, min_p, skip_without_jax):
        """Test Min-P sampling forward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_min_p

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX Min-P sampling
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_min_p(logits_mlx, min_p)
        mx.eval(mlx_out)

        # JAX/NumPy reference: min-p filtering
        # Use proper softmax for numerical stability (like MLX implementation)
        logits_shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
        probs = np.exp(logits_shifted)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        max_prob = probs.max(axis=-1, keepdims=True)
        threshold = min_p * max_prob

        # Create mask for tokens to keep (prob >= threshold OR is max prob)
        keep_mask = (probs >= threshold) | (probs == max_prob)
        jax_out = np.where(keep_mask, logits_np, -np.inf)

        rtol, atol = get_tolerance("sampling", "top_k", "fp32")

        mlx_np = _to_numpy(mlx_out)

        # Compare masked positions
        mlx_masked = ~np.isfinite(mlx_np)
        jax_masked = ~np.isfinite(jax_out)

        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg=f"Min-P mask mismatch (JAX) [{size}, min_p={min_p}]"
        )

        # Compare finite values
        mlx_finite = np.where(np.isfinite(mlx_np), mlx_np, 0)
        jax_finite = np.where(np.isfinite(jax_out), jax_out, 0)

        np.testing.assert_allclose(
            mlx_finite, jax_finite,
            rtol=rtol, atol=atol,
            err_msg=f"Min-P logits mismatch (JAX) [{size}, min_p={min_p}]"
        )


class TestTemperatureSamplingBackwardParity:
    """Temperature sampling backward parity tests.

    Temperature scaling is differentiable and should support backpropagation.
    """

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_backward_parity(self, size, temperature, skip_without_jax):
        """Test temperature sampling backward pass parity with JAX."""
        from mlx_primitives.generation.samplers import apply_temperature

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX backward
        def mlx_loss_fn(logits):
            scaled = apply_temperature(logits, temperature)
            return mx.sum(mx.softmax(scaled, axis=-1))

        logits_mlx = mx.array(logits_np)
        mlx_grad = mx.grad(mlx_loss_fn)(logits_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_loss_fn(logits):
            scaled = logits / temperature
            return jnp.sum(jax.nn.softmax(scaled, axis=-1))

        logits_jax = jnp.array(logits_np)
        jax_grad = jax.grad(jax_loss_fn)(logits_jax)

        rtol, atol = get_tolerance("sampling", "temperature", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,  # Looser tolerance for gradients
            err_msg=f"Temperature backward mismatch (JAX) [{size}, temp={temperature}]"
        )


class TestCombinedSamplingParity:
    """Combined sampling strategies parity tests."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_temperature_plus_top_p(self, size, skip_without_jax):
        """Test combined temperature + top-p sampling parity."""
        from mlx_primitives.generation.samplers import apply_temperature, apply_top_p

        config = SIZE_CONFIGS[size]["sampling"]
        batch = config["batch"]
        vocab_size = config["vocab_size"]
        temperature = 0.8
        top_p = 0.9

        np.random.seed(42)
        logits_np = np.random.randn(batch, vocab_size).astype(np.float32)

        # MLX combined sampling
        logits_mlx = mx.array(logits_np)
        mlx_temp = apply_temperature(logits_mlx, temperature)
        mlx_out = apply_top_p(mlx_temp, top_p)
        mx.eval(mlx_out)

        # JAX reference: temperature then top-p
        jax_temp = jax_temperature_sampling(logits_np, temperature)
        jax_out = jax_top_p_sampling(jax_temp, top_p)

        mlx_np = _to_numpy(mlx_out)

        # Compare masked positions
        mlx_masked = ~np.isfinite(mlx_np)
        jax_masked = ~np.isfinite(jax_out)

        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg=f"Combined temp+top_p mask mismatch [{size}]"
        )

        # Compare finite values
        rtol, atol = get_tolerance("sampling", "top_p", "fp32")
        mlx_finite = np.where(np.isfinite(mlx_np), mlx_np, 0)
        jax_finite = np.where(np.isfinite(jax_out), jax_out, 0)

        np.testing.assert_allclose(
            mlx_finite, jax_finite,
            rtol=rtol, atol=atol,
            err_msg=f"Combined temp+top_p values mismatch [{size}]"
        )
