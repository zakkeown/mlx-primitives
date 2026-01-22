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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_logits_scaling(self, skip_without_jax):
        """Test that logit scaling matches JAX with known values."""
        from mlx_primitives.generation.samplers import apply_temperature

        # Use known values for predictable output
        logits_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        temperature = 2.0
        expected = logits_np / temperature

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_temperature(logits_mlx, temperature)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_temperature_sampling(logits_np, temperature)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), expected, rtol=1e-6, atol=1e-7,
            err_msg="Temperature scaling mismatch with known values"
        )
        np.testing.assert_allclose(
            jax_out, expected, rtol=1e-6, atol=1e-7,
            err_msg="JAX temperature scaling mismatch"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_probability_distribution(self, skip_without_jax):
        """Test that resulting probability distribution matches."""
        from mlx_primitives.generation.samplers import apply_temperature
        from jax import nn as jnn

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)

        for temperature in [0.5, 1.0, 2.0]:
            # MLX
            logits_mlx = mx.array(logits_np)
            mlx_scaled = apply_temperature(logits_mlx, temperature)
            mlx_probs = mx.softmax(mlx_scaled, axis=-1)
            mx.eval(mlx_probs)

            # JAX
            jax_scaled = jax_temperature_sampling(logits_np, temperature)
            jax_probs = jnn.softmax(jnp.array(jax_scaled), axis=-1)

            np.testing.assert_allclose(
                _to_numpy(mlx_probs), np.array(jax_probs),
                rtol=1e-5, atol=1e-6,
                err_msg=f"Probability distribution mismatch at temp={temperature}"
            )

            # Verify probabilities sum to 1
            mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
            np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_temperature_zero(self, skip_without_jax):
        """Test temperature=0 (greedy) behavior - should return logits unchanged."""
        from mlx_primitives.generation.samplers import apply_temperature

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_temperature(logits_mlx, 0.0)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_temperature_sampling(logits_np, 0.0)

        # Both should return logits unchanged
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX temperature=0 should return logits unchanged"
        )
        np.testing.assert_allclose(
            jax_out, logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="JAX temperature=0 should return logits unchanged"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_temperature_very_high(self, skip_without_jax):
        """Test very high temperature (uniform-like) behavior."""
        from mlx_primitives.generation.samplers import apply_temperature
        from jax import nn as jnn

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 100.0

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_probs = mx.softmax(mlx_scaled, axis=-1)
        mx.eval(mlx_probs)

        # JAX
        jax_scaled = jax_temperature_sampling(logits_np, temperature)
        jax_probs = jnn.softmax(jnp.array(jax_scaled), axis=-1)

        # Check no NaN or Inf
        mlx_out_np = _to_numpy(mlx_probs)
        assert not np.any(np.isnan(mlx_out_np)), "MLX output contains NaN"
        assert not np.any(np.isinf(mlx_out_np)), "MLX output contains Inf"

        jax_out_np = np.array(jax_probs)
        assert not np.any(np.isnan(jax_out_np)), "JAX output contains NaN"
        assert not np.any(np.isinf(jax_out_np)), "JAX output contains Inf"

        # Verify distribution is more uniform (closer to 1/vocab_size)
        mlx_std = np.std(mlx_out_np, axis=-1)
        # High temperature should make std small (closer to uniform)
        assert np.all(mlx_std < 0.01), f"MLX distribution not uniform enough: std={mlx_std}"

        # Verify parity
        np.testing.assert_allclose(
            mlx_out_np, jax_out_np, rtol=1e-5, atol=1e-6,
            err_msg="High temperature parity mismatch"
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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_top_k_selection(self, skip_without_jax):
        """Test that top-K selection correctly keeps the k largest values."""
        from mlx_primitives.generation.samplers import apply_top_k

        # Use known values for predictable output
        logits_np = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]], dtype=np.float32)
        k = 3  # Should keep indices 1, 3, 4 (values 5, 8, 3)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mx.eval(mlx_out)
        mlx_np = _to_numpy(mlx_out)

        # JAX
        jax_out = jax_top_k_sampling(logits_np, k)

        # Count non-inf values
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        jax_kept = np.sum(~np.isneginf(jax_out), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [k], err_msg="MLX didn't keep exactly k values")
        np.testing.assert_array_equal(jax_kept, [k], err_msg="JAX didn't keep exactly k values")

        # Verify the correct values are kept (top k: 8, 5, 3)
        mlx_kept_values = sorted(mlx_np[~np.isneginf(mlx_np)])
        assert mlx_kept_values == [3.0, 5.0, 8.0], f"Wrong values kept: {mlx_kept_values}"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_probability_renormalization(self, skip_without_jax):
        """Test probability renormalization after top-k filtering."""
        from mlx_primitives.generation.samplers import apply_top_k
        from jax import nn as jnn

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 10

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_filtered = apply_top_k(logits_mlx, k)
        mlx_probs = mx.softmax(mlx_filtered, axis=-1)
        mx.eval(mlx_probs)

        # JAX
        jax_filtered = jax_top_k_sampling(logits_np, k)
        jax_probs = jnn.softmax(jnp.array(jax_filtered), axis=-1)

        # Probabilities should sum to 1
        mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
        jax_sum = np.array(jnp.sum(jax_probs, axis=-1))

        np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(jax_sum, np.ones(2), rtol=1e-5)

        # Verify parity
        np.testing.assert_allclose(
            _to_numpy(mlx_probs), np.array(jax_probs),
            rtol=1e-5, atol=1e-6,
            err_msg="Top-K probability renormalization mismatch"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_k_equals_1(self, skip_without_jax):
        """Test K=1 (greedy) behavior - only one value should be kept."""
        from mlx_primitives.generation.samplers import apply_top_k

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 1

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mx.eval(mlx_out)
        mlx_np = _to_numpy(mlx_out)

        # JAX
        jax_out = jax_top_k_sampling(logits_np, k)

        # Only 1 value should be non-inf per row
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        jax_kept = np.sum(~np.isneginf(jax_out), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [1, 1], err_msg="MLX k=1 didn't keep exactly 1 value")
        np.testing.assert_array_equal(jax_kept, [1, 1], err_msg="JAX k=1 didn't keep exactly 1 value")

        # The kept value should be the maximum
        for i in range(2):
            max_idx = np.argmax(logits_np[i])
            assert not np.isneginf(mlx_np[i, max_idx]), "MLX didn't keep the max value"
            assert not np.isneginf(jax_out[i, max_idx]), "JAX didn't keep the max value"

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_k_equals_vocab_size(self, skip_without_jax):
        """Test K=vocab_size (no filtering) behavior."""
        from mlx_primitives.generation.samplers import apply_top_k

        np.random.seed(42)
        vocab_size = 100
        logits_np = np.random.randn(2, vocab_size).astype(np.float32)
        k = vocab_size

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_k(logits_mlx, k)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_top_k_sampling(logits_np, k)

        # Output should equal input (no filtering)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX k=vocab_size should return logits unchanged"
        )
        np.testing.assert_allclose(
            jax_out, logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="JAX k=vocab_size should return logits unchanged"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_with_temperature(self, skip_without_jax):
        """Test Top-K combined with temperature scaling."""
        from mlx_primitives.generation.samplers import apply_temperature, apply_top_k

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 0.7
        k = 10

        # MLX: temperature then top-k
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_filtered = apply_top_k(mlx_scaled, k)
        mx.eval(mlx_filtered)

        # JAX
        jax_scaled = jax_temperature_sampling(logits_np, temperature)
        jax_filtered = jax_top_k_sampling(jax_scaled, k)

        mlx_np = _to_numpy(mlx_filtered)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        jax_masked = np.isneginf(jax_filtered)
        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg="Temperature+Top-K mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], jax_filtered[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Temperature+Top-K values mismatch"
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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_cumulative_probability(self, skip_without_jax):
        """Test cumulative probability computation matches."""
        from jax import nn as jnn

        # Use known values for predictable cumulative probs
        logits_np = np.array([[2.0, 1.0, 0.0, -1.0, -2.0]], dtype=np.float32)

        # Compute expected sorted probs
        sorted_logits = np.sort(logits_np, axis=-1)[:, ::-1]  # Descending
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True))
        sorted_probs_expected = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        cumulative_expected = np.cumsum(sorted_probs_expected, axis=-1)

        # MLX
        logits_mlx = mx.array(logits_np)
        sorted_indices_mlx = mx.argsort(logits_mlx, axis=-1)[:, ::-1]
        sorted_logits_mlx = mx.take_along_axis(logits_mlx, sorted_indices_mlx, axis=-1)
        sorted_probs_mlx = mx.softmax(sorted_logits_mlx, axis=-1)
        cumulative_mlx = mx.cumsum(sorted_probs_mlx, axis=-1)
        mx.eval(cumulative_mlx)

        # JAX
        logits_jax = jnp.array(logits_np)
        sorted_indices_jax = jnp.argsort(logits_jax, axis=-1)[:, ::-1]
        sorted_logits_jax = jnp.take_along_axis(logits_jax, sorted_indices_jax, axis=-1)
        sorted_probs_jax = jnn.softmax(sorted_logits_jax, axis=-1)
        cumulative_jax = jnp.cumsum(sorted_probs_jax, axis=-1)

        # Verify cumulative probs match
        np.testing.assert_allclose(
            _to_numpy(cumulative_mlx), np.array(cumulative_jax),
            rtol=1e-5, atol=1e-6,
            err_msg="Cumulative probability computation mismatch"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_nucleus_selection(self, skip_without_jax):
        """Test nucleus (smallest set with prob >= p) selection."""
        from mlx_primitives.generation.samplers import apply_top_p

        # Use known values: probs after softmax will be approximately [0.64, 0.24, 0.09, 0.02, 0.01]
        logits_np = np.array([[3.0, 2.0, 1.0, 0.0, -1.0]], dtype=np.float32)
        p = 0.9  # Should keep ~3 tokens (0.64 + 0.24 + 0.09 = 0.97 > 0.9)

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mx.eval(mlx_out)
        mlx_np = _to_numpy(mlx_out)

        # JAX
        jax_out = jax_top_p_sampling(logits_np, p)

        # Count kept tokens
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        jax_kept = np.sum(~np.isneginf(jax_out), axis=-1)

        np.testing.assert_array_equal(
            mlx_kept, jax_kept,
            err_msg=f"Nucleus size mismatch: MLX={mlx_kept}, JAX={jax_kept}"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_probability_renormalization(self, skip_without_jax):
        """Test probability renormalization after top-p filtering."""
        from mlx_primitives.generation.samplers import apply_top_p
        from jax import nn as jnn

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 0.9

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_filtered = apply_top_p(logits_mlx, p)
        mlx_probs = mx.softmax(mlx_filtered, axis=-1)
        mx.eval(mlx_probs)

        # JAX
        jax_filtered = jax_top_p_sampling(logits_np, p)
        jax_probs = jnn.softmax(jnp.array(jax_filtered), axis=-1)

        # Probabilities should sum to 1
        mlx_sum = _to_numpy(mx.sum(mlx_probs, axis=-1))
        jax_sum = np.array(jnp.sum(jax_probs, axis=-1))

        np.testing.assert_allclose(mlx_sum, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(jax_sum, np.ones(2), rtol=1e-5)

        # Verify parity
        np.testing.assert_allclose(
            _to_numpy(mlx_probs), np.array(jax_probs),
            rtol=1e-5, atol=1e-6,
            err_msg="Top-P probability renormalization mismatch"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_p_equals_0(self, skip_without_jax):
        """Test P=0 behavior - should keep at least one token (the top one)."""
        from mlx_primitives.generation.samplers import apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 0.0  # Edge case: p=0 should keep only the top token due to always-keep-first

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mx.eval(mlx_out)
        mlx_np = _to_numpy(mlx_out)

        # JAX
        jax_out = jax_top_p_sampling(logits_np, p)

        # Should keep exactly 1 token (the top probability one)
        mlx_kept = np.sum(~np.isneginf(mlx_np), axis=-1)
        jax_kept = np.sum(~np.isneginf(jax_out), axis=-1)

        np.testing.assert_array_equal(mlx_kept, [1, 1], err_msg="MLX p=0 didn't keep exactly 1 token")
        np.testing.assert_array_equal(jax_kept, [1, 1], err_msg="JAX p=0 didn't keep exactly 1 token")

        # Verify masks match
        np.testing.assert_array_equal(
            np.isneginf(mlx_np), np.isneginf(jax_out),
            err_msg="p=0 mask mismatch"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_p_equals_1(self, skip_without_jax):
        """Test P=1 (no filtering) behavior."""
        from mlx_primitives.generation.samplers import apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        p = 1.0

        # MLX
        logits_mlx = mx.array(logits_np)
        mlx_out = apply_top_p(logits_mlx, p)
        mx.eval(mlx_out)

        # JAX
        jax_out = jax_top_p_sampling(logits_np, p)

        # Output should equal input (no filtering)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="MLX p=1 should return logits unchanged"
        )
        np.testing.assert_allclose(
            jax_out, logits_np,
            rtol=1e-6, atol=1e-7,
            err_msg="JAX p=1 should return logits unchanged"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_with_temperature(self, skip_without_jax):
        """Test Top-P combined with temperature scaling."""
        from mlx_primitives.generation.samplers import apply_temperature, apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        temperature = 0.7
        p = 0.9

        # MLX: temperature then top-p
        logits_mlx = mx.array(logits_np)
        mlx_scaled = apply_temperature(logits_mlx, temperature)
        mlx_filtered = apply_top_p(mlx_scaled, p)
        mx.eval(mlx_filtered)

        # JAX
        jax_scaled = jax_temperature_sampling(logits_np, temperature)
        jax_filtered = jax_top_p_sampling(jax_scaled, p)

        mlx_np = _to_numpy(mlx_filtered)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        jax_masked = np.isneginf(jax_filtered)
        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg="Temperature+Top-P mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], jax_filtered[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Temperature+Top-P values mismatch"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_combined_top_k_top_p(self, skip_without_jax):
        """Test combined Top-K and Top-P filtering."""
        from mlx_primitives.generation.samplers import apply_top_k, apply_top_p

        np.random.seed(42)
        logits_np = np.random.randn(2, 100).astype(np.float32)
        k = 50
        p = 0.9

        # MLX: top-k then top-p
        logits_mlx = mx.array(logits_np)
        mlx_topk = apply_top_k(logits_mlx, k)
        mlx_topp = apply_top_p(mlx_topk, p)
        mx.eval(mlx_topp)

        # JAX
        jax_topk = jax_top_k_sampling(logits_np, k)
        jax_topp = jax_top_p_sampling(jax_topk, p)

        mlx_np = _to_numpy(mlx_topp)

        # Check mask positions match
        mlx_masked = np.isneginf(mlx_np)
        jax_masked = np.isneginf(jax_topp)
        np.testing.assert_array_equal(
            mlx_masked, jax_masked,
            err_msg="Top-K+Top-P mask mismatch"
        )

        # Check non-masked values match
        non_masked = ~mlx_masked
        if np.any(non_masked):
            np.testing.assert_allclose(
                mlx_np[non_masked], jax_topp[non_masked],
                rtol=1e-5, atol=1e-6,
                err_msg="Top-K+Top-P values mismatch"
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
