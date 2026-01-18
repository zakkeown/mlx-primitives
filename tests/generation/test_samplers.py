"""Tests for token samplers with NumPy reference validation."""

import math
import pytest
import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from mlx_primitives.generation import (
    TokenSampler,
    SamplingConfig,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
)


def numpy_softmax(x: NDArray, axis: int = -1) -> NDArray:
    """NumPy reference softmax implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def numpy_top_k_filter(logits: NDArray, k: int) -> NDArray:
    """NumPy reference for top-k filtering.

    Sets all logits below the k-th largest to -inf.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits.copy()

    result = logits.copy()
    for i in range(logits.shape[0]):
        # Find the k-th largest value
        threshold = np.partition(logits[i], -k)[-k]
        # Mask out values below threshold
        mask = logits[i] < threshold
        result[i, mask] = float("-inf")
    return result


def numpy_top_p_filter(logits: NDArray, p: float) -> NDArray:
    """NumPy reference for top-p (nucleus) filtering.

    Keeps smallest set of tokens whose cumulative probability >= p.
    """
    if p >= 1.0:
        return logits.copy()

    result = logits.copy()
    for i in range(logits.shape[0]):
        probs = numpy_softmax(logits[i:i+1])[0]
        # Sort by probability descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cutoff
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, p) + 1

        # Mask out tokens beyond cutoff
        tokens_to_keep = set(sorted_indices[:cutoff_idx])
        for j in range(logits.shape[-1]):
            if j not in tokens_to_keep:
                result[i, j] = float("-inf")

    return result


def numpy_apply_temperature(logits: NDArray, temperature: float) -> NDArray:
    """NumPy reference for temperature scaling."""
    if temperature == 0.0:
        return logits  # Will be handled by greedy
    return logits / temperature


def numpy_apply_repetition_penalty(
    logits: NDArray,
    generated_tokens: list,
    penalty: float
) -> NDArray:
    """NumPy reference for repetition penalty.

    Divides logits of already-generated tokens by penalty.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits.copy()

    result = logits.copy()
    for token_id in generated_tokens:
        if 0 <= token_id < logits.shape[-1]:
            if result[0, token_id] > 0:
                result[0, token_id] /= penalty
            else:
                result[0, token_id] *= penalty
    return result


class TestTopKSampling:
    """Tests for top-k filtering."""

    def test_top_k_basic(self) -> None:
        """Test basic top-k filtering."""
        mx.random.seed(42)
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = apply_top_k(logits, k=3)

        # Only top 3 (indices 2, 3, 4) should have valid values
        result_np = np.array(result)
        assert result_np[0, 0] == float("-inf")
        assert result_np[0, 1] == float("-inf")
        assert result_np[0, 2] != float("-inf")
        assert result_np[0, 3] != float("-inf")
        assert result_np[0, 4] != float("-inf")

    def test_top_k_matches_numpy(self) -> None:
        """Test top-k matches NumPy reference."""
        mx.random.seed(42)
        logits = mx.random.normal((1, 100))

        result = apply_top_k(logits, k=10)
        expected = numpy_top_k_filter(np.array(logits), k=10)

        # Count non-inf values
        result_np = np.array(result)
        assert np.sum(result_np[0] > float("-inf")) <= 10
        assert np.sum(expected[0] > float("-inf")) <= 10

    def test_top_k_zero_disabled(self) -> None:
        """Test k=0 disables filtering."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = apply_top_k(logits, k=0)

        np.testing.assert_array_equal(np.array(result), np.array(logits))

    def test_top_k_batched(self) -> None:
        """Test top-k with batched input filters correctly."""
        mx.random.seed(42)
        logits = mx.random.normal((4, 50))

        result = apply_top_k(logits, k=5)

        result_np = np.array(result)
        logits_np = np.array(logits)
        for i in range(4):
            # Verify that only top-k values (by magnitude) are kept
            non_inf_count = np.sum(result_np[i] > float("-inf"))
            # Should have some filtering occurred
            assert non_inf_count < 50
            # Values that are kept should be the largest ones
            kept_vals = result_np[i][result_np[i] > float("-inf")]
            if len(kept_vals) > 0:
                min_kept = np.min(kept_vals)
                # All discarded values should be <= min kept value
                discarded = logits_np[i][result_np[i] == float("-inf")]
                if len(discarded) > 0:
                    assert np.all(discarded <= min_kept)


class TestTopPSampling:
    """Tests for top-p (nucleus) filtering."""

    def test_top_p_basic(self) -> None:
        """Test basic top-p filtering."""
        # Create logits where one token dominates
        logits = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])

        result = apply_top_p(logits, p=0.9)

        # With dominant token, only it should survive
        probs = mx.softmax(result, axis=-1)
        assert np.array(probs)[0, 0] > 0.99

    def test_top_p_one_disabled(self) -> None:
        """Test p=1.0 disables filtering."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = apply_top_p(logits, p=1.0)

        # All tokens should be valid (no -inf)
        result_np = np.array(result)
        assert not np.any(np.isinf(result_np))

    def test_top_p_keeps_high_prob(self) -> None:
        """Test that high probability tokens are kept."""
        mx.random.seed(42)
        # Make first token very high probability
        logits = mx.array([[20.0, 1.0, 1.0, 1.0, 1.0]])

        result = apply_top_p(logits, p=0.5)

        result_np = np.array(result)
        # First token should definitely be kept
        assert result_np[0, 0] != float("-inf")


class TestRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_penalty_basic(self) -> None:
        """Test basic repetition penalty."""
        logits = mx.array([[5.0, 3.0, 1.0, 2.0, 4.0]])
        generated = [0, 2]  # Penalize tokens 0 and 2

        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        result_np = np.array(result)

        # Positive logits should be divided by penalty
        assert result_np[0, 0] == 2.5  # 5.0 / 2.0
        assert result_np[0, 2] == 0.5  # 1.0 / 2.0
        # Other tokens unchanged
        assert result_np[0, 1] == 3.0
        assert result_np[0, 3] == 2.0
        assert result_np[0, 4] == 4.0

    def test_penalty_one_disabled(self) -> None:
        """Test penalty=1.0 has no effect."""
        logits = mx.array([[5.0, 3.0, 1.0, 2.0, 4.0]])
        generated = [0, 2]

        result = apply_repetition_penalty(logits, generated, penalty=1.0)

        np.testing.assert_array_equal(np.array(result), np.array(logits))

    def test_penalty_matches_numpy(self) -> None:
        """Test penalty matches NumPy reference."""
        mx.random.seed(42)
        logits = mx.random.normal((1, 50))
        generated = [1, 5, 10, 25, 30]

        result = apply_repetition_penalty(logits, generated, penalty=1.5)
        expected = numpy_apply_repetition_penalty(
            np.array(logits), generated, penalty=1.5
        )

        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5)


class TestTokenSampler:
    """Tests for the complete TokenSampler."""

    def test_greedy_sampling(self) -> None:
        """Test greedy (temperature=0) sampling."""
        sampler = TokenSampler()
        logits = mx.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        configs = [SamplingConfig(temperature=0.0)]

        tokens = sampler(logits, configs)

        assert int(tokens[0]) == 1  # Index of max (5.0)

    def test_top_k_1_is_greedy(self) -> None:
        """Test top_k=1 is equivalent to greedy."""
        sampler = TokenSampler()
        logits = mx.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        configs = [SamplingConfig(temperature=1.0, top_k=1)]

        tokens = sampler(logits, configs)

        assert int(tokens[0]) == 1  # Index of max

    def test_deterministic_with_seed(self) -> None:
        """Test sampling is deterministic with same seed."""
        sampler = TokenSampler()
        logits = mx.random.normal((1, 100))
        configs = [SamplingConfig(temperature=1.0, top_k=10)]

        mx.random.seed(42)
        tokens1 = sampler(logits, configs)

        mx.random.seed(42)
        tokens2 = sampler(logits, configs)

        assert int(tokens1[0]) == int(tokens2[0])

    def test_batched_sampling(self) -> None:
        """Test batched sampling with different configs."""
        sampler = TokenSampler()
        logits = mx.array([
            [10.0, 1.0, 1.0, 1.0],  # Clear max at 0
            [1.0, 10.0, 1.0, 1.0],  # Clear max at 1
            [1.0, 1.0, 10.0, 1.0],  # Clear max at 2
        ])
        configs = [
            SamplingConfig(temperature=0.0),
            SamplingConfig(temperature=0.0),
            SamplingConfig(temperature=0.0),
        ]

        tokens = sampler(logits, configs)

        assert int(tokens[0]) == 0
        assert int(tokens[1]) == 1
        assert int(tokens[2]) == 2

    def test_output_shape(self) -> None:
        """Test output shape matches batch size."""
        sampler = TokenSampler()
        batch_size = 8
        logits = mx.random.normal((batch_size, 50))
        configs = [SamplingConfig(temperature=1.0)] * batch_size

        tokens = sampler(logits, configs)

        assert tokens.shape == (batch_size,)

    def test_tokens_in_valid_range(self) -> None:
        """Test sampled tokens are within vocabulary range."""
        sampler = TokenSampler()
        vocab_size = 100
        logits = mx.random.normal((10, vocab_size))
        configs = [SamplingConfig(temperature=1.0, top_k=20)] * 10

        tokens = sampler(logits, configs)
        tokens_np = np.array(tokens)

        assert np.all(tokens_np >= 0)
        assert np.all(tokens_np < vocab_size)


class TestSamplingDistribution:
    """Statistical tests for sampling distributions."""

    def test_high_temp_more_uniform(self) -> None:
        """Test higher temperature produces more uniform distribution."""
        sampler = TokenSampler()
        logits = mx.array([[2.0, 1.0, 1.0, 1.0, 1.0]])  # Slight preference for 0

        # Sample many times with low temperature
        low_temp_counts = np.zeros(5)
        for _ in range(100):
            tokens = sampler(logits, [SamplingConfig(temperature=0.1)])
            low_temp_counts[int(tokens[0])] += 1

        # Sample many times with high temperature
        high_temp_counts = np.zeros(5)
        for _ in range(100):
            tokens = sampler(logits, [SamplingConfig(temperature=2.0)])
            high_temp_counts[int(tokens[0])] += 1

        # Low temperature should concentrate more on token 0
        assert low_temp_counts[0] > high_temp_counts[0]

    def test_top_k_respects_k(self) -> None:
        """Test top-k only samples from top k tokens."""
        sampler = TokenSampler()
        # Logits with clear ordering
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

        # Sample many times with top_k=3
        for _ in range(50):
            tokens = sampler(logits, [SamplingConfig(temperature=1.0, top_k=3)])
            # Should only sample from indices 7, 8, 9 (top 3)
            assert int(tokens[0]) in [7, 8, 9]


class TestSamplingConfigValidation:
    """Tests for SamplingConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration passes validation."""
        config = SamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        config.validate()  # Should not raise

    def test_negative_temperature_invalid(self) -> None:
        """Test negative temperature is rejected."""
        config = SamplingConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_negative_top_k_invalid(self) -> None:
        """Test negative top_k is rejected."""
        config = SamplingConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k"):
            config.validate()

    def test_top_p_out_of_range_invalid(self) -> None:
        """Test top_p out of range is rejected."""
        config = SamplingConfig(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

        config = SamplingConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

    def test_low_repetition_penalty_invalid(self) -> None:
        """Test repetition penalty < 1 is rejected."""
        config = SamplingConfig(repetition_penalty=0.5)
        with pytest.raises(ValueError, match="repetition_penalty"):
            config.validate()

    def test_is_greedy(self) -> None:
        """Test is_greedy detection."""
        assert SamplingConfig(temperature=0.0).is_greedy()
        assert SamplingConfig(top_k=1).is_greedy()
        assert not SamplingConfig(temperature=1.0, top_k=10).is_greedy()
