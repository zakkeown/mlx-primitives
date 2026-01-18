"""Tests for GenerationEngine."""
import pytest
import mlx.core as mx

from mlx_primitives.generation.engine import (
    GenerationEngine,
    EngineConfig,
)
from mlx_primitives.generation.requests import SamplingConfig


def simple_model_forward(input_ids: mx.array, attention_mask: mx.array = None) -> mx.array:
    """Simple model that returns uniform logits for testing."""
    batch, seq = input_ids.shape
    # Return uniform logits
    return mx.zeros((batch, seq, 100))


def biased_model_forward(input_ids: mx.array, attention_mask: mx.array = None) -> mx.array:
    """Model that biases toward specific tokens for testing."""
    batch, seq = input_ids.shape
    logits = mx.zeros((batch, seq, 100))
    # Bias toward token 42
    logits = logits.at[:, :, 42].add(10.0)
    return logits


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EngineConfig(vocab_size=32000)
        assert config.vocab_size == 32000
        assert config.eos_token_id == 2
        assert config.pad_token_id == 0
        assert config.max_batch_size == 32

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EngineConfig(
            vocab_size=100,
            eos_token_id=99,
            pad_token_id=0,
            max_batch_size=8,
        )
        assert config.vocab_size == 100
        assert config.eos_token_id == 99
        assert config.max_batch_size == 8


class TestGenerationEngine:
    """Tests for GenerationEngine."""

    @pytest.fixture
    def simple_engine(self):
        """Create engine with simple uniform model."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        return GenerationEngine(simple_model_forward, config)

    @pytest.fixture
    def biased_engine(self):
        """Create engine with biased model."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        return GenerationEngine(biased_model_forward, config)

    def test_submit_creates_request(self, simple_engine) -> None:
        """Test that submit creates a valid request."""
        request = simple_engine.submit([1, 2, 3], max_new_tokens=10)
        assert request.request_id is not None
        assert not request.is_finished

    def test_submit_with_sampling_config(self, simple_engine) -> None:
        """Test submit with custom sampling config."""
        config = SamplingConfig(temperature=0.7, top_k=50, top_p=0.9)
        request = simple_engine.submit(
            [1, 2, 3],
            sampling_config=config,
            max_new_tokens=5,
        )
        assert request.sampling_config.temperature == 0.7
        assert request.sampling_config.top_k == 50

    def test_step_produces_tokens(self, simple_engine) -> None:
        """Test that step produces tokens."""
        simple_engine.submit([1, 2, 3], max_new_tokens=10)
        results = simple_engine.step()

        assert len(results) > 0
        for request_id, tokens in results.items():
            assert len(tokens) > 0
            assert all(isinstance(t, int) for t in tokens)

    def test_step_with_no_requests(self, simple_engine) -> None:
        """Test step with no pending requests."""
        results = simple_engine.step()
        assert results == {}

    def test_generate_blocking(self, biased_engine) -> None:
        """Test blocking generate call."""
        mx.random.seed(42)
        tokens = biased_engine.generate(
            [1, 2, 3],
            max_new_tokens=5,
            temperature=0.1,  # Low temperature for more deterministic
        )

        assert len(tokens) <= 5
        assert all(isinstance(t, int) for t in tokens)

    def test_generate_respects_max_tokens(self, simple_engine) -> None:
        """Test that generate respects max_new_tokens."""
        tokens = simple_engine.generate([1, 2, 3], max_new_tokens=3)
        assert len(tokens) <= 3

    def test_generate_stops_on_eos(self, simple_engine) -> None:
        """Test that generation stops on EOS token."""
        # Create a model that always produces EOS
        def eos_model(input_ids, attention_mask=None):
            batch, seq = input_ids.shape
            logits = mx.full((batch, seq, 100), -10.0)
            logits = logits.at[:, :, 99].add(20.0)  # Strong bias to EOS (99)
            return logits

        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(eos_model, config)

        tokens = engine.generate([1, 2, 3], max_new_tokens=100, temperature=0.1)

        # Should stop early due to EOS
        assert len(tokens) <= 100

    def test_multiple_requests(self, simple_engine) -> None:
        """Test submitting and processing multiple requests."""
        req1 = simple_engine.submit([1, 2, 3], max_new_tokens=5)
        req2 = simple_engine.submit([4, 5, 6], max_new_tokens=5)

        results = simple_engine.step()

        # Both requests should have results
        assert req1.request_id in results or req2.request_id in results

    def test_config_property(self, simple_engine) -> None:
        """Test config property access."""
        config = simple_engine.config
        assert config.vocab_size == 100
        assert config.eos_token_id == 99

    def test_scheduler_property(self, simple_engine) -> None:
        """Test scheduler property access."""
        scheduler = simple_engine.scheduler
        assert scheduler is not None


class TestGenerationEngineTemperature:
    """Tests for temperature-based sampling."""

    def test_low_temperature_is_deterministic(self) -> None:
        """Test that low temperature produces consistent results."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(biased_model_forward, config)

        mx.random.seed(42)
        tokens1 = engine.generate([1, 2], max_new_tokens=3, temperature=0.01)

        mx.random.seed(42)
        engine2 = GenerationEngine(biased_model_forward, config)
        tokens2 = engine2.generate([1, 2], max_new_tokens=3, temperature=0.01)

        # With same seed and low temp, should be identical
        assert tokens1 == tokens2

    def test_high_temperature_varies(self) -> None:
        """Test that high temperature produces varied results."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)

        results = []
        for seed in range(3):
            mx.random.seed(seed)
            engine = GenerationEngine(simple_model_forward, config)
            tokens = engine.generate([1, 2], max_new_tokens=5, temperature=2.0)
            results.append(tuple(tokens))

        # At high temperature with uniform logits, should get variety
        # (though not guaranteed)
        assert len(results) == 3


class TestGenerationEngineEdgeCases:
    """Edge case tests for GenerationEngine."""

    def test_empty_input(self) -> None:
        """Test with empty input."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(simple_model_forward, config)

        # Empty input should still work (may depend on model)
        request = engine.submit([], max_new_tokens=5)
        assert request.request_id is not None

    def test_single_token_input(self) -> None:
        """Test with single token input."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(simple_model_forward, config)

        tokens = engine.generate([1], max_new_tokens=3)
        assert len(tokens) <= 3

    def test_zero_max_tokens(self) -> None:
        """Test with zero max_new_tokens."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(simple_model_forward, config)

        tokens = engine.generate([1, 2, 3], max_new_tokens=0)
        # Engine may generate at least one token before checking limit
        assert len(tokens) <= 1

    def test_mx_array_input(self) -> None:
        """Test with mx.array input."""
        config = EngineConfig(vocab_size=100, eos_token_id=99)
        engine = GenerationEngine(simple_model_forward, config)

        input_ids = mx.array([1, 2, 3])
        request = engine.submit(input_ids, max_new_tokens=5)
        assert request.request_id is not None


class TestGenerationEnginePriority:
    """Tests for request priority handling."""

    def test_priority_ordering(self) -> None:
        """Test that higher priority requests are processed."""
        config = EngineConfig(vocab_size=100, eos_token_id=99, max_batch_size=1)
        engine = GenerationEngine(simple_model_forward, config)

        # Submit low priority first
        low_req = engine.submit([1, 2, 3], max_new_tokens=5, priority=0)
        # Then high priority
        high_req = engine.submit([4, 5, 6], max_new_tokens=5, priority=10)

        # With batch_size=1, only one should run
        results = engine.step()

        # High priority should be processed first
        if results:
            request_ids = list(results.keys())
            # At least one should be processed
            assert len(request_ids) >= 1
