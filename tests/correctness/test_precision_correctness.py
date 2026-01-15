"""Correctness tests for Precision system.

Tests verify:
1. FP16 safety detection heuristics
2. Sequence length thresholds
3. Overflow prevention
4. Context manager behavior
5. Integration with attention
"""

import pytest
import mlx.core as mx

from mlx_primitives.config.precision import (
    PrecisionMode,
    PrecisionConfig,
    get_precision_config,
    set_precision_config,
    set_precision_mode,
    precision_context,
    is_attention_safe_for_fp16,
    should_use_fp16,
)
from mlx_primitives.attention import flash_attention_forward


# =============================================================================
# Test Classes
# =============================================================================


class TestFP16SafetyDetection:
    """Test FP16 safety heuristics."""

    def test_small_values_safe(self):
        """Small input values should be identified as safe for FP16."""
        mx.random.seed(42)
        # Small values, well within fp16 range
        q = mx.random.normal((1, 128, 8, 64)) * 0.1
        k = mx.random.normal((1, 128, 8, 64)) * 0.1
        v = mx.random.normal((1, 128, 8, 64)) * 0.1
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        is_safe, reason = is_attention_safe_for_fp16(q, k, v, seq_len=128, scale=scale)

        assert is_safe, f"Small values should be safe for fp16, got: {reason}"

    def test_large_values_unsafe(self):
        """Large input values should be identified as unsafe for FP16."""
        mx.random.seed(42)
        # Values near fp16 max (65504)
        q = mx.random.normal((1, 128, 8, 64)) * 60000
        k = mx.random.normal((1, 128, 8, 64)) * 60000
        v = mx.random.normal((1, 128, 8, 64)) * 60000
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        is_safe, reason = is_attention_safe_for_fp16(q, k, v, seq_len=128, scale=scale)

        assert not is_safe, "Large values should be unsafe for fp16"
        assert "magnitude" in reason.lower() or "overflow" in reason.lower()

    def test_moderate_values_estimation(self):
        """Values that could cause attention score overflow should be detected."""
        mx.random.seed(42)
        # Moderate Q/K values that could overflow in Q@K^T
        # When Q*K*head_dim*scale > 11, exp() can overflow
        # With scale=1/8, head_dim=64, Q*K*64/8 > 11 means Q*K > 1.4
        # So Q=K=2 would give Q*K*head_dim*scale = 2*2*64*0.125 = 32 > 11
        q = mx.ones((1, 128, 8, 64)) * 2.0
        k = mx.ones((1, 128, 8, 64)) * 2.0
        v = mx.ones((1, 128, 8, 64))
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)  # 0.125
        is_safe, reason = is_attention_safe_for_fp16(q, k, v, seq_len=128, scale=scale)

        assert not is_safe, f"High attention scores should be unsafe: {reason}"
        assert "score" in reason.lower() or "overflow" in reason.lower()


class TestSequenceLengthThreshold:
    """Test sequence length bounds for FP16."""

    def test_short_sequence_below_min(self):
        """Very short sequences shouldn't use fp16 (overhead not worth it)."""
        mx.random.seed(42)
        q = mx.random.normal((1, 32, 8, 64)) * 0.1
        k = mx.random.normal((1, 32, 8, 64)) * 0.1
        v = mx.random.normal((1, 32, 8, 64)) * 0.1
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        config = PrecisionConfig(min_seq_len_fp16=64)
        is_safe, reason = is_attention_safe_for_fp16(
            q, k, v, seq_len=32, scale=scale, config=config
        )

        assert not is_safe, "Short sequences should not use fp16"
        assert "short" in reason.lower() or "too" in reason.lower()

    def test_long_sequence_above_max(self):
        """Very long sequences accumulate too much fp16 error."""
        mx.random.seed(42)
        # Don't actually allocate huge tensors, just test the threshold
        q = mx.random.normal((1, 64, 8, 64)) * 0.1
        k = mx.random.normal((1, 64, 8, 64)) * 0.1
        v = mx.random.normal((1, 64, 8, 64)) * 0.1
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        config = PrecisionConfig(max_seq_len_fp16=4096)
        is_safe, reason = is_attention_safe_for_fp16(
            q, k, v, seq_len=8192, scale=scale, config=config  # seq_len param, not tensor
        )

        assert not is_safe, "Long sequences should not use fp16"
        assert "exceed" in reason.lower() or "length" in reason.lower()

    def test_sequence_within_bounds(self):
        """Sequences within bounds should pass length check."""
        mx.random.seed(42)
        q = mx.random.normal((1, 256, 8, 64)) * 0.1
        k = mx.random.normal((1, 256, 8, 64)) * 0.1
        v = mx.random.normal((1, 256, 8, 64)) * 0.1
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        config = PrecisionConfig(min_seq_len_fp16=64, max_seq_len_fp16=8192)
        is_safe, reason = is_attention_safe_for_fp16(
            q, k, v, seq_len=256, scale=scale, config=config
        )

        assert is_safe, f"Sequence within bounds should be safe: {reason}"


class TestOverflowPrevention:
    """Test that overflow-prone inputs get fallback to fp32."""

    def test_inf_in_input_detected(self):
        """Inf values in input should be detected as unsafe."""
        q = mx.array([[[1.0, float("inf"), 0.0, 0.0]]]).reshape(1, 1, 1, 4)
        k = mx.random.normal((1, 1, 1, 4))
        v = mx.random.normal((1, 1, 1, 4))
        mx.eval(q, k, v)

        scale = 0.5
        config = PrecisionConfig(min_seq_len_fp16=1)  # Allow short seq for this test
        is_safe, reason = is_attention_safe_for_fp16(
            q, k, v, seq_len=1, scale=scale, config=config
        )

        assert not is_safe, "Inf in input should be unsafe"
        assert "inf" in reason.lower()

    def test_nan_in_input_detected(self):
        """NaN values in input should be detected as unsafe."""
        q = mx.array([[[1.0, float("nan"), 0.0, 0.0]]]).reshape(1, 1, 1, 4)
        k = mx.random.normal((1, 1, 1, 4))
        v = mx.random.normal((1, 1, 1, 4))
        mx.eval(q, k, v)

        scale = 0.5
        config = PrecisionConfig(min_seq_len_fp16=1)
        is_safe, reason = is_attention_safe_for_fp16(
            q, k, v, seq_len=1, scale=scale, config=config
        )

        assert not is_safe, "NaN in input should be unsafe"
        assert "nan" in reason.lower()

    def test_should_use_fp16_fallback(self):
        """should_use_fp16 falls back to fp32 when unsafe."""
        # Large values that would overflow
        q = mx.ones((1, 128, 8, 64)) * 50000
        k = mx.ones((1, 128, 8, 64)) * 50000
        v = mx.ones((1, 128, 8, 64))
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        config = PrecisionConfig(mode=PrecisionMode.AUTO, warn_on_fallback=False)
        use_fp16 = should_use_fp16(q, k, v, seq_len=128, scale=scale, config=config)

        assert not use_fp16, "Large values should trigger fp32 fallback"


class TestPrecisionContextManager:
    """Test precision context manager behavior."""

    def test_context_overrides_global(self):
        """Context should temporarily override global setting."""
        original = get_precision_config().mode

        # Set global to FLOAT32
        set_precision_mode(PrecisionMode.FLOAT32)

        # Inside context, should be FLOAT16
        with precision_context(mode=PrecisionMode.FLOAT16) as ctx:
            assert ctx.mode == PrecisionMode.FLOAT16
            assert get_precision_config().mode == PrecisionMode.FLOAT16

        # After context, should be back to FLOAT32
        assert get_precision_config().mode == PrecisionMode.FLOAT32

        # Restore original
        set_precision_mode(original)

    def test_context_restores_on_exception(self):
        """Context should restore settings even if exception occurs."""
        original = get_precision_config()
        set_precision_mode(PrecisionMode.FLOAT32)

        try:
            with precision_context(mode=PrecisionMode.FLOAT16):
                assert get_precision_config().mode == PrecisionMode.FLOAT16
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be restored
        assert get_precision_config().mode == PrecisionMode.FLOAT32
        set_precision_config(original)

    def test_context_with_additional_kwargs(self):
        """Context can override multiple config values."""
        original = get_precision_config()

        with precision_context(
            mode=PrecisionMode.AUTO,
            max_seq_len_fp16=2048,
            warn_on_fallback=False,
        ) as ctx:
            assert ctx.mode == PrecisionMode.AUTO
            assert ctx.max_seq_len_fp16 == 2048
            assert ctx.warn_on_fallback is False

        set_precision_config(original)


class TestPrecisionModes:
    """Test explicit precision mode behavior."""

    def test_float32_mode_always_uses_fp32(self):
        """FLOAT32 mode should never return True for fp16."""
        mx.random.seed(42)
        q = mx.random.normal((1, 128, 8, 64)) * 0.1
        k = mx.random.normal((1, 128, 8, 64)) * 0.1
        v = mx.random.normal((1, 128, 8, 64)) * 0.1
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        use_fp16 = should_use_fp16(
            q, k, v, seq_len=128, scale=scale,
            precision=PrecisionMode.FLOAT32
        )

        assert not use_fp16, "FLOAT32 mode should never use fp16"

    def test_float16_mode_always_uses_fp16(self):
        """FLOAT16 mode should always return True."""
        mx.random.seed(42)
        # Even with unsafe values
        q = mx.random.normal((1, 128, 8, 64)) * 50000
        k = mx.random.normal((1, 128, 8, 64)) * 50000
        v = mx.random.normal((1, 128, 8, 64))
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        use_fp16 = should_use_fp16(
            q, k, v, seq_len=128, scale=scale,
            precision=PrecisionMode.FLOAT16
        )

        assert use_fp16, "FLOAT16 mode should always use fp16"

    def test_auto_mode_respects_input_dtype(self):
        """AUTO mode should use fp16 if inputs are already fp16."""
        q = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
        k = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
        v = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        use_fp16 = should_use_fp16(
            q, k, v, seq_len=128, scale=scale,
            precision=PrecisionMode.AUTO
        )

        assert use_fp16, "AUTO should use fp16 for fp16 inputs"


class TestAttentionIntegration:
    """Test precision system integration with attention."""

    def test_attention_no_nan_with_auto_precision(self):
        """Attention with AUTO precision should never produce NaN."""
        mx.random.seed(42)
        q = mx.random.normal((1, 128, 8, 64))
        k = mx.random.normal((1, 128, 8, 64))
        v = mx.random.normal((1, 128, 8, 64))
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)
        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        has_nan = bool(mx.any(mx.isnan(out)))
        has_inf = bool(mx.any(mx.isinf(out)))

        assert not has_nan, "Attention output should not contain NaN"
        assert not has_inf, "Attention output should not contain Inf"

    def test_fp16_attention_matches_fp32_approximately(self):
        """FP16 and FP32 attention should produce similar results."""
        mx.random.seed(42)
        q = mx.random.normal((1, 128, 8, 64))
        k = mx.random.normal((1, 128, 8, 64))
        v = mx.random.normal((1, 128, 8, 64))
        mx.eval(q, k, v)

        scale = 1.0 / (64 ** 0.5)

        # FP32 attention
        out_fp32 = flash_attention_forward(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=scale,
            causal=False,
        )
        mx.eval(out_fp32)

        # FP16 attention
        out_fp16 = flash_attention_forward(
            q.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            scale=scale,
            causal=False,
        )
        mx.eval(out_fp16)

        # Compare (fp16 has lower precision)
        out_fp16_f32 = out_fp16.astype(mx.float32)
        max_diff = float(mx.max(mx.abs(out_fp32 - out_fp16_f32)))
        mean_diff = float(mx.mean(mx.abs(out_fp32 - out_fp16_f32)))

        # FP16 should be reasonably close to FP32
        # Typical tolerance for fp16 attention: 1-5% relative error
        relative_error = max_diff / (float(mx.max(mx.abs(out_fp32))) + 1e-10)
        assert relative_error < 0.05, f"FP16 vs FP32 relative error {relative_error:.4f} exceeds 5%"
        assert mean_diff < 0.01, f"Mean difference {mean_diff:.6f} too large"
