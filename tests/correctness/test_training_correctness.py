"""Correctness tests for training utilities.

Tests verify that training utilities (schedulers, EMA, gradient utilities)
produce mathematically correct results.
"""

import math
import pytest
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_primitives.training import (
    # Schedulers
    CosineAnnealingLR,
    WarmupCosineScheduler,
    OneCycleLR,
    PolynomialDecayLR,
    InverseSqrtScheduler,
    LinearWarmupLR,
    # Utilities
    EMA,
    GradientAccumulator,
    GradientClipper,
    compute_gradient_norm,
    clip_grad_norm,
    clip_grad_value,
    SWA,
    Lookahead,
    SAM,
    GradientNoiseInjection,
)


# ============================================================================
# Reference Implementations
# ============================================================================


def naive_cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Reference cosine annealing implementation with warm restarts."""
    # Implementation uses warm restarts (SGDR), so cycle through steps
    cycle_step = step % total_steps
    progress = cycle_step / total_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def naive_warmup_cosine(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float
) -> float:
    """Reference warmup + cosine annealing implementation."""
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / warmup_steps
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def naive_polynomial_decay(
    step: int,
    total_steps: int,
    lr_start: float,
    lr_end: float,
    power: float
) -> float:
    """Reference polynomial decay implementation."""
    if step >= total_steps:
        return lr_end
    decay = (1 - step / total_steps) ** power
    return (lr_start - lr_end) * decay + lr_end


def naive_inverse_sqrt(step: int, warmup_steps: int, lr_max: float) -> float:
    """Reference inverse sqrt scheduler implementation."""
    # Avoid division by zero (matches actual implementation)
    step = max(step, 1)
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    return lr_max * math.sqrt(warmup_steps / step)


def naive_gradient_norm(grads: dict) -> float:
    """Reference gradient norm computation."""
    total = 0.0
    for key, grad in grads.items():
        if isinstance(grad, dict):
            total += naive_gradient_norm(grad) ** 2
        else:
            total += float(mx.sum(grad ** 2))
    return math.sqrt(total)


def naive_ema_update(
    ema_weights: dict,
    model_weights: dict,
    decay: float
) -> dict:
    """Reference EMA update implementation."""
    result = {}
    for key in model_weights:
        if isinstance(model_weights[key], dict):
            result[key] = naive_ema_update(ema_weights[key], model_weights[key], decay)
        else:
            result[key] = decay * ema_weights[key] + (1 - decay) * model_weights[key]
    return result


# ============================================================================
# Learning Rate Scheduler Correctness Tests
# ============================================================================


class TestCosineAnnealingCorrectness:
    """Correctness tests for CosineAnnealingLR."""

    @pytest.mark.parametrize("total_steps", [100, 500, 1000])
    def test_cosine_vs_naive(self, total_steps):
        """Compare cosine annealing against naive implementation."""
        lr_max = 0.01
        lr_min = 0.0001

        scheduler = CosineAnnealingLR(
            base_lr=lr_max,
            T_max=total_steps,
            min_lr=lr_min
        )

        for step in [0, 10, 50, total_steps // 2, total_steps - 1, total_steps]:
            our_lr = scheduler.get_lr(step)
            naive_lr = naive_cosine_annealing(step, total_steps, lr_max, lr_min)

            assert abs(our_lr - naive_lr) < 1e-7, \
                f"Step {step}: ours={our_lr}, naive={naive_lr}"

    def test_cosine_endpoints(self):
        """Test cosine scheduler hits expected endpoints."""
        lr_max = 0.01
        lr_min = 0.0001
        total_steps = 100

        scheduler = CosineAnnealingLR(
            base_lr=lr_max,
            T_max=total_steps,
            min_lr=lr_min
        )

        # Start should be at lr_max
        assert abs(scheduler.get_lr(0) - lr_max) < 1e-7

        # End of first cycle (step=total_steps-1) approaches lr_min
        # Note: Implementation uses SGDR warm restarts, step=total_steps restarts cycle
        end_lr = scheduler.get_lr(total_steps - 1)
        # At step 99/100, we should be very close to lr_min
        assert end_lr < lr_max * 0.1, f"End LR should be near min: {end_lr}"

        # Middle should be average
        mid_lr = scheduler.get_lr(total_steps // 2)
        expected_mid = (lr_max + lr_min) / 2
        assert abs(mid_lr - expected_mid) < 0.001


class TestWarmupCosineCorrectness:
    """Correctness tests for WarmupCosineScheduler."""

    @pytest.mark.parametrize("warmup_steps", [10, 50, 100])
    def test_warmup_cosine_vs_naive(self, warmup_steps):
        """Compare warmup cosine against naive implementation."""
        lr_max = 0.01
        lr_min = 0.0001
        total_steps = 500

        scheduler = WarmupCosineScheduler(
            base_lr=lr_max,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=lr_min
        )

        test_steps = [0, warmup_steps // 2, warmup_steps, warmup_steps + 50,
                      total_steps // 2, total_steps - 1]

        for step in test_steps:
            our_lr = scheduler.get_lr(step)
            naive_lr = naive_warmup_cosine(step, warmup_steps, total_steps, lr_max, lr_min)

            assert abs(our_lr - naive_lr) < 1e-6, \
                f"Step {step}: ours={our_lr}, naive={naive_lr}"

    def test_warmup_is_linear(self):
        """Test warmup phase is linear."""
        warmup_steps = 100
        lr_max = 0.01

        scheduler = WarmupCosineScheduler(
            base_lr=lr_max,
            warmup_steps=warmup_steps,
            total_steps=1000
        )

        # Check linearity during warmup
        for step in range(1, warmup_steps):
            expected = lr_max * step / warmup_steps
            actual = scheduler.get_lr(step)
            assert abs(actual - expected) < 1e-7, \
                f"Step {step}: expected {expected}, got {actual}"


class TestPolynomialDecayCorrectness:
    """Correctness tests for PolynomialDecayLR."""

    @pytest.mark.parametrize("power", [0.5, 1.0, 2.0, 3.0])
    def test_polynomial_vs_naive(self, power):
        """Compare polynomial decay against naive implementation."""
        lr_start = 0.01
        lr_end = 0.0001
        total_steps = 1000

        scheduler = PolynomialDecayLR(
            base_lr=lr_start,
            total_steps=total_steps,
            end_lr=lr_end,
            power=power
        )

        for step in [0, 100, 500, 750, 999]:
            our_lr = scheduler.get_lr(step)
            naive_lr = naive_polynomial_decay(step, total_steps, lr_start, lr_end, power)

            assert abs(our_lr - naive_lr) < 1e-7, \
                f"Step {step}, power {power}: ours={our_lr}, naive={naive_lr}"

    def test_polynomial_power_1_is_linear(self):
        """Test polynomial decay with power=1 is linear."""
        lr_start = 0.01
        lr_end = 0.0
        total_steps = 100

        scheduler = PolynomialDecayLR(
            base_lr=lr_start,
            total_steps=total_steps,
            end_lr=lr_end,
            power=1.0
        )

        for step in range(0, total_steps, 10):
            expected = lr_start * (1 - step / total_steps)
            actual = scheduler.get_lr(step)
            assert abs(actual - expected) < 1e-7


class TestInverseSqrtCorrectness:
    """Correctness tests for InverseSqrtScheduler."""

    def test_inverse_sqrt_vs_naive(self):
        """Compare inverse sqrt against naive implementation."""
        lr_max = 0.01
        warmup_steps = 100

        scheduler = InverseSqrtScheduler(
            base_lr=lr_max,
            warmup_steps=warmup_steps
        )

        for step in [0, 50, 100, 200, 500, 1000]:
            our_lr = scheduler.get_lr(step)
            naive_lr = naive_inverse_sqrt(step, warmup_steps, lr_max)

            assert abs(our_lr - naive_lr) < 1e-7, \
                f"Step {step}: ours={our_lr}, naive={naive_lr}"

    def test_inverse_sqrt_decay_rate(self):
        """Test inverse sqrt decays at expected rate."""
        lr_max = 0.01
        warmup_steps = 100

        scheduler = InverseSqrtScheduler(
            base_lr=lr_max,
            warmup_steps=warmup_steps
        )

        # After warmup, lr should scale as 1/sqrt(step)
        lr_200 = scheduler.get_lr(200)
        lr_800 = scheduler.get_lr(800)

        # lr_800 / lr_200 should be sqrt(200/800) = 0.5
        ratio = lr_800 / lr_200
        expected_ratio = math.sqrt(200 / 800)

        assert abs(ratio - expected_ratio) < 1e-6


# ============================================================================
# Gradient Utilities Correctness Tests
# ============================================================================


class TestGradientNormCorrectness:
    """Correctness tests for gradient norm computation."""

    def test_gradient_norm_vs_naive(self):
        """Compare gradient norm against naive implementation."""
        mx.random.seed(42)

        grads = {
            "layer1": {
                "weight": mx.random.normal((64, 32)),
                "bias": mx.random.normal((64,))
            },
            "layer2": {
                "weight": mx.random.normal((128, 64)),
                "bias": mx.random.normal((128,))
            }
        }

        for key in grads:
            for subkey in grads[key]:
                mx.eval(grads[key][subkey])

        our_norm = compute_gradient_norm(grads)
        naive_norm = naive_gradient_norm(grads)

        assert abs(our_norm - naive_norm) < 1e-4, \
            f"Ours: {our_norm}, Naive: {naive_norm}"

    def test_gradient_norm_single_tensor(self):
        """Test gradient norm for a single tensor."""
        mx.random.seed(42)

        grads = {"weight": mx.random.normal((100, 100))}
        mx.eval(grads["weight"])

        our_norm = compute_gradient_norm(grads)
        expected = float(mx.sqrt(mx.sum(grads["weight"] ** 2)))

        assert abs(our_norm - expected) < 1e-4


class TestGradientClippingCorrectness:
    """Correctness tests for gradient clipping."""

    def test_clip_by_norm_scales_correctly(self):
        """Test gradient clipping scales gradients correctly."""
        mx.random.seed(42)

        # Create gradients with known norm
        grads = {"weight": mx.ones((10, 10))}  # norm = 10
        mx.eval(grads["weight"])

        original_norm = compute_gradient_norm(grads)
        max_norm = 5.0

        clipped, _ = clip_grad_norm(grads, max_norm=max_norm)
        clipped_norm = float(compute_gradient_norm(clipped))

        # Clipped norm should be max_norm
        assert abs(clipped_norm - max_norm) < 1e-4, \
            f"Clipped norm: {clipped_norm}, expected: {max_norm}"

        # Direction should be preserved (scaled uniformly)
        scale = max_norm / float(original_norm)
        expected = grads["weight"] * scale
        assert mx.allclose(clipped["weight"], expected, atol=1e-5)

    def test_clip_by_norm_no_change_if_below_threshold(self):
        """Test clipping doesn't change gradients below threshold."""
        mx.random.seed(42)

        grads = {"weight": mx.random.normal((10, 10)) * 0.1}
        mx.eval(grads["weight"])

        original_norm = compute_gradient_norm(grads)
        max_norm = 100.0  # Much larger than actual norm

        clipped, _ = clip_grad_norm(grads, max_norm=max_norm)

        # Should be unchanged
        assert mx.allclose(clipped["weight"], grads["weight"], atol=1e-6)

    def test_clip_by_value_clips_correctly(self):
        """Test value clipping clips individual values."""
        grads = {"weight": mx.array([[-10.0, 5.0], [3.0, 20.0]])}
        mx.eval(grads["weight"])

        clip_value = 8.0
        clipped = clip_grad_value(grads, max_value=clip_value)

        expected = mx.array([[-8.0, 5.0], [3.0, 8.0]])

        assert mx.allclose(clipped["weight"], expected, atol=1e-6)


# ============================================================================
# EMA Correctness Tests
# ============================================================================


class TestEMACorrectness:
    """Correctness tests for Exponential Moving Average."""

    def test_ema_update_vs_naive(self):
        """Compare EMA update against naive implementation."""
        mx.random.seed(42)

        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.99)

        # Get initial EMA weights (shadow_params is the internal storage)
        ema_weights = ema.shadow_params

        # Update model
        model.weight = model.weight + mx.random.normal((10, 10)) * 0.1
        mx.eval(model.weight)

        # Update EMA (requires step argument)
        ema.update(step=0)
        new_ema_weights = ema.shadow_params

        # Compute naive EMA
        decay = 0.99
        expected_weight = decay * ema_weights["weight"] + (1 - decay) * model.weight

        assert mx.allclose(new_ema_weights["weight"], expected_weight, atol=1e-5)

    def test_ema_converges_to_model(self):
        """Test EMA converges to model weights with repeated updates."""
        mx.random.seed(42)

        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)  # Lower decay for faster convergence

        # Update many times (same model weights)
        for step in range(100):
            ema.update(step=step)

        ema_weights = ema.shadow_params

        # Should converge to model weights
        assert mx.allclose(ema_weights["weight"], model.weight, atol=1e-3)


# ============================================================================
# Gradient Accumulator Correctness Tests
# ============================================================================


class TestGradientAccumulatorCorrectness:
    """Correctness tests for gradient accumulation."""

    def test_accumulation_equals_sum(self):
        """Test accumulated gradients equal sum of individual gradients."""
        mx.random.seed(42)

        accumulator = GradientAccumulator(accumulation_steps=4)

        # Accumulate 4 gradient batches
        all_grads = []
        for i in range(4):
            grads = {
                "weight": mx.random.normal((10, 10)),
                "bias": mx.random.normal((10,))
            }
            mx.eval(grads["weight"], grads["bias"])
            all_grads.append(grads)

            accumulator.accumulate(grads)
            # Check should_step at the correct iteration
            if i < 3:
                assert not accumulator.should_step(i)
            else:
                assert accumulator.should_step(i)

        # Get accumulated gradients
        accumulated = accumulator.get_accumulated()

        # Compute expected (average of all)
        expected_weight = sum(g["weight"] for g in all_grads) / 4
        expected_bias = sum(g["bias"] for g in all_grads) / 4

        assert mx.allclose(accumulated["weight"], expected_weight, atol=1e-5)
        assert mx.allclose(accumulated["bias"], expected_bias, atol=1e-5)

    def test_accumulator_reset(self):
        """Test accumulator properly resets."""
        mx.random.seed(42)

        accumulator = GradientAccumulator(accumulation_steps=2)

        # First accumulation cycle
        grads1 = {"weight": mx.ones((5, 5))}
        mx.eval(grads1["weight"])
        accumulator.accumulate(grads1)
        accumulator.accumulate(grads1)
        acc1 = accumulator.get_accumulated()
        accumulator.reset()

        # Second accumulation cycle with different gradients
        grads2 = {"weight": mx.ones((5, 5)) * 2}
        mx.eval(grads2["weight"])
        accumulator.accumulate(grads2)
        accumulator.accumulate(grads2)
        acc2 = accumulator.get_accumulated()

        # Should be independent
        assert float(mx.mean(acc1["weight"])) == 1.0
        assert float(mx.mean(acc2["weight"])) == 2.0


# ============================================================================
# SWA Correctness Tests
# ============================================================================


class TestSWACorrectness:
    """Correctness tests for Stochastic Weight Averaging."""

    def test_swa_average_is_correct(self):
        """Test SWA computes correct running average."""
        mx.random.seed(42)

        model = nn.Linear(10, 10)
        swa = SWA(model, swa_start=0, swa_freq=1)

        # Track all weights
        all_weights = []

        for step in range(5):
            # Update model with known pattern
            model.weight = mx.ones((10, 10)) * (step + 1)
            mx.eval(model.weight)
            all_weights.append(model.weight.tolist())
            swa.update(step=step)

        # Apply SWA
        swa.apply()

        # Expected: average of [1, 2, 3, 4, 5] = 3
        expected_mean = 3.0
        actual_mean = float(mx.mean(model.weight))

        assert abs(actual_mean - expected_mean) < 1e-4, \
            f"Expected mean {expected_mean}, got {actual_mean}"


# ============================================================================
# Gradient Noise Injection Correctness Tests
# ============================================================================


class TestGradientNoiseCorrectness:
    """Correctness tests for gradient noise injection."""

    def test_noise_variance_formula(self):
        """Test noise stddev follows the formula: eta / (1 + t)^gamma."""
        noise_injector = GradientNoiseInjection(eta=1.0, gamma=0.55)

        # Compute expected stddev at different steps
        for step in [1, 10, 100, 1000]:
            expected_stddev = 1.0 / (1 + step) ** 0.55
            # Variance is stddev squared
            expected_var = expected_stddev ** 2

            # Get noise by injecting into zero gradients
            grads = {"weight": mx.zeros((1000, 1000))}
            noisy = noise_injector.add_noise(grads, step=step)

            actual_var = float(mx.var(noisy["weight"]))

            # Allow some statistical variance (we're sampling)
            assert abs(actual_var - expected_var) / expected_var < 0.2, \
                f"Step {step}: expected var {expected_var}, got {actual_var}"

    def test_noise_has_zero_mean(self):
        """Test injected noise has approximately zero mean."""
        noise_injector = GradientNoiseInjection(eta=1.0, gamma=0.55)

        grads = {"weight": mx.zeros((1000, 1000))}
        noisy = noise_injector.add_noise(grads, step=1)

        mean = float(mx.mean(noisy["weight"]))

        # Mean should be close to zero
        assert abs(mean) < 0.01, f"Noise mean not zero: {mean}"


# ============================================================================
# Integration Tests
# ============================================================================


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_full_training_step(self):
        """Test a full training step with all components."""
        mx.random.seed(42)

        # Setup
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        optimizer = optim.SGD(learning_rate=0.01)
        scheduler = WarmupCosineScheduler(
            base_lr=0.01,
            warmup_steps=10,
            total_steps=100
        )
        ema = EMA(model, decay=0.99)
        clipper = GradientClipper(max_norm=1.0)

        # Training step
        x = mx.random.normal((4, 10))
        y = mx.random.normal((4, 10))
        mx.eval(x, y)

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss)

        # Clip gradients (returns tuple)
        grads, _ = clipper.clip_by_norm(grads)

        # Update model
        optimizer.update(model, grads)

        # Update EMA (requires step argument)
        ema.update(step=1)

        # Update learning rate
        new_lr = scheduler.get_lr(step=1)

        # All should complete without error
        assert loss is not None
        assert new_lr > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
