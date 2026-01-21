"""PyTorch parity tests for Learning Rate Schedulers.

This module tests parity between MLX scheduler implementations and PyTorch
equivalents for:
- CosineAnnealingLR
- OneCycleLR
- PolynomialDecayLR (vs PolynomialLR)
- MultiStepLR
- ExponentialDecayLR (vs ExponentialLR)
- LinearWarmupLR (manual comparison)

NOTE: Step counting conventions differ between MLX and PyTorch:
- MLX: get_lr(step) returns LR for step `step` (0-indexed, stateless)
- PyTorch: scheduler.step() advances internal epoch counter; get_last_lr()
  returns LR for current epoch

These tests verify formula equivalence by directly computing expected LRs
or by properly aligning epoch/step indices.
"""

import math
import numpy as np
import pytest

from tests.parity.conftest import HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler


# =============================================================================
# Helper Functions
# =============================================================================

def _create_dummy_optimizer(base_lr: float) -> "optim.Optimizer":
    """Create a dummy PyTorch optimizer for scheduler testing."""
    param = torch.zeros(1, requires_grad=True)
    return optim.SGD([param], lr=base_lr)


def _compute_cosine_lr(base_lr: float, min_lr: float, step: int, T_max: int) -> float:
    """Compute expected cosine annealing LR for a given step."""
    cos_value = math.cos(math.pi * step / T_max)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + cos_value)


def _compute_polynomial_lr(base_lr: float, end_lr: float, step: int, total_steps: int, power: float) -> float:
    """Compute expected polynomial decay LR for a given step."""
    if step >= total_steps:
        return end_lr
    progress = step / total_steps
    decay_factor = (1 - progress) ** power
    return (base_lr - end_lr) * decay_factor + end_lr


def _compute_exponential_lr(base_lr: float, gamma: float, step: int) -> float:
    """Compute expected exponential decay LR for a given step."""
    return base_lr * (gamma ** step)


# =============================================================================
# CosineAnnealingLR Parity Tests
# =============================================================================

class TestCosineAnnealingLRParity:
    """CosineAnnealingLR parity tests vs expected formula."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("T_max", [100, 1000, 10000])
    @pytest.mark.parametrize("min_lr", [0.0, 0.001])
    def test_basic_cosine_annealing(self, T_max, min_lr, skip_without_pytorch):
        """Test basic CosineAnnealingLR matches expected cosine formula.

        Formula: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / T_max))
        """
        from mlx_primitives.training import CosineAnnealingLR

        base_lr = 0.1

        # MLX scheduler
        mlx_sched = CosineAnnealingLR(base_lr=base_lr, T_max=T_max, min_lr=min_lr)

        # Compare at key steps: start, 25%, 50%, 75%, end
        test_steps = [0, T_max // 4, T_max // 2, 3 * T_max // 4, T_max - 1, T_max]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)
            expected_lr = _compute_cosine_lr(base_lr, min_lr, step % T_max, T_max)

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-10, atol=1e-12,
                err_msg=f"CosineAnnealingLR mismatch at step {step} (T_max={T_max}, min_lr={min_lr})"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("T_max", [100, 500])
    def test_cosine_with_warm_restarts(self, T_max, skip_without_pytorch):
        """Test CosineAnnealingLR with T_mult > 1 matches PyTorch CosineAnnealingWarmRestarts."""
        from mlx_primitives.training import CosineAnnealingLR

        base_lr = 0.1
        min_lr = 0.0
        T_mult = 2.0

        # MLX scheduler with restarts
        mlx_sched = CosineAnnealingLR(
            base_lr=base_lr, T_max=T_max, min_lr=min_lr, T_mult=T_mult
        )

        # PyTorch CosineAnnealingWarmRestarts
        optimizer = _create_dummy_optimizer(base_lr)
        torch_sched = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_max, T_mult=int(T_mult), eta_min=min_lr
        )

        # Test across multiple cycles
        test_steps = [0, T_max // 2, T_max - 1, T_max, T_max + T_max, T_max + 2 * T_max - 1]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)

            # PyTorch: use step(epoch) to compute LR at specific epoch
            optimizer = _create_dummy_optimizer(base_lr)
            torch_sched = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_max, T_mult=int(T_mult), eta_min=min_lr
            )
            torch_sched.step(step)
            torch_lr = torch_sched.get_last_lr()[0]

            np.testing.assert_allclose(
                mlx_lr, torch_lr, rtol=1e-5, atol=1e-7,
                err_msg=f"CosineAnnealingLR warm restarts mismatch at step {step}"
            )


# =============================================================================
# OneCycleLR Parity Tests
# =============================================================================

class TestOneCycleLRParity:
    """OneCycleLR parity tests vs expected formula."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("total_steps", [100, 1000, 10000])
    @pytest.mark.parametrize("pct_start", [0.3, 0.5])
    def test_one_cycle_lr(self, total_steps, pct_start, skip_without_pytorch):
        """Test OneCycleLR follows expected one-cycle policy.

        The 1cycle policy has two phases:
        1. Ramp up from initial_lr to max_lr (0 to pct_start * total_steps)
        2. Ramp down from max_lr to final_lr (remaining steps)
        """
        from mlx_primitives.training import OneCycleLR

        max_lr = 0.1
        div_factor = 25.0
        final_div_factor = 1e4
        initial_lr = max_lr / div_factor
        final_lr = initial_lr / final_div_factor

        # MLX scheduler
        mlx_sched = OneCycleLR(
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy="cos",
        )

        step_up = int(total_steps * pct_start)

        # Test at key points
        test_steps = [
            0,  # Start (should be initial_lr)
            step_up // 2,  # Mid ramp-up
            step_up,  # Peak (should be max_lr)
            (total_steps + step_up) // 2,  # Mid ramp-down
            total_steps - 1,  # End (should be final_lr)
        ]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)

            # Verify reasonable bounds
            assert mlx_lr >= final_lr * 0.99, f"LR too low at step {step}"
            assert mlx_lr <= max_lr * 1.01, f"LR too high at step {step}"

            # Verify phase behavior
            if step == 0:
                np.testing.assert_allclose(mlx_lr, initial_lr, rtol=1e-6,
                    err_msg=f"OneCycleLR initial LR mismatch")
            elif step == step_up:
                np.testing.assert_allclose(mlx_lr, max_lr, rtol=1e-6,
                    err_msg=f"OneCycleLR peak LR mismatch")

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_one_cycle_linear(self, skip_without_pytorch):
        """Test OneCycleLR with linear annealing produces monotonic transitions."""
        from mlx_primitives.training import OneCycleLR

        max_lr = 0.1
        total_steps = 1000
        pct_start = 0.3
        div_factor = 25.0
        initial_lr = max_lr / div_factor
        step_up = int(total_steps * pct_start)

        # MLX scheduler
        mlx_sched = OneCycleLR(
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="linear",
        )

        # Test monotonically increasing in ramp-up phase
        prev_lr = 0
        for step in range(0, step_up):
            lr = mlx_sched.get_lr(step)
            assert lr >= prev_lr, f"LR not increasing at step {step}"
            prev_lr = lr

        # Peak should be max_lr
        np.testing.assert_allclose(mlx_sched.get_lr(step_up), max_lr, rtol=1e-6)

        # Test monotonically decreasing in ramp-down phase
        prev_lr = max_lr
        for step in range(step_up, total_steps):
            lr = mlx_sched.get_lr(step)
            assert lr <= prev_lr * 1.001, f"LR not decreasing at step {step}"  # Small tolerance for FP
            prev_lr = lr


# =============================================================================
# MultiStepLR Parity Tests
# =============================================================================

class TestMultiStepLRParity:
    """MultiStepLR parity tests vs expected formula."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("milestones", [[30, 60, 90], [100, 200, 300, 400]])
    @pytest.mark.parametrize("gamma", [0.1, 0.5])
    def test_multi_step_lr(self, milestones, gamma, skip_without_pytorch):
        """Test MultiStepLR correctly decays at milestones.

        Formula: lr = base_lr * gamma^(num_milestones_passed)
        """
        from mlx_primitives.training import MultiStepLR

        base_lr = 0.1

        # MLX scheduler
        mlx_sched = MultiStepLR(base_lr=base_lr, milestones=milestones, gamma=gamma)

        # Test before, at, and after each milestone
        test_steps = [0]
        for m in milestones:
            test_steps.extend([m - 1, m, m + 1])
        test_steps.append(max(milestones) + 100)

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)

            # Compute expected LR
            num_decays = sum(1 for m in milestones if step >= m)
            expected_lr = base_lr * (gamma ** num_decays)

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-10, atol=1e-12,
                err_msg=f"MultiStepLR mismatch at step {step} (milestones={milestones}, gamma={gamma})"
            )


# =============================================================================
# ExponentialDecayLR Parity Tests
# =============================================================================

class TestExponentialDecayLRParity:
    """ExponentialDecayLR parity tests vs expected formula."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99])
    def test_exponential_lr(self, gamma, skip_without_pytorch):
        """Test ExponentialDecayLR matches expected formula.

        Formula: lr = base_lr * gamma^step
        """
        from mlx_primitives.training import ExponentialDecayLR

        base_lr = 0.1
        decay_steps = 1  # Decay every step

        # MLX scheduler
        mlx_sched = ExponentialDecayLR(
            base_lr=base_lr, decay_rate=gamma, decay_steps=decay_steps, staircase=False
        )

        test_steps = [0, 1, 10, 50, 100]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)
            expected_lr = _compute_exponential_lr(base_lr, gamma, step)

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-10, atol=1e-12,
                err_msg=f"ExponentialDecayLR mismatch at step {step} (gamma={gamma})"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_exponential_lr_staircase(self, skip_without_pytorch):
        """Test ExponentialDecayLR with staircase=True.

        Formula: lr = base_lr * gamma^(step // decay_steps)
        """
        from mlx_primitives.training import ExponentialDecayLR

        base_lr = 0.1
        gamma = 0.9
        decay_steps = 10

        # MLX scheduler with staircase
        mlx_sched = ExponentialDecayLR(
            base_lr=base_lr, decay_rate=gamma, decay_steps=decay_steps, staircase=True
        )

        test_steps = [0, 5, 9, 10, 11, 19, 20, 25]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)

            # Compute expected LR with staircase
            exponent = step // decay_steps
            expected_lr = base_lr * (gamma ** exponent)

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-10, atol=1e-12,
                err_msg=f"ExponentialDecayLR staircase mismatch at step {step}"
            )


# =============================================================================
# PolynomialDecayLR Parity Tests
# =============================================================================

class TestPolynomialDecayLRParity:
    """PolynomialDecayLR parity tests vs expected formula."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("power", [1.0, 2.0, 0.5])
    @pytest.mark.parametrize("total_steps", [100, 1000])
    def test_polynomial_lr(self, power, total_steps, skip_without_pytorch):
        """Test PolynomialDecayLR matches expected formula.

        Formula: lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr
        """
        from mlx_primitives.training import PolynomialDecayLR

        base_lr = 0.1
        end_lr = 0.0

        # MLX scheduler
        mlx_sched = PolynomialDecayLR(
            base_lr=base_lr, total_steps=total_steps, end_lr=end_lr, power=power
        )

        test_steps = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)
            expected_lr = _compute_polynomial_lr(base_lr, end_lr, step, total_steps, power)

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-10, atol=1e-12,
                err_msg=f"PolynomialDecayLR mismatch at step {step} (power={power}, total={total_steps})"
            )


# =============================================================================
# LinearWarmupLR Parity Tests
# =============================================================================

class TestLinearWarmupLRParity:
    """LinearWarmupLR parity tests (manual reference implementation)."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("warmup_steps", [100, 1000])
    def test_linear_warmup(self, warmup_steps, skip_without_pytorch):
        """Test LinearWarmupLR matches expected linear warmup formula."""
        from mlx_primitives.training import LinearWarmupLR

        base_lr = 0.1

        # MLX scheduler
        mlx_sched = LinearWarmupLR(base_lr=base_lr, warmup_steps=warmup_steps)

        # PyTorch LinearLR (linear warmup from start_factor to end_factor=1.0)
        optimizer = _create_dummy_optimizer(base_lr)
        # LinearLR multiplies base_lr by a factor that goes from start_factor to end_factor
        # We want LR to go from 0 to base_lr, but PyTorch LinearLR doesn't do exactly that
        # Instead, verify against the expected formula directly

        test_steps = [0, warmup_steps // 4, warmup_steps // 2, warmup_steps - 1, warmup_steps, warmup_steps + 100]

        for step in test_steps:
            mlx_lr = mlx_sched.get_lr(step)

            # Expected: during warmup, lr = base_lr * (step + 1) / warmup_steps
            # After warmup, lr = base_lr
            if step < warmup_steps:
                expected_lr = base_lr * (step + 1) / warmup_steps
            else:
                expected_lr = base_lr

            np.testing.assert_allclose(
                mlx_lr, expected_lr, rtol=1e-6, atol=1e-10,
                err_msg=f"LinearWarmupLR mismatch at step {step} (warmup_steps={warmup_steps})"
            )


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================

class TestSchedulerEdgeCases:
    """Edge case tests for all schedulers."""

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_step_beyond_total(self, skip_without_pytorch):
        """Test scheduler behavior when stepping beyond total_steps."""
        from mlx_primitives.training import (
            CosineAnnealingLR,
            OneCycleLR,
            PolynomialDecayLR,
        )

        total_steps = 100
        base_lr = 0.1

        # CosineAnnealing should cycle
        cos_sched = CosineAnnealingLR(base_lr=base_lr, T_max=total_steps)
        lr_at_end = cos_sched.get_lr(total_steps - 1)
        lr_beyond = cos_sched.get_lr(total_steps + 50)
        assert lr_beyond > 0, "CosineAnnealingLR should cycle beyond T_max"

        # OneCycleLR should clamp to final_lr
        one_cycle = OneCycleLR(max_lr=base_lr, total_steps=total_steps)
        final_lr = one_cycle.final_lr
        lr_beyond = one_cycle.get_lr(total_steps + 50)
        np.testing.assert_allclose(lr_beyond, final_lr, rtol=1e-6,
            err_msg="OneCycleLR should clamp to final_lr after total_steps")

        # PolynomialDecay should clamp to end_lr
        poly_sched = PolynomialDecayLR(base_lr=base_lr, total_steps=total_steps, end_lr=0.01)
        lr_beyond = poly_sched.get_lr(total_steps + 50)
        assert lr_beyond == 0.01, "PolynomialDecayLR should clamp to end_lr"

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_step_zero(self, skip_without_pytorch):
        """Test scheduler behavior at step 0."""
        from mlx_primitives.training import (
            CosineAnnealingLR,
            OneCycleLR,
            MultiStepLR,
            ExponentialDecayLR,
        )

        base_lr = 0.1

        # All should return meaningful values at step 0
        cos_sched = CosineAnnealingLR(base_lr=base_lr, T_max=100)
        assert cos_sched.get_lr(0) == base_lr

        multi_sched = MultiStepLR(base_lr=base_lr, milestones=[30, 60])
        assert multi_sched.get_lr(0) == base_lr

        exp_sched = ExponentialDecayLR(base_lr=base_lr, decay_rate=0.9, decay_steps=10)
        assert exp_sched.get_lr(0) == base_lr

        # OneCycleLR starts at initial_lr = max_lr / div_factor
        one_cycle = OneCycleLR(max_lr=base_lr, total_steps=100)
        initial_lr = base_lr / 25.0  # default div_factor
        np.testing.assert_allclose(one_cycle.get_lr(0), initial_lr, rtol=1e-6)

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_very_small_lr(self, skip_without_pytorch):
        """Test schedulers with very small learning rates."""
        from mlx_primitives.training import CosineAnnealingLR

        base_lr = 1e-8
        cos_sched = CosineAnnealingLR(base_lr=base_lr, T_max=100)

        # Should not produce NaN or inf
        for step in [0, 25, 50, 75, 99]:
            lr = cos_sched.get_lr(step)
            assert not np.isnan(lr), f"NaN at step {step}"
            assert not np.isinf(lr), f"Inf at step {step}"
            assert lr >= 0, f"Negative LR at step {step}"
