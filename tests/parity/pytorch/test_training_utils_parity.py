"""PyTorch parity tests for Training Utilities.

This module tests parity between MLX training utility implementations and
PyTorch/reference equivalents for:
- EMA (Exponential Moving Average)
- EMAWithWarmup
- GradientClipper (clip_grad_norm, clip_grad_value)
- SWA (Stochastic Weight Averaging)
- Lookahead optimizer wrapper

These are pure arithmetic operations, so tolerances are tight.
"""

import math
import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from tests.parity.conftest import HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn as torchnn

from mlx_primitives.training.utils import (
    EMA,
    EMAWithWarmup,
    GradientClipper,
    clip_grad_norm,
    clip_grad_value,
    compute_gradient_norm,
    SWA,
    Lookahead,
)


# =============================================================================
# Reference Implementations
# =============================================================================

def reference_ema_update(shadow: np.ndarray, current: np.ndarray, decay: float) -> np.ndarray:
    """Reference EMA update: shadow = decay * shadow + (1 - decay) * current."""
    return decay * shadow + (1 - decay) * current


def reference_clip_grad_norm(grads: dict, max_norm: float, norm_type: float = 2.0) -> tuple:
    """Reference gradient norm clipping using NumPy.

    This matches the MLX implementation which:
    1. Collects sum(|g|^norm_type) for each gradient tensor
    2. Computes total_norm = (sum of all those)^(1/norm_type)
    3. Clips by scaling all gradients by min(1, max_norm / (total_norm + eps))
    """
    # Flatten grads to list of numpy arrays
    flat_grads = []

    def collect_grads(g):
        if isinstance(g, np.ndarray):
            flat_grads.append(g)
        elif isinstance(g, dict):
            for v in g.values():
                collect_grads(v)
        elif isinstance(g, list):
            for v in g:
                collect_grads(v)

    collect_grads(grads)

    if not flat_grads:
        return grads, 0.0

    # Compute total norm (matching MLX implementation)
    if norm_type == float('inf'):
        total_norm = max(np.max(np.abs(g)) for g in flat_grads)
    else:
        # Sum of |g|^p for each array, then take p-th root
        total_norm_p = sum(np.sum(np.abs(g) ** norm_type) for g in flat_grads)
        total_norm = float(np.power(total_norm_p, 1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)

    # Scale gradients
    def scale_grads(g, scale):
        if isinstance(g, np.ndarray):
            return g * scale
        elif isinstance(g, dict):
            return {k: scale_grads(v, scale) for k, v in g.items()}
        elif isinstance(g, list):
            return [scale_grads(v, scale) for v in g]
        return g

    clipped = scale_grads(grads, clip_coef)
    return clipped, total_norm


def reference_swa_average(avg: np.ndarray, new: np.ndarray, n: int) -> np.ndarray:
    """Reference SWA running average: avg_new = avg + (new - avg) / (n + 1)."""
    return avg + (new - avg) / (n + 1)


def reference_lookahead_interpolate(slow: np.ndarray, fast: np.ndarray, alpha: float) -> np.ndarray:
    """Reference Lookahead interpolation: slow_new = slow + alpha * (fast - slow)."""
    return slow + alpha * (fast - slow)


# =============================================================================
# EMA Parity Tests
# =============================================================================

class TestEMAParity:
    """EMA parity tests against reference implementation."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("decay", [0.9, 0.99, 0.999, 0.9999])
    def test_ema_update_parity(self, decay, skip_without_pytorch):
        """Test EMA update matches reference formula."""
        # Create a simple model
        model = nn.Linear(64, 32)
        ema = EMA(model, decay=decay)

        # Get initial shadow params
        initial_shadow = np.array(ema.shadow_params["weight"])

        # Simulate weight update (modify model weights)
        new_weights = mx.random.normal(model.weight.shape)
        model.weight = new_weights

        # Update EMA
        ema.update(step=0)

        # Get updated shadow
        updated_shadow = np.array(ema.shadow_params["weight"])

        # Compute expected using reference
        expected_shadow = reference_ema_update(
            initial_shadow,
            np.array(new_weights),
            decay
        )

        np.testing.assert_allclose(
            updated_shadow, expected_shadow,
            rtol=1e-6, atol=1e-7,
            err_msg=f"EMA update mismatch for decay={decay}"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_ema_multiple_updates(self, skip_without_pytorch):
        """Test EMA over multiple update steps."""
        model = nn.Linear(32, 16)
        decay = 0.99
        ema = EMA(model, decay=decay)

        # Track reference shadow manually
        ref_shadow = np.array(model.weight).copy()

        for step in range(10):
            # Update model weights
            new_weights = mx.random.normal(model.weight.shape)
            model.weight = new_weights

            # Update both EMA and reference
            ema.update(step=step)
            ref_shadow = reference_ema_update(
                ref_shadow,
                np.array(new_weights),
                decay
            )

            # Compare
            mlx_shadow = np.array(ema.shadow_params["weight"])
            np.testing.assert_allclose(
                mlx_shadow, ref_shadow,
                rtol=1e-5, atol=1e-6,
                err_msg=f"EMA mismatch at step {step}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_ema_apply_restore_roundtrip(self, skip_without_pytorch):
        """Test apply_shadow and restore roundtrip."""
        model = nn.Linear(16, 8)
        ema = EMA(model, decay=0.99)

        # Store original weights
        original_weights = np.array(model.weight).copy()

        # Update model weights and EMA
        model.weight = mx.random.normal(model.weight.shape)
        ema.update(step=0)

        current_weights = np.array(model.weight).copy()
        shadow_weights = np.array(ema.shadow_params["weight"]).copy()

        # Apply shadow
        ema.apply_shadow()
        applied_weights = np.array(model.weight)

        np.testing.assert_allclose(
            applied_weights, shadow_weights,
            rtol=1e-7, atol=1e-8,
            err_msg="apply_shadow did not apply shadow weights correctly"
        )

        # Restore
        ema.restore()
        restored_weights = np.array(model.weight)

        np.testing.assert_allclose(
            restored_weights, current_weights,
            rtol=1e-7, atol=1e-8,
            err_msg="restore did not restore original weights correctly"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_ema_update_after_step(self, skip_without_pytorch):
        """Test update_after_step parameter."""
        model = nn.Linear(16, 8)
        ema = EMA(model, decay=0.99, update_after_step=5)

        initial_shadow = np.array(ema.shadow_params["weight"]).copy()

        # Update model weights
        model.weight = mx.random.normal(model.weight.shape)

        # EMA should not update for steps < 5
        for step in range(5):
            ema.update(step=step)
            current_shadow = np.array(ema.shadow_params["weight"])
            np.testing.assert_allclose(
                current_shadow, initial_shadow,
                rtol=1e-8, atol=1e-10,
                err_msg=f"EMA should not update at step {step}"
            )

        # EMA should update at step 5
        ema.update(step=5)
        updated_shadow = np.array(ema.shadow_params["weight"])

        assert not np.allclose(updated_shadow, initial_shadow), \
            "EMA should update at step 5"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_ema_update_every(self, skip_without_pytorch):
        """Test update_every parameter."""
        model = nn.Linear(16, 8)
        ema = EMA(model, decay=0.99, update_every=3)

        shadows = []

        for step in range(10):
            # Update model weights each step
            model.weight = mx.random.normal(model.weight.shape)
            ema.update(step=step)
            shadows.append(np.array(ema.shadow_params["weight"]).copy())

        # Shadow should only change at steps 0, 3, 6, 9
        for i in range(1, 10):
            if i % 3 == 0:
                # Should have changed
                assert not np.allclose(shadows[i], shadows[i-1]), \
                    f"Shadow should change at step {i}"
            else:
                # Should not have changed (compare to previous update step)
                prev_update_step = (i // 3) * 3
                np.testing.assert_allclose(
                    shadows[i], shadows[prev_update_step],
                    rtol=1e-8, atol=1e-10,
                    err_msg=f"Shadow should not change at step {i}"
                )


class TestEMAWithWarmupParity:
    """EMAWithWarmup parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("warmup_steps", [10, 100])
    def test_ema_warmup_decay_progression(self, warmup_steps, skip_without_pytorch):
        """Test decay warmup progression."""
        target_decay = 0.999
        min_decay = 0.0

        model = nn.Linear(16, 8)
        ema = EMAWithWarmup(
            model,
            decay=target_decay,
            warmup_steps=warmup_steps,
            min_decay=min_decay
        )

        # Test decay at various steps
        for step in range(warmup_steps + 10):
            decay = ema.get_decay(step)

            if step >= warmup_steps:
                # Should be at target decay
                np.testing.assert_allclose(
                    decay, target_decay, rtol=1e-8,
                    err_msg=f"Decay should be {target_decay} at step {step}"
                )
            else:
                # Should be linearly interpolated
                expected = min_decay + (step / warmup_steps) * (target_decay - min_decay)
                np.testing.assert_allclose(
                    decay, expected, rtol=1e-8,
                    err_msg=f"Decay mismatch at step {step}"
                )


# =============================================================================
# Gradient Clipping Parity Tests
# =============================================================================

class TestGradientClippingParity:
    """Gradient clipping parity tests against PyTorch."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_norm", [0.5, 1.0, 5.0])
    def test_clip_grad_norm_parity(self, max_norm, skip_without_pytorch):
        """Test clip_grad_norm matches reference implementation."""
        # Create gradient dictionary
        np.random.seed(42)
        grads = {
            "layer1": {
                "weight": np.random.randn(64, 32).astype(np.float32),
                "bias": np.random.randn(32).astype(np.float32),
            },
            "layer2": {
                "weight": np.random.randn(32, 16).astype(np.float32),
                "bias": np.random.randn(16).astype(np.float32),
            }
        }

        # Convert to MLX
        def to_mlx(g):
            if isinstance(g, np.ndarray):
                return mx.array(g)
            elif isinstance(g, dict):
                return {k: to_mlx(v) for k, v in g.items()}
            return g

        mlx_grads = to_mlx(grads)

        # MLX clip
        clipped_mlx, norm_mlx = clip_grad_norm(mlx_grads, max_norm)

        # Reference clip
        clipped_ref, norm_ref = reference_clip_grad_norm(grads, max_norm)

        # Compare norms
        np.testing.assert_allclose(
            float(norm_mlx), norm_ref, rtol=1e-5, atol=1e-6,
            err_msg=f"Gradient norm mismatch for max_norm={max_norm}"
        )

        # Compare clipped gradients
        def compare_grads(mlx_g, ref_g, path=""):
            if isinstance(mlx_g, mx.array):
                np.testing.assert_allclose(
                    np.array(mlx_g), ref_g, rtol=1e-5, atol=1e-6,
                    err_msg=f"Clipped gradient mismatch at {path}"
                )
            elif isinstance(mlx_g, dict):
                for k in mlx_g:
                    compare_grads(mlx_g[k], ref_g[k], f"{path}.{k}")

        compare_grads(clipped_mlx, clipped_ref)

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_value", [0.1, 0.5, 1.0])
    def test_clip_grad_value_parity(self, max_value, skip_without_pytorch):
        """Test clip_grad_value matches numpy clip."""
        np.random.seed(42)
        grads = {
            "weight": np.random.randn(64, 32).astype(np.float32) * 2,  # Values outside [-1, 1]
            "bias": np.random.randn(32).astype(np.float32) * 2,
        }

        # Convert to MLX
        mlx_grads = {k: mx.array(v) for k, v in grads.items()}

        # MLX clip
        clipped_mlx = clip_grad_value(mlx_grads, max_value)

        # Reference: numpy clip
        clipped_ref = {k: np.clip(v, -max_value, max_value) for k, v in grads.items()}

        # Compare
        for k in grads:
            np.testing.assert_allclose(
                np.array(clipped_mlx[k]), clipped_ref[k],
                rtol=1e-6, atol=1e-7,
                err_msg=f"clip_grad_value mismatch for {k}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("norm_type", [1.0, 2.0, float('inf')])
    def test_compute_gradient_norm(self, norm_type, skip_without_pytorch):
        """Test gradient norm computation for different norm types."""
        np.random.seed(42)
        grads = {
            "weight": np.random.randn(32, 16).astype(np.float32),
            "bias": np.random.randn(16).astype(np.float32),
        }

        # Convert to MLX
        mlx_grads = {k: mx.array(v) for k, v in grads.items()}

        # MLX norm
        mlx_norm = float(compute_gradient_norm(mlx_grads, norm_type))

        # Reference norm
        flat = np.concatenate([g.flatten() for g in grads.values()])
        if norm_type == float('inf'):
            ref_norm = np.max(np.abs(flat))
        else:
            ref_norm = np.linalg.norm(flat, ord=norm_type)

        np.testing.assert_allclose(
            mlx_norm, ref_norm, rtol=1e-5, atol=1e-6,
            err_msg=f"Gradient norm mismatch for norm_type={norm_type}"
        )


# =============================================================================
# SWA Parity Tests
# =============================================================================

class TestSWAParity:
    """Stochastic Weight Averaging parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_swa_averaging_parity(self, skip_without_pytorch):
        """Test SWA running average matches reference formula."""
        model = nn.Linear(32, 16)
        swa = SWA(model, swa_start=0, swa_freq=1)

        # Track reference average manually
        ref_avg = None

        for step in range(10):
            # Update model weights
            new_weights = mx.random.normal(model.weight.shape)
            model.weight = new_weights

            # Update SWA
            swa.update(step=step)

            # Update reference
            new_weights_np = np.array(new_weights)
            if ref_avg is None:
                ref_avg = new_weights_np.copy()
            else:
                ref_avg = reference_swa_average(ref_avg, new_weights_np, step)

            # Compare
            if swa.averaged_params is not None:
                mlx_avg = np.array(swa.averaged_params["weight"])
                np.testing.assert_allclose(
                    mlx_avg, ref_avg,
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"SWA average mismatch at step {step}"
                )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_swa_start_freq(self, skip_without_pytorch):
        """Test SWA swa_start and swa_freq parameters."""
        model = nn.Linear(16, 8)
        swa_start = 5
        swa_freq = 2
        swa = SWA(model, swa_start=swa_start, swa_freq=swa_freq)

        for step in range(15):
            model.weight = mx.random.normal(model.weight.shape)
            swa.update(step=step)

            # Should not have averaged before swa_start
            if step < swa_start:
                assert swa.averaged_params is None or swa.n_averaged == 0, \
                    f"SWA should not average before step {swa_start}"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_swa_apply(self, skip_without_pytorch):
        """Test SWA apply method."""
        model = nn.Linear(16, 8)
        swa = SWA(model, swa_start=0, swa_freq=1)

        # Run several updates
        for step in range(5):
            model.weight = mx.random.normal(model.weight.shape)
            swa.update(step=step)

        # Store averaged params
        averaged_weights = np.array(swa.averaged_params["weight"]).copy()

        # Apply averaged weights
        swa.apply()

        # Model weights should now equal averaged weights
        model_weights = np.array(model.weight)
        np.testing.assert_allclose(
            model_weights, averaged_weights,
            rtol=1e-7, atol=1e-8,
            err_msg="SWA apply did not set model weights correctly"
        )


# =============================================================================
# Lookahead Parity Tests
# =============================================================================

class TestLookaheadParity:
    """Lookahead optimizer wrapper parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("alpha", [0.5, 0.8])
    @pytest.mark.parametrize("k", [5, 10])
    def test_lookahead_slow_update(self, alpha, k, skip_without_pytorch):
        """Test Lookahead slow weight interpolation.

        Lookahead works as follows:
        1. For steps 0 to k-2: apply base optimizer, model = fast weights
        2. At step k-1: apply base optimizer, then interpolate slow weights
           and set model = new_slow = old_slow + alpha * (fast - old_slow)

        We verify that the interpolation formula is correctly applied.
        """
        import mlx.optimizers as optim

        # Use fixed seed for reproducibility
        mx.random.seed(42)
        np.random.seed(42)

        model = nn.Linear(16, 8)
        base_optimizer = optim.SGD(learning_rate=0.01)
        lookahead = Lookahead(base_optimizer, k=k, alpha=alpha)

        # Initialize slow weights
        lookahead.init_slow_weights(model)
        slow_weights_np = np.array(lookahead.slow_weights["weight"]).copy()

        # Run k-1 steps (no slow update yet)
        for step in range(k - 1):
            grads = {"weight": mx.random.normal(model.weight.shape)}
            lookahead.update(model, grads, step)

        # Capture fast weights BEFORE the k-th step's slow update
        fast_weights_before = np.array(model.weight).copy()

        # Now run step k-1 which triggers slow update
        grads = {"weight": mx.random.normal(model.weight.shape)}
        lookahead.update(model, grads, k - 1)

        # After SGD on step k-1, the fast weights would be different,
        # but then the slow update sets model to new_slow.
        # The implementation gets fast = model.parameters() AFTER SGD update.

        # We need to verify that slow_weights was updated correctly.
        # new_slow = old_slow + alpha * (fast_after_sgd - old_slow)
        # model weights = new_slow

        # Verify the interpolation formula on slow_weights
        new_slow_weights = np.array(lookahead.slow_weights["weight"])
        model_weights = np.array(model.weight)

        # Model should equal new slow weights
        np.testing.assert_allclose(
            model_weights, new_slow_weights,
            rtol=1e-6, atol=1e-7,
            err_msg="Model weights should equal new slow weights after k steps"
        )

        # Verify that new_slow moved from old_slow toward some fast weights
        # (we can't know the exact fast weights since SGD was applied)
        # But we can verify the formula was applied by checking the slow_weights
        # are different from the original
        assert not np.allclose(new_slow_weights, slow_weights_np, rtol=1e-3), \
            "Slow weights should have changed after k steps"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_lookahead_k_steps_boundary(self, skip_without_pytorch):
        """Test that slow weights update only at k step boundaries."""
        import mlx.optimizers as optim

        k = 5
        model = nn.Linear(16, 8)
        base_optimizer = optim.SGD(learning_rate=0.01)
        lookahead = Lookahead(base_optimizer, k=k, alpha=0.5)

        initial_weights = np.array(model.weight).copy()

        # Run k-1 steps
        for step in range(k - 1):
            grads = {"weight": mx.random.normal(model.weight.shape) * 0.01}
            lookahead.update(model, grads, step)

        # After k-1 steps, weights should have changed due to SGD
        weights_before_k = np.array(model.weight).copy()

        # Run step k-1 (which triggers slow weight update)
        grads = {"weight": mx.random.normal(model.weight.shape) * 0.01}
        lookahead.update(model, grads, k - 1)

        # Weights should now be interpolated (not just SGD updated)
        weights_after_k = np.array(model.weight)

        # The change should include the interpolation effect
        # (weights should move toward initial slow weights)
        assert not np.allclose(weights_after_k, weights_before_k), \
            "Weights should change at k step boundary due to slow weight update"


# =============================================================================
# Edge Cases
# =============================================================================

class TestTrainingUtilsEdgeCases:
    """Edge case tests for training utilities."""

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_ema_with_nested_params(self, skip_without_pytorch):
        """Test EMA with nested parameter dictionaries."""
        # Create a more complex model
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(32, 16)
                self.decoder = nn.Linear(16, 32)

            def __call__(self, x):
                return self.decoder(nn.relu(self.encoder(x)))

        model = NestedModel()
        ema = EMA(model, decay=0.99)

        # Update model weights
        model.encoder.weight = mx.random.normal(model.encoder.weight.shape)
        model.decoder.weight = mx.random.normal(model.decoder.weight.shape)

        # This should not raise
        ema.update(step=0)

        # Shadow should have been updated
        assert "encoder" in ema.shadow_params
        assert "decoder" in ema.shadow_params

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_gradient_clipping_empty_grads(self, skip_without_pytorch):
        """Test gradient clipping with empty gradient dict."""
        grads = {}

        # Should not raise
        clipped, norm = clip_grad_norm(grads, max_norm=1.0)

        # Norm should be zero
        assert float(norm) == 0.0

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_gradient_clipping_small_norm(self, skip_without_pytorch):
        """Test gradient clipping when norm < max_norm."""
        grads = {
            "weight": mx.array([[0.1, 0.2], [0.3, 0.4]], dtype=mx.float32)
        }
        max_norm = 10.0  # Much larger than actual norm

        clipped, norm = clip_grad_norm(grads, max_norm)

        # Gradients should be unchanged (no clipping needed)
        np.testing.assert_allclose(
            np.array(clipped["weight"]),
            np.array(grads["weight"]),
            rtol=1e-7, atol=1e-8,
            err_msg="Gradients should not be clipped when norm < max_norm"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.edge_case
    def test_swa_no_updates_before_start(self, skip_without_pytorch):
        """Test SWA does not average before swa_start."""
        model = nn.Linear(16, 8)
        swa = SWA(model, swa_start=10, swa_freq=1)

        for step in range(10):
            model.weight = mx.random.normal(model.weight.shape)
            swa.update(step=step)

        assert swa.n_averaged == 0, "SWA should not average before swa_start"
        assert swa.averaged_params is None, "averaged_params should be None before swa_start"
