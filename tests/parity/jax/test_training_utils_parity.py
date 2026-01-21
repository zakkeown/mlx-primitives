"""JAX parity tests for Training Utilities.

This module tests parity between MLX training utility implementations and
reference implementations for:
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

from tests.parity.conftest import HAS_JAX

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
    """Reference gradient norm clipping using NumPy."""
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

    if norm_type == float('inf'):
        total_norm = max(np.max(np.abs(g)) for g in flat_grads)
    else:
        total_norm_p = sum(np.sum(np.abs(g) ** norm_type) for g in flat_grads)
        total_norm = float(np.power(total_norm_p, 1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)

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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("decay", [0.9, 0.99, 0.999, 0.9999])
    def test_ema_update_parity(self, decay, skip_without_jax):
        """Test EMA update matches reference formula."""
        model = nn.Linear(64, 32)
        ema = EMA(model, decay=decay)

        initial_shadow = np.array(ema.shadow_params["weight"])

        new_weights = mx.random.normal(model.weight.shape)
        model.weight = new_weights

        ema.update(step=0)

        updated_shadow = np.array(ema.shadow_params["weight"])

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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_ema_multiple_updates(self, skip_without_jax):
        """Test EMA over multiple update steps."""
        model = nn.Linear(32, 16)
        decay = 0.99
        ema = EMA(model, decay=decay)

        ref_shadow = np.array(model.weight).copy()

        for step in range(10):
            new_weights = mx.random.normal(model.weight.shape)
            model.weight = new_weights

            ema.update(step=step)
            ref_shadow = reference_ema_update(
                ref_shadow,
                np.array(new_weights),
                decay
            )

            mlx_shadow = np.array(ema.shadow_params["weight"])
            np.testing.assert_allclose(
                mlx_shadow, ref_shadow,
                rtol=1e-5, atol=1e-6,
                err_msg=f"EMA mismatch at step {step}"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_ema_apply_restore_roundtrip(self, skip_without_jax):
        """Test apply_shadow and restore roundtrip."""
        model = nn.Linear(16, 8)
        ema = EMA(model, decay=0.99)

        original_weights = np.array(model.weight).copy()

        model.weight = mx.random.normal(model.weight.shape)
        ema.update(step=0)

        current_weights = np.array(model.weight).copy()
        shadow_weights = np.array(ema.shadow_params["weight"]).copy()

        ema.apply_shadow()
        applied_weights = np.array(model.weight)

        np.testing.assert_allclose(
            applied_weights, shadow_weights,
            rtol=1e-7, atol=1e-8,
            err_msg="apply_shadow did not apply shadow weights correctly"
        )

        ema.restore()
        restored_weights = np.array(model.weight)

        np.testing.assert_allclose(
            restored_weights, current_weights,
            rtol=1e-7, atol=1e-8,
            err_msg="restore did not restore original weights correctly"
        )


class TestEMAWithWarmupParity:
    """EMAWithWarmup parity tests."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("warmup_steps", [10, 100])
    def test_ema_warmup_decay_progression(self, warmup_steps, skip_without_jax):
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

        for step in range(warmup_steps + 10):
            decay = ema.get_decay(step)

            if step >= warmup_steps:
                np.testing.assert_allclose(
                    decay, target_decay, rtol=1e-8,
                    err_msg=f"Decay should be {target_decay} at step {step}"
                )
            else:
                expected = min_decay + (step / warmup_steps) * (target_decay - min_decay)
                np.testing.assert_allclose(
                    decay, expected, rtol=1e-8,
                    err_msg=f"Decay mismatch at step {step}"
                )


# =============================================================================
# Gradient Clipping Parity Tests
# =============================================================================

class TestGradientClippingParity:
    """Gradient clipping parity tests against reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_norm", [0.5, 1.0, 5.0])
    def test_clip_grad_norm_parity(self, max_norm, skip_without_jax):
        """Test clip_grad_norm matches reference implementation."""
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

        def to_mlx(g):
            if isinstance(g, np.ndarray):
                return mx.array(g)
            elif isinstance(g, dict):
                return {k: to_mlx(v) for k, v in g.items()}
            return g

        mlx_grads = to_mlx(grads)

        clipped_mlx, norm_mlx = clip_grad_norm(mlx_grads, max_norm)
        clipped_ref, norm_ref = reference_clip_grad_norm(grads, max_norm)

        np.testing.assert_allclose(
            float(norm_mlx), norm_ref, rtol=1e-5, atol=1e-6,
            err_msg=f"Gradient norm mismatch for max_norm={max_norm}"
        )

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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_value", [0.1, 0.5, 1.0])
    def test_clip_grad_value_parity(self, max_value, skip_without_jax):
        """Test clip_grad_value matches numpy clip."""
        np.random.seed(42)
        grads = {
            "weight": np.random.randn(64, 32).astype(np.float32) * 2,
            "bias": np.random.randn(32).astype(np.float32) * 2,
        }

        mlx_grads = {k: mx.array(v) for k, v in grads.items()}

        clipped_mlx = clip_grad_value(mlx_grads, max_value)

        clipped_ref = {k: np.clip(v, -max_value, max_value) for k, v in grads.items()}

        for k in grads:
            np.testing.assert_allclose(
                np.array(clipped_mlx[k]), clipped_ref[k],
                rtol=1e-6, atol=1e-7,
                err_msg=f"clip_grad_value mismatch for {k}"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("norm_type", [1.0, 2.0, float('inf')])
    def test_compute_gradient_norm(self, norm_type, skip_without_jax):
        """Test gradient norm computation for different norm types."""
        np.random.seed(42)
        grads = {
            "weight": np.random.randn(32, 16).astype(np.float32),
            "bias": np.random.randn(16).astype(np.float32),
        }

        mlx_grads = {k: mx.array(v) for k, v in grads.items()}

        mlx_norm = float(compute_gradient_norm(mlx_grads, norm_type))

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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_swa_averaging_parity(self, skip_without_jax):
        """Test SWA running average matches reference formula."""
        model = nn.Linear(32, 16)
        swa = SWA(model, swa_start=0, swa_freq=1)

        ref_avg = None

        for step in range(10):
            new_weights = mx.random.normal(model.weight.shape)
            model.weight = new_weights

            swa.update(step=step)

            new_weights_np = np.array(new_weights)
            if ref_avg is None:
                ref_avg = new_weights_np.copy()
            else:
                ref_avg = reference_swa_average(ref_avg, new_weights_np, step)

            if swa.averaged_params is not None:
                mlx_avg = np.array(swa.averaged_params["weight"])
                np.testing.assert_allclose(
                    mlx_avg, ref_avg,
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"SWA average mismatch at step {step}"
                )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_swa_apply(self, skip_without_jax):
        """Test SWA apply method."""
        model = nn.Linear(16, 8)
        swa = SWA(model, swa_start=0, swa_freq=1)

        for step in range(5):
            model.weight = mx.random.normal(model.weight.shape)
            swa.update(step=step)

        averaged_weights = np.array(swa.averaged_params["weight"]).copy()

        swa.apply()

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

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("alpha", [0.5, 0.8])
    @pytest.mark.parametrize("k", [5, 10])
    def test_lookahead_slow_update(self, alpha, k, skip_without_jax):
        """Test Lookahead slow weight interpolation."""
        import mlx.optimizers as optim

        mx.random.seed(42)
        np.random.seed(42)

        model = nn.Linear(16, 8)
        base_optimizer = optim.SGD(learning_rate=0.01)
        lookahead = Lookahead(base_optimizer, k=k, alpha=alpha)

        lookahead.init_slow_weights(model)
        slow_weights_np = np.array(lookahead.slow_weights["weight"]).copy()

        for step in range(k - 1):
            grads = {"weight": mx.random.normal(model.weight.shape)}
            lookahead.update(model, grads, step)

        grads = {"weight": mx.random.normal(model.weight.shape)}
        lookahead.update(model, grads, k - 1)

        new_slow_weights = np.array(lookahead.slow_weights["weight"])
        model_weights = np.array(model.weight)

        np.testing.assert_allclose(
            model_weights, new_slow_weights,
            rtol=1e-6, atol=1e-7,
            err_msg="Model weights should equal new slow weights after k steps"
        )

        assert not np.allclose(new_slow_weights, slow_weights_np, rtol=1e-3), \
            "Slow weights should have changed after k steps"


# =============================================================================
# Edge Cases
# =============================================================================

class TestTrainingUtilsEdgeCases:
    """Edge case tests for training utilities."""

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_gradient_clipping_empty_grads(self, skip_without_jax):
        """Test gradient clipping with empty gradient dict."""
        grads = {}

        clipped, norm = clip_grad_norm(grads, max_norm=1.0)

        assert float(norm) == 0.0

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_gradient_clipping_small_norm(self, skip_without_jax):
        """Test gradient clipping when norm < max_norm."""
        grads = {
            "weight": mx.array([[0.1, 0.2], [0.3, 0.4]], dtype=mx.float32)
        }
        max_norm = 10.0

        clipped, norm = clip_grad_norm(grads, max_norm)

        np.testing.assert_allclose(
            np.array(clipped["weight"]),
            np.array(grads["weight"]),
            rtol=1e-7, atol=1e-8,
            err_msg="Gradients should not be clipped when norm < max_norm"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_swa_no_updates_before_start(self, skip_without_jax):
        """Test SWA does not average before swa_start."""
        model = nn.Linear(16, 8)
        swa = SWA(model, swa_start=10, swa_freq=1)

        for step in range(10):
            model.weight = mx.random.normal(model.weight.shape)
            swa.update(step=step)

        assert swa.n_averaged == 0, "SWA should not average before swa_start"
        assert swa.averaged_params is None, "averaged_params should be None before swa_start"
