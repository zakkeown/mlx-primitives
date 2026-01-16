"""Golden file tests for normalization layers.

These tests compare MLX normalization implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category normalization

To run tests:
    pytest tests/correctness/test_normalization_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# RMSNorm
# =============================================================================


class TestRMSNormGolden:
    """Test RMSNorm against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_rmsnorm_sizes(self, size):
        """RMSNorm output matches PyTorch for various sizes."""
        golden = load_golden("normalization", f"rmsnorm_{size}")

        x = mx.array(golden["x"])
        weight = mx.array(golden["weight"])
        eps = golden["__metadata__"]["params"]["eps"]

        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
        out = x / rms * weight
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    @pytest.mark.parametrize("edge_case", ["zeros", "small_values", "large_values"])
    def test_rmsnorm_edge_cases(self, edge_case):
        """RMSNorm handles edge cases correctly."""
        test_name = f"rmsnorm_{edge_case}"
        if not golden_exists("normalization", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("normalization", test_name)

        x = mx.array(golden["x"])
        weight = mx.array(golden["weight"])
        eps = golden["__metadata__"]["params"]["eps"]

        rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
        out = x / rms * weight
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# GroupNorm
# =============================================================================


class TestGroupNormGolden:
    """Test GroupNorm against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "groupnorm_c32_g8_2d",
            "groupnorm_c64_g8_2d",
            "groupnorm_c128_g16_2d",
            "groupnorm_c256_g32_2d",
        ],
    )
    def test_groupnorm_2d(self, config):
        """GroupNorm 2D matches PyTorch."""
        if not golden_exists("normalization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("normalization", config)

        x = mx.array(golden["x"])  # NCHW format from PyTorch
        num_groups = golden["__metadata__"]["params"]["num_groups"]
        eps = golden["__metadata__"]["params"]["eps"]

        # Manual GroupNorm implementation matching PyTorch exactly
        # PyTorch GroupNorm: normalize over (C/G, H, W) for each group
        batch, C, H, W = x.shape
        channels_per_group = C // num_groups

        # Reshape to (N, G, C/G, H, W)
        x_grouped = x.reshape(batch, num_groups, channels_per_group, H, W)

        # Compute mean and var over (C/G, H, W) dimensions
        mean = mx.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
        var = mx.var(x_grouped, axis=(2, 3, 4), keepdims=True)

        # Normalize
        x_norm = (x_grouped - mean) / mx.sqrt(var + eps)

        # Reshape back to (N, C, H, W)
        out = x_norm.reshape(batch, C, H, W)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    @pytest.mark.parametrize(
        "config",
        [
            "groupnorm_c32_g8_1d",
            "groupnorm_c64_g8_1d",
            "groupnorm_c128_g16_1d",
            "groupnorm_c256_g32_1d",
        ],
    )
    def test_groupnorm_1d(self, config):
        """GroupNorm 1D matches PyTorch."""
        if not golden_exists("normalization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("normalization", config)

        x = mx.array(golden["x"])  # NCL format from PyTorch
        num_groups = golden["__metadata__"]["params"]["num_groups"]
        eps = golden["__metadata__"]["params"]["eps"]

        # Manual GroupNorm implementation matching PyTorch exactly
        # PyTorch GroupNorm: normalize over (C/G, L) for each group
        batch, C, L = x.shape
        channels_per_group = C // num_groups

        # Reshape to (N, G, C/G, L)
        x_grouped = x.reshape(batch, num_groups, channels_per_group, L)

        # Compute mean and var over (C/G, L) dimensions
        mean = mx.mean(x_grouped, axis=(2, 3), keepdims=True)
        var = mx.var(x_grouped, axis=(2, 3), keepdims=True)

        # Normalize
        x_norm = (x_grouped - mean) / mx.sqrt(var + eps)

        # Reshape back to (N, C, L)
        out = x_norm.reshape(batch, C, L)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# InstanceNorm
# =============================================================================


class TestInstanceNormGolden:
    """Test InstanceNorm against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_instancenorm_2d(self, size):
        """InstanceNorm 2D matches PyTorch."""
        for suffix in ["_no_affine", "_affine"]:
            test_name = f"instancenorm_{size}{suffix}"
            if not golden_exists("normalization", test_name):
                continue

            golden = load_golden("normalization", test_name)

            x = mx.array(golden["x"])
            eps = golden["__metadata__"]["params"]["eps"]
            affine = golden["__metadata__"]["params"]["affine"]

            # InstanceNorm: normalize over spatial dims
            # For 2D: (N, C, H, W) -> normalize over H, W
            mean = mx.mean(x, axis=(2, 3), keepdims=True)
            var = mx.var(x, axis=(2, 3), keepdims=True)
            out = (x - mean) / mx.sqrt(var + eps)

            if affine and "weight" in golden:
                weight = mx.array(golden["weight"]).reshape(1, -1, 1, 1)
                bias = mx.array(golden["bias"]).reshape(1, -1, 1, 1)
                out = out * weight + bias

            mx.eval(out)
            assert_close_golden(out, golden, "out")


# =============================================================================
# QKNorm
# =============================================================================


class TestQKNormGolden:
    """Test QKNorm against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_qknorm_sizes(self, size):
        """QKNorm output matches PyTorch."""
        golden = load_golden("normalization", f"qknorm_{size}")

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        q_scale = mx.array(golden["q_scale"])
        k_scale = mx.array(golden["k_scale"])
        eps = golden["__metadata__"]["params"]["eps"]

        # QKNorm: RMSNorm on last dimension
        def rms_norm(x, scale, eps):
            rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
            return (x / rms) * scale

        q_norm = rms_norm(q, q_scale, eps)
        k_norm = rms_norm(k, k_scale, eps)
        mx.eval(q_norm, k_norm)

        assert_close_golden(q_norm, golden, "q_norm")
        assert_close_golden(k_norm, golden, "k_norm")

    @pytest.mark.parametrize("config", ["qknorm_single_head", "qknorm_many_heads"])
    def test_qknorm_edge_cases(self, config):
        """QKNorm handles edge cases correctly."""
        if not golden_exists("normalization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("normalization", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        q_scale = mx.array(golden["q_scale"])
        k_scale = mx.array(golden["k_scale"])
        eps = golden["__metadata__"]["params"]["eps"]

        def rms_norm(x, scale, eps):
            rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
            return (x / rms) * scale

        q_norm = rms_norm(q, q_scale, eps)
        k_norm = rms_norm(k, k_scale, eps)
        mx.eval(q_norm, k_norm)

        assert_close_golden(q_norm, golden, "q_norm")
        assert_close_golden(k_norm, golden, "k_norm")
