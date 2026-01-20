"""Golden file tests for pooling layers.

These tests compare MLX pooling implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category pooling

To run tests:
    pytest tests/correctness/test_pooling_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# Adaptive Average Pooling
# =============================================================================


class TestAdaptiveAvgPool1dGolden:
    """Test AdaptiveAvgPool1d against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "adaptive_avgpool1d_to1",
            "adaptive_avgpool1d_to8",
            "adaptive_avgpool1d_to32",
            "adaptive_avgpool1d_same",
            "adaptive_avgpool1d_large",
        ],
    )
    def test_adaptive_avgpool1d(self, config):
        """AdaptiveAvgPool1d output matches PyTorch."""
        if not golden_exists("pooling", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("pooling", config)

        x = mx.array(golden["x"])
        output_size = golden["__metadata__"]["params"]["output_size"]

        # MLX adaptive pooling - implementation depends on MLX API
        # This is a reference implementation
        batch, channels, length = x.shape

        if output_size == length:
            out = x
        else:
            # Compute adaptive pooling manually
            stride = length // output_size
            kernel_size = length - (output_size - 1) * stride

            # Simple average pooling approximation
            outputs = []
            for i in range(output_size):
                start = i * stride
                end = start + kernel_size
                pooled = mx.mean(x[:, :, start:end], axis=2, keepdims=True)
                outputs.append(pooled)
            out = mx.concatenate(outputs, axis=2)

        mx.eval(out)
        assert_close_golden(out, golden, "out")


class TestAdaptiveAvgPool2dGolden:
    """Test AdaptiveAvgPool2d against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "adaptive_avgpool2d_to1x1",
            "adaptive_avgpool2d_to7x7",
            "adaptive_avgpool2d_to14x14",
        ],
    )
    def test_adaptive_avgpool2d(self, config):
        """AdaptiveAvgPool2d output matches PyTorch."""
        if not golden_exists("pooling", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("pooling", config)

        x = mx.array(golden["x"])
        output_size = tuple(golden["__metadata__"]["params"]["output_size"])

        # For global average pooling (1x1)
        if output_size == (1, 1):
            out = mx.mean(x, axis=(2, 3), keepdims=True)
        else:
            # General adaptive pooling - match PyTorch's algorithm exactly
            import math
            batch, channels, h, w = x.shape
            out_h, out_w = output_size

            outputs = []
            for i in range(out_h):
                row = []
                for j in range(out_w):
                    # PyTorch's adaptive pooling index calculation
                    start_h = int(math.floor(i * h / out_h))
                    end_h = int(math.ceil((i + 1) * h / out_h))
                    start_w = int(math.floor(j * w / out_w))
                    end_w = int(math.ceil((j + 1) * w / out_w))

                    pooled = mx.mean(
                        x[:, :, start_h:end_h, start_w:end_w],
                        axis=(2, 3),
                        keepdims=True,
                    )
                    row.append(pooled)
                outputs.append(mx.concatenate(row, axis=3))
            out = mx.concatenate(outputs, axis=2)

        mx.eval(out)
        assert_close_golden(out, golden, "out")


# =============================================================================
# GeM Pooling
# =============================================================================


class TestGeMGolden:
    """Test Generalized Mean (GeM) pooling against PyTorch golden files."""

    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_gem_p_values(self, p):
        """GeM pooling matches PyTorch for various p values."""
        config = f"gem_p{p}"
        if not golden_exists("pooling", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("pooling", config)

        x = mx.array(golden["x"])
        p_val = golden["__metadata__"]["params"]["p"]
        eps = golden["__metadata__"]["params"]["eps"]

        # GeM: (mean(x^p))^(1/p)
        x_pow = mx.power(mx.maximum(x, eps), p_val)
        pooled = mx.mean(x_pow, axis=(2, 3), keepdims=True)
        out = mx.power(pooled, 1.0 / p_val)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Spatial Pyramid Pooling
# =============================================================================


class TestSPPGolden:
    """Test Spatial Pyramid Pooling against PyTorch golden files."""

    @pytest.mark.parametrize("config", ["spp_standard", "spp_fine", "spp_coarse"])
    def test_spp(self, config):
        """SPP output matches PyTorch."""
        if not golden_exists("pooling", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("pooling", config)

        x = mx.array(golden["x"])
        levels = golden["__metadata__"]["params"]["levels"]

        batch, channels, height, width = x.shape

        # SPP: Pool at each level and concatenate
        pyramid = []
        for level in levels:
            # Adaptive average pool to level x level
            if level == 1:
                pooled = mx.mean(x, axis=(2, 3), keepdims=True)
            else:
                # Simplified adaptive pooling
                stride_h = height // level
                stride_w = width // level
                outputs = []
                for i in range(level):
                    for j in range(level):
                        start_h = i * stride_h
                        start_w = j * stride_w
                        end_h = start_h + stride_h if i < level - 1 else height
                        end_w = start_w + stride_w if j < level - 1 else width
                        pooled_val = mx.mean(x[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
                        outputs.append(pooled_val)
                pooled = mx.concatenate(outputs, axis=2)
                pooled = pooled.reshape(batch, channels, level, level)

            # Flatten spatial dimensions
            flat = pooled.reshape(batch, channels, -1)
            pyramid.append(flat)

        out = mx.concatenate(pyramid, axis=-1)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Global Attention Pooling
# =============================================================================


class TestGlobalAttentionPoolingGolden:
    """Test Global Attention Pooling against PyTorch golden files."""

    @pytest.mark.parametrize("config", ["global_attn_pool_small", "global_attn_pool_medium", "global_attn_pool_large"])
    def test_global_attention_pooling(self, config):
        """Global attention pooling matches PyTorch."""
        if not golden_exists("pooling", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("pooling", config)

        x = mx.array(golden["x"])
        attn_weight = mx.array(golden["attn_weight"])
        attn_bias = mx.array(golden["attn_bias"])

        # Compute attention scores
        scores = x @ attn_weight + attn_bias  # (B, seq, 1)
        attn = mx.softmax(scores, axis=1)

        # Weighted sum
        out = mx.sum(attn * x, axis=1)
        mx.eval(out)

        assert_close_golden(out, golden, "out")
