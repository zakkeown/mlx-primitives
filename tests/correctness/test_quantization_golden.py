"""Golden file tests for quantization operations.

These tests compare MLX quantization implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category quantization

To run tests:
    pytest tests/correctness/test_quantization_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# Basic Quantization
# =============================================================================


class TestQuantizeDequantizeGolden:
    """Test basic quantize/dequantize against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "quant_int8_small",
            "quant_int8_medium",
            "quant_int8_large",
            "quant_int4_small",
            "quant_int4_medium",
        ],
    )
    def test_quantize_dequantize(self, config):
        """Quantize/dequantize matches PyTorch."""
        if not golden_exists("quantization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("quantization", config)

        x = mx.array(golden["x"])
        bits = golden["__metadata__"]["params"]["bits"]

        # Symmetric quantization
        qmax = 2 ** (bits - 1) - 1
        scale = mx.max(mx.abs(x)) / qmax

        # Quantize
        q = mx.clip(mx.round(x / scale), -qmax - 1, qmax)

        # Dequantize
        x_dequant = q * scale
        mx.eval(x_dequant)

        assert_close_golden(x_dequant, golden, "dequantized")

    @pytest.mark.parametrize("config", ["quant_uniform", "quant_sparse"])
    def test_quantize_distributions(self, config):
        """Quantization handles different distributions correctly."""
        if not golden_exists("quantization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("quantization", config)

        x = mx.array(golden["x"])
        bits = golden["__metadata__"]["params"]["bits"]

        qmax = 2 ** (bits - 1) - 1
        scale = mx.max(mx.abs(x)) / qmax
        q = mx.clip(mx.round(x / scale), -qmax - 1, qmax)
        x_dequant = q * scale
        mx.eval(x_dequant)

        assert_close_golden(x_dequant, golden, "dequantized")


# =============================================================================
# INT8 Linear
# =============================================================================


class TestInt8LinearGolden:
    """Test INT8 linear layer against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "int8_linear_small",
            "int8_linear_medium",
            "int8_linear_large",
            "int8_linear_square",
        ],
    )
    def test_int8_linear(self, config):
        """INT8 linear output matches PyTorch."""
        if not golden_exists("quantization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("quantization", config)

        x = mx.array(golden["x"])
        weight = mx.array(golden["weight"])

        # Full precision output
        fp_out = x @ weight.T
        mx.eval(fp_out)

        assert_close_golden(fp_out, golden, "fp_out")

        # INT8 quantized output (using stored scales)
        w_scales = mx.array(golden["expected_w_scales"])
        x_scales = mx.array(golden["expected_x_scales"])

        # Quantize weight (per-output-channel)
        w_quant = mx.clip(mx.round(weight / w_scales), -128, 127)
        w_dequant = w_quant * w_scales

        # Quantize activation (per-token)
        batch, seq, in_features = x.shape
        x_flat = x.reshape(-1, in_features)
        x_quant = mx.clip(mx.round(x_flat / x_scales.reshape(-1, 1)), -128, 127)
        x_dequant = x_quant * x_scales.reshape(-1, 1)
        x_dequant = x_dequant.reshape(batch, seq, in_features)

        int8_out = x_dequant @ w_dequant.T
        mx.eval(int8_out)

        assert_close_golden(int8_out, golden, "int8_out")


# =============================================================================
# INT4 Linear
# =============================================================================


class TestInt4LinearGolden:
    """Test INT4 linear layer against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "int4_linear_g128",
            "int4_linear_g64",
            "int4_linear_g32",
            "int4_linear_large",
        ],
    )
    def test_int4_linear(self, config):
        """INT4 linear output matches PyTorch."""
        if not golden_exists("quantization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("quantization", config)

        x = mx.array(golden["x"])
        weight = mx.array(golden["weight"])
        group_size = golden["__metadata__"]["params"]["group_size"]

        # Full precision output
        fp_out = x @ weight.T
        mx.eval(fp_out)

        assert_close_golden(fp_out, golden, "fp_out")

        # INT4 quantized output
        out_features, in_features = weight.shape
        num_groups = in_features // group_size

        weight_grouped = weight.reshape(out_features, num_groups, group_size)
        scales = mx.max(mx.abs(weight_grouped), axis=2, keepdims=True) / 7

        w_quant = mx.clip(mx.round(weight_grouped / scales), -8, 7)
        w_dequant = (w_quant * scales).reshape(out_features, in_features)

        int4_out = x @ w_dequant.T
        mx.eval(int4_out)

        assert_close_golden(int4_out, golden, "int4_out")


# =============================================================================
# QLoRA Linear
# =============================================================================


class TestQLoRALinearGolden:
    """Test QLoRA linear layer against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "qlora_r8",
            "qlora_r16",
            "qlora_r32",
            "qlora_small_rank",
        ],
    )
    def test_qlora_linear(self, config):
        """QLoRA linear output matches PyTorch."""
        if not golden_exists("quantization", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("quantization", config)

        x = mx.array(golden["x"])
        base_weight = mx.array(golden["base_weight"])
        lora_A = mx.array(golden["lora_A"])
        lora_B = mx.array(golden["lora_B"])
        rank = golden["__metadata__"]["params"]["rank"]
        alpha = golden["__metadata__"]["params"]["alpha"]
        group_size = golden["__metadata__"]["params"]["group_size"]
        scaling = alpha / rank

        # Quantize base weight
        out_features, in_features = base_weight.shape
        num_groups = in_features // group_size

        weight_grouped = base_weight.reshape(out_features, num_groups, group_size)
        scales = mx.max(mx.abs(weight_grouped), axis=2, keepdims=True) / 7
        w_quant = mx.clip(mx.round(weight_grouped / scales), -8, 7)
        w_dequant = (w_quant * scales).reshape(out_features, in_features)

        # QLoRA output: quantized base + LoRA
        qlora_out = x @ w_dequant.T + scaling * (x @ lora_A.T @ lora_B.T)
        mx.eval(qlora_out)

        assert_close_golden(qlora_out, golden, "qlora_out")
