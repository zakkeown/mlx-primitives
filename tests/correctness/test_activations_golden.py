"""Golden file tests for activation functions.

These tests compare MLX activation implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category activations

To run tests:
    pytest tests/correctness/test_activations_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# GLU Variants
# =============================================================================


class TestSwiGLUGolden:
    """Test SwiGLU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_swiglu_sizes(self, size):
        """SwiGLU output matches PyTorch for various sizes."""
        golden = load_golden("activations", f"swiglu_{size}")

        x = mx.array(golden["x"])

        # Split and apply SwiGLU: silu(x1) * x2
        dims = golden["__metadata__"]["params"]["dims"]
        x1 = x[..., :dims]
        x2 = x[..., dims:]

        out = nn.silu(x1) * x2
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    @pytest.mark.parametrize("edge_case", ["zeros", "negative", "large"])
    def test_swiglu_edge_cases(self, edge_case):
        """SwiGLU handles edge cases correctly."""
        test_name = f"swiglu_{edge_case}"
        if not golden_exists("activations", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("activations", test_name)

        x = mx.array(golden["x"])
        dims = golden["__metadata__"]["params"]["dims"]
        x1 = x[..., :dims]
        x2 = x[..., dims:]

        out = nn.silu(x1) * x2
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestGeGLUGolden:
    """Test GeGLU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_geglu_sizes(self, size):
        """GeGLU output matches PyTorch for various sizes."""
        golden = load_golden("activations", f"geglu_{size}")

        x = mx.array(golden["x"])
        dims = golden["__metadata__"]["params"]["dims"]
        x1 = x[..., :dims]
        x2 = x[..., dims:]

        out = nn.gelu(x1) * x2
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestReGLUGolden:
    """Test ReGLU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_reglu_sizes(self, size):
        """ReGLU output matches PyTorch for various sizes."""
        golden = load_golden("activations", f"reglu_{size}")

        x = mx.array(golden["x"])
        dims = golden["__metadata__"]["params"]["dims"]
        x1 = x[..., :dims]
        x2 = x[..., dims:]

        out = nn.relu(x1) * x2
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# GELU Variants
# =============================================================================


class TestGELUGolden:
    """Test GELU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_gelu_sizes(self, size):
        """GELU output matches PyTorch for various sizes."""
        golden = load_golden("activations", f"gelu_{size}")

        x = mx.array(golden["x"])
        out = nn.gelu(x)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    @pytest.mark.parametrize("edge_case", ["zeros", "negative", "large_positive", "large_negative"])
    def test_gelu_edge_cases(self, edge_case):
        """GELU handles edge cases correctly."""
        test_name = f"gelu_{edge_case}"
        if not golden_exists("activations", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("activations", test_name)
        x = mx.array(golden["x"])
        out = nn.gelu(x)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestGELUTanhGolden:
    """Test GELU with tanh approximation against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_gelu_tanh_sizes(self, size):
        """GELU tanh approximation matches PyTorch."""
        golden = load_golden("activations", f"gelu_tanh_{size}")

        x = mx.array(golden["x"])
        out = nn.gelu_approx(x)  # MLX's approximate GELU
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestQuickGELUGolden:
    """Test QuickGELU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_quick_gelu_sizes(self, size):
        """QuickGELU matches PyTorch reference."""
        golden = load_golden("activations", f"quick_gelu_{size}")

        x = mx.array(golden["x"])
        # QuickGELU: x * sigmoid(1.702 * x)
        out = x * mx.sigmoid(1.702 * x)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Miscellaneous Activations
# =============================================================================


class TestMishGolden:
    """Test Mish activation against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_mish_sizes(self, size):
        """Mish output matches PyTorch."""
        golden = load_golden("activations", f"mish_{size}")

        x = mx.array(golden["x"])
        # Mish: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
        softplus_x = mx.log(1 + mx.exp(x))
        out = x * mx.tanh(softplus_x)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestSquaredReLUGolden:
    """Test SquaredReLU against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_squared_relu_sizes(self, size):
        """SquaredReLU matches PyTorch."""
        golden = load_golden("activations", f"squared_relu_{size}")

        x = mx.array(golden["x"])
        out = nn.relu(x) ** 2
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestSwishGolden:
    """Test Swish activation against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "swish_beta0.5_small",
            "swish_beta1.0_small",
            "swish_beta1.5_small",
            "swish_beta2.0_small",
        ],
    )
    def test_swish_betas(self, config):
        """Swish with various beta values matches PyTorch."""
        if not golden_exists("activations", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("activations", config)
        x = mx.array(golden["x"])
        beta = golden["__metadata__"]["params"]["beta"]

        # Swish: x * sigmoid(beta * x)
        out = x * mx.sigmoid(beta * x)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestHardSwishGolden:
    """Test HardSwish against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_hard_swish_sizes(self, size):
        """HardSwish matches PyTorch."""
        golden = load_golden("activations", f"hard_swish_{size}")

        x = mx.array(golden["x"])
        # HardSwish: x * relu6(x + 3) / 6
        out = x * mx.minimum(mx.maximum(x + 3, 0), 6) / 6
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestHardSigmoidGolden:
    """Test HardSigmoid against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_hard_sigmoid_sizes(self, size):
        """HardSigmoid matches PyTorch."""
        golden = load_golden("activations", f"hard_sigmoid_{size}")

        x = mx.array(golden["x"])
        # HardSigmoid: relu6(x + 3) / 6
        out = mx.minimum(mx.maximum(x + 3, 0), 6) / 6
        mx.eval(out)

        assert_close_golden(out, golden, "out")
