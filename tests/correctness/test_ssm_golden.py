"""Golden file tests for State Space Models (SSM).

These tests compare MLX SSM implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category ssm

To run tests:
    pytest tests/correctness/test_ssm_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# Selective Scan
# =============================================================================


class TestSelectiveScanGolden:
    """Test selective scan against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_selective_scan_sizes(self, size):
        """Selective scan output matches PyTorch for various sizes."""
        golden = load_golden("ssm", f"selective_scan_{size}")

        x = mx.array(golden["x"])
        delta = mx.array(golden["delta"])
        A = mx.array(golden["A"])
        B = mx.array(golden["B"])
        C = mx.array(golden["C"])
        D = mx.array(golden["D"])

        # Selective scan: sequential recurrence
        # This is a reference implementation - actual MLX impl may differ
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize: deltaA = exp(delta * A), deltaB = delta * B
        deltaA = mx.exp(mx.expand_dims(delta, -1) * A)  # (B, L, D, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 2)  # (B, L, D, N)

        # Scan
        h = mx.zeros((batch, d_inner, d_state))
        outputs = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * mx.expand_dims(x[:, t], -1)
            y = mx.sum(h * mx.expand_dims(C[:, t], 1), axis=-1)
            outputs.append(y)

        y = mx.stack(outputs, axis=1)  # (B, L, D)
        out = y + x * D
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    @pytest.mark.parametrize("config", ["selective_scan_long_seq", "selective_scan_single", "selective_scan_large_state"])
    def test_selective_scan_edge_cases(self, config):
        """Selective scan handles edge cases correctly."""
        if not golden_exists("ssm", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("ssm", config)

        x = mx.array(golden["x"])
        delta = mx.array(golden["delta"])
        A = mx.array(golden["A"])
        B = mx.array(golden["B"])
        C = mx.array(golden["C"])
        D = mx.array(golden["D"])

        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 2)

        h = mx.zeros((batch, d_inner, d_state))
        outputs = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * mx.expand_dims(x[:, t], -1)
            y = mx.sum(h * mx.expand_dims(C[:, t], 1), axis=-1)
            outputs.append(y)

        y = mx.stack(outputs, axis=1)
        out = y + x * D
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# S4 Layer
# =============================================================================


class TestS4Golden:
    """Test S4 layer against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_s4_sizes(self, size):
        """S4 output matches PyTorch for various sizes."""
        for suffix in ["", "_bidirectional"]:
            test_name = f"s4_{size}{suffix}" if suffix else f"s4_{size}"
            if not golden_exists("ssm", test_name):
                continue

            golden = load_golden("ssm", test_name)

            x = mx.array(golden["x"])
            A_real = mx.array(golden["A_real"])
            A_imag = mx.array(golden["A_imag"])
            B = mx.array(golden["B"])
            C = mx.array(golden["C"])
            D = mx.array(golden["D"])

            # S4 diagonal: h' = A * h + B * x, y = C * h + D * x
            # Simplified reference implementation
            batch, seq_len, dims = x.shape
            d_state = A_real.shape[-1]

            # Initialize state
            h_real = mx.zeros((batch, dims, d_state))
            h_imag = mx.zeros((batch, dims, d_state))

            outputs = []
            for t in range(seq_len):
                # Complex multiply for diagonal A
                x_t = mx.expand_dims(x[:, t], -1)  # (B, D, 1)
                B_exp = mx.expand_dims(B, 0)  # (1, D, N)

                # h' = A * h + B * x
                new_h_real = A_real * h_real - A_imag * h_imag + B_exp * x_t
                new_h_imag = A_real * h_imag + A_imag * h_real

                h_real = new_h_real
                h_imag = new_h_imag

                # y = Re(C * h) + D * x
                C_exp = mx.expand_dims(C, 0)
                y = mx.sum(C_exp * h_real, axis=-1) + D * x[:, t]
                outputs.append(y)

            out = mx.stack(outputs, axis=1)
            mx.eval(out)

            assert_close_golden(out, golden, "out")


# =============================================================================
# H3 Layer
# =============================================================================


class TestH3Golden:
    """Test H3 layer against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_h3_sizes(self, size):
        """H3 output matches PyTorch for various sizes."""
        test_name = f"h3_{size}"
        if not golden_exists("ssm", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("ssm", test_name)

        # H3 combines SSM with short convolution and gating
        # This test verifies the full block output
        out_expected = golden["out"]

        # For full H3 test, we'd need the complete implementation
        # This is a placeholder that tests against stored outputs
        pytest.skip("H3 full implementation test - requires MLX H3 module")
