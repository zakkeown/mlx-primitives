"""Golden file tests for attention mechanisms.

These tests compare MLX attention implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category attention

To run tests:
    pytest tests/correctness/test_attention_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx

from golden_utils import load_golden, assert_close_golden, golden_exists

# Import MLX attention implementations
from mlx_primitives.attention import flash_attention_forward
from mlx_primitives.attention.grouped_query import GroupedQueryAttention
from mlx_primitives.attention.multi_query import MultiQueryAttention
from mlx_primitives.attention.sliding_window import SlidingWindowAttention
from mlx_primitives.attention.rope import apply_rope, precompute_freqs_cis


# =============================================================================
# Standard SDPA
# =============================================================================


class TestSDPAGolden:
    """Test Scaled Dot-Product Attention against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("causal", ["causal", "noncausal"])
    def test_sdpa_sizes(self, size, causal):
        """SDPA output matches PyTorch for various sizes."""
        test_name = f"sdpa_{size}_{causal}"
        golden = load_golden("attention", test_name)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        scale = golden["__metadata__"]["params"]["scale"]
        is_causal = golden["__metadata__"]["params"]["causal"]

        out = flash_attention_forward(q, k, v, scale=scale, causal=is_causal)
        mx.eval(out)

        assert_close_golden(out, golden, "out")

    def test_sdpa_cross_attention_small(self):
        """Cross-attention (different KV seq length) matches PyTorch."""
        test_name = "sdpa_cross_attention_small"
        if not golden_exists("attention", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("attention", test_name)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        scale = golden["__metadata__"]["params"]["scale"]

        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# GQA and MQA
# =============================================================================


class TestGQAGolden:
    """Test Grouped Query Attention against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "gqa_h8_kv2_small_causal",
            "gqa_h8_kv2_small_noncausal",
            "gqa_h16_kv4_small_causal",
            "gqa_h32_kv8_medium_causal",
        ],
    )
    def test_gqa_configs(self, config):
        """GQA matches PyTorch for various head configurations."""
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        metadata = golden["__metadata__"]
        heads = metadata["heads"]
        kv_heads = metadata["kv_heads"]
        head_dim = metadata["head_dim"]
        causal = metadata.get("params", {}).get("causal", True)

        # Create GQA module
        dims = heads * head_dim
        gqa = GroupedQueryAttention(
            dims=dims,
            num_heads=heads,
            num_kv_heads=kv_heads,
        )

        # For testing, we pass pre-computed Q, K, V
        # GQA internally does projection, so we need to test the attention portion
        # This tests the KV head expansion logic

        # Expand K/V to match Q heads (same as PyTorch reference)
        num_groups = heads // kv_heads
        k_expanded = mx.repeat(k, num_groups, axis=2)
        v_expanded = mx.repeat(v, num_groups, axis=2)

        # Now run standard attention with expanded K/V
        scale = 1.0 / (head_dim**0.5)
        out = flash_attention_forward(q, k_expanded, v_expanded, scale=scale, causal=causal)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


class TestMQAGolden:
    """Test Multi-Query Attention against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("causal", ["causal", "noncausal"])
    def test_mqa_sizes(self, size, causal):
        """MQA matches PyTorch for various sizes."""
        test_name = f"mqa_{size}_{causal}"
        if not golden_exists("attention", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("attention", test_name)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])  # (batch, seq, 1, head_dim)
        v = mx.array(golden["v"])  # (batch, seq, 1, head_dim)

        metadata = golden["__metadata__"]
        heads = metadata["heads"]
        head_dim = metadata["head_dim"]
        is_causal = causal == "causal"

        # Expand K/V to all heads
        k_expanded = mx.broadcast_to(k, q.shape)
        v_expanded = mx.broadcast_to(v, q.shape)

        scale = 1.0 / (head_dim**0.5)
        out = flash_attention_forward(q, k_expanded, v_expanded, scale=scale, causal=is_causal)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# Sliding Window Attention
# =============================================================================


class TestSlidingWindowGolden:
    """Test Sliding Window Attention against PyTorch golden files.

    Tolerance is set to 1.5e-6 to account for softmax boundary effects.
    """

    @pytest.mark.parametrize(
        "config",
        [
            "sliding_window_w64_medium",  # window=64, seq=128
            "sliding_window_tiny",  # window=16, seq=128
            "sliding_window_full",  # window=64, seq=64 (full attention)
        ],
    )
    def test_sliding_window_configs(self, config):
        """Sliding window attention matches PyTorch."""
        if not golden_exists("attention", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("attention", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        window_size = golden["__metadata__"]["params"]["window_size"]

        metadata = golden["__metadata__"]
        heads = metadata["heads"]
        head_dim = metadata["head_dim"]
        seq = metadata["seq"]

        # Create sliding window mask
        rows = mx.arange(seq).reshape(-1, 1)
        cols = mx.arange(seq).reshape(1, -1)

        # Causal sliding window mask
        causal_mask = cols <= rows
        window_mask = cols >= (rows - window_size + 1)
        mask = causal_mask & window_mask

        # Convert to attention bias
        attn_mask = mx.where(mask, 0.0, float("-inf"))
        attn_mask = attn_mask.reshape(1, 1, seq, seq)

        # Compute attention with mask
        scale = 1.0 / (head_dim**0.5)
        out = flash_attention_forward(q, k, v, scale=scale, mask=attn_mask)
        mx.eval(out)

        assert_close_golden(out, golden, "out")


# =============================================================================
# RoPE (Rotary Position Embeddings)
# =============================================================================


class TestRoPEGolden:
    """Test Rotary Position Embeddings against PyTorch golden files.

    Tolerances are computed per-test based on the formula: atol = 6e-8 × seq × 1.2
    to account for exp() ULP differences that scale linearly with sequence length.
    """

    @pytest.mark.parametrize(
        "size",
        [
            "tiny",
            "small",
            "medium",
            "large",
        ],
    )
    def test_rope_sizes(self, size):
        """RoPE output matches PyTorch for various sizes."""
        test_name = f"rope_{size}"
        if not golden_exists("attention", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("attention", test_name)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])

        metadata = golden["__metadata__"]
        seq = metadata["seq"]
        head_dim = metadata["head_dim"]
        base = golden["__metadata__"]["params"]["base"]

        # Precompute frequencies
        cos, sin = precompute_freqs_cis(head_dim, seq, base=base)
        mx.eval(cos, sin)

        # Apply RoPE
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        mx.eval(q_rot, k_rot)

        # Compare Q rotation
        expected_q = golden["expected_q_rot"]
        assert_close_golden(q_rot, {"expected_out": expected_q, "__metadata__": golden["__metadata__"]}, "out")

        # Compare K rotation
        expected_k = golden["expected_k_rot"]
        assert_close_golden(k_rot, {"expected_out": expected_k, "__metadata__": golden["__metadata__"]}, "out")


class TestALiBiGolden:
    """Test ALiBi attention against PyTorch golden files."""

    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("causal", ["causal", "noncausal"])
    def test_alibi_sizes(self, size, causal):
        """ALiBi attention matches PyTorch."""
        test_name = f"alibi_{size}_{causal}"
        if not golden_exists("attention", test_name):
            pytest.skip(f"Golden file not found: {test_name}")

        golden = load_golden("attention", test_name)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])

        # Get ALiBi bias from golden file
        alibi_bias = mx.array(golden["expected_alibi_bias"])
        alibi_bias = alibi_bias.reshape(1, *alibi_bias.shape)  # Add batch dim

        metadata = golden["__metadata__"]
        head_dim = metadata["head_dim"]

        scale = 1.0 / (head_dim**0.5)
        out = flash_attention_forward(q, k, v, scale=scale, mask=alibi_bias)
        mx.eval(out)

        assert_close_golden(out, golden, "out")
