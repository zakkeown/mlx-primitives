"""Golden file tests for embedding layers.

These tests compare MLX embedding implementations against
PyTorch reference outputs stored in golden files.

To generate golden files:
    python scripts/validation/generate_all.py --category embeddings

To run tests:
    pytest tests/correctness/test_embeddings_golden.py -v
"""

import pytest
import numpy as np
import mlx.core as mx

from golden_utils import load_golden, assert_close_golden, golden_exists


# =============================================================================
# Sinusoidal Embeddings
# =============================================================================


class TestSinusoidalEmbeddingGolden:
    """Test sinusoidal positional embeddings against PyTorch golden files.

    Tolerances are computed per-test based on the formula: atol = 6e-8 × max_len × 1.2
    to account for exp() ULP differences that scale linearly with position.
    """

    @pytest.mark.parametrize(
        "config",
        [
            "sinusoidal_emb_small",
            "sinusoidal_emb_medium",
            "sinusoidal_emb_large",
            "sinusoidal_emb_bert",
            "sinusoidal_emb_gpt2",
        ],
    )
    def test_sinusoidal_embedding(self, config):
        """Sinusoidal embedding matches PyTorch."""
        if not golden_exists("embeddings", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("embeddings", config)

        max_len = golden["__metadata__"]["params"]["max_len"]
        dims = golden["__metadata__"]["params"]["dims"]

        # Create sinusoidal position encodings
        position = mx.arange(max_len).reshape(-1, 1).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, dims, 2).astype(mx.float32) * (-np.log(10000.0) / dims)
        )

        pe = mx.zeros((max_len, dims))
        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)

        # Interleave sin and cos
        pe = mx.zeros((max_len, dims))
        indices_even = mx.arange(0, dims, 2)
        indices_odd = mx.arange(1, dims, 2)

        # Manual interleaving since MLX may not have scatter
        out = mx.concatenate([pe_sin, pe_cos], axis=1)
        # Reorder columns to interleave
        reorder = []
        for i in range(dims // 2):
            reorder.extend([i, i + dims // 2])
        out = out[:, reorder]

        mx.eval(out)
        assert_close_golden(out, golden, "embeddings")


# =============================================================================
# Learned Positional Embeddings
# =============================================================================


class TestLearnedPositionalEmbeddingGolden:
    """Test learned positional embeddings against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "learned_pos_emb_small",
            "learned_pos_emb_medium",
            "learned_pos_emb_large",
            "learned_pos_emb_offset",
        ],
    )
    def test_learned_positional_embedding(self, config):
        """Learned positional embedding matches PyTorch."""
        if not golden_exists("embeddings", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("embeddings", config)

        positions = mx.array(golden["positions"])
        weight = mx.array(golden["weight"])

        # Simple embedding lookup
        out = weight[positions]
        mx.eval(out)

        assert_close_golden(out, golden, "embeddings")


# =============================================================================
# Rotary Embeddings
# =============================================================================


class TestRotaryEmbeddingGolden:
    """Test rotary positional embeddings against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "rope_small",
            "rope_medium",
            "rope_long",
            "rope_ntk_scaled",
        ],
    )
    def test_rotary_embedding(self, config):
        """Rotary embedding matches PyTorch."""
        if not golden_exists("embeddings", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("embeddings", config)

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        cos = mx.array(golden["cos"])
        sin = mx.array(golden["sin"])

        # Apply rotary embedding
        def apply_rope(x, cos, sin):
            x1, x2 = x[..., ::2], x[..., 1::2]
            x_rotated = mx.stack([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ], axis=-1)
            return x_rotated.reshape(x.shape)

        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)
        mx.eval(q_rope, k_rope)

        assert_close_golden(q_rope, golden, "q_rope")
        assert_close_golden(k_rope, golden, "k_rope")


# =============================================================================
# ALiBi Embeddings
# =============================================================================


class TestALiBiEmbeddingGolden:
    """Test ALiBi embeddings against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "alibi_small",
            "alibi_medium",
            "alibi_large",
            "alibi_many_heads",
            "alibi_single_head",
        ],
    )
    def test_alibi_embedding(self, config):
        """ALiBi embedding matches PyTorch."""
        if not golden_exists("embeddings", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("embeddings", config)

        slopes = mx.array(golden["slopes"])
        seq = golden["__metadata__"]["params"]["seq"]
        heads = golden["__metadata__"]["params"]["heads"]

        # Create position difference matrix
        positions = mx.arange(seq).astype(mx.float32)
        pos_diff = positions.reshape(1, -1) - positions.reshape(-1, 1)

        # Compute biases: -slope * |i - j|
        alibi_bias = -slopes.reshape(-1, 1, 1) * mx.abs(pos_diff).reshape(1, seq, seq)
        mx.eval(alibi_bias)

        assert_close_golden(alibi_bias, golden, "alibi_bias")


# =============================================================================
# Relative Positional Embeddings
# =============================================================================


class TestRelativePositionalEmbeddingGolden:
    """Test relative positional embeddings against PyTorch golden files."""

    @pytest.mark.parametrize(
        "config",
        [
            "rel_pos_small",
            "rel_pos_medium",
            "rel_pos_large",
            "rel_pos_bidirectional",
        ],
    )
    def test_relative_positional_embedding(self, config):
        """Relative positional embedding matches PyTorch."""
        if not golden_exists("embeddings", config):
            pytest.skip(f"Golden file not found: {config}")

        golden = load_golden("embeddings", config)

        bucket_indices = mx.array(golden["bucket_indices"])
        weight = mx.array(golden["weight"])

        # Look up embeddings from bucket indices
        rel_pos_bias = weight[bucket_indices]  # (seq, seq, heads)
        rel_pos_bias = mx.transpose(rel_pos_bias, axes=(2, 0, 1))  # (heads, seq, seq)
        mx.eval(rel_pos_bias)

        assert_close_golden(rel_pos_bias, golden, "rel_pos_bias")
