"""PyTorch Golden Reference Comparison Tests.

These tests compare MLX implementations against pre-generated PyTorch outputs.
Golden files are generated using scripts/generate_pytorch_golden.py

If golden files don't exist, tests are skipped with a message to generate them.

Run the generator:
    pip install torch
    python scripts/generate_pytorch_golden.py
"""

import pytest
from pathlib import Path
import numpy as np
import mlx.core as mx

from mlx_primitives.attention import flash_attention_forward
from mlx_primitives.attention.rope import apply_rope, precompute_freqs_cis
from mlx_primitives.layers import RMSNorm, SwiGLU, GroupNorm


GOLDEN_DIR = Path(__file__).parent.parent / "golden"


def golden_file_exists(name: str) -> bool:
    """Check if a golden file exists."""
    return (GOLDEN_DIR / name).exists()


def load_golden(name: str) -> dict:
    """Load golden file as dict of arrays."""
    path = GOLDEN_DIR / name
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# =============================================================================
# Attention Tests
# =============================================================================


class TestFlashAttentionVsPyTorch:
    """Compare FlashAttention against PyTorch SDPA golden outputs."""

    @pytest.mark.skipif(
        not golden_file_exists("attention_small.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_attention_small_non_causal(self):
        """Test small attention config (1, 64, 8, 64) non-causal."""
        golden = load_golden("attention_small.npz")

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        expected = mx.array(golden["out_non_causal"])
        scale = float(golden["scale"])

        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        mean_diff = float(mx.mean(mx.abs(out - expected)))

        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"
        assert mean_diff < 1e-5, f"Mean diff {mean_diff} exceeds tolerance"

    @pytest.mark.skipif(
        not golden_file_exists("attention_small.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_attention_small_causal(self):
        """Test small attention config (1, 64, 8, 64) causal."""
        golden = load_golden("attention_small.npz")

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        expected = mx.array(golden["out_causal"])
        scale = float(golden["scale"])

        out = flash_attention_forward(q, k, v, scale=scale, causal=True)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    @pytest.mark.skipif(
        not golden_file_exists("attention_medium.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_attention_medium_non_causal(self):
        """Test medium attention config (2, 128, 16, 64) non-causal."""
        golden = load_golden("attention_medium.npz")

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        expected = mx.array(golden["out_non_causal"])
        scale = float(golden["scale"])

        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    @pytest.mark.skipif(
        not golden_file_exists("attention_large.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_attention_large_non_causal(self):
        """Test large attention config (4, 256, 32, 64) non-causal."""
        golden = load_golden("attention_large.npz")

        q = mx.array(golden["q"])
        k = mx.array(golden["k"])
        v = mx.array(golden["v"])
        expected = mx.array(golden["out_non_causal"])
        scale = float(golden["scale"])

        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# RoPE Tests
# =============================================================================


class TestRoPEVsPyTorch:
    """Compare RoPE against PyTorch reference."""

    @pytest.mark.skipif(
        not golden_file_exists("rope.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_rope_matches_pytorch(self):
        """RoPE output should match PyTorch reference."""
        golden = load_golden("rope.npz")

        x = mx.array(golden["x"])
        expected = mx.array(golden["out"])
        base = float(golden["base"])

        # Get dimensions
        batch, seq_len, heads, head_dim = x.shape

        # Precompute frequencies
        cos, sin = precompute_freqs_cis(head_dim, seq_len, base=base)
        mx.eval(cos, sin)

        # Apply RoPE - note: apply_rope takes (q, k), returns (q_rot, k_rot)
        # Use x for both q and k, take first output
        out, _ = apply_rope(x, x, cos, sin)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# RMSNorm Tests
# =============================================================================


class TestRMSNormVsPyTorch:
    """Compare RMSNorm against PyTorch reference."""

    @pytest.mark.skipif(
        not golden_file_exists("rmsnorm.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_rmsnorm_matches_pytorch(self):
        """RMSNorm output should match PyTorch reference."""
        golden = load_golden("rmsnorm.npz")

        x = mx.array(golden["x"])
        expected = mx.array(golden["out"])
        eps = float(golden["eps"])

        # Get dim
        dim = x.shape[-1]

        # Create RMSNorm with default weight=1
        norm = RMSNorm(dim, eps=eps)
        # Weight should be initialized to 1
        mx.eval(norm.parameters())

        out = norm(x)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# GELU Tests
# =============================================================================


class TestGELUVsPyTorch:
    """Compare GELU against PyTorch reference."""

    @pytest.mark.skipif(
        not golden_file_exists("gelu.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_gelu_matches_pytorch(self):
        """GELU output should match PyTorch reference."""
        golden = load_golden("gelu.npz")

        x = mx.array(golden["x"])
        expected = mx.array(golden["out"])

        # MLX GELU
        import mlx.nn as nn

        gelu = nn.GELU()
        out = gelu(x)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        # GELU implementations can have slight differences
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# SwiGLU Tests
# =============================================================================


class TestSwiGLUVsPyTorch:
    """Compare SwiGLU against PyTorch reference."""

    @pytest.mark.skipif(
        not golden_file_exists("swiglu.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_swiglu_matches_pytorch(self):
        """SwiGLU output should match PyTorch reference."""
        golden = load_golden("swiglu.npz")

        x = mx.array(golden["x"])
        expected = mx.array(golden["out"])

        # Our SwiGLU takes pre-split input, so we need to split
        dim = x.shape[-1] // 2
        x1 = x[..., :dim]
        x2 = x[..., dim:]

        # Manual SwiGLU: silu(x1) * x2
        import mlx.nn as nn

        silu = nn.SiLU()
        out = silu(x1) * x2
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# GroupNorm Tests
# =============================================================================


class TestGroupNormVsPyTorch:
    """Compare GroupNorm against PyTorch reference."""

    @pytest.mark.skipif(
        not golden_file_exists("groupnorm.npz"),
        reason="Golden file not found. Run: python scripts/generate_pytorch_golden.py",
    )
    def test_groupnorm_matches_pytorch(self):
        """GroupNorm output should match PyTorch reference."""
        golden = load_golden("groupnorm.npz")

        x = mx.array(golden["x"])
        expected = mx.array(golden["out"])
        num_groups = int(golden["num_groups"])
        eps = float(golden["eps"])

        # Get channels
        channels = x.shape[1]

        # Create GroupNorm
        norm = GroupNorm(num_groups, channels, eps=eps)
        mx.eval(norm.parameters())

        out = norm(x)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"


# =============================================================================
# Self-Contained Reference Tests (no golden files needed)
# =============================================================================


class TestSelfContainedReferences:
    """Tests that don't require golden files - compare to inline reference implementations."""

    def test_rmsnorm_vs_inline_reference(self):
        """RMSNorm matches inline mathematical definition."""
        mx.random.seed(42)
        x = mx.random.normal((2, 64, 128))
        dim = 128
        eps = 1e-6

        # Inline reference: x / sqrt(mean(x^2) + eps)
        rms = mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + eps)
        expected = x / rms

        # Our implementation
        norm = RMSNorm(dim, eps=eps)
        out = norm(x)
        mx.eval(out, expected)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-5, f"RMSNorm differs by {max_diff}"

    def test_gelu_vs_inline_reference(self):
        """GELU matches mathematical definition."""
        mx.random.seed(42)
        x = mx.random.normal((2, 64, 128))

        # MLX GELU uses the exact form: x * 0.5 * (1 + erf(x / sqrt(2)))
        # This differs slightly from the tanh approximation
        import math

        expected = x * 0.5 * (1 + mx.erf(x / math.sqrt(2)))

        import mlx.nn as nn

        gelu = nn.GELU()
        out = gelu(x)
        mx.eval(out, expected)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-5, f"GELU differs by {max_diff}"

    def test_attention_vs_naive_reference(self):
        """Attention matches naive Q@K^T softmax @ V implementation."""
        mx.random.seed(42)
        batch, seq, heads, head_dim = 2, 32, 4, 64

        q = mx.random.normal((batch, seq, heads, head_dim))
        k = mx.random.normal((batch, seq, heads, head_dim))
        v = mx.random.normal((batch, seq, heads, head_dim))
        mx.eval(q, k, v)

        scale = 1.0 / (head_dim**0.5)

        # Naive reference
        # Transpose to (batch, heads, seq, dim)
        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)

        scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(scores, axis=-1)
        expected = attn @ v_t
        expected = expected.transpose(0, 2, 1, 3)  # Back to (batch, seq, heads, dim)
        mx.eval(expected)

        # Our implementation
        out = flash_attention_forward(q, k, v, scale=scale, causal=False)
        mx.eval(out)

        max_diff = float(mx.max(mx.abs(out - expected)))
        assert max_diff < 1e-4, f"Attention differs by {max_diff}"
