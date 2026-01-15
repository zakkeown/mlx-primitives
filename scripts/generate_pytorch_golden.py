#!/usr/bin/env python3
"""Generate PyTorch golden reference outputs for correctness testing.

This script generates reference outputs from PyTorch implementations
and saves them as .npz files for comparison with MLX implementations.

Usage:
    pip install torch  # Requires PyTorch
    python scripts/generate_pytorch_golden.py

Output files are saved to tests/golden/
"""

import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    print("Error: This script requires PyTorch and NumPy.")
    print("Install with: pip install torch numpy")
    sys.exit(1)

# Output directory
GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)


def generate_attention_golden():
    """Generate golden outputs for scaled dot product attention."""
    print("Generating attention golden files...")

    configs = [
        {"batch": 1, "seq": 64, "heads": 8, "head_dim": 64, "name": "small"},
        {"batch": 2, "seq": 128, "heads": 16, "head_dim": 64, "name": "medium"},
        {"batch": 4, "seq": 256, "heads": 32, "head_dim": 64, "name": "large"},
    ]

    for cfg in configs:
        torch.manual_seed(42)

        # Shape: (batch, seq, heads, head_dim)
        q = torch.randn(cfg["batch"], cfg["seq"], cfg["heads"], cfg["head_dim"])
        k = torch.randn(cfg["batch"], cfg["seq"], cfg["heads"], cfg["head_dim"])
        v = torch.randn(cfg["batch"], cfg["seq"], cfg["heads"], cfg["head_dim"])

        # PyTorch SDPA expects (batch, heads, seq, dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Non-causal
        out_non_causal = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        out_non_causal = out_non_causal.transpose(1, 2)  # Back to (batch, seq, heads, dim)

        # Causal
        out_causal = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        out_causal = out_causal.transpose(1, 2)

        np.savez(
            GOLDEN_DIR / f"attention_{cfg['name']}.npz",
            q=q.numpy(),
            k=k.numpy(),
            v=v.numpy(),
            out_non_causal=out_non_causal.numpy(),
            out_causal=out_causal.numpy(),
            scale=1.0 / (cfg["head_dim"] ** 0.5),
        )
        print(f"  Saved attention_{cfg['name']}.npz")


def generate_rope_golden():
    """Generate golden outputs for Rotary Position Embeddings."""
    print("Generating RoPE golden files...")

    torch.manual_seed(42)

    seq_len = 128
    head_dim = 64
    batch = 2
    heads = 8

    # Create position indices
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    # Create input
    x = torch.randn(batch, seq_len, heads, head_dim)

    # Compute RoPE using standard implementation
    def compute_rope_frequencies(head_dim, seq_len, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    cos, sin = compute_rope_frequencies(head_dim, seq_len)

    def apply_rope(x, cos, sin):
        # x: (batch, seq, heads, dim)
        seq_len = x.shape[1]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim/2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        # Rotate
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    out = apply_rope(x, cos, sin)

    np.savez(
        GOLDEN_DIR / "rope.npz",
        x=x.numpy(),
        cos=cos.numpy(),
        sin=sin.numpy(),
        out=out.numpy(),
        base=10000.0,
    )
    print("  Saved rope.npz")


def generate_rmsnorm_golden():
    """Generate golden outputs for RMS Normalization."""
    print("Generating RMSNorm golden files...")

    torch.manual_seed(42)

    batch, seq, dim = 2, 128, 256
    eps = 1e-6

    x = torch.randn(batch, seq, dim)
    weight = torch.ones(dim)

    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    out = x / rms * weight

    np.savez(
        GOLDEN_DIR / "rmsnorm.npz",
        x=x.numpy(),
        weight=weight.numpy(),
        out=out.numpy(),
        eps=eps,
    )
    print("  Saved rmsnorm.npz")


def generate_gelu_golden():
    """Generate golden outputs for GELU activation."""
    print("Generating GELU golden files...")

    torch.manual_seed(42)

    x = torch.randn(2, 128, 256)
    out = F.gelu(x)

    np.savez(
        GOLDEN_DIR / "gelu.npz",
        x=x.numpy(),
        out=out.numpy(),
    )
    print("  Saved gelu.npz")


def generate_swiglu_golden():
    """Generate golden outputs for SwiGLU activation."""
    print("Generating SwiGLU golden files...")

    torch.manual_seed(42)

    batch, seq, dim = 2, 128, 256

    # SwiGLU: silu(x1) * x2 where input is split
    x = torch.randn(batch, seq, dim * 2)
    x1, x2 = x.chunk(2, dim=-1)
    out = F.silu(x1) * x2

    np.savez(
        GOLDEN_DIR / "swiglu.npz",
        x=x.numpy(),
        out=out.numpy(),
    )
    print("  Saved swiglu.npz")


def generate_groupnorm_golden():
    """Generate golden outputs for Group Normalization."""
    print("Generating GroupNorm golden files...")

    torch.manual_seed(42)

    batch, channels, height, width = 2, 32, 16, 16
    num_groups = 8
    eps = 1e-5

    x = torch.randn(batch, channels, height, width)
    gn = torch.nn.GroupNorm(num_groups, channels, eps=eps)

    # Use default weight=1, bias=0
    with torch.no_grad():
        gn.weight.fill_(1.0)
        gn.bias.fill_(0.0)

    out = gn(x)

    np.savez(
        GOLDEN_DIR / "groupnorm.npz",
        x=x.detach().numpy(),
        out=out.detach().numpy(),
        num_groups=num_groups,
        eps=eps,
    )
    print("  Saved groupnorm.npz")


def main():
    """Generate all golden files."""
    print(f"Output directory: {GOLDEN_DIR}")
    print()

    generate_attention_golden()
    generate_rope_golden()
    generate_rmsnorm_golden()
    generate_gelu_golden()
    generate_swiglu_golden()
    generate_groupnorm_golden()

    print()
    print("Done! Golden files generated.")
    print(f"Files saved to: {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
