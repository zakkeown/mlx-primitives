"""Positional encoding generators: ALiBi, RoPE variants."""

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class ALiBiGenerator(GoldenGenerator):
    """Generate golden files for ALiBi (Attention with Linear Biases).

    ALiBi adds linear position-based biases to attention scores.
    bias[i,j] = -slope * |i - j|
    """

    @property
    def name(self) -> str:
        return "alibi"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["alibi"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            for causal in [False, True]:
                causal_str = "causal" if causal else "noncausal"
                configs.append(
                    {
                        "name": f"alibi_{size_name}_{causal_str}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "causal": causal,
                    }
                )

        # Long sequence extrapolation test
        configs.append(
            {
                "name": "alibi_extrapolation",
                "batch": 1,
                "seq": 2048,
                "heads": 8,
                "head_dim": 64,
                "causal": True,
            }
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)
        v = torch.randn(batch, seq, heads, head_dim)

        # Compute ALiBi slopes
        def get_alibi_slopes(num_heads):
            """Get ALiBi slopes for each head."""
            # Slopes follow geometric sequence: 2^(-8/n), 2^(-16/n), ...
            ratio = 2 ** (-8 / num_heads)
            slopes = torch.tensor([ratio ** (i + 1) for i in range(num_heads)])
            return slopes

        slopes = get_alibi_slopes(heads)

        # Compute ALiBi bias matrix
        # bias[h, i, j] = -slopes[h] * |i - j|
        positions = torch.arange(seq)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq, seq)
        rel_pos = rel_pos.float()

        # For causal, we use i - j (not absolute value)
        if causal:
            alibi_bias = -slopes.view(-1, 1, 1) * rel_pos.unsqueeze(0)
            # Add causal mask
            causal_mask = torch.triu(torch.ones(seq, seq) * float("-inf"), diagonal=1)
            alibi_bias = alibi_bias + causal_mask.unsqueeze(0)
        else:
            alibi_bias = -slopes.view(-1, 1, 1) * rel_pos.abs().unsqueeze(0)

        # Shape: (heads, seq, seq) -> (1, heads, seq, seq)
        alibi_bias = alibi_bias.unsqueeze(0)

        # Compute attention with ALiBi
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=alibi_bias)
        out = out_t.transpose(1, 2)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={"causal": causal},
            expected_outputs={
                "out": out.numpy(),
                "slopes": slopes.numpy(),
                "alibi_bias": alibi_bias.squeeze(0).numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class RoPEGenerator(GoldenGenerator):
    """Generate golden files for standard Rotary Position Embeddings.

    RoPE applies rotation to Q and K based on position.
    """

    @property
    def name(self) -> str:
        return "rope"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["rope"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"rope_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "heads": shape["heads"],
                    "head_dim": shape["head_dim"],
                    "base": 10000.0,
                }
            )

        # Different base frequencies
        for base in [10000.0, 500000.0, 1000000.0]:
            configs.append(
                {
                    "name": f"rope_base{int(base)}",
                    "batch": 2,
                    "seq": 128,
                    "heads": 8,
                    "head_dim": 64,
                    "base": base,
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = config["base"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)

        # Compute RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq).float()
        freqs = torch.outer(t, inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Apply RoPE
        def apply_rope(x, cos, sin):
            # x: (batch, seq, heads, dim)
            # cos, sin: (seq, dim/2)
            seq_len = x.shape[1]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim/2)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]

            # Rotate
            x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return x_rot

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        # Mathematically justified tolerance:
        # exp() ULP error (~6e-8) × seq × 1.2 safety margin
        computed_atol = 6e-8 * seq * 1.2

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
            },
            params={"base": base},
            expected_outputs={
                "q_rot": q_rot.numpy(),
                "k_rot": k_rot.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
            tolerance=ToleranceConfig(
                rtol_fp32=1e-5,
                atol_fp32=computed_atol,
                rtol_fp16=1e-3,
                atol_fp16=max(1e-4, computed_atol * 10),
            ),
        )


class RoPENTKGenerator(GoldenGenerator):
    """Generate golden files for NTK-aware RoPE.

    Scales the base frequency for longer context windows.
    """

    @property
    def name(self) -> str:
        return "rope_ntk"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["rope"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Different scaling factors
        for scale_factor in [2.0, 4.0, 8.0]:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"rope_ntk_scale{scale_factor}_{size_name}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "base": 10000.0,
                        "scale_factor": scale_factor,
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = config["base"]
        scale_factor = config["scale_factor"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)

        # NTK-aware scaling: scale base by alpha
        # alpha = (scale_factor * seq / original_seq) ^ (dim / (dim - 2))
        alpha = scale_factor ** (head_dim / (head_dim - 2))
        scaled_base = base * alpha

        # Compute frequencies with scaled base
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq).float()
        freqs = torch.outer(t, inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        def apply_rope(x, cos, sin):
            seq_len = x.shape[1]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]

            x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return x_rot

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
            },
            params={
                "base": base,
                "scale_factor": scale_factor,
                "scaled_base": scaled_base,
            },
            expected_outputs={
                "q_rot": q_rot.numpy(),
                "k_rot": k_rot.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )


class RoPEYaRNGenerator(GoldenGenerator):
    """Generate golden files for YaRN RoPE.

    YaRN uses attention scaling and frequency interpolation.
    """

    @property
    def name(self) -> str:
        return "rope_yarn"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["rope"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for scale_factor in [2.0, 4.0]:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"rope_yarn_scale{scale_factor}_{size_name}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "base": 10000.0,
                        "scale_factor": scale_factor,
                        "original_max_seq": 2048,
                        "beta_fast": 32,
                        "beta_slow": 1,
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = config["base"]
        scale_factor = config["scale_factor"]
        original_max_seq = config["original_max_seq"]
        beta_fast = config["beta_fast"]
        beta_slow = config["beta_slow"]

        q = torch.randn(batch, seq, heads, head_dim)
        k = torch.randn(batch, seq, heads, head_dim)

        # YaRN frequency computation
        def yarn_find_correction_range(low_rot, high_rot, dim, base, max_seq):
            low = math.floor(dim * math.log(max_seq / (low_rot * 2 * math.pi)) / (2 * math.log(base)))
            high = math.ceil(dim * math.log(max_seq / (high_rot * 2 * math.pi)) / (2 * math.log(base)))
            return max(low, 0), min(high, dim // 2 - 1)

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001
            linear_func = (torch.arange(dim // 2, dtype=torch.float32) - min_val) / (max_val - min_val)
            return torch.clamp(linear_func, 0, 1)

        low, high = yarn_find_correction_range(beta_fast, beta_slow, head_dim, base, original_max_seq)
        ramp_mask = yarn_linear_ramp_mask(low, high, head_dim)

        # Compute inverse frequencies with YaRN scaling
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Interpolation factor
        inv_freq_interpolated = inv_freq / scale_factor

        # Blend based on ramp
        inv_freq_yarn = inv_freq_interpolated * (1 - ramp_mask) + inv_freq * ramp_mask

        t = torch.arange(seq).float()
        freqs = torch.outer(t, inv_freq_yarn)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Attention scaling factor
        attn_scale = 0.1 * math.log(scale_factor) + 1.0

        def apply_rope(x, cos, sin):
            seq_len = x.shape[1]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]

            x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return x_rot

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
            },
            params={
                "base": base,
                "scale_factor": scale_factor,
                "attn_scale": attn_scale,
            },
            expected_outputs={
                "q_rot": q_rot.numpy(),
                "k_rot": k_rot.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
            },
            metadata={
                "batch": batch,
                "seq": seq,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
