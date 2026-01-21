"""Fused attention generators: FusedRoPEAttention."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class FusedRoPEAttentionGenerator(GoldenGenerator):
    """Generate golden files for fused RoPE + Attention.

    FusedRoPEAttention: Apply RoPE to Q and K, then compute scaled dot-product attention.
    Computes: softmax(RoPE(Q) @ RoPE(K)^T / sqrt(d_k)) @ V

    This fused operation avoids materializing intermediate rotated Q/K tensors,
    reducing memory bandwidth for large sequence lengths.
    """

    @property
    def name(self) -> str:
        return "fused_rope_attention"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["attention_standard"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Standard shapes with causal and non-causal
        for size_name, shape in STANDARD_SHAPES.items():
            for causal in [False, True]:
                causal_str = "causal" if causal else "noncausal"
                configs.append(
                    {
                        "name": f"fused_rope_attention_{size_name}_{causal_str}",
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "heads": shape["heads"],
                        "head_dim": shape["head_dim"],
                        "causal": causal,
                        "base": 10000.0,
                    }
                )

        # Position offset tests (for incremental decoding)
        configs.extend(
            [
                {
                    "name": "fused_rope_attention_offset_q",
                    "batch": 2,
                    "seq_q": 1,  # Single token (decode step)
                    "seq_kv": 64,
                    "heads": 8,
                    "head_dim": 64,
                    "causal": False,
                    "base": 10000.0,
                    "q_offset": 63,  # Position of new token
                    "kv_offset": 0,
                },
                {
                    "name": "fused_rope_attention_offset_both",
                    "batch": 2,
                    "seq_q": 4,
                    "seq_kv": 64,
                    "heads": 8,
                    "head_dim": 64,
                    "causal": False,
                    "base": 10000.0,
                    "q_offset": 60,
                    "kv_offset": 0,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq_q = config.get("seq_q", config.get("seq"))
        seq_kv = config.get("seq_kv", seq_q)
        heads = config["heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]
        base = config["base"]
        q_offset = config.get("q_offset", 0)
        kv_offset = config.get("kv_offset", 0)

        # Generate Q, K, V tensors
        # Shape: (batch, seq, heads, head_dim) - MLX convention
        q = torch.randn(batch, seq_q, heads, head_dim)
        k = torch.randn(batch, seq_kv, heads, head_dim)
        v = torch.randn(batch, seq_kv, heads, head_dim)

        # Compute RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Position indices for Q and K (with offsets)
        q_positions = torch.arange(q_offset, q_offset + seq_q).float()
        kv_positions = torch.arange(kv_offset, kv_offset + seq_kv).float()

        q_freqs = torch.outer(q_positions, inv_freq)
        kv_freqs = torch.outer(kv_positions, inv_freq)

        q_cos = torch.cos(q_freqs)
        q_sin = torch.sin(q_freqs)
        k_cos = torch.cos(kv_freqs)
        k_sin = torch.sin(kv_freqs)

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

        q_rot = apply_rope(q, q_cos, q_sin)
        k_rot = apply_rope(k, k_cos, k_sin)

        # Transpose to PyTorch convention: (batch, heads, seq, head_dim)
        q_t = q_rot.transpose(1, 2)
        k_t = k_rot.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Use PyTorch SDPA
        # Note: is_causal only works when seq_q == seq_kv
        if causal and seq_q == seq_kv:
            out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        else:
            # Manual attention with optional causal mask
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

            if causal:
                # Create causal mask for cross-attention case
                mask = torch.triu(torch.ones(seq_q, seq_kv), diagonal=1).bool()
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            out_t = torch.matmul(attn_weights, v_t)

        # Transpose back to MLX convention
        out = out_t.transpose(1, 2)

        scale = 1.0 / (head_dim ** 0.5)

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q.numpy(),
                "k": k.numpy(),
                "v": v.numpy(),
            },
            params={
                "causal": causal,
                "scale": scale,
                "base": base,
                "q_offset": q_offset,
                "kv_offset": kv_offset,
            },
            expected_outputs={
                "out": out.numpy(),
                "q_cos": q_cos.numpy(),
                "q_sin": q_sin.numpy(),
                "k_cos": k_cos.numpy(),
                "k_sin": k_sin.numpy(),
            },
            metadata={
                "batch": batch,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "heads": heads,
                "head_dim": head_dim,
            },
        )

    def generate_numpy_reference(self, config: Dict[str, Any]) -> TestConfig:
        np.random.seed(self.seed)

        batch = config["batch"]
        seq_q = config.get("seq_q", config.get("seq"))
        seq_kv = config.get("seq_kv", seq_q)
        heads = config["heads"]
        head_dim = config["head_dim"]
        causal = config["causal"]
        base = config["base"]
        q_offset = config.get("q_offset", 0)
        kv_offset = config.get("kv_offset", 0)

        # Generate Q, K, V tensors
        q = np.random.randn(batch, seq_q, heads, head_dim).astype(np.float32)
        k = np.random.randn(batch, seq_kv, heads, head_dim).astype(np.float32)
        v = np.random.randn(batch, seq_kv, heads, head_dim).astype(np.float32)

        # Compute RoPE frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))

        # Position indices for Q and K (with offsets)
        q_positions = np.arange(q_offset, q_offset + seq_q).astype(np.float32)
        kv_positions = np.arange(kv_offset, kv_offset + seq_kv).astype(np.float32)

        q_freqs = np.outer(q_positions, inv_freq)
        kv_freqs = np.outer(kv_positions, inv_freq)

        q_cos = np.cos(q_freqs)
        q_sin = np.sin(q_freqs)
        k_cos = np.cos(kv_freqs)
        k_sin = np.sin(kv_freqs)

        # Apply RoPE
        def apply_rope(x, cos, sin):
            seq_len = x.shape[1]
            cos = cos[:seq_len][np.newaxis, :, np.newaxis, :]  # (1, seq, 1, dim/2)
            sin = sin[:seq_len][np.newaxis, :, np.newaxis, :]

            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]

            x_rot = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
            return x_rot

        q_rot = apply_rope(q, q_cos, q_sin)
        k_rot = apply_rope(k, k_cos, k_sin)

        # Transpose to (batch, heads, seq, head_dim)
        q_t = np.transpose(q_rot, (0, 2, 1, 3))
        k_t = np.transpose(k_rot, (0, 2, 1, 3))
        v_t = np.transpose(v, (0, 2, 1, 3))

        # Compute attention
        scale = 1.0 / (head_dim ** 0.5)
        scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) * scale

        if causal:
            # Create causal mask
            mask = np.triu(np.ones((seq_q, seq_kv)), k=1)
            scores = np.where(mask[np.newaxis, np.newaxis, :, :] == 1, -np.inf, scores)

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        out_t = np.matmul(attn_weights, v_t)

        # Transpose back to MLX convention
        out = np.transpose(out_t, (0, 2, 1, 3))

        return TestConfig(
            name=config["name"],
            inputs={
                "q": q,
                "k": k,
                "v": v,
            },
            params={
                "causal": causal,
                "scale": scale,
                "base": base,
                "q_offset": q_offset,
                "kv_offset": kv_offset,
            },
            expected_outputs={
                "out": out,
                "q_cos": q_cos,
                "q_sin": q_sin,
                "k_cos": k_cos,
                "k_sin": k_sin,
            },
            metadata={
                "batch": batch,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "heads": heads,
                "head_dim": head_dim,
            },
        )
