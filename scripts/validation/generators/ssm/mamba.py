"""Mamba/SSM generators: selective_scan, MambaBlock.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


def selective_scan_reference(x, delta, A, B, C, D):
    """Reference implementation of selective scan in PyTorch.

    This is the core operation in Mamba that replaces attention.

    Args:
        x: Input tensor (batch, seq, d_inner)
        delta: Time step tensor (batch, seq, d_inner)
        A: State transition matrix (d_inner, d_state)
        B: Input projection (batch, seq, d_state)
        C: Output projection (batch, seq, d_state)
        D: Skip connection (d_inner,)

    Returns:
        Output tensor (batch, seq, d_inner)
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize: convert continuous A, B to discrete form
    # deltaA = exp(delta * A)
    # deltaB = delta * B
    delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (batch, seq, d_inner, d_state)
    delta_A = torch.exp(delta_A)
    delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_inner, d_state)

    # Sequential scan (the core recurrence)
    h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
    outputs = []

    for t in range(seq_len):
        # h_t = deltaA_t * h_{t-1} + deltaB_t * x_t
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t, :, None]
        # y_t = h_t @ C_t
        y = (h * C[:, t, None, :]).sum(dim=-1)
        outputs.append(y)

    y = torch.stack(outputs, dim=1)

    # Add skip connection: y = y + D * x
    y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


class SelectiveScanGenerator(GoldenGenerator):
    """Generate golden files for selective scan operation.

    The selective scan is the core operation in Mamba that enables
    linear-time sequence modeling with input-dependent state transitions.
    """

    @property
    def name(self) -> str:
        return "selective_scan"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["ssm_scan"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Various sizes
        ssm_configs = [
            {"batch": 1, "seq": 16, "d_inner": 32, "d_state": 8, "name": "tiny"},
            {"batch": 2, "seq": 64, "d_inner": 128, "d_state": 16, "name": "small"},
            {"batch": 2, "seq": 128, "d_inner": 256, "d_state": 16, "name": "medium"},
            {"batch": 1, "seq": 256, "d_inner": 512, "d_state": 16, "name": "large"},
        ]

        for cfg in ssm_configs:
            configs.append(
                {
                    **cfg,
                    "name": f"selective_scan_{cfg['name']}",  # Must be after **cfg to override
                }
            )

        # Edge cases
        configs.extend(
            [
                # Long sequence for numerical stability
                {
                    "name": "selective_scan_long_seq",
                    "batch": 1,
                    "seq": 1024,
                    "d_inner": 64,
                    "d_state": 16,
                },
                # Single token
                {
                    "name": "selective_scan_single",
                    "batch": 1,
                    "seq": 1,
                    "d_inner": 64,
                    "d_state": 16,
                },
                # Large d_state
                {
                    "name": "selective_scan_large_state",
                    "batch": 2,
                    "seq": 64,
                    "d_inner": 128,
                    "d_state": 64,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq_len = config["seq"]
        d_inner = config["d_inner"]
        d_state = config["d_state"]

        # Generate inputs
        x = torch.randn(batch, seq_len, d_inner)

        # Delta should be positive (time step), use softplus
        delta_raw = torch.randn(batch, seq_len, d_inner)
        delta = F.softplus(delta_raw)

        # A should be negative for stability (ensures state decays)
        A = -torch.exp(torch.randn(d_inner, d_state))

        # B and C are input-dependent projections
        B = torch.randn(batch, seq_len, d_state)
        C = torch.randn(batch, seq_len, d_state)

        # D is skip connection weight
        D = torch.ones(d_inner)

        # Compute reference output
        out = selective_scan_reference(x, delta, A, B, C, D)

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "delta": delta.numpy(),
                "A": A.numpy(),
                "B": B.numpy(),
                "C": C.numpy(),
                "D": D.numpy(),
            },
            params={"d_state": d_state},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq_len": seq_len,
                "d_inner": d_inner,
                "d_state": d_state,
            },
        )


class MambaBlockGenerator(GoldenGenerator):
    """Generate golden files for full Mamba block.

    A Mamba block consists of:
    1. Linear projection to expand dimensions
    2. Conv1D for local context
    3. Selective scan for sequence modeling
    4. Output projection
    """

    @property
    def name(self) -> str:
        return "mamba_block"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["ssm"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        mamba_configs = [
            {"dims": 64, "d_state": 16, "d_conv": 4, "expand": 2, "name": "tiny"},
            {"dims": 256, "d_state": 16, "d_conv": 4, "expand": 2, "name": "small"},
            {"dims": 512, "d_state": 16, "d_conv": 4, "expand": 2, "name": "medium"},
        ]

        for cfg in mamba_configs:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        **cfg,
                        "batch": shape["batch"],
                        "seq": shape["seq"],
                        "name": f"mamba_block_{cfg['name']}_{size_name}",  # Must be after **cfg
                    }
                )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]
        expand = config["expand"]

        d_inner = dims * expand

        # Input
        x = torch.randn(batch, seq, dims)

        # Mamba block weights
        # Input projection: project to 2 * d_inner (for gate and main path)
        in_proj_weight = torch.randn(2 * d_inner, dims) * 0.02

        # Conv1D weights
        conv_weight = torch.randn(d_inner, 1, d_conv) * 0.02
        conv_bias = torch.zeros(d_inner)

        # SSM parameters
        dt_proj_weight = torch.randn(d_inner, d_inner) * 0.02
        dt_proj_bias = torch.randn(d_inner) * 0.02
        A = -torch.exp(torch.randn(d_inner, d_state))
        D = torch.ones(d_inner)

        # Output projection
        out_proj_weight = torch.randn(dims, d_inner) * 0.02

        # x_and_res projections for B and C
        x_proj_weight = torch.randn(d_state * 2, d_inner) * 0.02

        # Forward pass
        # 1. Input projection
        xz = x @ in_proj_weight.T  # (batch, seq, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (batch, seq, d_inner)

        # 2. Conv1D (causal, so we need padding)
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq)
        x_conv = F.pad(x_conv, (d_conv - 1, 0))  # left pad
        x_conv = F.conv1d(x_conv, conv_weight, conv_bias, groups=d_inner)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)

        # 3. SiLU activation
        x_conv = F.silu(x_conv)

        # 4. SSM projection to get B, C
        x_dbl = x_conv @ x_proj_weight.T  # (batch, seq, 2*d_state)
        B, C = x_dbl.split(d_state, dim=-1)

        # 5. Delta projection
        delta = x_conv @ dt_proj_weight.T + dt_proj_bias
        delta = F.softplus(delta)

        # 6. Selective scan
        y = selective_scan_reference(x_conv, delta, A, B, C, D)

        # 7. Gate with z
        y = y * F.silu(z)

        # 8. Output projection
        out = y @ out_proj_weight.T

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "in_proj_weight": in_proj_weight.numpy(),
                "conv_weight": conv_weight.numpy(),
                "conv_bias": conv_bias.numpy(),
                "dt_proj_weight": dt_proj_weight.numpy(),
                "dt_proj_bias": dt_proj_bias.numpy(),
                "A": A.numpy(),
                "D": D.numpy(),
                "x_proj_weight": x_proj_weight.numpy(),
                "out_proj_weight": out_proj_weight.numpy(),
            },
            params={
                "dims": dims,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            },
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "d_inner": d_inner,
            },
        )
