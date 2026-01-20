"""S4 (Structured State Space) generator.

Reference: "Efficiently Modeling Long Sequences with Structured State Spaces"
https://arxiv.org/abs/2111.00396
"""

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


def s4_kernel_reference(A, B, C, L, step):
    """Compute S4 convolution kernel using DPLR (Diagonal Plus Low-Rank).

    The S4 kernel is computed by:
    1. Discretize continuous-time SSM using bilinear transform
    2. Compute convolution kernel K = C @ (I - Ab)^{-L} @ ... @ (I - Ab) @ Bb

    Simplified reference for testing - uses eigendecomposition approach.
    """
    N = A.shape[0]

    # Discretize using bilinear transform (simplified)
    # Ab = (I - step/2 * A)^{-1} @ (I + step/2 * A)
    I = torch.eye(N, dtype=A.dtype)
    Ab = torch.linalg.solve(I - step / 2 * A, I + step / 2 * A)
    Bb = torch.linalg.solve(I - step / 2 * A, step * B)

    # Compute kernel via power iteration
    K = torch.zeros(L, dtype=A.dtype)
    Ab_power = torch.eye(N, dtype=A.dtype)

    for i in range(L):
        K[i] = (C @ Ab_power @ Bb).item()
        Ab_power = Ab_power @ Ab

    return K


class S4Generator(GoldenGenerator):
    """Generate golden files for S4 layer.

    S4 uses structured state spaces with HIPPO initialization
    for efficient long-range sequence modeling.
    """

    @property
    def name(self) -> str:
        return "s4"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["ssm"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        s4_configs = [
            {"d_model": 64, "d_state": 32, "name": "tiny"},
            {"d_model": 128, "d_state": 64, "name": "small"},
            {"d_model": 256, "d_state": 64, "name": "medium"},
        ]

        for cfg in s4_configs:
            # Use "small" shape for all S4 tests (test expects s4_tiny, s4_small, s4_medium)
            shape = STANDARD_SHAPES["small"]
            configs.append(
                {
                    **cfg,
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "name": f"s4_{cfg['name']}",  # Test expects simple names like s4_tiny
                }
            )

        # Bidirectional test
        configs.append(
            {
                "name": "s4_bidirectional",
                "batch": 2,
                "seq": 64,
                "d_model": 128,
                "d_state": 64,
                "bidirectional": True,
            }
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        d_model = config["d_model"]
        d_state = config["d_state"]
        bidirectional = config.get("bidirectional", False)

        # Input
        x = torch.randn(batch, seq, d_model)

        # S4 parameters per channel (simplified - one SSM per feature)
        # In practice, S4 uses HIPPO initialization, but for testing we use random

        # Discretization step
        step = 1.0 / seq

        # Generate SSM parameters (simplified - diagonal A)
        # Real S4 uses structured HIPPO matrix
        Lambda_real = -torch.exp(torch.randn(d_model, d_state))
        Lambda_imag = torch.randn(d_model, d_state) * math.pi

        # B, C are learnable
        B = torch.randn(d_model, d_state) * 0.02
        C = torch.randn(d_model, d_state) * 0.02

        # D is skip connection
        D = torch.ones(d_model)

        # Simplified forward: compute as recurrence
        # For each channel, run SSM
        outputs = []
        for b_idx in range(batch):
            channel_outputs = []
            for d in range(d_model):
                # Simple diagonal SSM for this channel
                h = torch.zeros(d_state)
                y_channel = []

                # Discretized A (diagonal case)
                A_d = Lambda_real[d] + 1j * Lambda_imag[d]
                A_discrete = torch.exp(step * A_d)

                for t in range(seq):
                    # Update state
                    h = A_discrete.real * h + step * B[d] * x[b_idx, t, d]
                    # Output
                    y_t = (C[d] * h).sum() + D[d] * x[b_idx, t, d]
                    y_channel.append(y_t)

                channel_outputs.append(torch.stack(y_channel))
            outputs.append(torch.stack(channel_outputs, dim=-1))

        out = torch.stack(outputs)  # (batch, seq, d_model)

        if bidirectional:
            # Run backward pass and combine
            outputs_bwd = []
            for b_idx in range(batch):
                channel_outputs = []
                for d in range(d_model):
                    h = torch.zeros(d_state)
                    y_channel = []

                    A_d = Lambda_real[d] + 1j * Lambda_imag[d]
                    A_discrete = torch.exp(step * A_d)

                    for t in range(seq - 1, -1, -1):
                        h = A_discrete.real * h + step * B[d] * x[b_idx, t, d]
                        y_t = (C[d] * h).sum()
                        y_channel.append(y_t)

                    y_channel = y_channel[::-1]
                    channel_outputs.append(torch.stack(y_channel))
                outputs_bwd.append(torch.stack(channel_outputs, dim=-1))

            out_bwd = torch.stack(outputs_bwd)
            out = out + out_bwd

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "Lambda_real": Lambda_real.numpy(),
                "Lambda_imag": Lambda_imag.numpy(),
                "B": B.numpy(),
                "C": C.numpy(),
                "D": D.numpy(),
            },
            params={
                "d_state": d_state,
                "step": step,
                "bidirectional": bidirectional,
            },
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "d_model": d_model,
            },
        )
