"""GLU variant generators: SwiGLU, GeGLU, ReGLU, FusedSwiGLU."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class SwiGLUGenerator(GoldenGenerator):
    """Generate golden files for SwiGLU activation.

    SwiGLU: silu(gate(x)) * linear(x)
    """

    @property
    def name(self) -> str:
        return "swiglu"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_glu"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Standard shapes
        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"swiglu_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "dims": shape["dims"],
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "swiglu_zeros",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "zeros",
                },
                {
                    "name": "swiglu_negative",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "negative",
                },
                {
                    "name": "swiglu_large",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "large",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        input_type = config.get("input_type", "random")

        # SwiGLU takes input of shape (batch, seq, 2*dims) split into gate and linear parts
        shape = (batch, seq, dims * 2)

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        elif input_type == "large":
            x = torch.full(shape, 10.0)
        else:
            x = torch.randn(shape)

        # Split and apply SwiGLU: silu(x1) * x2
        x1, x2 = x.chunk(2, dim=-1)
        out = F.silu(x1) * x2

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"dims": dims},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "dims": dims,
                "input_type": input_type,
            },
        )


class GeGLUGenerator(GoldenGenerator):
    """Generate golden files for GeGLU activation.

    GeGLU: gelu(gate(x)) * linear(x)
    """

    @property
    def name(self) -> str:
        return "geglu"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_glu"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"geglu_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "dims": shape["dims"],
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "geglu_zeros",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "zeros",
                },
                {
                    "name": "geglu_negative",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "negative",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        input_type = config.get("input_type", "random")

        shape = (batch, seq, dims * 2)

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        else:
            x = torch.randn(shape)

        # Split and apply GeGLU: gelu(x1) * x2
        x1, x2 = x.chunk(2, dim=-1)
        out = F.gelu(x1) * x2

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"dims": dims},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "dims": dims,
                "input_type": input_type,
            },
        )


class ReGLUGenerator(GoldenGenerator):
    """Generate golden files for ReGLU activation.

    ReGLU: relu(gate(x)) * linear(x)
    """

    @property
    def name(self) -> str:
        return "reglu"

    def get_tolerance_config(self) -> ToleranceConfig:
        # ReGLU uses ReLU which is exact
        return TOLERANCE_CONFIGS["activations_exact"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"reglu_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "dims": shape["dims"],
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "reglu_zeros",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "zeros",
                },
                {
                    "name": "reglu_negative",
                    "batch": 2,
                    "seq": 64,
                    "dims": 256,
                    "input_type": "negative",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        dims = config["dims"]
        input_type = config.get("input_type", "random")

        shape = (batch, seq, dims * 2)

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        else:
            x = torch.randn(shape)

        # Split and apply ReGLU: relu(x1) * x2
        x1, x2 = x.chunk(2, dim=-1)
        out = F.relu(x1) * x2

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"dims": dims},
            expected_outputs={"out": out.numpy()},
            metadata={
                "batch": batch,
                "seq": seq,
                "dims": dims,
                "input_type": input_type,
            },
        )


class FusedSwiGLUGenerator(GoldenGenerator):
    """Generate golden files for FusedSwiGLU.

    Same computation as SwiGLU but tests the fused (optimized) implementation.
    """

    @property
    def name(self) -> str:
        return "fused_swiglu"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_glu"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"fused_swiglu_{size_name}",
                    "batch": shape["batch"],
                    "seq": shape["seq"],
                    "in_dims": shape["dims"],
                    "hidden_dims": shape["hidden"],
                    "out_dims": shape["dims"],
                }
            )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        seq = config["seq"]
        in_dims = config["in_dims"]
        hidden_dims = config["hidden_dims"]
        out_dims = config["out_dims"]

        # Input
        x = torch.randn(batch, seq, in_dims)

        # Weight matrices for fused SwiGLU
        w1 = torch.randn(in_dims, hidden_dims) * 0.02
        w_gate = torch.randn(in_dims, hidden_dims) * 0.02
        w2 = torch.randn(hidden_dims, out_dims) * 0.02

        # FusedSwiGLU forward: w2 @ (silu(x @ w_gate) * (x @ w1))
        gate = F.silu(x @ w_gate)
        hidden = gate * (x @ w1)
        out = hidden @ w2

        return TestConfig(
            name=config["name"],
            inputs={
                "x": x.numpy(),
                "w1": w1.numpy(),
                "w_gate": w_gate.numpy(),
                "w2": w2.numpy(),
            },
            params={
                "in_dims": in_dims,
                "hidden_dims": hidden_dims,
                "out_dims": out_dims,
            },
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "seq": seq},
        )
