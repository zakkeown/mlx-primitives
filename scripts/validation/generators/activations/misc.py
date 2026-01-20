"""Miscellaneous activation generators: Mish, SquaredReLU, Swish, HardSwish, HardSigmoid."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class MishGenerator(GoldenGenerator):
    """Generate golden files for Mish activation.

    Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    """

    @property
    def name(self) -> str:
        return "mish"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_transcendental"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"mish_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "mish_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "mish_negative",
                    "shape": (2, 64, 256),
                    "input_type": "negative",
                },
                {
                    "name": "mish_large_negative",
                    "shape": (2, 64, 256),
                    "input_range": (-10, -5),
                },
                {
                    "name": "mish_large_positive",
                    "shape": (2, 64, 256),
                    "input_range": (5, 10),
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = config["shape"]
        input_type = config.get("input_type", "random")
        input_range = config.get("input_range")

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        elif input_range:
            low, high = input_range
            x = torch.rand(shape) * (high - low) + low
        else:
            x = torch.randn(shape)

        # Mish: x * tanh(softplus(x))
        out = F.mish(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )


class SquaredReLUGenerator(GoldenGenerator):
    """Generate golden files for SquaredReLU activation.

    SquaredReLU: relu(x)^2

    From "Primer: Searching for Efficient Transformers for Language Modeling"
    """

    @property
    def name(self) -> str:
        return "squared_relu"

    def get_tolerance_config(self) -> ToleranceConfig:
        # Squared ReLU is exact (piecewise polynomial)
        return TOLERANCE_CONFIGS["activations_exact"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"squared_relu_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "squared_relu_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "squared_relu_negative",
                    "shape": (2, 64, 256),
                    "input_type": "negative",
                },
                {
                    "name": "squared_relu_positive",
                    "shape": (2, 64, 256),
                    "input_type": "positive",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = config["shape"]
        input_type = config.get("input_type", "random")

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        elif input_type == "positive":
            x = torch.abs(torch.randn(shape)) + 0.1
        else:
            x = torch.randn(shape)

        # SquaredReLU: relu(x)^2
        out = F.relu(x) ** 2

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )


class SwishGenerator(GoldenGenerator):
    """Generate golden files for Swish activation (learnable beta).

    Swish: x * sigmoid(beta * x)

    When beta=1, this is equivalent to SiLU.
    """

    @property
    def name(self) -> str:
        return "swish"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_transcendental"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        # Test with different beta values
        for beta in [0.5, 1.0, 1.5, 2.0]:
            for size_name in ["small", "medium"]:
                shape = STANDARD_SHAPES[size_name]
                configs.append(
                    {
                        "name": f"swish_beta{beta}_{size_name}",
                        "shape": (shape["batch"], shape["seq"], shape["dims"]),
                        "beta": beta,
                    }
                )

        # Edge cases with beta=1 (SiLU)
        configs.extend(
            [
                {
                    "name": "swish_zeros",
                    "shape": (2, 64, 256),
                    "beta": 1.0,
                    "input_type": "zeros",
                },
                {
                    "name": "swish_negative",
                    "shape": (2, 64, 256),
                    "beta": 1.0,
                    "input_type": "negative",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = config["shape"]
        beta = config.get("beta", 1.0)
        input_type = config.get("input_type", "random")

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        else:
            x = torch.randn(shape)

        # Swish: x * sigmoid(beta * x)
        out = x * torch.sigmoid(beta * x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"beta": beta},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "beta": beta, "input_type": input_type},
        )


class HardSwishGenerator(GoldenGenerator):
    """Generate golden files for HardSwish activation.

    HardSwish: x * relu6(x + 3) / 6

    Efficient approximation of Swish for mobile networks.
    """

    @property
    def name(self) -> str:
        return "hard_swish"

    def get_tolerance_config(self) -> ToleranceConfig:
        # HardSwish is piecewise linear, should be exact
        return TOLERANCE_CONFIGS["activations_exact"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"hard_swish_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases - test the piecewise boundaries
        configs.extend(
            [
                {
                    "name": "hard_swish_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "hard_swish_at_boundary_neg3",
                    "shape": (2, 64, 256),
                    "input_value": -3.0,
                },
                {
                    "name": "hard_swish_at_boundary_pos3",
                    "shape": (2, 64, 256),
                    "input_value": 3.0,
                },
                {
                    "name": "hard_swish_negative",
                    "shape": (2, 64, 256),
                    "input_type": "negative",
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = config["shape"]
        input_type = config.get("input_type", "random")
        input_value = config.get("input_value")

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -5.0)
        elif input_value is not None:
            x = torch.full(shape, input_value)
        else:
            x = torch.randn(shape)

        # HardSwish
        out = F.hardswish(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )


class HardSigmoidGenerator(GoldenGenerator):
    """Generate golden files for HardSigmoid activation.

    HardSigmoid: relu6(x + 3) / 6

    Efficient approximation of sigmoid.
    """

    @property
    def name(self) -> str:
        return "hard_sigmoid"

    def get_tolerance_config(self) -> ToleranceConfig:
        # HardSigmoid is piecewise linear, should be exact
        return TOLERANCE_CONFIGS["activations_exact"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"hard_sigmoid_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "hard_sigmoid_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "hard_sigmoid_at_boundary_neg3",
                    "shape": (2, 64, 256),
                    "input_value": -3.0,
                },
                {
                    "name": "hard_sigmoid_at_boundary_pos3",
                    "shape": (2, 64, 256),
                    "input_value": 3.0,
                },
                {
                    "name": "hard_sigmoid_very_negative",
                    "shape": (2, 64, 256),
                    "input_value": -10.0,
                },
                {
                    "name": "hard_sigmoid_very_positive",
                    "shape": (2, 64, 256),
                    "input_value": 10.0,
                },
            ]
        )

        return configs

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = config["shape"]
        input_type = config.get("input_type", "random")
        input_value = config.get("input_value")

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_value is not None:
            x = torch.full(shape, input_value)
        else:
            x = torch.randn(shape)

        # HardSigmoid
        out = F.hardsigmoid(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )
