"""GELU variant generators: GELU exact, GELU tanh, QuickGELU."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS, STANDARD_SHAPES


class GELUGenerator(GoldenGenerator):
    """Generate golden files for exact GELU activation.

    GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    """

    @property
    def name(self) -> str:
        return "gelu"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_transcendental"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"gelu_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "gelu_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "gelu_negative",
                    "shape": (2, 64, 256),
                    "input_type": "negative",
                },
                {
                    "name": "gelu_large_positive",
                    "shape": (2, 64, 256),
                    "input_type": "large",
                },
                {
                    "name": "gelu_large_negative",
                    "shape": (2, 64, 256),
                    "input_value": -10.0,
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
            x = torch.full(shape, -1.0)
        elif input_type == "large":
            x = torch.full(shape, 10.0)
        elif input_value is not None:
            x = torch.full(shape, input_value)
        else:
            x = torch.randn(shape)

        # Exact GELU (not tanh approximation)
        out = F.gelu(x, approximate="none")

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )


class GELUTanhGenerator(GoldenGenerator):
    """Generate golden files for GELU with tanh approximation.

    GELU_tanh: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    @property
    def name(self) -> str:
        return "gelu_tanh"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_transcendental"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"gelu_tanh_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "gelu_tanh_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "gelu_tanh_negative",
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

        if input_type == "zeros":
            x = torch.zeros(shape)
        elif input_type == "negative":
            x = torch.full(shape, -1.0)
        else:
            x = torch.randn(shape)

        # Tanh approximation of GELU
        out = F.gelu(x, approximate="tanh")

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )


class QuickGELUGenerator(GoldenGenerator):
    """Generate golden files for QuickGELU activation.

    QuickGELU: x * sigmoid(1.702 * x)

    Used in OpenAI CLIP and other models.
    """

    @property
    def name(self) -> str:
        return "quick_gelu"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["activations_transcendental"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        configs = []

        for size_name, shape in STANDARD_SHAPES.items():
            configs.append(
                {
                    "name": f"quick_gelu_{size_name}",
                    "shape": (shape["batch"], shape["seq"], shape["dims"]),
                }
            )

        # Edge cases
        configs.extend(
            [
                {
                    "name": "quick_gelu_zeros",
                    "shape": (2, 64, 256),
                    "input_type": "zeros",
                },
                {
                    "name": "quick_gelu_negative",
                    "shape": (2, 64, 256),
                    "input_type": "negative",
                },
                {
                    "name": "quick_gelu_large",
                    "shape": (2, 64, 256),
                    "input_type": "large",
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
        elif input_type == "large":
            x = torch.full(shape, 10.0)
        else:
            x = torch.randn(shape)

        # QuickGELU: x * sigmoid(1.702 * x)
        out = x * torch.sigmoid(1.702 * x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={},
            expected_outputs={"out": out.numpy()},
            metadata={"shape": shape, "input_type": input_type},
        )
