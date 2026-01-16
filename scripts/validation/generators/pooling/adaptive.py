"""Adaptive pooling generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class AdaptiveAvgPool1dGenerator(GoldenGenerator):
    """Generate golden files for AdaptiveAvgPool1d."""

    @property
    def name(self) -> str:
        return "adaptive_avg_pool1d"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "adaptive_avgpool1d_to1", "batch": 2, "channels": 64, "length": 128, "output_size": 1},
            {"name": "adaptive_avgpool1d_to8", "batch": 2, "channels": 64, "length": 128, "output_size": 8},
            {"name": "adaptive_avgpool1d_to32", "batch": 2, "channels": 64, "length": 128, "output_size": 32},
            {"name": "adaptive_avgpool1d_same", "batch": 2, "channels": 64, "length": 64, "output_size": 64},
            {"name": "adaptive_avgpool1d_large", "batch": 4, "channels": 128, "length": 512, "output_size": 16},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        length = config["length"]
        output_size = config["output_size"]

        x = torch.randn(batch, channels, length)
        pool = nn.AdaptiveAvgPool1d(output_size)
        out = pool(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"output_size": output_size},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "length": length},
        )


class AdaptiveAvgPool2dGenerator(GoldenGenerator):
    """Generate golden files for AdaptiveAvgPool2d."""

    @property
    def name(self) -> str:
        return "adaptive_avg_pool2d"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "adaptive_avgpool2d_to1x1", "batch": 2, "channels": 64, "height": 32, "width": 32, "output_size": (1, 1)},
            {"name": "adaptive_avgpool2d_to7x7", "batch": 2, "channels": 64, "height": 32, "width": 32, "output_size": (7, 7)},
            {"name": "adaptive_avgpool2d_to14x14", "batch": 2, "channels": 64, "height": 56, "width": 56, "output_size": (14, 14)},
            {"name": "adaptive_avgpool2d_asymmetric", "batch": 2, "channels": 64, "height": 64, "width": 32, "output_size": (8, 4)},
            {"name": "adaptive_avgpool2d_large", "batch": 4, "channels": 256, "height": 128, "width": 128, "output_size": (1, 1)},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        output_size = config["output_size"]

        x = torch.randn(batch, channels, height, width)
        pool = nn.AdaptiveAvgPool2d(output_size)
        out = pool(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"output_size": list(output_size)},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "height": height, "width": width},
        )


class AdaptiveMaxPool1dGenerator(GoldenGenerator):
    """Generate golden files for AdaptiveMaxPool1d."""

    @property
    def name(self) -> str:
        return "adaptive_max_pool1d"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "adaptive_maxpool1d_to1", "batch": 2, "channels": 64, "length": 128, "output_size": 1},
            {"name": "adaptive_maxpool1d_to8", "batch": 2, "channels": 64, "length": 128, "output_size": 8},
            {"name": "adaptive_maxpool1d_to32", "batch": 2, "channels": 64, "length": 128, "output_size": 32},
            {"name": "adaptive_maxpool1d_same", "batch": 2, "channels": 64, "length": 64, "output_size": 64},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        length = config["length"]
        output_size = config["output_size"]

        x = torch.randn(batch, channels, length)
        pool = nn.AdaptiveMaxPool1d(output_size)
        out = pool(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"output_size": output_size},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "length": length},
        )


class AdaptiveMaxPool2dGenerator(GoldenGenerator):
    """Generate golden files for AdaptiveMaxPool2d."""

    @property
    def name(self) -> str:
        return "adaptive_max_pool2d"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["pooling"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "adaptive_maxpool2d_to1x1", "batch": 2, "channels": 64, "height": 32, "width": 32, "output_size": (1, 1)},
            {"name": "adaptive_maxpool2d_to7x7", "batch": 2, "channels": 64, "height": 32, "width": 32, "output_size": (7, 7)},
            {"name": "adaptive_maxpool2d_to14x14", "batch": 2, "channels": 64, "height": 56, "width": 56, "output_size": (14, 14)},
            {"name": "adaptive_maxpool2d_asymmetric", "batch": 2, "channels": 64, "height": 64, "width": 32, "output_size": (8, 4)},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        output_size = config["output_size"]

        x = torch.randn(batch, channels, height, width)
        pool = nn.AdaptiveMaxPool2d(output_size)
        out = pool(x)

        return TestConfig(
            name=config["name"],
            inputs={"x": x.numpy()},
            params={"output_size": list(output_size)},
            expected_outputs={"out": out.numpy()},
            metadata={"batch": batch, "channels": channels, "height": height, "width": width},
        )
