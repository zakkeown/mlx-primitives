"""EMA (Exponential Moving Average) generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class EMAGenerator(GoldenGenerator):
    """Generate golden files for Exponential Moving Average.

    EMA update: ema = decay * ema + (1 - decay) * current
    """

    @property
    def name(self) -> str:
        return "ema"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["elementwise"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "ema_standard", "shape": (256, 512), "decay": 0.999, "num_updates": 100},
            {"name": "ema_fast", "shape": (256, 512), "decay": 0.99, "num_updates": 100},
            {"name": "ema_slow", "shape": (256, 512), "decay": 0.9999, "num_updates": 100},
            {"name": "ema_large", "shape": (1024, 2048), "decay": 0.999, "num_updates": 50},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = tuple(config["shape"])
        decay = config["decay"]
        num_updates = config["num_updates"]

        # Initial parameter
        initial_param = torch.randn(shape)
        ema_param = initial_param.clone()

        # Store trajectory
        all_params = [initial_param.numpy().copy()]
        all_ema = [ema_param.numpy().copy()]

        # Simulate updates
        for _ in range(num_updates):
            # New parameter value (simulating training update)
            new_param = initial_param + torch.randn(shape) * 0.01

            # EMA update
            ema_param = decay * ema_param + (1 - decay) * new_param

            all_params.append(new_param.numpy().copy())
            all_ema.append(ema_param.numpy().copy())

            initial_param = new_param

        return TestConfig(
            name=config["name"],
            inputs={"initial_param": all_params[0]},
            params={"decay": decay, "num_updates": num_updates},
            expected_outputs={
                "final_ema": all_ema[-1],
                "final_param": all_params[-1],
            },
            metadata={"shape": shape},
        )


class EMAWithWarmupGenerator(GoldenGenerator):
    """Generate golden files for EMA with warmup.

    During warmup, decay ramps up from 0 to target decay.
    decay_t = min(decay, (1 + t) / (10 + t))
    """

    @property
    def name(self) -> str:
        return "ema_with_warmup"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["elementwise"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "ema_warmup_standard", "shape": (256, 512), "decay": 0.999, "warmup_steps": 10, "num_updates": 100},
            {"name": "ema_warmup_long", "shape": (256, 512), "decay": 0.999, "warmup_steps": 50, "num_updates": 100},
            {"name": "ema_warmup_short", "shape": (256, 512), "decay": 0.9999, "warmup_steps": 5, "num_updates": 100},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        torch.manual_seed(self.seed)

        shape = tuple(config["shape"])
        target_decay = config["decay"]
        warmup_steps = config["warmup_steps"]
        num_updates = config["num_updates"]

        # Initial parameter
        initial_param = torch.randn(shape)
        ema_param = initial_param.clone()

        # Store decay values
        decay_values = []

        # Simulate updates
        for t in range(num_updates):
            # Warmup decay
            decay = min(target_decay, (1 + t) / (warmup_steps + t))
            decay_values.append(decay)

            # New parameter value
            new_param = initial_param + torch.randn(shape) * 0.01

            # EMA update
            ema_param = decay * ema_param + (1 - decay) * new_param

            initial_param = new_param

        return TestConfig(
            name=config["name"],
            inputs={"initial_param": torch.randn(shape).numpy()},
            params={"decay": target_decay, "warmup_steps": warmup_steps, "num_updates": num_updates},
            expected_outputs={
                "final_ema": ema_param.numpy(),
                "decay_values": np.array(decay_values),
            },
            metadata={"shape": shape},
        )
