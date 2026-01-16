"""Learning rate scheduler generators."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim

from base import GoldenGenerator, TestConfig, ToleranceConfig
from config import TOLERANCE_CONFIGS


class CosineAnnealingLRGenerator(GoldenGenerator):
    """Generate golden files for Cosine Annealing LR scheduler."""

    @property
    def name(self) -> str:
        return "cosine_annealing_lr"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "cosine_short", "base_lr": 1e-3, "T_max": 100, "eta_min": 0},
            {"name": "cosine_long", "base_lr": 1e-3, "T_max": 1000, "eta_min": 0},
            {"name": "cosine_with_min", "base_lr": 1e-3, "T_max": 100, "eta_min": 1e-5},
            {"name": "cosine_high_lr", "base_lr": 1e-2, "T_max": 200, "eta_min": 1e-4},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        base_lr = config["base_lr"]
        T_max = config["T_max"]
        eta_min = config["eta_min"]

        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=base_lr)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

        # Collect LR values at each step
        lr_values = []
        for step in range(T_max + 10):  # Go a bit past T_max
            lr_values.append(scheduler.get_last_lr()[0])
            scheduler.step()

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(len(lr_values))},
            params={"base_lr": base_lr, "T_max": T_max, "eta_min": eta_min},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": len(lr_values)},
        )


class WarmupCosineSchedulerGenerator(GoldenGenerator):
    """Generate golden files for Warmup + Cosine scheduler."""

    @property
    def name(self) -> str:
        return "warmup_cosine_scheduler"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "warmup_cosine_short", "base_lr": 1e-3, "warmup_steps": 100, "total_steps": 1000},
            {"name": "warmup_cosine_long", "base_lr": 1e-3, "warmup_steps": 500, "total_steps": 5000},
            {"name": "warmup_cosine_quick", "base_lr": 5e-4, "warmup_steps": 50, "total_steps": 500},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        base_lr = config["base_lr"]
        warmup_steps = config["warmup_steps"]
        total_steps = config["total_steps"]

        def get_lr(step):
            if step < warmup_steps:
                # Linear warmup
                return base_lr * step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        lr_values = [get_lr(step) for step in range(total_steps)]

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(total_steps)},
            params={"base_lr": base_lr, "warmup_steps": warmup_steps, "total_steps": total_steps},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": total_steps},
        )


class OneCycleLRGenerator(GoldenGenerator):
    """Generate golden files for OneCycle LR scheduler."""

    @property
    def name(self) -> str:
        return "one_cycle_lr"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "onecycle_default", "max_lr": 1e-3, "total_steps": 1000, "pct_start": 0.3},
            {"name": "onecycle_fast_warmup", "max_lr": 1e-3, "total_steps": 1000, "pct_start": 0.1},
            {"name": "onecycle_slow_warmup", "max_lr": 1e-3, "total_steps": 1000, "pct_start": 0.5},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        max_lr = config["max_lr"]
        total_steps = config["total_steps"]
        pct_start = config["pct_start"]

        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=max_lr)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
        )

        # Collect LR values at each step
        lr_values = []
        for step in range(total_steps):
            lr_values.append(scheduler.get_last_lr()[0])
            scheduler.step()

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(total_steps)},
            params={"max_lr": max_lr, "total_steps": total_steps, "pct_start": pct_start},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": total_steps},
        )


class PolynomialDecayLRGenerator(GoldenGenerator):
    """Generate golden files for Polynomial Decay LR scheduler."""

    @property
    def name(self) -> str:
        return "polynomial_decay_lr"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "poly_linear", "base_lr": 1e-3, "total_steps": 1000, "power": 1.0, "end_lr": 0},
            {"name": "poly_quadratic", "base_lr": 1e-3, "total_steps": 1000, "power": 2.0, "end_lr": 0},
            {"name": "poly_sqrt", "base_lr": 1e-3, "total_steps": 1000, "power": 0.5, "end_lr": 1e-5},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        base_lr = config["base_lr"]
        total_steps = config["total_steps"]
        power = config["power"]
        end_lr = config["end_lr"]

        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=base_lr)

        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_steps,
            power=power,
        )

        # Collect LR values at each step
        lr_values = []
        for step in range(total_steps + 10):
            lr_values.append(scheduler.get_last_lr()[0])
            scheduler.step()

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(len(lr_values))},
            params={"base_lr": base_lr, "total_steps": total_steps, "power": power, "end_lr": end_lr},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": len(lr_values)},
        )


class MultiStepLRGenerator(GoldenGenerator):
    """Generate golden files for MultiStep LR scheduler."""

    @property
    def name(self) -> str:
        return "multi_step_lr"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "multistep_2", "base_lr": 1e-3, "milestones": [30, 60], "gamma": 0.1, "total_steps": 100},
            {"name": "multistep_3", "base_lr": 1e-3, "milestones": [30, 60, 90], "gamma": 0.1, "total_steps": 120},
            {"name": "multistep_gentle", "base_lr": 1e-3, "milestones": [50, 100], "gamma": 0.5, "total_steps": 150},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        base_lr = config["base_lr"]
        milestones = config["milestones"]
        gamma = config["gamma"]
        total_steps = config["total_steps"]

        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=base_lr)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

        # Collect LR values at each step
        lr_values = []
        for step in range(total_steps):
            lr_values.append(scheduler.get_last_lr()[0])
            scheduler.step()

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(total_steps)},
            params={"base_lr": base_lr, "milestones": milestones, "gamma": gamma},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": total_steps},
        )


class InverseSqrtSchedulerGenerator(GoldenGenerator):
    """Generate golden files for Inverse Square Root scheduler.

    Used in original Transformer paper:
    lr = base_lr * min(step^-0.5, step * warmup_steps^-1.5)
    """

    @property
    def name(self) -> str:
        return "inverse_sqrt_scheduler"

    def get_tolerance_config(self) -> ToleranceConfig:
        return TOLERANCE_CONFIGS["schedulers"]

    def get_test_configs(self) -> List[Dict[str, Any]]:
        return [
            {"name": "inv_sqrt_short", "base_lr": 1e-3, "warmup_steps": 4000, "total_steps": 10000},
            {"name": "inv_sqrt_long", "base_lr": 1e-3, "warmup_steps": 4000, "total_steps": 100000},
            {"name": "inv_sqrt_quick_warmup", "base_lr": 5e-4, "warmup_steps": 1000, "total_steps": 20000},
        ]

    def generate_pytorch_reference(self, config: Dict[str, Any]) -> TestConfig:
        base_lr = config["base_lr"]
        warmup_steps = config["warmup_steps"]
        total_steps = config["total_steps"]

        def get_lr(step):
            # Avoid division by zero at step 0
            step = max(step, 1)
            return base_lr * min(step ** (-0.5), step * warmup_steps ** (-1.5))

        lr_values = [get_lr(step) for step in range(total_steps)]

        return TestConfig(
            name=config["name"],
            inputs={"steps": np.arange(total_steps)},
            params={"base_lr": base_lr, "warmup_steps": warmup_steps},
            expected_outputs={"lr_values": np.array(lr_values)},
            metadata={"total_steps": total_steps},
        )
