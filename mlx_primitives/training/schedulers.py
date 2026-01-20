"""Learning rate schedulers for MLX.

This module provides learning rate schedulers commonly used in deep learning:
- CosineAnnealingLR: Cosine annealing with optional restarts
- WarmupCosineScheduler: Linear warmup followed by cosine decay
- OneCycleLR: Super-convergence one-cycle policy
- PolynomialDecayLR: Polynomial learning rate decay
- InverseSqrtScheduler: Inverse square root decay (transformers)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional


class LRScheduler(ABC):
    """Base class for learning rate schedulers.

    All schedulers maintain an internal step counter and can return
    the learning rate for any step.

    Args:
        base_lr: Initial/base learning rate.

    Raises:
        ValueError: If base_lr is not positive.
    """

    def __init__(self, base_lr: float):
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}")
        self.base_lr = base_lr
        self._step = 0

    @abstractmethod
    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for a given step.

        Args:
            step: Step number. If None, uses internal counter.

        Returns:
            Learning rate for the step.
        """
        pass

    def step(self) -> float:
        """Advance internal step counter and return current LR.

        Returns:
            Learning rate after stepping.
        """
        lr = self.get_lr(self._step)
        self._step += 1
        return lr

    def set_step(self, step: int) -> None:
        """Set the internal step counter.

        Args:
            step: Step number to set.
        """
        self._step = step

    @property
    def current_step(self) -> int:
        """Get current step count."""
        return self._step


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    Decays learning rate following a cosine curve from base_lr to min_lr.
    Optionally supports warm restarts (SGDR).

    Args:
        base_lr: Initial learning rate.
        T_max: Maximum number of steps for one cycle.
        min_lr: Minimum learning rate (default: 0).
        T_mult: Multiplier for cycle length after each restart (default: 1).
        warmup_steps: Number of warmup steps at the start (default: 0).

    Reference:
        "SGDR: Stochastic Gradient Descent with Warm Restarts"
        https://arxiv.org/abs/1608.03983

    Example:
        >>> scheduler = CosineAnnealingLR(base_lr=1e-3, T_max=1000)
        >>> for step in range(1000):
        ...     lr = scheduler.step()
        ...     optimizer.learning_rate = lr
    """

    def __init__(
        self,
        base_lr: float,
        T_max: int,
        min_lr: float = 0.0,
        T_mult: float = 1.0,
        warmup_steps: int = 0,
    ):
        super().__init__(base_lr)
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        if T_mult < 1.0:
            raise ValueError(f"T_mult must be >= 1.0, got {T_mult}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        self.T_max = T_max
        self.min_lr = min_lr
        self.T_mult = T_mult
        self.warmup_steps = warmup_steps

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        # Handle warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        # Adjust step for warmup
        step = step - self.warmup_steps

        # Handle warm restarts with T_mult
        if self.T_mult == 1.0:
            # No restarts or constant cycle length
            cycle_step = step % self.T_max
            T_cur = self.T_max
        else:
            # Variable cycle length with restarts
            # Find which cycle we're in and the step within that cycle
            T_cur = self.T_max
            cycle_start = 0
            while step >= cycle_start + T_cur:
                cycle_start += T_cur
                T_cur = int(T_cur * self.T_mult)
            cycle_step = step - cycle_start

        # Cosine annealing formula
        cos_value = math.cos(math.pi * cycle_step / T_cur)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos_value)


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay.

    Commonly used scheduler that linearly increases learning rate during
    warmup, then decays following a cosine curve.

    Args:
        base_lr: Peak learning rate (reached after warmup).
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay (default: 0).
        warmup_init_lr: Initial learning rate at step 0 (default: 0).

    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     base_lr=3e-4,
        ...     warmup_steps=1000,
        ...     total_steps=100000,
        ... )
        >>> for step in range(100000):
        ...     lr = scheduler.step()
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_init_lr: float = 0.0,
    ):
        super().__init__(base_lr)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
            )
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return self.warmup_init_lr + progress * (self.base_lr - self.warmup_init_lr)

        # Cosine decay phase
        if step >= self.total_steps:
            return self.min_lr

        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        progress = decay_step / decay_steps

        cos_value = math.cos(math.pi * progress)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos_value)


class OneCycleLR(LRScheduler):
    """One-cycle learning rate scheduler.

    Implements the 1cycle policy from Leslie Smith's paper for super-convergence.
    The learning rate increases from initial_lr to max_lr in the first phase,
    then decreases to min_lr in the second phase. Optionally includes a final
    annihilation phase.

    Args:
        max_lr: Maximum learning rate (peak of the cycle).
        total_steps: Total number of training steps.
        pct_start: Percentage of cycle spent increasing LR (default: 0.3).
        div_factor: Initial LR = max_lr / div_factor (default: 25).
        final_div_factor: Final LR = initial_lr / final_div_factor (default: 1e4).
        anneal_strategy: Annealing strategy, 'cos' or 'linear' (default: 'cos').

    Reference:
        "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
        https://arxiv.org/abs/1708.07120

    Example:
        >>> scheduler = OneCycleLR(
        ...     max_lr=1e-3,
        ...     total_steps=10000,
        ...     pct_start=0.3,
        ... )
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: str = "cos",
    ):
        if max_lr <= 0:
            raise ValueError(f"max_lr must be positive, got {max_lr}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not (0.0 < pct_start < 1.0):
            raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")
        if div_factor <= 0:
            raise ValueError(f"div_factor must be positive, got {div_factor}")
        if final_div_factor <= 0:
            raise ValueError(f"final_div_factor must be positive, got {final_div_factor}")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy}")

        initial_lr = max_lr / div_factor
        super().__init__(initial_lr)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy

        # Derived values
        self.initial_lr = initial_lr
        self.final_lr = initial_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

    def _anneal(self, start: float, end: float, pct: float) -> float:
        """Anneal from start to end given percentage progress."""
        if self.anneal_strategy == "cos":
            cos_value = math.cos(math.pi * pct)
            return end + 0.5 * (start - end) * (1 + cos_value)
        else:  # linear
            return start + pct * (end - start)

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        if step >= self.total_steps:
            return self.final_lr

        if step < self.step_up:
            # Increasing phase
            pct = step / self.step_up
            return self._anneal(self.initial_lr, self.max_lr, pct)
        else:
            # Decreasing phase
            pct = (step - self.step_up) / self.step_down
            return self._anneal(self.max_lr, self.final_lr, pct)


class PolynomialDecayLR(LRScheduler):
    """Polynomial learning rate decay.

    Decays learning rate using a polynomial function.

    Formula: lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr

    Args:
        base_lr: Initial learning rate.
        total_steps: Total number of training steps.
        end_lr: Final learning rate (default: 0).
        power: Power of the polynomial (default: 1.0 for linear decay).
        warmup_steps: Number of warmup steps (default: 0).

    Example:
        >>> scheduler = PolynomialDecayLR(
        ...     base_lr=1e-3,
        ...     total_steps=10000,
        ...     power=2.0,  # Quadratic decay
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        end_lr: float = 0.0,
        power: float = 1.0,
        warmup_steps: int = 0,
    ):
        super().__init__(base_lr)
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if end_lr < 0:
            raise ValueError(f"end_lr must be non-negative, got {end_lr}")
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
            )
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        self.warmup_steps = warmup_steps

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        # Handle warmup
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        # Adjust step for warmup
        step = step - self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps

        if step >= decay_steps:
            return self.end_lr

        # Polynomial decay
        progress = step / decay_steps
        decay_factor = (1 - progress) ** self.power
        return (self.base_lr - self.end_lr) * decay_factor + self.end_lr


class InverseSqrtScheduler(LRScheduler):
    """Inverse square root learning rate scheduler.

    Decays learning rate proportional to 1/sqrt(step). Commonly used
    in transformer training (e.g., original "Attention is All You Need").

    Formula: lr = base_lr * sqrt(warmup_steps) / sqrt(step)

    During warmup: lr = base_lr * step / warmup_steps
    After warmup: lr = base_lr * sqrt(warmup_steps / step)

    Args:
        base_lr: Learning rate at the end of warmup.
        warmup_steps: Number of warmup steps.

    Reference:
        "Attention Is All You Need"
        https://arxiv.org/abs/1706.03762

    Example:
        >>> scheduler = InverseSqrtScheduler(
        ...     base_lr=1e-3,
        ...     warmup_steps=4000,
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
    ):
        super().__init__(base_lr)
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.warmup_steps = warmup_steps

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        # Avoid division by zero
        step = max(step, 1)

        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / self.warmup_steps

        # Inverse sqrt decay
        return self.base_lr * math.sqrt(self.warmup_steps / step)


class LinearWarmupLR(LRScheduler):
    """Simple linear warmup scheduler.

    Linearly increases learning rate from 0 to base_lr over warmup_steps,
    then maintains base_lr.

    Args:
        base_lr: Target learning rate after warmup.
        warmup_steps: Number of warmup steps.

    Example:
        >>> scheduler = LinearWarmupLR(base_lr=1e-3, warmup_steps=1000)
    """

    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__(base_lr)
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.warmup_steps = warmup_steps

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        return self.base_lr


class ConstantLR(LRScheduler):
    """Constant learning rate (no scheduling).

    Useful as a baseline or when combined with other scheduling strategies.

    Args:
        base_lr: The constant learning rate.

    Example:
        >>> scheduler = ConstantLR(base_lr=1e-3)
    """

    def __init__(self, base_lr: float):
        super().__init__(base_lr)

    def get_lr(self, step: Optional[int] = None) -> float:
        return self.base_lr


class ExponentialDecayLR(LRScheduler):
    """Exponential learning rate decay.

    Decays learning rate exponentially: lr = base_lr * decay_rate^(step/decay_steps)

    Args:
        base_lr: Initial learning rate.
        decay_rate: Multiplicative decay factor.
        decay_steps: Number of steps for one decay factor application.
        staircase: If True, decay at discrete intervals (default: False).
        min_lr: Minimum learning rate (default: 0).

    Example:
        >>> scheduler = ExponentialDecayLR(
        ...     base_lr=1e-3,
        ...     decay_rate=0.96,
        ...     decay_steps=1000,
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        decay_rate: float,
        decay_steps: int,
        staircase: bool = False,
        min_lr: float = 0.0,
    ):
        super().__init__(base_lr)
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {decay_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.min_lr = min_lr

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        if self.staircase:
            exponent = step // self.decay_steps
        else:
            exponent = step / self.decay_steps

        lr = self.base_lr * (self.decay_rate ** exponent)
        return max(lr, self.min_lr)


class MultiStepLR(LRScheduler):
    """Multi-step learning rate scheduler.

    Decays learning rate by gamma at each milestone.

    Args:
        base_lr: Initial learning rate.
        milestones: List of step indices at which to decay.
        gamma: Multiplicative decay factor (default: 0.1).

    Example:
        >>> scheduler = MultiStepLR(
        ...     base_lr=1e-3,
        ...     milestones=[30000, 60000, 90000],
        ...     gamma=0.1,
        ... )
    """

    def __init__(
        self,
        base_lr: float,
        milestones: list[int],
        gamma: float = 0.1,
    ):
        super().__init__(base_lr)
        if not milestones:
            raise ValueError("milestones list cannot be empty")
        if any(m < 0 for m in milestones):
            raise ValueError("all milestones must be non-negative")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        # Count how many milestones we've passed
        num_decays = sum(1 for m in self.milestones if step >= m)

        return self.base_lr * (self.gamma ** num_decays)


class ChainedScheduler(LRScheduler):
    """Chain multiple schedulers together sequentially.

    Applies schedulers in sequence, switching to the next when the
    current one's steps are exhausted.

    Args:
        schedulers: List of (scheduler, num_steps) tuples.

    Example:
        >>> warmup = LinearWarmupLR(base_lr=1e-3, warmup_steps=1000)
        >>> cosine = CosineAnnealingLR(base_lr=1e-3, T_max=9000)
        >>> scheduler = ChainedScheduler([
        ...     (warmup, 1000),
        ...     (cosine, 9000),
        ... ])
    """

    def __init__(self, schedulers: list[tuple[LRScheduler, int]]):
        if not schedulers:
            raise ValueError("schedulers list cannot be empty")
        for i, (sched, steps) in enumerate(schedulers):
            if steps <= 0:
                raise ValueError(f"scheduler {i} has invalid steps: {steps}")
            if not isinstance(sched, LRScheduler):
                raise TypeError(f"scheduler {i} must be an LRScheduler, got {type(sched)}")

        # Use the first scheduler's base_lr
        super().__init__(schedulers[0][0].base_lr)
        self.schedulers = schedulers

        # Precompute cumulative steps
        self.cumulative_steps = []
        total = 0
        for _, steps in schedulers:
            total += steps
            self.cumulative_steps.append(total)

    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step

        # Find which scheduler to use
        prev_cumulative = 0
        for i, (scheduler, _) in enumerate(self.schedulers):
            if step < self.cumulative_steps[i]:
                local_step = step - prev_cumulative
                return scheduler.get_lr(local_step)
            prev_cumulative = self.cumulative_steps[i]

        # Past all schedulers, return last scheduler's final value
        if self.schedulers:
            last_scheduler, last_steps = self.schedulers[-1]
            return last_scheduler.get_lr(last_steps - 1)

        return self.base_lr
