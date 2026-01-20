"""NumPy reference implementations for training utilities."""

import numpy as np


def cosine_annealing_lr(
    steps: np.ndarray,
    base_lr: float,
    total_steps: int,
    min_lr: float = 0.0,
) -> np.ndarray:
    """Cosine Annealing learning rate schedule.

    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / total_steps))

    Args:
        steps: Step numbers to compute LR for
        base_lr: Base (maximum) learning rate
        total_steps: Total number of training steps
        min_lr: Minimum learning rate

    Returns:
        Learning rates for each step
    """
    steps = np.asarray(steps, dtype=np.float32)
    progress = np.clip(steps / total_steps, 0, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def warmup_cosine_lr(
    steps: np.ndarray,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> np.ndarray:
    """Warmup + Cosine Annealing learning rate schedule.

    Linear warmup followed by cosine decay.

    Args:
        steps: Step numbers to compute LR for
        base_lr: Base (maximum) learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate

    Returns:
        Learning rates for each step
    """
    steps = np.asarray(steps, dtype=np.float32)
    lr_values = np.zeros_like(steps)

    # Warmup phase
    warmup_mask = steps < warmup_steps
    lr_values[warmup_mask] = base_lr * steps[warmup_mask] / warmup_steps

    # Cosine phase
    cosine_mask = ~warmup_mask
    progress = (steps[cosine_mask] - warmup_steps) / (total_steps - warmup_steps)
    progress = np.clip(progress, 0, 1)
    lr_values[cosine_mask] = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

    return lr_values


def polynomial_decay_lr(
    steps: np.ndarray,
    base_lr: float,
    total_steps: int,
    end_lr: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Polynomial Decay learning rate schedule.

    lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr

    Args:
        steps: Step numbers to compute LR for
        base_lr: Base (maximum) learning rate
        total_steps: Total number of training steps
        end_lr: Final learning rate
        power: Polynomial power (1.0 = linear decay)

    Returns:
        Learning rates for each step
    """
    steps = np.asarray(steps, dtype=np.float32)
    progress = np.clip(steps / total_steps, 0, 1)
    return (base_lr - end_lr) * (1 - progress) ** power + end_lr


def multistep_lr(
    steps: np.ndarray,
    base_lr: float,
    milestones: list,
    gamma: float = 0.1,
) -> np.ndarray:
    """Multi-Step learning rate schedule.

    Multiplies learning rate by gamma at each milestone.

    Args:
        steps: Step numbers to compute LR for
        base_lr: Base learning rate
        milestones: List of step numbers where to decay
        gamma: Decay factor

    Returns:
        Learning rates for each step
    """
    steps = np.asarray(steps, dtype=np.float32)
    lr_values = np.full_like(steps, base_lr)

    milestones = sorted(milestones)

    for milestone in milestones:
        lr_values = np.where(steps >= milestone, lr_values * gamma, lr_values)

    return lr_values


def inverse_sqrt_lr(
    steps: np.ndarray,
    base_lr: float,
    warmup_steps: int,
) -> np.ndarray:
    """Inverse Square Root learning rate schedule.

    Common in transformer training (e.g., original Transformer paper).

    lr = base_lr * min(1/sqrt(step), step * warmup_steps^(-1.5))

    Args:
        steps: Step numbers to compute LR for
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps

    Returns:
        Learning rates for each step
    """
    steps = np.asarray(steps, dtype=np.float32)
    # Avoid division by zero at step 0
    steps = np.maximum(steps, 1)

    inv_sqrt = 1.0 / np.sqrt(steps)
    warmup_factor = steps * (warmup_steps ** -1.5)

    return base_lr * np.minimum(inv_sqrt, warmup_factor)


def ema_update(
    ema_params: dict,
    model_params: dict,
    decay: float,
) -> dict:
    """Exponential Moving Average update.

    ema_param = decay * ema_param + (1 - decay) * model_param

    Args:
        ema_params: Dictionary of EMA parameter arrays
        model_params: Dictionary of model parameter arrays
        decay: EMA decay factor (e.g., 0.999)

    Returns:
        Updated EMA parameters dictionary
    """
    updated = {}
    for name in ema_params:
        if name in model_params:
            updated[name] = decay * ema_params[name] + (1 - decay) * model_params[name]
        else:
            updated[name] = ema_params[name]
    return updated


def ema_with_warmup_decay(
    step: int,
    base_decay: float = 0.9999,
    warmup_steps: int = 1000,
    power: float = 1.0,
) -> float:
    """Compute EMA decay with warmup.

    Gradually increases decay from 0 to base_decay during warmup.

    Args:
        step: Current training step
        base_decay: Target decay value
        warmup_steps: Number of steps to warm up decay
        power: Power for warmup schedule

    Returns:
        Current decay value
    """
    if step < warmup_steps:
        progress = step / warmup_steps
        return base_decay * (progress ** power)
    return base_decay
