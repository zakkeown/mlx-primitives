"""Training utilities for MLX.

This module provides utilities commonly used during training:
- EMA: Exponential Moving Average of model weights
- GradientAccumulator: Accumulate gradients over multiple steps
- GradientClipper: Clip gradients by norm or value
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average of the training parameters. The EMA
    parameters often provide better generalization.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (default: 0.9999).
        update_after_step: Start EMA updates after this many steps (default: 0).
        update_every: Update EMA every N steps (default: 1).

    Example:
        >>> model = nn.Linear(768, 768)
        >>> ema = EMA(model, decay=0.9999)
        >>> for step, batch in enumerate(dataloader):
        ...     # Train step
        ...     loss, grads = loss_fn(model, batch)
        ...     optimizer.update(model, grads)
        ...     # Update EMA
        ...     ema.update(step)
        >>> # For evaluation, use EMA weights
        >>> ema.apply_shadow()
        >>> eval_output = model(eval_batch)
        >>> ema.restore()  # Restore training weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every

        # Store shadow parameters (EMA weights)
        self.shadow_params = self._copy_params(model.parameters())

        # Store original parameters for restore
        self.backup_params: Optional[Dict[str, Any]] = None

    def _copy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy parameter dictionary."""

        def copy_recursive(p: Any) -> Any:
            if isinstance(p, mx.array):
                return mx.array(p)
            elif isinstance(p, dict):
                return {k: copy_recursive(v) for k, v in p.items()}
            elif isinstance(p, list):
                return [copy_recursive(v) for v in p]
            else:
                return p

        return copy_recursive(params)

    def _ema_update(
        self, shadow: Dict[str, Any], current: Dict[str, Any], decay: float
    ) -> Dict[str, Any]:
        """Recursively update shadow params with EMA."""

        def update_recursive(s: Any, c: Any) -> Any:
            if isinstance(s, mx.array):
                return decay * s + (1 - decay) * c
            elif isinstance(s, dict):
                return {k: update_recursive(s[k], c[k]) for k in s.keys()}
            elif isinstance(s, list):
                return [update_recursive(sv, cv) for sv, cv in zip(s, c)]
            else:
                return s

        return update_recursive(shadow, current)

    def get_decay(self, step: int) -> float:
        """Get decay rate for current step (can implement warmup)."""
        return self.decay

    def update(self, step: int) -> None:
        """Update EMA parameters.

        Args:
            step: Current training step.
        """
        if step < self.update_after_step:
            return

        if step % self.update_every != 0:
            return

        decay = self.get_decay(step)
        current_params = self.model.parameters()
        self.shadow_params = self._ema_update(self.shadow_params, current_params, decay)

    def apply_shadow(self) -> None:
        """Apply EMA weights to model (backup current weights first)."""
        self.backup_params = self._copy_params(self.model.parameters())
        self.model.update(self.shadow_params)

    def restore(self) -> None:
        """Restore original training weights after apply_shadow()."""
        if self.backup_params is not None:
            self.model.update(self.backup_params)
            self.backup_params = None

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights to another model instance.

        Args:
            model: Model to copy EMA weights to.
        """
        model.update(self.shadow_params)

    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state for checkpointing."""
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.shadow_params = state["shadow_params"]
        if "decay" in state:
            self.decay = state["decay"]


class EMAWithWarmup(EMA):
    """EMA with decay warmup.

    Starts with a lower decay and increases to the target decay over
    warmup steps. This helps stabilize early training.

    Args:
        model: The model to track.
        decay: Target EMA decay rate.
        warmup_steps: Steps to warm up to target decay.
        min_decay: Initial decay rate (default: 0.0).
        update_after_step: Start EMA after this step (default: 0).
        update_every: Update frequency (default: 1).

    Example:
        >>> ema = EMAWithWarmup(model, decay=0.9999, warmup_steps=1000)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 1000,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        super().__init__(model, decay, update_after_step, update_every)
        self.warmup_steps = warmup_steps
        self.min_decay = min_decay

    def get_decay(self, step: int) -> float:
        """Get decay with warmup."""
        if step >= self.warmup_steps:
            return self.decay

        # Linear warmup of decay
        progress = step / self.warmup_steps
        return self.min_decay + progress * (self.decay - self.min_decay)


class GradientAccumulator:
    """Gradient accumulation helper.

    Accumulates gradients over multiple forward/backward passes before
    applying an optimizer step. Useful for simulating larger batch sizes.

    Args:
        accumulation_steps: Number of steps to accumulate before update.

    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=4)
        >>> for step, batch in enumerate(dataloader):
        ...     loss, grads = loss_fn(model, batch)
        ...     if accumulator.should_step(step):
        ...         avg_grads = accumulator.get_accumulated(normalize=True)
        ...         optimizer.update(model, avg_grads)
        ...         accumulator.reset()
        ...     else:
        ...         accumulator.accumulate(grads)
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads: Optional[Dict[str, Any]] = None
        self.num_accumulated = 0

    def _add_grads(
        self, accumulated: Optional[Dict[str, Any]], grads: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add gradients to accumulated gradients."""

        def add_recursive(a: Any, g: Any) -> Any:
            if g is None:
                return a
            if isinstance(g, mx.array):
                if a is None:
                    return g
                return a + g
            elif isinstance(g, dict):
                if a is None:
                    a = {}
                return {k: add_recursive(a.get(k), v) for k, v in g.items()}
            elif isinstance(g, list):
                if a is None:
                    a = [None] * len(g)
                return [add_recursive(av, gv) for av, gv in zip(a, g)]
            else:
                return a

        return add_recursive(accumulated, grads)

    def _scale_grads(self, grads: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """Scale gradients by a factor."""

        def scale_recursive(g: Any) -> Any:
            if g is None:
                return None
            if isinstance(g, mx.array):
                return g * scale
            elif isinstance(g, dict):
                return {k: scale_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [scale_recursive(v) for v in g]
            else:
                return g

        return scale_recursive(grads)

    def accumulate(self, grads: Dict[str, Any]) -> None:
        """Add gradients to accumulator.

        Args:
            grads: Gradients from backward pass.
        """
        self.accumulated_grads = self._add_grads(self.accumulated_grads, grads)
        self.num_accumulated += 1

    def should_step(self, step: int) -> bool:
        """Check if optimizer should step at this iteration.

        Args:
            step: Current training step (0-indexed).

        Returns:
            True if should apply optimizer update.
        """
        return (step + 1) % self.accumulation_steps == 0

    def get_accumulated(self, normalize: bool = True) -> Dict[str, Any]:
        """Get accumulated gradients.

        Args:
            normalize: If True, divide by number of accumulation steps.

        Returns:
            Accumulated (and optionally normalized) gradients.
        """
        if self.accumulated_grads is None:
            return {}

        if normalize and self.num_accumulated > 1:
            return self._scale_grads(
                self.accumulated_grads, 1.0 / self.num_accumulated
            )

        return self.accumulated_grads

    def reset(self) -> None:
        """Reset accumulator after optimizer step."""
        self.accumulated_grads = None
        self.num_accumulated = 0


class GradientClipper:
    """Gradient clipping utilities.

    Provides methods for clipping gradients by global norm or by value.

    Args:
        max_norm: Maximum gradient norm (for norm clipping).
        max_value: Maximum gradient value (for value clipping).
        norm_type: Norm type for norm clipping (default: 2.0).

    Example:
        >>> clipper = GradientClipper(max_norm=1.0)
        >>> loss, grads = loss_fn(model, batch)
        >>> grads, grad_norm = clipper.clip_by_norm(grads)
        >>> optimizer.update(model, grads)
    """

    def __init__(
        self,
        max_norm: Optional[float] = None,
        max_value: Optional[float] = None,
        norm_type: float = 2.0,
    ):
        self.max_norm = max_norm
        self.max_value = max_value
        self.norm_type = norm_type

    def _compute_grad_norm(
        self, grads: Dict[str, Any], norm_type: float
    ) -> mx.array:
        """Compute total gradient norm."""
        norms = []

        def collect_norms(g: Any) -> None:
            if g is None:
                return
            if isinstance(g, mx.array):
                if norm_type == float("inf"):
                    norms.append(mx.max(mx.abs(g)))
                else:
                    norms.append(mx.sum(mx.abs(g) ** norm_type))
            elif isinstance(g, dict):
                for v in g.values():
                    collect_norms(v)
            elif isinstance(g, list):
                for v in g:
                    collect_norms(v)

        collect_norms(grads)

        if not norms:
            return mx.array(0.0)

        if norm_type == float("inf"):
            return mx.max(mx.stack(norms))
        else:
            total_norm = mx.sum(mx.stack(norms))
            return mx.power(total_norm, 1.0 / norm_type)

    def _scale_grads(self, grads: Dict[str, Any], scale: mx.array) -> Dict[str, Any]:
        """Scale all gradients by a factor."""

        def scale_recursive(g: Any) -> Any:
            if g is None:
                return None
            if isinstance(g, mx.array):
                return g * scale
            elif isinstance(g, dict):
                return {k: scale_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [scale_recursive(v) for v in g]
            else:
                return g

        return scale_recursive(grads)

    def clip_by_norm(
        self, grads: Dict[str, Any], max_norm: Optional[float] = None
    ) -> tuple[Dict[str, Any], mx.array]:
        """Clip gradients by global norm.

        Args:
            grads: Gradient dictionary.
            max_norm: Maximum norm (uses instance default if None).

        Returns:
            Tuple of (clipped gradients, original gradient norm).
        """
        max_norm = max_norm if max_norm is not None else self.max_norm
        if max_norm is None:
            raise ValueError("max_norm must be specified")

        grad_norm = self._compute_grad_norm(grads, self.norm_type)

        # Compute clip coefficient
        clip_coef = max_norm / (grad_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        clipped_grads = self._scale_grads(grads, clip_coef)
        return clipped_grads, grad_norm

    def clip_by_value(
        self, grads: Dict[str, Any], max_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clip gradients by value.

        Args:
            grads: Gradient dictionary.
            max_value: Maximum absolute value (uses instance default if None).

        Returns:
            Clipped gradients.
        """
        max_value = max_value if max_value is not None else self.max_value
        if max_value is None:
            raise ValueError("max_value must be specified")

        def clip_recursive(g: Any) -> Any:
            if g is None:
                return None
            if isinstance(g, mx.array):
                return mx.clip(g, -max_value, max_value)
            elif isinstance(g, dict):
                return {k: clip_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [clip_recursive(v) for v in g]
            else:
                return g

        return clip_recursive(grads)


def compute_gradient_norm(
    grads: Dict[str, Any], norm_type: float = 2.0
) -> mx.array:
    """Compute total gradient norm.

    Args:
        grads: Gradient dictionary.
        norm_type: Type of norm to compute.

    Returns:
        Total gradient norm.
    """
    clipper = GradientClipper(norm_type=norm_type)
    return clipper._compute_grad_norm(grads, norm_type)


def clip_grad_norm(
    grads: Dict[str, Any], max_norm: float, norm_type: float = 2.0
) -> tuple[Dict[str, Any], mx.array]:
    """Clip gradients by global norm (functional interface).

    Args:
        grads: Gradient dictionary.
        max_norm: Maximum gradient norm.
        norm_type: Type of norm to compute.

    Returns:
        Tuple of (clipped gradients, original gradient norm).
    """
    clipper = GradientClipper(max_norm=max_norm, norm_type=norm_type)
    return clipper.clip_by_norm(grads)


def clip_grad_value(grads: Dict[str, Any], max_value: float) -> Dict[str, Any]:
    """Clip gradients by value (functional interface).

    Args:
        grads: Gradient dictionary.
        max_value: Maximum absolute gradient value.

    Returns:
        Clipped gradients.
    """
    clipper = GradientClipper(max_value=max_value)
    return clipper.clip_by_value(grads)


class MixedPrecisionManager:
    """Mixed precision training manager.

    Handles automatic mixed precision (AMP) training with loss scaling
    to prevent underflow in fp16/bf16.

    Args:
        enabled: Whether mixed precision is enabled.
        dtype: Target dtype for mixed precision (float16 or bfloat16).
        init_scale: Initial loss scale.
        growth_factor: Factor to increase scale when no overflow.
        backoff_factor: Factor to decrease scale on overflow.
        growth_interval: Steps between scale increases.

    Example:
        >>> amp = MixedPrecisionManager(enabled=True, dtype=mx.float16)
        >>> with amp.autocast():
        ...     output = model(input)
        >>> scaled_loss = amp.scale_loss(loss)
        >>> grads = mx.grad(lambda: scaled_loss)(model.parameters())
        >>> grads = amp.unscale_grads(grads)
        >>> amp.update_scale(grads)
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: mx.Dtype = mx.float16,
        init_scale: float = 65536.0,
        loss_scale: Optional[float] = None,  # Alias for init_scale
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled
        self.dtype = dtype
        # Support both init_scale and loss_scale for compatibility
        self.scale = loss_scale if loss_scale is not None else init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        self._growth_tracker = 0
        self._found_inf = False

    def cast_forward(self, x: mx.array) -> mx.array:
        """Cast input to mixed precision dtype.

        Args:
            x: Input tensor.

        Returns:
            Tensor cast to mixed precision dtype.
        """
        if not self.enabled:
            return x
        return x.astype(self.dtype)

    def autocast_forward(self, fn: Callable, *args, **kwargs) -> Any:
        """Run function with mixed precision.

        Args:
            fn: Function to run.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function output.
        """
        if not self.enabled:
            return fn(*args, **kwargs)

        # Cast inputs to lower precision
        def cast_input(x: Any) -> Any:
            if isinstance(x, mx.array) and x.dtype == mx.float32:
                return x.astype(self.dtype)
            return x

        args = tuple(cast_input(a) for a in args)
        kwargs = {k: cast_input(v) for k, v in kwargs.items()}

        output = fn(*args, **kwargs)

        # Cast output back to float32 for loss computation
        if isinstance(output, mx.array):
            return output.astype(mx.float32)
        return output

    def scale_loss(self, loss: mx.array) -> mx.array:
        """Scale loss for mixed precision training.

        Args:
            loss: Original loss value.

        Returns:
            Scaled loss.
        """
        if not self.enabled:
            return loss
        return loss * self.scale

    def unscale_grads(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """Unscale gradients after backward pass.

        Args:
            grads: Scaled gradients.

        Returns:
            Unscaled gradients.
        """
        if not self.enabled:
            return grads

        inv_scale = 1.0 / self.scale

        def unscale_recursive(g: Any) -> Any:
            if g is None:
                return None
            if isinstance(g, mx.array):
                return g * inv_scale
            elif isinstance(g, dict):
                return {k: unscale_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [unscale_recursive(v) for v in g]
            return g

        return unscale_recursive(grads)

    def _check_for_inf(self, grads: Dict[str, Any]) -> bool:
        """Check if gradients contain inf or nan."""
        def check_recursive(g: Any) -> bool:
            if g is None:
                return False
            if isinstance(g, mx.array):
                return bool(mx.any(mx.isinf(g) | mx.isnan(g)))
            elif isinstance(g, dict):
                return any(check_recursive(v) for v in g.values())
            elif isinstance(g, list):
                return any(check_recursive(v) for v in g)
            return False

        return check_recursive(grads)

    def update_scale(self, grads: Dict[str, Any]) -> bool:
        """Update loss scale based on gradient overflow.

        Args:
            grads: Unscaled gradients.

        Returns:
            True if gradients are valid (no overflow).
        """
        if not self.enabled:
            return True

        self._found_inf = self._check_for_inf(grads)

        if self._found_inf:
            # Reduce scale
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            return False
        else:
            # Check if we should increase scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
            return True

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "scale": self.scale,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.scale = state["scale"]
        self._growth_tracker = state["growth_tracker"]


class Checkpointer:
    """Model and optimizer checkpointing utility.

    Handles saving and loading of model weights, optimizer state,
    and training state.

    Args:
        save_dir: Directory to save checkpoints.
        model: Model to checkpoint.
        optimizer: Optimizer to checkpoint (optional).
        max_to_keep: Maximum checkpoints to keep (0 = keep all).

    Example:
        >>> checkpointer = Checkpointer("./checkpoints", model, optimizer)
        >>> # Save checkpoint
        >>> checkpointer.save(step=1000, metrics={"loss": 0.5})
        >>> # Load latest checkpoint
        >>> step = checkpointer.load_latest()
    """

    def __init__(
        self,
        save_dir: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        max_to_keep: int = 5,
    ):
        import os
        self.save_dir = save_dir
        self.model = model
        self.optimizer = optimizer
        self.max_to_keep = max_to_keep

        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        step: int,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint.

        Args:
            step: Training step.
            model: Model to checkpoint (overrides instance model).
            optimizer: Optimizer to checkpoint (overrides instance optimizer).
            metrics: Optional metrics to save.
            extra_state: Additional state to save.

        Returns:
            Path to saved checkpoint.
        """
        import os
        import json

        # Use passed model/optimizer or fall back to instance attributes
        save_model = model if model is not None else self.model
        save_optimizer = optimizer if optimizer is not None else self.optimizer

        checkpoint_name = f"checkpoint_step_{step}"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name + ".safetensors")

        # Save model weights
        if save_model is not None:
            save_model.save_weights(checkpoint_path)

        # Save optimizer state if available
        if save_optimizer is not None:
            optimizer_state = save_optimizer.state
            if optimizer_state:
                # Flatten optimizer state and save with savez
                opt_path = os.path.join(self.save_dir, f"checkpoint_step_{step}_optimizer.npz")
                mx.savez(opt_path, **optimizer_state)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Metadata dictionary with step, metrics, etc.
        """
        import os
        import json

        # Load model weights
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        self.model.load_weights(model_path)

        # Load optimizer state if available
        opt_path = os.path.join(checkpoint_path, "optimizer.safetensors")
        if os.path.exists(opt_path) and self.optimizer is not None:
            optimizer_state = mx.load(opt_path)
            self.optimizer.state = optimizer_state

        # Load metadata
        with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        return metadata

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        import os

        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        return os.path.join(self.save_dir, checkpoints[-1])

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Returns:
            Metadata if checkpoint found, None otherwise.
        """
        latest = self.get_latest_checkpoint()
        if latest is None:
            return None
        return self.load(latest)

    def _list_checkpoints(self) -> list:
        """List all checkpoints sorted by step."""
        import os

        checkpoints = []
        for name in os.listdir(self.save_dir):
            if name.startswith("checkpoint_"):
                checkpoints.append(name)

        return sorted(checkpoints)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        import os

        if self.max_to_keep <= 0:
            return

        checkpoints = self._list_checkpoints()
        while len(checkpoints) > self.max_to_keep:
            oldest = checkpoints.pop(0)
            oldest_path = os.path.join(self.save_dir, oldest)
            if os.path.isfile(oldest_path):
                os.remove(oldest_path)
            elif os.path.isdir(oldest_path):
                import shutil
                shutil.rmtree(oldest_path)


class SWA:
    """Stochastic Weight Averaging.

    Maintains a running average of model weights during training,
    which often provides better generalization.

    Args:
        model: Model to track.
        swa_start: Step to start SWA.
        swa_freq: Steps between SWA updates.

    Reference:
        "Averaging Weights Leads to Wider Optima and Better Generalization"
        https://arxiv.org/abs/1803.05407

    Example:
        >>> swa = SWA(model, swa_start=1000, swa_freq=100)
        >>> for step in range(num_steps):
        ...     train_step(...)
        ...     swa.update(step)
        >>> swa.apply()  # Apply averaged weights
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start: int = 0,
        swa_freq: int = 1,
    ):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq

        self.averaged_params: Optional[Dict[str, Any]] = None
        self.n_averaged = 0

    def _copy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy parameters."""
        def copy_recursive(p: Any) -> Any:
            if isinstance(p, mx.array):
                return mx.array(p)
            elif isinstance(p, dict):
                return {k: copy_recursive(v) for k, v in p.items()}
            elif isinstance(p, list):
                return [copy_recursive(v) for v in p]
            return p
        return copy_recursive(params)

    def _average_params(
        self,
        avg: Dict[str, Any],
        new: Dict[str, Any],
        n: int,
    ) -> Dict[str, Any]:
        """Update running average."""
        def avg_recursive(a: Any, p: Any) -> Any:
            if isinstance(p, mx.array):
                return a + (p - a) / (n + 1)
            elif isinstance(p, dict):
                return {k: avg_recursive(a[k], v) for k, v in p.items()}
            elif isinstance(p, list):
                return [avg_recursive(av, pv) for av, pv in zip(a, p)]
            return a
        return avg_recursive(avg, new)

    def update(self, step: int) -> None:
        """Update SWA average.

        Args:
            step: Current training step.
        """
        if step < self.swa_start:
            return

        if (step - self.swa_start) % self.swa_freq != 0:
            return

        current_params = self.model.parameters()

        if self.averaged_params is None:
            self.averaged_params = self._copy_params(current_params)
        else:
            self.averaged_params = self._average_params(
                self.averaged_params, current_params, self.n_averaged
            )

        self.n_averaged += 1

    def apply(self) -> None:
        """Apply averaged weights to model."""
        if self.averaged_params is not None:
            self.model.update(self.averaged_params)

    def apply_swa(self) -> None:
        """Alias for apply() - Apply averaged weights to model."""
        self.apply()

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "averaged_params": self.averaged_params,
            "n_averaged": self.n_averaged,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.averaged_params = state["averaged_params"]
        self.n_averaged = state["n_averaged"]


class Lookahead:
    """Lookahead optimizer wrapper.

    Wraps any optimizer and performs slow weight updates by
    interpolating between fast and slow weights.

    Args:
        optimizer: Base optimizer to wrap.
        k: Steps between slow weight updates.
        alpha: Interpolation coefficient.

    Reference:
        "Lookahead Optimizer: k steps forward, 1 step back"
        https://arxiv.org/abs/1907.08610

    Example:
        >>> base_optimizer = optim.AdamW(learning_rate=1e-4)
        >>> optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        >>> for step, batch in enumerate(dataloader):
        ...     loss, grads = loss_fn(model, batch)
        ...     optimizer.update(model, grads, step)
    """

    def __init__(
        self,
        optimizer: Any,
        k: int = 5,
        alpha: float = 0.5,
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        self.slow_weights: Optional[Dict[str, Any]] = None
        self._step = 0

    def _copy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy parameters."""
        def copy_recursive(p: Any) -> Any:
            if isinstance(p, mx.array):
                return mx.array(p)
            elif isinstance(p, dict):
                return {k: copy_recursive(v) for k, v in p.items()}
            elif isinstance(p, list):
                return [copy_recursive(v) for v in p]
            return p
        return copy_recursive(params)

    def _interpolate(
        self,
        slow: Dict[str, Any],
        fast: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Interpolate between slow and fast weights."""
        def interp_recursive(s: Any, f: Any) -> Any:
            if isinstance(f, mx.array):
                return s + self.alpha * (f - s)
            elif isinstance(f, dict):
                return {k: interp_recursive(s[k], v) for k, v in f.items()}
            elif isinstance(f, list):
                return [interp_recursive(sv, fv) for sv, fv in zip(s, f)]
            return s
        return interp_recursive(slow, fast)

    def init_slow_weights(self, model: nn.Module) -> None:
        """Initialize slow weights from model.

        Args:
            model: Model to copy weights from.
        """
        self.slow_weights = self._copy_params(model.parameters())

    def step(
        self,
        model: nn.Module,
        grads: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Alias for update() - perform a lookahead step."""
        self.update(model, grads, step)

    def update(
        self,
        model: nn.Module,
        grads: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Update model with lookahead.

        Args:
            model: Model to update.
            grads: Gradients.
            step: Current step (optional, uses internal counter if None).
        """
        # First, apply base optimizer
        self.optimizer.update(model, grads)

        if step is None:
            step = self._step
            self._step += 1

        # Initialize slow weights on first step
        if self.slow_weights is None:
            self.slow_weights = self._copy_params(model.parameters())

        # Every k steps, update slow weights and reset fast weights
        if (step + 1) % self.k == 0:
            fast_weights = model.parameters()
            self.slow_weights = self._interpolate(self.slow_weights, fast_weights)
            model.update(self.slow_weights)

    @property
    def state(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "base_state": self.optimizer.state,
            "slow_weights": self.slow_weights,
            "step": self._step,
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """Set optimizer state from checkpoint."""
        self.optimizer.state = value["base_state"]
        self.slow_weights = value["slow_weights"]
        self._step = value["step"]


class SAM:
    """Sharpness-Aware Minimization.

    Seeks parameters that lie in neighborhoods with uniformly low loss,
    leading to better generalization.

    Args:
        optimizer: Base optimizer.
        rho: Neighborhood size.
        adaptive: Use adaptive SAM.

    Reference:
        "Sharpness-Aware Minimization for Efficiently Improving Generalization"
        https://arxiv.org/abs/2010.01412

    Example:
        >>> base_optimizer = optim.AdamW(learning_rate=1e-4)
        >>> sam = SAM(base_optimizer, rho=0.05)
        >>> for batch in dataloader:
        ...     # First forward-backward pass
        ...     loss1, grads1 = loss_fn(model, batch)
        ...     sam.first_step(model, grads1)
        ...     # Second forward-backward pass
        ...     loss2, grads2 = loss_fn(model, batch)
        ...     sam.second_step(model, grads2)
    """

    def __init__(
        self,
        optimizer: Any,
        rho: float = 0.05,
        adaptive: bool = False,
    ):
        self.optimizer = optimizer
        self.rho = rho
        self.adaptive = adaptive

        self._backup_params: Optional[Dict[str, Any]] = None

    def _copy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy parameters."""
        def copy_recursive(p: Any) -> Any:
            if isinstance(p, mx.array):
                return mx.array(p)
            elif isinstance(p, dict):
                return {k: copy_recursive(v) for k, v in p.items()}
            elif isinstance(p, list):
                return [copy_recursive(v) for v in p]
            return p
        return copy_recursive(params)

    def _compute_grad_norm(self, grads: Dict[str, Any]) -> float:
        """Compute gradient norm."""
        norms = []

        def collect_norms(g: Any) -> None:
            if isinstance(g, mx.array):
                norms.append(float(mx.sum(g ** 2)))
            elif isinstance(g, dict):
                for v in g.values():
                    collect_norms(v)
            elif isinstance(g, list):
                for v in g:
                    collect_norms(v)

        collect_norms(grads)
        return math.sqrt(sum(norms))

    def _perturbation(
        self,
        grads: Dict[str, Any],
        scale: float,
    ) -> Dict[str, Any]:
        """Compute perturbation from gradients."""
        def perturb_recursive(g: Any) -> Any:
            if isinstance(g, mx.array):
                return g * scale
            elif isinstance(g, dict):
                return {k: perturb_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [perturb_recursive(v) for v in g]
            return g
        return perturb_recursive(grads)

    def _add_perturbation(
        self,
        params: Dict[str, Any],
        perturbation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add perturbation to parameters (only for keys in perturbation)."""
        def add_recursive(p: Any, d: Any) -> Any:
            if isinstance(p, mx.array):
                return p + d if d is not None else p
            elif isinstance(p, dict):
                # Only perturb keys that have corresponding gradients
                return {k: add_recursive(p[k], d.get(k)) if k in d else p[k] for k in p.keys()}
            elif isinstance(p, list):
                return [add_recursive(pv, dv) for pv, dv in zip(p, d)]
            return p
        return add_recursive(params, perturbation)

    def first_step(self, model: nn.Module, grads: Dict[str, Any]) -> None:
        """First step: perturb weights in gradient direction.

        Args:
            model: Model to perturb.
            grads: Gradients from first forward-backward.
        """
        # Backup original weights
        self._backup_params = self._copy_params(model.parameters())

        # Compute perturbation scale
        grad_norm = self._compute_grad_norm(grads)
        scale = self.rho / (grad_norm + 1e-12)

        # Apply perturbation
        perturbation = self._perturbation(grads, scale)
        perturbed_params = self._add_perturbation(
            model.parameters(), perturbation
        )
        model.update(perturbed_params)

    def second_step(self, model: nn.Module, grads: Dict[str, Any]) -> None:
        """Second step: restore weights and apply optimizer.

        Args:
            model: Model to update.
            grads: Gradients from second forward-backward.
        """
        # Restore original weights
        if self._backup_params is not None:
            model.update(self._backup_params)
            self._backup_params = None

        # Apply base optimizer with second gradients
        self.optimizer.update(model, grads)

    @property
    def state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return self.optimizer.state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """Set optimizer state."""
        self.optimizer.state = value


class GradientNoiseInjection:
    """Add gradient noise for regularization.

    Adds Gaussian noise to gradients during training, which can
    help escape sharp minima and improve generalization.

    Args:
        eta: Noise scale.
        gamma: Decay rate for noise.

    Reference:
        "Adding Gradient Noise Improves Learning for Very Deep Networks"
        https://arxiv.org/abs/1511.06807

    Example:
        >>> noise = GradientNoiseInjection(eta=0.01, gamma=0.55)
        >>> for step, batch in enumerate(dataloader):
        ...     loss, grads = loss_fn(model, batch)
        ...     grads = noise.add_noise(grads, step)
        ...     optimizer.update(model, grads)
    """

    def __init__(
        self,
        eta: float = 0.01,
        gamma: float = 0.55,
    ):
        self.eta = eta
        self.gamma = gamma

    def get_noise_stddev(self, step: int) -> float:
        """Get noise standard deviation for current step."""
        return self.eta / ((1 + step) ** self.gamma)

    def add_noise(
        self,
        grads: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Add Gaussian noise to gradients.

        Args:
            grads: Gradient dictionary.
            step: Current training step.

        Returns:
            Noisy gradients.
        """
        stddev = self.get_noise_stddev(step)

        def add_noise_recursive(g: Any) -> Any:
            if g is None:
                return None
            if isinstance(g, mx.array):
                noise = mx.random.normal(g.shape) * stddev
                return g + noise
            elif isinstance(g, dict):
                return {k: add_noise_recursive(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [add_noise_recursive(v) for v in g]
            return g

        return add_noise_recursive(grads)
