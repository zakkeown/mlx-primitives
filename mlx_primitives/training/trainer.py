"""Trainer class for MLX.

This module provides a flexible training loop abstraction:
- Trainer: Configurable training loop with callbacks, schedulers, etc.
- TrainingConfig: Configuration dataclass for training hyperparameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_primitives.training.callbacks import (
    Callback,
    CallbackList,
    ProgressCallback,
    TrainingState,
)
from mlx_primitives.training.schedulers import LRScheduler
from mlx_primitives.training.utils import (
    EMA,
    GradientAccumulator,
    GradientClipper,
)


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        learning_rate: Initial learning rate.
        epochs: Number of training epochs.
        max_steps: Maximum training steps (overrides epochs if set).
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping (None to disable).
        max_grad_value: Maximum gradient value for clipping (None to disable).
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps.
        log_every_n_steps: Log metrics every N steps.
        eval_every_n_steps: Run evaluation every N steps.
        save_every_n_steps: Save checkpoint every N steps.
        use_ema: Whether to use Exponential Moving Average.
        ema_decay: EMA decay rate.
        seed: Random seed for reproducibility.
    """

    learning_rate: float = 1e-4
    epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    max_grad_value: Optional[float] = None
    weight_decay: float = 0.0
    warmup_steps: int = 0
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    use_ema: bool = False
    ema_decay: float = 0.9999
    seed: Optional[int] = None


class Trainer:
    """Flexible training loop for MLX models.

    Handles the training loop with support for:
    - Learning rate scheduling
    - Gradient accumulation and clipping
    - Callbacks (early stopping, checkpointing, logging)
    - Exponential Moving Average (EMA)
    - Validation during training

    Args:
        model: The model to train.
        optimizer: MLX optimizer instance.
        loss_fn: Loss function that takes (model, batch) and returns (loss, aux).
        config: Training configuration.
        scheduler: Optional learning rate scheduler.
        callbacks: List of callbacks.

    Example:
        >>> model = MyModel()
        >>> optimizer = optim.AdamW(learning_rate=3e-4)
        >>> config = TrainingConfig(epochs=10, max_grad_norm=1.0)
        >>>
        >>> def loss_fn(model, batch):
        ...     x, y = batch
        ...     pred = model(x)
        ...     loss = mx.mean((pred - y) ** 2)
        ...     return loss, {}
        >>>
        >>> trainer = Trainer(model, optimizer, loss_fn, config)
        >>> trainer.add_callback(EarlyStopping(patience=5))
        >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[[nn.Module, Any], tuple[mx.array, Dict[str, Any]]],
        config: Optional[TrainingConfig] = None,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()
        self.scheduler = scheduler

        # Set up callbacks
        self.callbacks = CallbackList(callbacks or [])
        if self.config.log_every_n_steps > 0:
            self.callbacks.add(ProgressCallback(self.config.log_every_n_steps))

        # Set up gradient utilities
        self.grad_accumulator = None
        if self.config.gradient_accumulation_steps > 1:
            self.grad_accumulator = GradientAccumulator(
                self.config.gradient_accumulation_steps
            )

        self.grad_clipper = None
        if self.config.max_grad_norm is not None or self.config.max_grad_value is not None:
            self.grad_clipper = GradientClipper(
                max_norm=self.config.max_grad_norm,
                max_value=self.config.max_grad_value,
            )

        # Set up EMA
        self.ema = None
        if self.config.use_ema:
            self.ema = EMA(model, decay=self.config.ema_decay)

        # Training state
        self.state = TrainingState(model=model, optimizer=optimizer)
        self.global_step = 0
        self.current_epoch = 0

    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the trainer.

        Args:
            callback: Callback instance to add.
        """
        self.callbacks.add(callback)

    def _create_train_step(self) -> Callable:
        """Create the training step function with value_and_grad."""

        def loss_wrapper(model: nn.Module, batch: Any) -> mx.array:
            loss, _ = self.loss_fn(model, batch)
            return loss

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_wrapper)

        def train_step(batch: Any) -> tuple[mx.array, Dict[str, Any]]:
            loss, grads = loss_and_grad_fn(self.model, batch)
            return loss, grads

        return train_step

    def _update_learning_rate(self) -> float:
        """Update learning rate from scheduler."""
        if self.scheduler is not None:
            lr = self.scheduler.step()
            self.optimizer.learning_rate = lr
            return lr
        return float(self.optimizer.learning_rate)

    def _apply_gradients(
        self, grads: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional[mx.array]]:
        """Apply gradient clipping and return processed gradients."""
        grad_norm = None

        if self.grad_clipper is not None:
            if self.config.max_grad_norm is not None:
                grads, grad_norm = self.grad_clipper.clip_by_norm(grads)
            if self.config.max_grad_value is not None:
                grads = self.grad_clipper.clip_by_value(grads)

        return grads, grad_norm

    def train_step(
        self, batch: Any, train_step_fn: Callable
    ) -> tuple[float, Optional[float]]:
        """Execute a single training step.

        Args:
            batch: Training batch.
            train_step_fn: Compiled training step function.

        Returns:
            Tuple of (loss, gradient_norm).
        """
        loss, grads = train_step_fn(batch)

        # Handle gradient accumulation
        if self.grad_accumulator is not None:
            self.grad_accumulator.accumulate(grads)

            if not self.grad_accumulator.should_step(self.global_step):
                return float(loss), None

            grads = self.grad_accumulator.get_accumulated(normalize=True)
            self.grad_accumulator.reset()

        # Apply gradient clipping
        grads, grad_norm = self._apply_gradients(grads)

        # Update model
        self.optimizer.update(self.model, grads)

        # Evaluate lazy computation
        mx.eval(self.model.parameters(), self.optimizer.state)

        # Update EMA
        if self.ema is not None:
            self.ema.update(self.global_step)

        return float(loss), float(grad_norm) if grad_norm is not None else None

    def evaluate(
        self,
        eval_loader: Iterator,
        eval_fn: Optional[Callable[[nn.Module, Any], Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Run evaluation.

        Args:
            eval_loader: Evaluation data iterator.
            eval_fn: Optional custom evaluation function. If None, uses loss_fn.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Use EMA weights for evaluation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        num_batches = 0
        all_metrics: Dict[str, List[float]] = {}

        for batch in eval_loader:
            if eval_fn is not None:
                metrics = eval_fn(self.model, batch)
            else:
                loss, aux = self.loss_fn(self.model, batch)
                metrics = {"loss": float(loss)}
                metrics.update({k: float(v) for k, v in aux.items()})

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            if "loss" in metrics:
                total_loss += metrics["loss"]
            num_batches += 1

        # Restore training weights
        if self.ema is not None:
            self.ema.restore()

        # Compute averages
        avg_metrics = {
            f"val_{key}": sum(values) / len(values)
            for key, values in all_metrics.items()
        }

        return avg_metrics

    def fit(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator] = None,
        epochs: Optional[int] = None,
        eval_fn: Optional[Callable[[nn.Module, Any], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_loader: Training data iterator (must be re-iterable for epochs > 1).
            val_loader: Optional validation data iterator.
            epochs: Number of epochs (overrides config if provided).
            eval_fn: Optional custom evaluation function.

        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.epochs

        # Set random seed if provided
        if self.config.seed is not None:
            mx.random.seed(self.config.seed)

        # Create training step function
        train_step_fn = self._create_train_step()

        # Initialize state
        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
        )

        # Training history
        history: Dict[str, List[float]] = {
            "loss": [],
            "learning_rate": [],
        }

        # Callbacks: training begin
        self.callbacks.on_train_begin(self.state)

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                self.state.epoch = epoch

                # Callbacks: epoch begin
                self.callbacks.on_epoch_begin(self.state)

                epoch_losses: List[float] = []

                for batch in train_loader:
                    # Callbacks: step begin
                    self.state.step = self.global_step
                    self.callbacks.on_step_begin(self.state)

                    # Update learning rate
                    lr = self._update_learning_rate()

                    # Training step
                    loss, grad_norm = self.train_step(batch, train_step_fn)

                    # Update state
                    self.state.loss = loss
                    self.state.learning_rate = lr
                    self.state.grad_norm = grad_norm

                    epoch_losses.append(loss)
                    history["loss"].append(loss)
                    history["learning_rate"].append(lr)

                    # Callbacks: step end
                    self.callbacks.on_step_end(self.state)

                    # Check for early stopping
                    for callback in self.callbacks.callbacks:
                        if hasattr(callback, "stop_training") and callback.stop_training:
                            self.callbacks.on_train_end(self.state)
                            return history

                    # Run evaluation if configured
                    if (
                        val_loader is not None
                        and self.config.eval_every_n_steps is not None
                        and self.global_step > 0
                        and self.global_step % self.config.eval_every_n_steps == 0
                    ):
                        self.callbacks.on_validation_begin(self.state)
                        val_metrics = self.evaluate(val_loader, eval_fn)
                        self.state.metrics.update(val_metrics)
                        self.callbacks.on_validation_end(self.state)

                    self.global_step += 1

                    # Check max_steps
                    if (
                        self.config.max_steps is not None
                        and self.global_step >= self.config.max_steps
                    ):
                        break

                # End of epoch validation
                if val_loader is not None:
                    self.callbacks.on_validation_begin(self.state)
                    val_metrics = self.evaluate(val_loader, eval_fn)
                    self.state.metrics.update(val_metrics)
                    self.callbacks.on_validation_end(self.state)

                    # Store validation metrics in history
                    for key, value in val_metrics.items():
                        if key not in history:
                            history[key] = []
                        history[key].append(value)

                # Callbacks: epoch end
                self.callbacks.on_epoch_end(self.state)

                # Check for early stopping from callbacks
                for callback in self.callbacks.callbacks:
                    if hasattr(callback, "stop_training") and callback.stop_training:
                        self.callbacks.on_train_end(self.state)
                        return history

                # Check max_steps
                if (
                    self.config.max_steps is not None
                    and self.global_step >= self.config.max_steps
                ):
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        # Callbacks: training end
        self.callbacks.on_train_end(self.state)

        return history

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights = self._flatten_dict(self.model.parameters())
        mx.save_safetensors(str(path), weights)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint.
        """
        weights = mx.load(str(path))
        # Note: This is a simplified load - proper implementation would
        # need to unflatten the dictionary to match model structure
        self.model.update(weights)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, mx.array]:
        """Flatten nested dictionary for saving."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, mx.array):
                items.append((new_key, v))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, mx.array):
                        items.append((f"{new_key}{sep}{i}", item))
                    elif isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(item, f"{new_key}{sep}{i}", sep).items()
                        )
        return dict(items)
