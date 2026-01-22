"""Training callbacks for MLX.

This module provides callbacks for monitoring and controlling training:
- EarlyStopping: Stop training when metric plateaus
- ModelCheckpoint: Save best/periodic model checkpoints
- LRMonitor: Log learning rate changes
- GradientMonitor: Track gradient statistics
- ProgressCallback: Training progress display
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.training.utils import copy_params, flatten_dict

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state passed to callbacks.

    Attributes:
        step: Current global step.
        epoch: Current epoch.
        loss: Current loss value.
        metrics: Dictionary of metric values.
        learning_rate: Current learning rate.
        grad_norm: Current gradient norm (if computed).
        model: Reference to the model.
        optimizer: Reference to the optimizer.
    """

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    model: Optional[nn.Module] = None
    optimizer: Any = None
    stop_training: bool = False  # Flag to signal early stopping


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks receive training state at various points during training
    and can perform actions or modify training behavior.

    Methods:
        on_train_begin: Called at the start of training.
        on_train_end: Called at the end of training.
        on_epoch_begin: Called at the start of each epoch.
        on_epoch_end: Called at the end of each epoch.
        on_step_begin: Called before each training step.
        on_step_end: Called after each training step.
        on_validation_begin: Called before validation.
        on_validation_end: Called after validation.
    """

    def on_train_begin(self, state: TrainingState) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, state: TrainingState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, state: TrainingState) -> None:
        """Called before each training step."""
        pass

    def on_step_end(self, state: TrainingState) -> None:
        """Called after each training step."""
        pass

    def on_validation_begin(self, state: TrainingState) -> None:
        """Called before validation."""
        pass

    def on_validation_end(self, state: TrainingState) -> None:
        """Called after validation."""
        pass


class CallbackList:
    """Container for multiple callbacks.

    Manages a list of callbacks and dispatches events to all of them.

    Args:
        callbacks: List of Callback instances.

    Example:
        >>> callbacks = CallbackList([
        ...     EarlyStopping(patience=5),
        ...     ModelCheckpoint(save_dir="checkpoints"),
        ... ])
        >>> callbacks.on_epoch_end(state)
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.stop_training = False

    def add(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(state)

    def on_train_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)

    def on_epoch_begin(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(state)

    def on_epoch_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_step_begin(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_step_begin(state)

    def on_step_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_step_end(state)

    def on_validation_begin(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(state)

    def on_validation_end(self, state: TrainingState) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(state)


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to monitor (default: 'val_loss').
        min_delta: Minimum change to qualify as improvement (default: 0.0).
        patience: Number of epochs with no improvement before stopping (default: 5).
        mode: 'min' or 'max' - whether lower or higher is better (default: 'min').
        restore_best_weights: Whether to restore best weights at end (default: True).
        save_best_to_disk: Path to save best weights to disk instead of memory.
            Use this for large models to avoid doubling memory usage.
            If provided and restore_best_weights=True, weights are loaded from
            disk at the end of training. (default: None = store in memory)
        verbose: Whether to print messages (default: True).

    Example:
        >>> # Store best weights in memory (default, may OOM on large models)
        >>> early_stopping = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=5,
        ...     mode='min',
        ... )
        >>>
        >>> # Store best weights on disk (memory efficient for large models)
        >>> early_stopping = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=5,
        ...     save_best_to_disk='/tmp/best_model.safetensors',
        ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        restore_best_weights: bool = True,
        save_best_to_disk: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_to_disk = Path(save_best_to_disk) if save_best_to_disk else None
        self.verbose = verbose

        self.best_value: Optional[float] = None
        self.best_weights: Optional[Dict[str, Any]] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.stop_training: bool = False

        # Set comparison function based on mode
        if mode == "min":
            self._is_better = lambda new, best: new < best - min_delta
            self.best_value = float("inf")
        else:
            self._is_better = lambda new, best: new > best + min_delta
            self.best_value = float("-inf")

    def on_train_begin(self, state: TrainingState) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

    def on_epoch_end(self, state: TrainingState) -> None:
        current = state.metrics.get(self.monitor)
        if current is None:
            return

        if self._is_better(current, self.best_value):
            self.best_value = current
            self.best_epoch = state.epoch
            self.wait = 0
            if self.restore_best_weights and state.model is not None:
                if self.save_best_to_disk is not None:
                    # Save to disk to avoid memory overhead
                    self.save_best_to_disk.parent.mkdir(parents=True, exist_ok=True)
                    mx.save_safetensors(
                        str(self.save_best_to_disk),
                        flatten_dict(state.model.parameters())
                    )
                    if self.verbose:
                        logger.info(
                            f"EarlyStopping: {self.monitor} improved to {current:.6f}, "
                            f"saved to {self.save_best_to_disk}"
                        )
                else:
                    # Store in memory (can OOM on large models)
                    self.best_weights = copy_params(state.model.parameters())
                    if self.verbose:
                        logger.info(
                            f"EarlyStopping: {self.monitor} improved to {current:.6f}"
                        )
            elif self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved to {current:.6f}"
                )
        else:
            self.wait += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve. "
                    f"Patience: {self.wait}/{self.patience}"
                )
            if self.wait >= self.patience:
                self.stopped_epoch = state.epoch
                self.stop_training = True
                state.stop_training = True  # Set state flag for trainer to check
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training at epoch {state.epoch}. "
                        f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}"
                    )

    def on_train_end(self, state: TrainingState) -> None:
        if not self.restore_best_weights or state.model is None:
            return

        if self.save_best_to_disk is not None and self.save_best_to_disk.exists():
            # Load from disk
            state.model.load_weights(str(self.save_best_to_disk))
            if self.verbose:
                logger.info(
                    f"EarlyStopping: Restored best weights from {self.save_best_to_disk} "
                    f"(epoch {self.best_epoch})"
                )
        elif self.best_weights is not None:
            # Load from memory
            state.model.update(self.best_weights)
            if self.verbose:
                logger.info(f"EarlyStopping: Restored best weights from epoch {self.best_epoch}")


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric to monitor for best model (default: 'val_loss').
        mode: 'min' or 'max' (default: 'min').
        save_best_only: Only save when monitored metric improves (default: True).
        save_weights_only: Only save model weights, not optimizer state (default: False).
        save_every_n_epochs: Save every N epochs regardless of metric (default: None).
        save_every_n_steps: Save every N steps (default: None).
        max_checkpoints: Maximum number of checkpoints to keep (default: 5).
        verbose: Whether to print messages (default: True).

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     save_dir="checkpoints",
        ...     monitor="val_loss",
        ...     save_best_only=True,
        ... )
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_steps: Optional[int] = None,
        max_checkpoints: int = 5,
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose

        self.best_value: Optional[float] = None
        self.saved_checkpoints: List[Path] = []

        if mode == "min":
            self._is_better = lambda new, best: new < best
            self.best_value = float("inf")
        else:
            self._is_better = lambda new, best: new > best
            self.best_value = float("-inf")

    def on_train_begin(self, state: TrainingState) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

    def on_step_end(self, state: TrainingState) -> None:
        if self.save_every_n_steps is not None:
            if state.step > 0 and state.step % self.save_every_n_steps == 0:
                self._save_checkpoint(state, f"step_{state.step}")

    def on_epoch_end(self, state: TrainingState) -> None:
        # Check if we should save based on metric
        if not self.save_best_only:
            if self.save_every_n_epochs is not None:
                if (state.epoch + 1) % self.save_every_n_epochs == 0:
                    self._save_checkpoint(state, f"epoch_{state.epoch}")
        else:
            current = state.metrics.get(self.monitor)
            if current is not None and self._is_better(current, self.best_value):
                self.best_value = current
                self._save_checkpoint(state, "best")
                if self.verbose:
                    logger.info(
                        f"ModelCheckpoint: {self.monitor} improved to {current:.6f}, "
                        f"saving checkpoint"
                    )

    def _save_checkpoint(self, state: TrainingState, name: str) -> None:
        """Save a checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_{name}.safetensors"

        if state.model is not None:
            # Save model weights
            weights = state.model.parameters()
            mx.save_safetensors(str(checkpoint_path), flatten_dict(weights))

            if self.verbose:
                logger.info(f"ModelCheckpoint: Saved checkpoint to {checkpoint_path}")

            # Track saved checkpoints for cleanup
            if name != "best" and checkpoint_path not in self.saved_checkpoints:
                self.saved_checkpoints.append(checkpoint_path)
                self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    logger.info(f"ModelCheckpoint: Removed old checkpoint {old_checkpoint}")


class LRMonitor(Callback):
    """Monitor and log learning rate during training.

    Args:
        log_every_n_steps: Log learning rate every N steps (default: 100).
        verbose: Whether to print to console (default: True).

    Example:
        >>> lr_monitor = LRMonitor(log_every_n_steps=100)
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        verbose: bool = True,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.lr_history: List[tuple[int, float]] = []

    def on_step_end(self, state: TrainingState) -> None:
        if state.step % self.log_every_n_steps == 0:
            self.lr_history.append((state.step, state.learning_rate))
            if self.verbose:
                logger.info(f"Step {state.step}: lr = {state.learning_rate:.2e}")


class GradientMonitor(Callback):
    """Monitor gradient statistics during training.

    Tracks gradient norms, detects vanishing/exploding gradients.

    Args:
        log_every_n_steps: Log gradient stats every N steps (default: 100).
        warn_threshold_low: Warn if grad norm below this (default: 1e-7).
        warn_threshold_high: Warn if grad norm above this (default: 1000).
        verbose: Whether to print to console (default: True).

    Example:
        >>> grad_monitor = GradientMonitor(log_every_n_steps=100)
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        warn_threshold_low: float = 1e-7,
        warn_threshold_high: float = 1000.0,
        verbose: bool = True,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.warn_threshold_low = warn_threshold_low
        self.warn_threshold_high = warn_threshold_high
        self.verbose = verbose
        self.grad_norm_history: List[tuple[int, float]] = []

    def on_step_end(self, state: TrainingState) -> None:
        if state.grad_norm is None:
            return

        grad_norm = float(state.grad_norm)

        if state.step % self.log_every_n_steps == 0:
            self.grad_norm_history.append((state.step, grad_norm))
            if self.verbose:
                logger.info(f"Step {state.step}: grad_norm = {grad_norm:.4f}")

        # Warn about potential issues
        if grad_norm < self.warn_threshold_low:
            if self.verbose:
                logger.warning(
                    f"Vanishing gradients detected at step {state.step}. "
                    f"Grad norm: {grad_norm:.2e}"
                )
        elif grad_norm > self.warn_threshold_high:
            if self.verbose:
                logger.warning(
                    f"Exploding gradients detected at step {state.step}. "
                    f"Grad norm: {grad_norm:.2e}"
                )


class ProgressCallback(Callback):
    """Display training progress.

    Shows loss, metrics, and timing information during training.

    Args:
        log_every_n_steps: Update progress every N steps (default: 10).
        show_eta: Show estimated time remaining (default: True).

    Example:
        >>> progress = ProgressCallback(log_every_n_steps=10)
    """

    def __init__(
        self,
        log_every_n_steps: int = 10,
        show_eta: bool = True,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.show_eta = show_eta

        self.train_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.total_steps: Optional[int] = None

    def on_train_begin(self, state: TrainingState) -> None:
        self.train_start_time = time.time()
        logger.info("Training started")

    def on_train_end(self, state: TrainingState) -> None:
        if self.train_start_time is not None:
            elapsed = time.time() - self.train_start_time
            logger.info(f"Training completed in {self._format_time(elapsed)}")

    def on_epoch_begin(self, state: TrainingState) -> None:
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {state.epoch + 1}")
        logger.info("-" * 40)

    def on_epoch_end(self, state: TrainingState) -> None:
        if self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in state.metrics.items()
            )
            logger.info(f"Epoch {state.epoch + 1} completed in {self._format_time(elapsed)}")
            if metrics_str:
                logger.info(f"Metrics: {metrics_str}")

    def on_step_end(self, state: TrainingState) -> None:
        if state.step % self.log_every_n_steps == 0:
            loss_str = f"loss: {state.loss:.4f}"
            lr_str = f"lr: {state.learning_rate:.2e}"

            eta_str = ""
            if self.show_eta and self.train_start_time and self.total_steps:
                elapsed = time.time() - self.train_start_time
                if state.step > 0:
                    time_per_step = elapsed / state.step
                    remaining_steps = self.total_steps - state.step
                    eta = time_per_step * remaining_steps
                    eta_str = f" | ETA: {self._format_time(eta)}"

            logger.info(f"Step {state.step}: {loss_str} | {lr_str}{eta_str}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class MetricLogger(Callback):
    """Log metrics to a file.

    Saves metrics history to a JSON file for later analysis.

    Args:
        log_file: Path to the log file.
        log_every_n_steps: Log every N steps (default: 1).

    Example:
        >>> logger = MetricLogger("training_log.json")
    """

    def __init__(
        self,
        log_file: Union[str, Path],
        log_every_n_steps: int = 1,
    ):
        self.log_file = Path(log_file)
        self.log_every_n_steps = log_every_n_steps
        self.history: List[Dict[str, Any]] = []

    def on_train_begin(self, state: TrainingState) -> None:
        self.history = []
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, state: TrainingState) -> None:
        if state.step % self.log_every_n_steps == 0:
            entry = {
                "step": state.step,
                "epoch": state.epoch,
                "loss": state.loss,
                "learning_rate": state.learning_rate,
                "timestamp": time.time(),
            }
            if state.grad_norm is not None:
                entry["grad_norm"] = float(state.grad_norm)
            entry.update(state.metrics)
            self.history.append(entry)

    def on_train_end(self, state: TrainingState) -> None:
        self._save()

    def _save(self) -> None:
        """Save history to file."""
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)


class LambdaCallback(Callback):
    """Create a callback from lambda functions.

    Args:
        on_train_begin: Function to call at training start.
        on_train_end: Function to call at training end.
        on_epoch_begin: Function to call at epoch start.
        on_epoch_end: Function to call at epoch end.
        on_step_begin: Function to call at step start.
        on_step_end: Function to call at step end.

    Example:
        >>> callback = LambdaCallback(
        ...     on_epoch_end=lambda state: print(f"Epoch {state.epoch} done!")
        ... )
    """

    def __init__(
        self,
        on_train_begin: Optional[Callable[[TrainingState], None]] = None,
        on_train_end: Optional[Callable[[TrainingState], None]] = None,
        on_epoch_begin: Optional[Callable[[TrainingState], None]] = None,
        on_epoch_end: Optional[Callable[[TrainingState], None]] = None,
        on_step_begin: Optional[Callable[[TrainingState], None]] = None,
        on_step_end: Optional[Callable[[TrainingState], None]] = None,
    ):
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_step_begin = on_step_begin
        self._on_step_end = on_step_end

    def on_train_begin(self, state: TrainingState) -> None:
        if self._on_train_begin:
            self._on_train_begin(state)

    def on_train_end(self, state: TrainingState) -> None:
        if self._on_train_end:
            self._on_train_end(state)

    def on_epoch_begin(self, state: TrainingState) -> None:
        if self._on_epoch_begin:
            self._on_epoch_begin(state)

    def on_epoch_end(self, state: TrainingState) -> None:
        if self._on_epoch_end:
            self._on_epoch_end(state)

    def on_step_begin(self, state: TrainingState) -> None:
        if self._on_step_begin:
            self._on_step_begin(state)

    def on_step_end(self, state: TrainingState) -> None:
        if self._on_step_end:
            self._on_step_end(state)


class WandbCallback(Callback):
    """Weights & Biases logging callback.

    Logs training metrics, gradients, and system information to W&B.

    Args:
        project: W&B project name.
        name: Run name (optional, auto-generated if None).
        config: Configuration dict to log.
        log_every_n_steps: Log metrics every N steps (default: 1).
        log_gradients: Whether to log gradient statistics (default: False).
        log_model: Whether to log model checkpoints (default: False).
        entity: W&B entity (team or username).
        tags: List of tags for the run.
        notes: Notes for the run.

    Example:
        >>> callback = WandbCallback(
        ...     project="my-project",
        ...     name="experiment-1",
        ...     config={"lr": 0.001, "batch_size": 32},
        ... )
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_every_n_steps: int = 1,
        log_gradients: bool = False,
        log_model: bool = False,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_every_n_steps = log_every_n_steps
        self.log_gradients = log_gradients
        self.log_model = log_model
        self.entity = entity
        self.tags = tags
        self.notes = notes

        self._wandb = None
        self._run = None

    def on_train_begin(self, state: TrainingState) -> None:
        try:
            import wandb
            self._wandb = wandb

            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                entity=self.entity,
                tags=self.tags,
                notes=self.notes,
                reinit=True,
            )
            logger.info(f"WandbCallback: Initialized run {self._run.name}")
        except ImportError:
            logger.warning("WandbCallback: wandb not installed. Install with 'pip install wandb'")
            self._wandb = None

    def on_train_end(self, state: TrainingState) -> None:
        if self._wandb is not None and self._run is not None:
            # Log final metrics
            final_metrics = {
                "final_loss": state.loss,
                "total_steps": state.step,
                "total_epochs": state.epoch,
            }
            final_metrics.update({f"final_{k}": v for k, v in state.metrics.items()})
            self._wandb.log(final_metrics)

            if self.log_model and state.model is not None:
                # Save and log model artifact
                artifact = self._wandb.Artifact(
                    name=f"model-{self._run.name}",
                    type="model",
                )
                # Save model weights to temp file
                import tempfile
                import os
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
                        temp_path = f.name
                        weights = state.model.parameters()
                        flat_weights = flatten_dict(weights)
                        mx.save_safetensors(f.name, flat_weights)
                        artifact.add_file(f.name, name="model.safetensors")
                    self._wandb.log_artifact(artifact)
                finally:
                    # Clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            self._wandb.finish()
            logger.info("WandbCallback: Run finished")

    def on_step_end(self, state: TrainingState) -> None:
        if self._wandb is None:
            return

        if state.step % self.log_every_n_steps == 0:
            log_dict = {
                "train/loss": state.loss,
                "train/step": state.step,
                "train/epoch": state.epoch,
                "train/learning_rate": state.learning_rate,
            }

            if self.log_gradients and state.grad_norm is not None:
                log_dict["train/grad_norm"] = float(state.grad_norm)

            # Add custom metrics
            for k, v in state.metrics.items():
                log_dict[f"train/{k}"] = v

            self._wandb.log(log_dict, step=state.step)

    def on_epoch_end(self, state: TrainingState) -> None:
        if self._wandb is None:
            return

        log_dict = {"epoch": state.epoch}
        for k, v in state.metrics.items():
            log_dict[f"epoch/{k}"] = v

        self._wandb.log(log_dict, step=state.step)

    def on_validation_end(self, state: TrainingState) -> None:
        if self._wandb is None:
            return

        log_dict = {}
        for k, v in state.metrics.items():
            if k.startswith("val_"):
                log_dict[f"val/{k[4:]}"] = v
            else:
                log_dict[f"val/{k}"] = v

        if log_dict:
            self._wandb.log(log_dict, step=state.step)


class TensorBoardCallback(Callback):
    """TensorBoard logging callback.

    Logs training metrics, scalars, and histograms to TensorBoard.

    Args:
        log_dir: Directory to save TensorBoard logs.
        log_every_n_steps: Log metrics every N steps (default: 1).
        log_gradients: Whether to log gradient histograms (default: False).
        log_weights: Whether to log weight histograms (default: False).
        histogram_every_n_steps: Log histograms every N steps (default: 100).
        flush_secs: How often to flush to disk in seconds (default: 120).

    Example:
        >>> callback = TensorBoardCallback(
        ...     log_dir="runs/experiment-1",
        ...     log_gradients=True,
        ... )
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        log_every_n_steps: int = 1,
        log_gradients: bool = False,
        log_weights: bool = False,
        histogram_every_n_steps: int = 100,
        flush_secs: int = 120,
    ):
        self.log_dir = Path(log_dir)
        self.log_every_n_steps = log_every_n_steps
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.histogram_every_n_steps = histogram_every_n_steps
        self.flush_secs = flush_secs

        self._writer = None

    def on_train_begin(self, state: TrainingState) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=self.flush_secs,
            )
            logger.info(f"TensorBoardCallback: Logging to {self.log_dir}")
        except ImportError:
            logger.warning(
                "TensorBoardCallback: tensorboard not installed. "
                "Install with 'pip install tensorboard'"
            )
            self._writer = None

    def on_train_end(self, state: TrainingState) -> None:
        if self._writer is not None:
            self._writer.close()
            logger.info("TensorBoardCallback: Writer closed")

    def on_step_end(self, state: TrainingState) -> None:
        if self._writer is None:
            return

        if state.step % self.log_every_n_steps == 0:
            self._writer.add_scalar("train/loss", state.loss, state.step)
            self._writer.add_scalar("train/learning_rate", state.learning_rate, state.step)

            if state.grad_norm is not None:
                self._writer.add_scalar("train/grad_norm", float(state.grad_norm), state.step)

            # Log custom metrics
            for k, v in state.metrics.items():
                self._writer.add_scalar(f"train/{k}", v, state.step)

        # Log histograms less frequently
        if state.step % self.histogram_every_n_steps == 0:
            if self.log_weights and state.model is not None:
                self._log_weight_histograms(state.model, state.step)

    def on_epoch_end(self, state: TrainingState) -> None:
        if self._writer is None:
            return

        for k, v in state.metrics.items():
            self._writer.add_scalar(f"epoch/{k}", v, state.epoch)

    def on_validation_end(self, state: TrainingState) -> None:
        if self._writer is None:
            return

        for k, v in state.metrics.items():
            if k.startswith("val_"):
                self._writer.add_scalar(f"val/{k[4:]}", v, state.step)
            else:
                self._writer.add_scalar(f"val/{k}", v, state.step)

    def _log_weight_histograms(self, model: nn.Module, step: int) -> None:
        """Log weight histograms to TensorBoard."""
        try:
            import numpy as np
            for name, param in self._iterate_params(model.parameters()):
                values = np.array(param.tolist())
                self._writer.add_histogram(f"weights/{name}", values, step)
        except ImportError:
            # numpy not available - skip histogram logging
            pass
        except Exception as e:
            # Log the error once rather than silently swallowing
            import warnings
            warnings.warn(
                f"TensorBoardCallback: Failed to log weight histograms: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _iterate_params(
        self, params: Dict[str, Any], prefix: str = ""
    ) -> List[tuple[str, mx.array]]:
        """Iterate over all parameters in a nested dict."""
        items = []
        for k, v in params.items():
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, mx.array):
                items.append((name, v))
            elif isinstance(v, dict):
                items.extend(self._iterate_params(v, name))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, mx.array):
                        items.append((f"{name}.{i}", item))
                    elif isinstance(item, dict):
                        items.extend(self._iterate_params(item, f"{name}.{i}"))
        return items