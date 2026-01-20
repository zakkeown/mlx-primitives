"""Tests for training utilities."""

import math
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from mlx_primitives.training import (
    # Schedulers
    LRScheduler,
    CosineAnnealingLR,
    WarmupCosineScheduler,
    OneCycleLR,
    PolynomialDecayLR,
    InverseSqrtScheduler,
    LinearWarmupLR,
    ConstantLR,
    ExponentialDecayLR,
    MultiStepLR,
    ChainedScheduler,
    # Callbacks
    Callback,
    CallbackList,
    TrainingState,
    EarlyStopping,
    ModelCheckpoint,
    LRMonitor,
    GradientMonitor,
    ProgressCallback,
    LambdaCallback,
    WandbCallback,
    TensorBoardCallback,
    # Utilities
    EMA,
    EMAWithWarmup,
    GradientAccumulator,
    GradientClipper,
    MixedPrecisionManager,
    Checkpointer,
    SWA,
    Lookahead,
    SAM,
    GradientNoiseInjection,
    compute_gradient_norm,
    clip_grad_norm,
    clip_grad_value,
    # Trainer
    Trainer,
    TrainingConfig,
)


# ============================================================================
# Scheduler Tests
# ============================================================================


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_basic_cosine(self):
        """Test basic cosine annealing."""
        scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100)

        # At step 0, should be base_lr
        assert scheduler.get_lr(0) == pytest.approx(1.0)

        # At step T_max/2, should be (base_lr + min_lr) / 2
        assert scheduler.get_lr(50) == pytest.approx(0.5)

        # At step T_max-1, should be near min_lr
        assert scheduler.get_lr(99) == pytest.approx(0.0, abs=0.01)

        # At step T_max, restarts cycle (warm restart behavior)
        assert scheduler.get_lr(100) == pytest.approx(1.0)

    def test_cosine_with_min_lr(self):
        """Test cosine annealing with min_lr."""
        scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, min_lr=0.1)

        assert scheduler.get_lr(0) == pytest.approx(1.0)
        assert scheduler.get_lr(50) == pytest.approx(0.55)
        # At step T_max-1, should be near min_lr
        assert scheduler.get_lr(99) == pytest.approx(0.1, abs=0.01)

    def test_cosine_with_warmup(self):
        """Test cosine annealing with warmup."""
        scheduler = CosineAnnealingLR(
            base_lr=1.0, T_max=100, warmup_steps=10
        )

        # During warmup - linear increase
        assert scheduler.get_lr(0) == pytest.approx(0.1)
        assert scheduler.get_lr(5) == pytest.approx(0.6)
        assert scheduler.get_lr(9) == pytest.approx(1.0)

        # After warmup - cosine decay
        assert scheduler.get_lr(10) == pytest.approx(1.0)

    def test_step_method(self):
        """Test step() method updates internal counter."""
        scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100)

        lr1 = scheduler.step()
        lr2 = scheduler.step()

        assert scheduler.current_step == 2
        assert lr1 == pytest.approx(scheduler.get_lr(0))
        assert lr2 == pytest.approx(scheduler.get_lr(1))


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_warmup_cosine(self):
        """Test warmup followed by cosine decay."""
        scheduler = WarmupCosineScheduler(
            base_lr=1.0,
            warmup_steps=100,
            total_steps=1000,
        )

        # During warmup - linear increase
        assert scheduler.get_lr(0) == pytest.approx(0.0)
        assert scheduler.get_lr(50) == pytest.approx(0.5)
        assert scheduler.get_lr(100) == pytest.approx(1.0)

        # After warmup - cosine decay
        assert scheduler.get_lr(550) == pytest.approx(0.5)
        assert scheduler.get_lr(1000) == pytest.approx(0.0)

    def test_warmup_with_init_lr(self):
        """Test warmup starting from non-zero LR."""
        scheduler = WarmupCosineScheduler(
            base_lr=1.0,
            warmup_steps=100,
            total_steps=1000,
            warmup_init_lr=0.1,
        )

        assert scheduler.get_lr(0) == pytest.approx(0.1)
        assert scheduler.get_lr(50) == pytest.approx(0.55)


class TestOneCycleLR:
    """Tests for OneCycleLR scheduler."""

    def test_one_cycle_shape(self):
        """Test one cycle LR has correct shape."""
        scheduler = OneCycleLR(
            max_lr=1.0,
            total_steps=1000,
            pct_start=0.3,
            div_factor=25.0,
        )

        # Initial LR should be max_lr / div_factor
        assert scheduler.get_lr(0) == pytest.approx(1.0 / 25.0, rel=0.01)

        # At pct_start, should be at max_lr
        assert scheduler.get_lr(300) == pytest.approx(1.0, rel=0.01)

        # Final LR should be very small
        final_lr = scheduler.get_lr(999)
        assert final_lr < 0.001

    def test_one_cycle_linear(self):
        """Test one cycle with linear annealing."""
        scheduler = OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.5,
            anneal_strategy="linear",
        )

        # Should increase linearly in first half
        lr_0 = scheduler.get_lr(0)
        lr_25 = scheduler.get_lr(25)
        lr_50 = scheduler.get_lr(50)

        # Linear increase
        assert lr_25 > lr_0
        assert lr_50 > lr_25


class TestPolynomialDecayLR:
    """Tests for PolynomialDecayLR."""

    def test_linear_decay(self):
        """Test linear decay (power=1)."""
        scheduler = PolynomialDecayLR(
            base_lr=1.0,
            total_steps=100,
            power=1.0,
        )

        assert scheduler.get_lr(0) == pytest.approx(1.0)
        assert scheduler.get_lr(50) == pytest.approx(0.5)
        assert scheduler.get_lr(100) == pytest.approx(0.0)

    def test_quadratic_decay(self):
        """Test quadratic decay (power=2)."""
        scheduler = PolynomialDecayLR(
            base_lr=1.0,
            total_steps=100,
            power=2.0,
        )

        assert scheduler.get_lr(0) == pytest.approx(1.0)
        assert scheduler.get_lr(50) == pytest.approx(0.25)


class TestInverseSqrtScheduler:
    """Tests for InverseSqrtScheduler."""

    def test_inverse_sqrt(self):
        """Test inverse sqrt decay."""
        scheduler = InverseSqrtScheduler(
            base_lr=1.0,
            warmup_steps=100,
        )

        # During warmup
        assert scheduler.get_lr(50) == pytest.approx(0.5)
        assert scheduler.get_lr(100) == pytest.approx(1.0)

        # After warmup - inverse sqrt decay
        # lr = base_lr * sqrt(warmup_steps / step)
        assert scheduler.get_lr(400) == pytest.approx(0.5)


class TestMultiStepLR:
    """Tests for MultiStepLR."""

    def test_multi_step(self):
        """Test multi-step decay."""
        scheduler = MultiStepLR(
            base_lr=1.0,
            milestones=[100, 200, 300],
            gamma=0.1,
        )

        assert scheduler.get_lr(50) == pytest.approx(1.0)
        assert scheduler.get_lr(150) == pytest.approx(0.1)
        assert scheduler.get_lr(250) == pytest.approx(0.01)
        assert scheduler.get_lr(350) == pytest.approx(0.001)


class TestChainedScheduler:
    """Tests for ChainedScheduler."""

    def test_chained_scheduler(self):
        """Test chaining multiple schedulers."""
        warmup = LinearWarmupLR(base_lr=1.0, warmup_steps=100)
        constant = ConstantLR(base_lr=1.0)

        scheduler = ChainedScheduler([
            (warmup, 100),
            (constant, 100),
        ])

        # During warmup phase
        assert scheduler.get_lr(50) == pytest.approx(0.51, rel=0.01)

        # During constant phase
        assert scheduler.get_lr(150) == pytest.approx(1.0)


# ============================================================================
# Callback Tests
# ============================================================================


class TestCallbacks:
    """Tests for callback classes."""

    def test_callback_list(self):
        """Test CallbackList dispatches to all callbacks."""
        events = []

        class TestCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_step_end(self, state):
                events.append(f"{self.name}_step_{state.step}")

        callbacks = CallbackList([
            TestCallback("a"),
            TestCallback("b"),
        ])

        state = TrainingState(step=0)
        callbacks.on_step_end(state)

        assert "a_step_0" in events
        assert "b_step_0" in events

    def test_early_stopping_min_mode(self):
        """Test early stopping with min mode."""
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min",
            verbose=False,
        )

        state = TrainingState()
        early_stop.on_train_begin(state)

        # Improvement
        state.metrics = {"val_loss": 1.0}
        early_stop.on_epoch_end(state)
        assert not early_stop.stop_training

        # No improvement
        state.metrics = {"val_loss": 1.1}
        early_stop.on_epoch_end(state)
        assert not early_stop.stop_training
        assert early_stop.wait == 1

        # Still no improvement - should trigger stop
        state.metrics = {"val_loss": 1.2}
        early_stop.on_epoch_end(state)
        assert early_stop.stop_training

    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode."""
        early_stop = EarlyStopping(
            monitor="val_acc",
            patience=2,
            mode="max",
            verbose=False,
        )

        state = TrainingState()
        early_stop.on_train_begin(state)

        # Improvement
        state.metrics = {"val_acc": 0.9}
        early_stop.on_epoch_end(state)
        assert not early_stop.stop_training

        # No improvement
        state.metrics = {"val_acc": 0.85}
        early_stop.on_epoch_end(state)
        assert early_stop.wait == 1

    def test_lr_monitor(self):
        """Test LR monitor logs correctly."""
        lr_monitor = LRMonitor(log_every_n_steps=1, verbose=False)

        state = TrainingState(step=0, learning_rate=0.001)
        lr_monitor.on_step_end(state)

        state.step = 1
        state.learning_rate = 0.0005
        lr_monitor.on_step_end(state)

        assert len(lr_monitor.lr_history) == 2
        assert lr_monitor.lr_history[0] == (0, 0.001)
        assert lr_monitor.lr_history[1] == (1, 0.0005)

    def test_lambda_callback(self):
        """Test lambda callback."""
        events = []

        callback = LambdaCallback(
            on_step_end=lambda state: events.append(f"step_{state.step}")
        )

        state = TrainingState(step=0)
        callback.on_step_end(state)

        assert events == ["step_0"]


class TestModelCheckpoint:
    """Tests for ModelCheckpoint."""

    def test_checkpoint_saves_best(self):
        """Test checkpoint saves on improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                save_dir=tmpdir,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=False,
            )

            model = nn.Linear(10, 10)
            state = TrainingState(model=model)

            checkpoint.on_train_begin(state)

            # First epoch - should save
            state.epoch = 0
            state.metrics = {"val_loss": 1.0}
            checkpoint.on_epoch_end(state)

            best_path = Path(tmpdir) / "checkpoint_best.safetensors"
            assert best_path.exists()

            # Second epoch - worse, should not update
            state.epoch = 1
            state.metrics = {"val_loss": 1.5}
            initial_mtime = best_path.stat().st_mtime_ns

            checkpoint.on_epoch_end(state)

            # File should not be modified
            assert best_path.stat().st_mtime_ns == initial_mtime


# ============================================================================
# EMA Tests
# ============================================================================


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_update(self):
        """Test EMA updates shadow params."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Get initial shadow params
        initial_shadow = ema.shadow_params["weight"][0, 0].item()

        # Modify model params
        model.weight = model.weight + 1.0
        mx.eval(model.weight)

        # Update EMA
        ema.update(step=0)

        # Shadow should be updated (0.9 * old + 0.1 * new)
        new_shadow = ema.shadow_params["weight"][0, 0].item()
        assert new_shadow != initial_shadow

    def test_ema_apply_restore(self):
        """Test apply_shadow and restore."""
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Store original weight
        original_weight = model.weight[0, 0].item()

        # Modify model
        model.weight = model.weight + 10.0
        mx.eval(model.weight)

        # Update EMA
        ema.update(step=0)

        # Apply shadow (should change model weights)
        ema.apply_shadow()
        shadow_weight = model.weight[0, 0].item()
        assert shadow_weight != original_weight + 10.0

        # Restore (should bring back training weights)
        ema.restore()
        restored_weight = model.weight[0, 0].item()
        assert restored_weight == pytest.approx(original_weight + 10.0)

    def test_ema_with_warmup(self):
        """Test EMA with decay warmup."""
        model = nn.Linear(10, 10)
        ema = EMAWithWarmup(
            model,
            decay=0.999,
            warmup_steps=100,
            min_decay=0.0,
        )

        # At step 0, decay should be min_decay
        assert ema.get_decay(0) == pytest.approx(0.0)

        # At step 50, decay should be 0.5 * target
        assert ema.get_decay(50) == pytest.approx(0.4995)

        # At step 100+, decay should be target
        assert ema.get_decay(100) == pytest.approx(0.999)


# ============================================================================
# Gradient Utility Tests
# ============================================================================


class TestGradientAccumulator:
    """Tests for GradientAccumulator."""

    def test_accumulation(self):
        """Test gradient accumulation."""
        accumulator = GradientAccumulator(accumulation_steps=4)

        grads = {"layer": {"weight": mx.ones((2, 2))}}

        # Accumulate 4 times
        for i in range(4):
            accumulator.accumulate(grads)

            if i < 3:
                assert not accumulator.should_step(i)
            else:
                assert accumulator.should_step(i)

        # Get accumulated (normalized)
        result = accumulator.get_accumulated(normalize=True)
        expected = mx.ones((2, 2))  # 4 * 1 / 4 = 1

        assert mx.allclose(result["layer"]["weight"], expected)

    def test_accumulation_reset(self):
        """Test accumulator reset."""
        accumulator = GradientAccumulator(accumulation_steps=2)

        grads = {"weight": mx.ones((2, 2))}
        accumulator.accumulate(grads)
        accumulator.accumulate(grads)
        accumulator.reset()

        assert accumulator.accumulated_grads is None
        assert accumulator.num_accumulated == 0


class TestGradientClipper:
    """Tests for GradientClipper."""

    def test_clip_by_norm(self):
        """Test gradient clipping by norm."""
        clipper = GradientClipper(max_norm=1.0)

        # Gradients with large norm
        grads = {"weight": mx.array([[3.0, 4.0]])}  # norm = 5

        clipped, norm = clipper.clip_by_norm(grads)

        # Original norm should be 5
        assert float(norm) == pytest.approx(5.0)

        # Clipped norm should be <= 1
        clipped_norm = mx.sqrt(mx.sum(clipped["weight"] ** 2))
        assert float(clipped_norm) <= 1.0 + 1e-6

    def test_clip_by_value(self):
        """Test gradient clipping by value."""
        clipper = GradientClipper(max_value=1.0)

        grads = {"weight": mx.array([[5.0, -5.0, 0.5]])}

        clipped = clipper.clip_by_value(grads)

        expected = mx.array([[1.0, -1.0, 0.5]])
        assert mx.allclose(clipped["weight"], expected)

    def test_functional_clip_grad_norm(self):
        """Test functional clip_grad_norm."""
        grads = {"weight": mx.array([[3.0, 4.0]])}

        clipped, norm = clip_grad_norm(grads, max_norm=1.0)

        assert float(norm) == pytest.approx(5.0)
        clipped_norm = mx.sqrt(mx.sum(clipped["weight"] ** 2))
        assert float(clipped_norm) <= 1.0 + 1e-6

    def test_functional_clip_grad_value(self):
        """Test functional clip_grad_value."""
        grads = {"weight": mx.array([[5.0, -5.0]])}

        clipped = clip_grad_value(grads, max_value=2.0)

        expected = mx.array([[2.0, -2.0]])
        assert mx.allclose(clipped["weight"], expected)


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(learning_rate=0.01)

        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            return mx.mean((pred - y) ** 2), {}

        config = TrainingConfig(
            learning_rate=0.01,
            epochs=1,
            max_grad_norm=1.0,
        )

        trainer = Trainer(model, optimizer, loss_fn, config)

        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.grad_clipper is not None

    def test_trainer_with_scheduler(self):
        """Test trainer with learning rate scheduler."""
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(learning_rate=0.01)

        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            return mx.mean((pred - y) ** 2), {}

        scheduler = CosineAnnealingLR(base_lr=0.01, T_max=100)

        trainer = Trainer(
            model, optimizer, loss_fn,
            scheduler=scheduler
        )

        # Scheduler should update LR
        initial_lr = trainer._update_learning_rate()
        assert initial_lr == pytest.approx(0.01)

    def test_trainer_simple_training(self):
        """Test simple training loop."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(learning_rate=0.1)

        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            return mx.mean((pred - y) ** 2), {}

        config = TrainingConfig(
            epochs=1,
            log_every_n_steps=100,  # Don't log during test
        )

        trainer = Trainer(model, optimizer, loss_fn, config)

        # Create simple data
        data = [
            (mx.random.normal((4, 10)), mx.random.normal((4, 5)))
            for _ in range(5)
        ]

        # Train
        history = trainer.fit(data)

        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_trainer_with_callbacks(self):
        """Test trainer with callbacks."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(learning_rate=0.1)

        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            return mx.mean((pred - y) ** 2), {}

        events = []

        callback = LambdaCallback(
            on_train_begin=lambda s: events.append("train_begin"),
            on_train_end=lambda s: events.append("train_end"),
            on_epoch_begin=lambda s: events.append(f"epoch_{s.epoch}_begin"),
            on_epoch_end=lambda s: events.append(f"epoch_{s.epoch}_end"),
        )

        config = TrainingConfig(epochs=2, log_every_n_steps=100)

        trainer = Trainer(model, optimizer, loss_fn, config, callbacks=[callback])

        data = [(mx.random.normal((4, 10)), mx.random.normal((4, 5)))]

        trainer.fit(data)

        assert "train_begin" in events
        assert "train_end" in events
        assert "epoch_0_begin" in events
        assert "epoch_0_end" in events
        assert "epoch_1_begin" in events
        assert "epoch_1_end" in events


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.learning_rate == 1e-4
        assert config.epochs == 10
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            learning_rate=3e-4,
            epochs=5,
            max_grad_norm=0.5,
            use_ema=True,
        )

        assert config.learning_rate == 3e-4
        assert config.epochs == 5
        assert config.max_grad_norm == 0.5
        assert config.use_ema is True


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark
class TestSchedulerBenchmarks:
    """Benchmark tests for schedulers."""

    def test_cosine_annealing_benchmark(self, benchmark):
        """Benchmark cosine annealing scheduler."""
        scheduler = CosineAnnealingLR(base_lr=1.0, T_max=10000)

        def run_scheduler():
            for _ in range(1000):
                scheduler.step()

        benchmark(run_scheduler)

    def test_warmup_cosine_benchmark(self, benchmark):
        """Benchmark warmup cosine scheduler."""
        scheduler = WarmupCosineScheduler(
            base_lr=1.0,
            warmup_steps=1000,
            total_steps=10000,
        )

        def run_scheduler():
            for _ in range(1000):
                scheduler.step()

        benchmark(run_scheduler)


@pytest.mark.benchmark
class TestGradientBenchmarks:
    """Benchmark tests for gradient utilities."""

    def test_gradient_clipping_benchmark(self, benchmark):
        """Benchmark gradient clipping."""
        clipper = GradientClipper(max_norm=1.0)

        # Large gradient dictionary
        grads = {
            f"layer_{i}": {
                "weight": mx.random.normal((512, 512)),
                "bias": mx.random.normal((512,)),
            }
            for i in range(10)
        }

        def clip_grads():
            return clipper.clip_by_norm(grads)

        benchmark(clip_grads)


# ============================================================================
# SWA (Stochastic Weight Averaging) Tests
# ============================================================================


class TestSWA:
    """Tests for Stochastic Weight Averaging."""

    def test_swa_update(self):
        """Test SWA weight averaging."""
        model = nn.Linear(10, 10)
        swa = SWA(model, swa_start=0, swa_freq=1)

        # Initial weights
        initial_weight = model.weight[0, 0].item()

        # Take a few steps
        for _ in range(5):
            model.weight = model.weight + 0.1
            mx.eval(model.weight)
            swa.update(step=0)

        # SWA weights should be different from current
        swa.apply_swa()
        swa_weight = model.weight[0, 0].item()

        # Should be average of seen weights
        assert swa_weight != initial_weight + 0.5

    def test_swa_with_bn_update(self):
        """Test SWA updates batch norm statistics."""
        model = nn.Sequential(
            nn.Linear(10, 10),
        )
        swa = SWA(model, swa_start=0, swa_freq=1)

        # Should work without error
        swa.update(step=0)


# ============================================================================
# Lookahead Tests
# ============================================================================


class TestLookahead:
    """Tests for Lookahead optimizer wrapper."""

    def test_lookahead_wrapping(self):
        """Test Lookahead wraps optimizer correctly."""
        base_optimizer = optim.SGD(learning_rate=0.1)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        assert optimizer.k == 5
        assert optimizer.alpha == 0.5

    def test_lookahead_step(self):
        """Test Lookahead step updates correctly."""
        model = nn.Linear(10, 10)
        base_optimizer = optim.SGD(learning_rate=0.1)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        # Initialize slow weights
        optimizer.init_slow_weights(model)

        # Perform updates
        for i in range(10):
            grads = {"weight": mx.random.normal((10, 10))}
            optimizer.step(model, grads, step=i)

        # Should not error
        assert True


# ============================================================================
# SAM (Sharpness-Aware Minimization) Tests
# ============================================================================


class TestSAM:
    """Tests for Sharpness-Aware Minimization."""

    def test_sam_initialization(self):
        """Test SAM initializes correctly."""
        base_optimizer = optim.SGD(learning_rate=0.1)
        optimizer = SAM(base_optimizer, rho=0.05)

        assert optimizer.rho == 0.05

    def test_sam_two_step(self):
        """Test SAM performs two-step optimization."""
        model = nn.Linear(10, 10)
        base_optimizer = optim.SGD(learning_rate=0.1)
        optimizer = SAM(base_optimizer, rho=0.05)

        # First step - compute gradient at perturbed weights
        grads = {"weight": mx.random.normal((10, 10))}

        # SAM first step (perturb)
        optimizer.first_step(model, grads)

        # SAM second step (update)
        grads2 = {"weight": mx.random.normal((10, 10))}
        optimizer.second_step(model, grads2)

        # Should complete without error
        assert True


# ============================================================================
# GradientNoiseInjection Tests
# ============================================================================


class TestGradientNoiseInjection:
    """Tests for gradient noise injection."""

    def test_noise_addition(self):
        """Test noise is added to gradients."""
        noise_injector = GradientNoiseInjection(eta=0.1, gamma=0.55)

        grads = {"weight": mx.zeros((10, 10))}

        noisy_grads = noise_injector.add_noise(grads, step=0)

        # Should no longer be all zeros
        assert not mx.all(noisy_grads["weight"] == 0)

    def test_noise_decay(self):
        """Test noise decays over steps."""
        noise_injector = GradientNoiseInjection(eta=1.0, gamma=0.55)

        grads = {"weight": mx.zeros((100, 100))}

        # Get noise at different steps
        noisy_1 = noise_injector.add_noise(grads, step=1)
        noisy_1000 = noise_injector.add_noise(grads, step=1000)

        # Later step should have smaller noise variance
        var_1 = float(mx.var(noisy_1["weight"]))
        var_1000 = float(mx.var(noisy_1000["weight"]))

        assert var_1000 < var_1


# ============================================================================
# MixedPrecisionManager Tests
# ============================================================================


class TestMixedPrecisionManager:
    """Tests for MixedPrecisionManager."""

    def test_cast_to_fp16(self):
        """Test casting to fp16."""
        mp = MixedPrecisionManager(dtype=mx.float16)

        x = mx.random.normal((10, 10), dtype=mx.float32)
        x_fp16 = mp.cast_forward(x)

        assert x_fp16.dtype == mx.float16

    def test_scale_loss(self):
        """Test loss scaling."""
        mp = MixedPrecisionManager(dtype=mx.float16, loss_scale=128.0)

        loss = mx.array(1.0)
        scaled = mp.scale_loss(loss)

        assert float(scaled) == 128.0

    def test_unscale_grads(self):
        """Test gradient unscaling."""
        mp = MixedPrecisionManager(dtype=mx.float16, loss_scale=128.0)

        grads = {"weight": mx.array([[128.0, 256.0]])}
        unscaled = mp.unscale_grads(grads)

        assert float(unscaled["weight"][0, 0]) == 1.0
        assert float(unscaled["weight"][0, 1]) == 2.0


# ============================================================================
# Checkpointer Tests
# ============================================================================


class TestCheckpointer:
    """Tests for Checkpointer utility."""

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = Checkpointer(save_dir=tmpdir)

            model = nn.Linear(10, 10)
            optimizer = optim.SGD(learning_rate=0.01)

            # Save checkpoint
            checkpointer.save(
                model=model,
                optimizer=optimizer,
                step=100,
                metrics={"loss": 0.5},
            )

            # Check file exists
            checkpoint_path = Path(tmpdir) / "checkpoint_step_100.safetensors"
            assert checkpoint_path.exists()

    def test_load_latest(self):
        """Test loading latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = Checkpointer(save_dir=tmpdir)

            model = nn.Linear(10, 10)
            optimizer = optim.SGD(learning_rate=0.01)

            # Save multiple checkpoints
            for step in [100, 200, 300]:
                checkpointer.save(model=model, optimizer=optimizer, step=step)

            # Load latest
            latest = checkpointer.get_latest_checkpoint()

            assert "300" in str(latest)
