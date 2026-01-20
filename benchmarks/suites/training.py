"""Training utilities benchmark suite."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from benchmarks.config import BenchmarkConfig, TrainingSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn


class SimpleMLP(nn.Module):
    """Simple MLP for EMA benchmarks."""

    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        return self.layer2(x)


class TrainingBenchmarks:
    """Benchmark suite for training utility operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[TrainingSizes] = None,
    ):
        """Initialize training benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for training benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or TrainingSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all training benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_ema_benchmarks())
        results.extend(self.run_gradient_benchmarks())
        results.extend(self.run_scheduler_benchmarks())
        return results

    def run_ema_benchmarks(self) -> list[BenchmarkResult]:
        """Run EMA (Exponential Moving Average) benchmarks."""
        results = []

        for param_count in self.sizes.param_counts[:4]:
            # EMA update
            result = self._benchmark_ema_update(param_count)
            if result:
                results.append(result)

            # EMA with warmup
            result = self._benchmark_ema_warmup(param_count)
            if result:
                results.append(result)

        return results

    def run_gradient_benchmarks(self) -> list[BenchmarkResult]:
        """Run gradient operation benchmarks."""
        results = []

        for param_count in self.sizes.param_counts[:4]:
            # Gradient clipping by norm
            result = self._benchmark_grad_clip_norm(param_count)
            if result:
                results.append(result)

            # Gradient clipping by value
            result = self._benchmark_grad_clip_value(param_count)
            if result:
                results.append(result)

            # Compute gradient norm
            result = self._benchmark_compute_grad_norm(param_count)
            if result:
                results.append(result)

        return results

    def run_scheduler_benchmarks(self) -> list[BenchmarkResult]:
        """Run learning rate scheduler benchmarks."""
        results = []

        # Cosine annealing
        result = self._benchmark_cosine_scheduler()
        if result:
            results.append(result)

        # Warmup cosine
        result = self._benchmark_warmup_cosine_scheduler()
        if result:
            results.append(result)

        # OneCycle
        result = self._benchmark_onecycle_scheduler()
        if result:
            results.append(result)

        return results

    def _create_model(self, param_count: int) -> nn.Module:
        """Create mock model with specified approximate parameter count.

        Args:
            param_count: Approximate number of parameters.

        Returns:
            nn.Module with approximately that many parameters.
        """
        # Create model to approximate the count
        # SimpleMLP has 2 * dim * dim + 2 * dim parameters
        dim = int((param_count / 2) ** 0.5)
        return SimpleMLP(dim)

    def _create_grad_dict(self, param_count: int) -> dict:
        """Create mock gradient dictionary with specified count.

        Args:
            param_count: Approximate number of parameters.

        Returns:
            Dictionary of gradients.
        """
        # Create gradients to approximate the count
        dim = int((param_count / 2) ** 0.5)
        return {
            "layer1": {"weight": mx.random.normal((dim, dim)), "bias": mx.zeros((dim,))},
            "layer2": {"weight": mx.random.normal((dim, dim)), "bias": mx.zeros((dim,))},
        }

    def _benchmark_ema_update(self, param_count: int) -> Optional[BenchmarkResult]:
        """Benchmark EMA parameter update."""
        try:
            from mlx_primitives.training import EMA
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        model = self._create_model(param_count)
        ema = EMA(model, decay=0.999)

        step = 0

        def fn():
            nonlocal step
            ema.update(step)
            step += 1
            return ema.shadow_params

        name = f"ema_update_p{param_count}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "param_count": param_count,
            "decay": 0.999,
            "type": "ema",
            "operation": "update",
        }
        return result

    def _benchmark_ema_warmup(self, param_count: int) -> Optional[BenchmarkResult]:
        """Benchmark EMA with warmup."""
        try:
            from mlx_primitives.training import EMAWithWarmup
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        model = self._create_model(param_count)
        ema = EMAWithWarmup(model, decay=0.999, warmup_steps=100)

        step = 0

        def fn():
            nonlocal step
            ema.update(step)
            step += 1
            return ema.shadow_params

        name = f"ema_warmup_p{param_count}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "param_count": param_count,
            "decay": 0.999,
            "warmup_steps": 100,
            "type": "ema",
            "operation": "warmup_update",
        }
        return result

    def _benchmark_grad_clip_norm(self, param_count: int) -> Optional[BenchmarkResult]:
        """Benchmark gradient clipping by norm."""
        try:
            from mlx_primitives.training import clip_grad_norm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        grads = self._create_grad_dict(param_count)
        max_norm = 1.0

        def fn():
            return clip_grad_norm(grads, max_norm)

        name = f"grad_clip_norm_p{param_count}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "param_count": param_count,
            "max_norm": max_norm,
            "type": "gradient",
            "operation": "clip_norm",
        }
        return result

    def _benchmark_grad_clip_value(self, param_count: int) -> Optional[BenchmarkResult]:
        """Benchmark gradient clipping by value."""
        try:
            from mlx_primitives.training import clip_grad_value
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        grads = self._create_grad_dict(param_count)
        max_value = 1.0

        def fn():
            return clip_grad_value(grads, max_value)

        name = f"grad_clip_value_p{param_count}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "param_count": param_count,
            "max_value": max_value,
            "type": "gradient",
            "operation": "clip_value",
        }
        return result

    def _benchmark_compute_grad_norm(self, param_count: int) -> Optional[BenchmarkResult]:
        """Benchmark gradient norm computation."""
        try:
            from mlx_primitives.training import compute_gradient_norm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        grads = self._create_grad_dict(param_count)

        def fn():
            return compute_gradient_norm(grads)

        name = f"compute_grad_norm_p{param_count}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "param_count": param_count,
            "type": "gradient",
            "operation": "compute_norm",
        }
        return result

    def _benchmark_cosine_scheduler(self) -> Optional[BenchmarkResult]:
        """Benchmark cosine annealing scheduler."""
        try:
            from mlx_primitives.training import CosineAnnealingLR
        except ImportError:
            return None

        scheduler = CosineAnnealingLR(
            base_lr=1e-3,
            T_max=10000,
            min_lr=1e-5,
        )

        def fn():
            lrs = []
            for step in range(1000):
                lrs.append(scheduler.get_lr(step))
            return lrs

        name = "cosine_scheduler_1000steps"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "total_steps": 10000,
            "query_steps": 1000,
            "type": "scheduler",
            "operation": "CosineAnnealingLR",
        }
        return result

    def _benchmark_warmup_cosine_scheduler(self) -> Optional[BenchmarkResult]:
        """Benchmark warmup + cosine scheduler."""
        try:
            from mlx_primitives.training import WarmupCosineScheduler
        except ImportError:
            return None

        scheduler = WarmupCosineScheduler(
            base_lr=1e-3,
            warmup_steps=1000,
            total_steps=10000,
            min_lr=1e-5,
        )

        def fn():
            lrs = []
            for step in range(1000):
                lrs.append(scheduler.get_lr(step))
            return lrs

        name = "warmup_cosine_scheduler_1000steps"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "warmup_steps": 1000,
            "total_steps": 10000,
            "query_steps": 1000,
            "type": "scheduler",
            "operation": "WarmupCosineScheduler",
        }
        return result

    def _benchmark_onecycle_scheduler(self) -> Optional[BenchmarkResult]:
        """Benchmark OneCycleLR scheduler."""
        try:
            from mlx_primitives.training import OneCycleLR
        except ImportError:
            return None

        scheduler = OneCycleLR(
            max_lr=1e-3,
            total_steps=10000,
            pct_start=0.3,
        )

        def fn():
            lrs = []
            for step in range(1000):
                lrs.append(scheduler.get_lr(step))
            return lrs

        name = "onecycle_scheduler_1000steps"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "total_steps": 10000,
            "query_steps": 1000,
            "pct_start": 0.3,
            "type": "scheduler",
            "operation": "OneCycleLR",
        }
        return result
