"""MoE (Mixture of Experts) benchmark suite."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from benchmarks.config import BenchmarkConfig, MoESizes
from benchmarks.utils import BenchmarkResult, benchmark_fn, benchmark_backward


class MoEBenchmarks:
    """Benchmark suite for Mixture of Experts operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[MoESizes] = None,
    ):
        """Initialize MoE benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for MoE benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or MoESizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all MoE benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_router_benchmarks())
        results.extend(self.run_moe_layer_benchmarks())
        results.extend(self.run_backward_benchmarks())
        results.extend(self.run_gather_scatter_benchmarks())
        return results

    def run_router_benchmarks(self) -> list[BenchmarkResult]:
        """Run router benchmarks."""
        results = []

        for num_experts in self.sizes.num_experts[:3]:
            for hidden_dim in self.sizes.hidden_dims[:3]:
                batch_size = 4
                seq_len = 512

                # TopK Router
                result = self._benchmark_topk_router(batch_size, seq_len, hidden_dim, num_experts, top_k=2)
                if result:
                    results.append(result)

                # Expert Choice Router
                result = self._benchmark_expert_choice_router(batch_size, seq_len, hidden_dim, num_experts)
                if result:
                    results.append(result)

        return results

    def run_moe_layer_benchmarks(self) -> list[BenchmarkResult]:
        """Run full MoE layer benchmarks."""
        results = []

        for num_experts in self.sizes.num_experts[:3]:
            for hidden_dim in self.sizes.hidden_dims[:3]:
                batch_size = 4
                seq_len = 512

                # MoE Layer
                result = self._benchmark_moe_layer(batch_size, seq_len, hidden_dim, num_experts, top_k=2)
                if result:
                    results.append(result)

                # Switch MoE
                result = self._benchmark_switch_moe(batch_size, seq_len, hidden_dim, num_experts)
                if result:
                    results.append(result)

        return results

    def run_backward_benchmarks(self) -> list[BenchmarkResult]:
        """Run backward pass benchmarks for MoE."""
        results = []

        # Use smaller sizes for backward benchmarks
        for num_experts in self.sizes.num_experts[:2]:
            hidden_dim = 512
            batch_size = 2
            seq_len = 256

            result = self._benchmark_moe_backward(batch_size, seq_len, hidden_dim, num_experts, top_k=2)
            if result:
                results.append(result)

        return results

    def _benchmark_topk_router(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark TopK router."""
        try:
            from mlx_primitives.advanced import TopKRouter
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        router = TopKRouter(dims=hidden_dim, num_experts=num_experts, top_k=top_k)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return router(x)

        name = f"topk_router_b{batch_size}_s{seq_len}_e{num_experts}_k{top_k}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "type": "router",
            "layer": "TopKRouter",
        }
        return result

    def _benchmark_expert_choice_router(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Expert Choice router."""
        try:
            from mlx_primitives.advanced import ExpertChoiceRouter
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        capacity_factor = 1.25
        router = ExpertChoiceRouter(dims=hidden_dim, num_experts=num_experts, capacity_factor=capacity_factor)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return router(x)

        name = f"expert_choice_router_b{batch_size}_s{seq_len}_e{num_experts}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "capacity_factor": capacity_factor,
            "type": "router",
            "layer": "ExpertChoiceRouter",
        }
        return result

    def _benchmark_moe_layer(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark full MoE layer."""
        try:
            from mlx_primitives.advanced import MoELayer
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        expert_dim = hidden_dim * 4
        layer = MoELayer(
            dims=hidden_dim,
            hidden_dims=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"moe_layer_b{batch_size}_s{seq_len}_e{num_experts}_k{top_k}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "expert_dim": expert_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "type": "moe",
            "layer": "MoELayer",
        }
        return result

    def _benchmark_switch_moe(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Switch MoE (top-1 routing)."""
        try:
            from mlx_primitives.advanced import SwitchMoE
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        expert_dim = hidden_dim * 4
        layer = SwitchMoE(
            dims=hidden_dim,
            hidden_dims=expert_dim,
            num_experts=num_experts,
        )
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"switch_moe_b{batch_size}_s{seq_len}_e{num_experts}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "expert_dim": expert_dim,
            "num_experts": num_experts,
            "type": "moe",
            "layer": "SwitchMoE",
        }
        return result

    def _benchmark_moe_backward(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark MoE backward pass."""
        try:
            from mlx_primitives.advanced import MoELayer
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        expert_dim = hidden_dim * 4
        layer = MoELayer(
            dims=hidden_dim,
            hidden_dims=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def forward(x):
            output, aux_loss, router_logits = layer(x)  # MoE returns MoEOutput
            return output

        name = f"moe_backward_b{batch_size}_s{seq_len}_e{num_experts}_k{top_k}"
        try:
            result = benchmark_backward(
                forward,
                [x],
                iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                name=name,
            )
        except ValueError as e:
            # VJP not implemented for custom kernels
            if "Not implemented" in str(e):
                return None
            raise
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "expert_dim": expert_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "type": "backward",
            "layer": "MoELayer",
        }
        return result

    def run_gather_scatter_benchmarks(self) -> list[BenchmarkResult]:
        """Run gather/scatter benchmarks for MoE dispatch."""
        results = []

        configs = [
            (4, 512, 256, 8),    # (batch, seq, hidden, num_experts)
            (4, 1024, 512, 8),
            (8, 512, 256, 16),
            (8, 1024, 512, 16),
        ]

        for batch, seq, hidden, experts in configs:
            # Selective gather
            result = self._benchmark_selective_gather(batch, seq, hidden, experts)
            if result:
                results.append(result)

            # Selective scatter add
            result = self._benchmark_selective_scatter_add(batch, seq, hidden, experts)
            if result:
                results.append(result)

        return results

    def _benchmark_selective_gather(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark selective_gather for MoE token dispatch."""
        try:
            from mlx_primitives.primitives.gather_scatter import selective_gather
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Flatten tokens: (batch, seq, hidden) -> (n_tokens, hidden)
        n_tokens = batch_size * seq_len
        x = mx.random.normal((n_tokens, hidden_dim))

        # Simulate MoE routing: select a subset of tokens (capacity per expert)
        # Typically capacity = (n_tokens / num_experts) * capacity_factor
        capacity = n_tokens // num_experts
        indices = mx.random.randint(0, n_tokens, shape=(capacity,))

        def fn():
            return selective_gather(x, indices)

        name = f"selective_gather_b{batch_size}_s{seq_len}_h{hidden_dim}_e{num_experts}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "n_tokens": n_tokens,
            "capacity": capacity,
            "type": "optimized",
            "operation": "selective_gather",
        }
        return result

    def _benchmark_selective_scatter_add(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark selective_scatter_add for MoE output combination."""
        try:
            from mlx_primitives.primitives.gather_scatter import selective_scatter_add
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        # Flatten tokens: (batch, seq) -> n_tokens
        n_tokens = batch_size * seq_len
        # Capacity per expert (tokens routed to each expert)
        capacity = n_tokens // num_experts

        # Output accumulator: shape (n_tokens, hidden_dim)
        output = mx.zeros((n_tokens, hidden_dim))
        # Expert outputs to scatter: shape (capacity, hidden_dim)
        values = mx.random.normal((capacity, hidden_dim))
        # Where to scatter each value: 1D indices
        indices = mx.random.randint(0, n_tokens, shape=(capacity,))
        # Routing weights for each value
        weights = mx.random.uniform(shape=(capacity,))

        def fn():
            return selective_scatter_add(output, values, indices, weights)

        name = f"selective_scatter_add_b{batch_size}_s{seq_len}_h{hidden_dim}_e{num_experts}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "n_tokens": n_tokens,
            "capacity": capacity,
            "type": "optimized",
            "operation": "selective_scatter_add",
        }
        return result
