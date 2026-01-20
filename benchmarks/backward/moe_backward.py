"""Backward pass benchmarks for MoE operations."""

from typing import Optional

import mlx.core as mx

from benchmarks.config import BenchmarkConfig, MoESizes
from benchmarks.utils import BenchmarkResult, benchmark_backward


class MoEBackwardBenchmarks:
    """Backward pass benchmarks for MoE operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[MoESizes] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or MoESizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all MoE backward benchmarks."""
        results = []
        results.extend(self.run_moe_layer_backward())
        results.extend(self.run_router_backward())
        return results

    def run_moe_layer_backward(self) -> list[BenchmarkResult]:
        """Benchmark MoE layer backward passes."""
        results = []

        for num_experts in self.sizes.num_experts[:2]:
            for hidden_dim in self.sizes.hidden_dims[:2]:
                batch_size = 2
                seq_len = 256

                result = self._benchmark_moe_layer_backward(
                    batch_size, seq_len, hidden_dim, num_experts, top_k=2
                )
                if result:
                    results.append(result)

        return results

    def run_router_backward(self) -> list[BenchmarkResult]:
        """Benchmark router backward passes."""
        results = []

        for num_experts in self.sizes.num_experts[:2]:
            for hidden_dim in self.sizes.hidden_dims[:2]:
                batch_size = 4
                seq_len = 512

                result = self._benchmark_topk_router_backward(
                    batch_size, seq_len, hidden_dim, num_experts, top_k=2
                )
                if result:
                    results.append(result)

        return results

    def _benchmark_moe_layer_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark MoE layer backward."""
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

        def fn(x):
            output, aux_loss, router_logits = layer(x)  # MoE returns MoEOutput
            return output

        name = f"moe_layer_backward_b{batch_size}_s{seq_len}_e{num_experts}"
        try:
            result = benchmark_backward(
                fn, [x],
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
            "operation": "moe_layer",
        }
        return result

    def _benchmark_topk_router_backward(
        self, batch_size: int, seq_len: int, hidden_dim: int, num_experts: int, top_k: int
    ) -> Optional[BenchmarkResult]:
        """Benchmark TopK router backward."""
        try:
            from mlx_primitives.advanced import TopKRouter
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        router = TopKRouter(dims=hidden_dim, num_experts=num_experts, top_k=top_k)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn(x):
            weights, indices, router_logits = router(x)  # Returns 3 values
            return weights

        name = f"topk_router_backward_b{batch_size}_s{seq_len}_e{num_experts}"
        try:
            result = benchmark_backward(
                fn, [x],
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
            "num_experts": num_experts,
            "top_k": top_k,
            "type": "backward",
            "operation": "topk_router",
        }
        return result
