"""MoE (Mixture of Experts) parity benchmarks."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from benchmarks.parity.config import ParityBenchmarkConfig, DEFAULT_SIZES
from benchmarks.utils import BenchmarkResult

# Framework availability checks
try:
    import torch
    import torch.nn.functional as F
    HAS_PYTORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False
    HAS_MPS = False
    torch = None
    F = None

try:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    jnn = None


class MoEParityBenchmarks:
    """Multi-framework MoE benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all MoE parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            results[f"topk_routing_{size}"] = self._benchmark_to_list(
                self.benchmark_topk_routing(size)
            )
            results[f"expert_dispatch_{size}"] = self._benchmark_to_list(
                self.benchmark_expert_dispatch(size)
            )
            results[f"load_balancing_loss_{size}"] = self._benchmark_to_list(
                self.benchmark_load_balancing_loss(size)
            )
            results[f"full_moe_forward_{size}"] = self._benchmark_to_list(
                self.benchmark_full_moe_forward(size)
            )

        # Scaling analysis
        for num_experts in [4, 8, 16, 32]:
            results[f"routing_experts_{num_experts}"] = self._benchmark_to_list(
                self._benchmark_routing_scaling(num_experts)
            )

        return results

    def _benchmark_to_list(self, results: Dict[str, BenchmarkResult]) -> List[BenchmarkResult]:
        """Convert dict of results to list."""
        return list(results.values())

    def _benchmark_mlx(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> BenchmarkResult:
        """Benchmark an MLX function."""
        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            if isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, tuple):
                mx.eval(*[r for r in result if isinstance(r, mx.array)])

        # Timed iterations
        times = []
        for _ in range(iterations):
            mx.synchronize()
            start = time.perf_counter()
            result = fn()
            if isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, tuple):
                mx.eval(*[r for r in result if isinstance(r, mx.array)])
            mx.synchronize()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "mlx"},
        )

    def _benchmark_pytorch(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> Optional[BenchmarkResult]:
        """Benchmark a PyTorch function on MPS."""
        if not HAS_PYTORCH or not HAS_MPS:
            return None

        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            fn()
            torch.mps.synchronize()

        # Timed iterations
        times = []
        for _ in range(iterations):
            torch.mps.synchronize()
            start = time.perf_counter()
            fn()
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "pytorch_mps"},
        )

    def _benchmark_jax(
        self,
        fn,
        name: str,
        iterations: int = None,
        warmup_iterations: int = None,
    ) -> Optional[BenchmarkResult]:
        """Benchmark a JAX function."""
        if not HAS_JAX:
            return None

        iterations = iterations or self.config.benchmark_iterations
        warmup_iterations = warmup_iterations or self.config.warmup_iterations

        # Warmup
        for _ in range(warmup_iterations):
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
            times.append(time.perf_counter() - start)

        benchmark_result = BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "jax"},
        )

        # Clear JAX JIT cache
        jax.clear_caches()

        return benchmark_result

    # =========================================================================
    # Top-K Routing Benchmarks
    # =========================================================================

    def benchmark_topk_routing(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark TopK routing across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced.moe import TopKRouter

        config = self.sizes.get_config("moe", size)
        batch, seq, dim, num_experts, top_k = config
        results = {}

        # MLX benchmark
        router = TopKRouter(dims=dim, num_experts=num_experts, top_k=top_k)
        mx.eval(router.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: router(x_mlx),
            name=f"topk_routing_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")
            gate_weight = torch.randn(dim, num_experts, device="mps") * 0.02

            def pytorch_topk_routing():
                logits = x_torch @ gate_weight
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                gate_weights = F.softmax(topk_values, dim=-1)
                return gate_weights, topk_indices, logits

            result = self._benchmark_pytorch(
                pytorch_topk_routing,
                name=f"topk_routing_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))
            gate_weight_jax = jnp.array(np.random.randn(dim, num_experts).astype(np.float32)) * 0.02

            @jax.jit
            def jax_topk_routing():
                logits = x_jax @ gate_weight_jax
                topk_values, topk_indices = jax.lax.top_k(logits, top_k)
                gate_weights = jnn.softmax(topk_values, axis=-1)
                return gate_weights, topk_indices, logits

            result = self._benchmark_jax(
                jax_topk_routing,
                name=f"topk_routing_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Expert Dispatch Benchmarks
    # =========================================================================

    def benchmark_expert_dispatch(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark expert dispatch across MLX, PyTorch MPS, and JAX."""
        config = self.sizes.get_config("moe", size)
        batch, seq, dim, num_experts, top_k = config
        hidden_dim = dim * 4
        n_tokens = batch * seq
        results = {}

        # MLX benchmark - simplified dispatch (routing + gather)
        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        gate_weight_np = np.random.randn(dim, num_experts).astype(np.float32) * 0.02

        x_mlx = mx.array(x_np)
        gate_weight_mlx = mx.array(gate_weight_np)
        mx.eval(x_mlx, gate_weight_mlx)

        def mlx_dispatch():
            x_flat = x_mlx.reshape(-1, dim)
            logits = x_flat @ gate_weight_mlx
            sorted_indices = mx.argsort(-logits, axis=-1)
            expert_indices = sorted_indices[..., :top_k]
            flat_logits = logits
            flat_indices = expert_indices
            gathered = []
            for k in range(top_k):
                idx = flat_indices[:, k]
                vals = mx.take_along_axis(flat_logits, idx[:, None], axis=1).squeeze(-1)
                gathered.append(vals)
            selected_logits = mx.stack(gathered, axis=-1)
            gate_weights = mx.softmax(selected_logits, axis=-1)
            return gate_weights, expert_indices

        results["mlx"] = self._benchmark_mlx(
            mlx_dispatch,
            name=f"expert_dispatch_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.from_numpy(x_np).to("mps")
            gate_weight_torch = torch.from_numpy(gate_weight_np).to("mps")

            def pytorch_dispatch():
                x_flat = x_torch.reshape(-1, dim)
                logits = x_flat @ gate_weight_torch
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                gate_weights = F.softmax(topk_values, dim=-1)
                return gate_weights, topk_indices

            result = self._benchmark_pytorch(
                pytorch_dispatch,
                name=f"expert_dispatch_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            x_jax = jnp.array(x_np)
            gate_weight_jax = jnp.array(gate_weight_np)

            @jax.jit
            def jax_dispatch():
                x_flat = x_jax.reshape(-1, dim)
                logits = x_flat @ gate_weight_jax
                topk_values, topk_indices = jax.lax.top_k(logits, top_k)
                gate_weights = jnn.softmax(topk_values, axis=-1)
                return gate_weights, topk_indices

            result = self._benchmark_jax(
                jax_dispatch,
                name=f"expert_dispatch_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Load Balancing Loss Benchmarks
    # =========================================================================

    def benchmark_load_balancing_loss(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark load balancing loss across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced.moe import load_balancing_loss

        config = self.sizes.get_config("moe", size)
        batch, seq, dim, num_experts, top_k = config
        results = {}

        # Generate inputs
        np.random.seed(42)
        router_logits_np = np.random.randn(batch, seq, num_experts).astype(np.float32)
        expert_indices_np = np.random.randint(0, num_experts, (batch, seq, top_k))

        # MLX benchmark
        router_logits_mlx = mx.array(router_logits_np)
        expert_indices_mlx = mx.array(expert_indices_np)
        mx.eval(router_logits_mlx, expert_indices_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: load_balancing_loss(router_logits_mlx, expert_indices_mlx, num_experts),
            name=f"load_balancing_loss_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            router_logits_torch = torch.from_numpy(router_logits_np).to("mps")
            expert_indices_torch = torch.from_numpy(expert_indices_np).to("mps")

            def pytorch_load_balancing_loss():
                router_probs = F.softmax(router_logits_torch, dim=-1)
                total_tokens = batch * seq * top_k
                expert_counts = torch.zeros(num_experts, device="mps", dtype=torch.float32)
                for e in range(num_experts):
                    expert_counts[e] = (expert_indices_torch == e).sum().float()
                expert_fraction = expert_counts / total_tokens
                mean_prob = router_probs.mean(dim=(0, 1))
                return num_experts * (expert_fraction * mean_prob).sum()

            result = self._benchmark_pytorch(
                pytorch_load_balancing_loss,
                name=f"load_balancing_loss_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            router_logits_jax = jnp.array(router_logits_np)
            expert_indices_jax = jnp.array(expert_indices_np)

            @jax.jit
            def jax_load_balancing_loss():
                router_probs = jnn.softmax(router_logits_jax, axis=-1)
                total_tokens = batch * seq * top_k
                # Use one-hot encoding for counting
                one_hot = jnn.one_hot(expert_indices_jax, num_experts)
                expert_counts = one_hot.sum(axis=(0, 1, 2))
                expert_fraction = expert_counts / total_tokens
                mean_prob = router_probs.mean(axis=(0, 1))
                return num_experts * jnp.sum(expert_fraction * mean_prob)

            result = self._benchmark_jax(
                jax_load_balancing_loss,
                name=f"load_balancing_loss_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Full MoE Forward Benchmarks
    # =========================================================================

    def benchmark_full_moe_forward(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark full MoE layer forward pass across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced.moe import MoELayer

        config = self.sizes.get_config("moe", size)
        batch, seq, dim, num_experts, top_k = config
        hidden_dim = dim * 4
        results = {}

        # MLX benchmark
        moe = MoELayer(dims=dim, hidden_dims=hidden_dim, num_experts=num_experts, top_k=top_k)
        mx.eval(moe.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        def mlx_moe_forward():
            out = moe(x_mlx)
            return out.output

        results["mlx"] = self._benchmark_mlx(
            mlx_moe_forward,
            name=f"full_moe_forward_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            np.random.seed(42)
            x_np = np.random.randn(batch, seq, dim).astype(np.float32)
            gate_weight_np = np.random.randn(dim, num_experts).astype(np.float32) * 0.02
            expert_w1_np = np.random.randn(num_experts, dim, hidden_dim).astype(np.float32) * 0.02
            expert_w2_np = np.random.randn(num_experts, hidden_dim, dim).astype(np.float32) * 0.02

            x_torch = torch.from_numpy(x_np).to("mps")
            gate_weight_torch = torch.from_numpy(gate_weight_np).to("mps")
            expert_w1_torch = torch.from_numpy(expert_w1_np).to("mps")
            expert_w2_torch = torch.from_numpy(expert_w2_np).to("mps")

            n_tokens = batch * seq

            def pytorch_moe_forward():
                x_flat = x_torch.reshape(n_tokens, dim)
                logits = x_flat @ gate_weight_torch
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                gate_weights = F.softmax(topk_values, dim=-1)
                output = torch.zeros_like(x_flat)
                for e in range(num_experts):
                    expert_mask = (topk_indices == e)
                    weights_for_expert = torch.where(expert_mask, gate_weights, torch.zeros_like(gate_weights))
                    token_weights = weights_for_expert.sum(dim=-1)
                    routed_mask = token_weights > 0
                    if routed_mask.any():
                        x_expert = x_flat[routed_mask]
                        w_expert = token_weights[routed_mask]
                        hidden = F.silu(x_expert @ expert_w1_torch[e])
                        expert_out = hidden @ expert_w2_torch[e]
                        output[routed_mask] += expert_out * w_expert.unsqueeze(-1)
                return output.reshape(batch, seq, dim)

            result = self._benchmark_pytorch(
                pytorch_moe_forward,
                name=f"full_moe_forward_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark (simplified - just routing overhead)
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))
            gate_weight_jax = jnp.array(np.random.randn(dim, num_experts).astype(np.float32)) * 0.02
            expert_w1_jax = jnp.array(np.random.randn(num_experts, dim, hidden_dim).astype(np.float32)) * 0.02
            expert_w2_jax = jnp.array(np.random.randn(num_experts, hidden_dim, dim).astype(np.float32)) * 0.02

            # JAX MoE is more complex to implement efficiently, use simplified version
            @jax.jit
            def jax_moe_forward():
                x_flat = x_jax.reshape(-1, dim)
                logits = x_flat @ gate_weight_jax
                topk_values, topk_indices = jax.lax.top_k(logits, top_k)
                gate_weights = jnn.softmax(topk_values, axis=-1)
                # Simplified: just compute routing cost
                return gate_weights.sum()

            result = self._benchmark_jax(
                jax_moe_forward,
                name=f"full_moe_forward_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Expert Scaling Analysis
    # =========================================================================

    def run_expert_scaling(self, operation: str) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark scaling behavior as num_experts increases."""
        results = {}
        for num_experts in [4, 8, 16, 32]:
            if operation == "routing":
                results[f"experts_{num_experts}"] = self._benchmark_to_list(
                    self._benchmark_routing_scaling(num_experts)
                )
            elif operation == "dispatch":
                results[f"experts_{num_experts}"] = self._benchmark_to_list(
                    self._benchmark_dispatch_scaling(num_experts)
                )
        return results

    def _benchmark_routing_scaling(self, num_experts: int) -> Dict[str, BenchmarkResult]:
        """Benchmark routing with specific expert count."""
        from mlx_primitives.advanced.moe import TopKRouter

        # Fixed size for scaling analysis
        batch, seq, dim, top_k = 4, 256, 512, 2
        results = {}

        # MLX benchmark
        router = TopKRouter(dims=dim, num_experts=num_experts, top_k=top_k)
        mx.eval(router.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: router(x_mlx),
            name=f"routing_scaling_e{num_experts}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")
            gate_weight = torch.randn(dim, num_experts, device="mps") * 0.02

            def pytorch_routing():
                logits = x_torch @ gate_weight
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                return F.softmax(topk_values, dim=-1), topk_indices

            result = self._benchmark_pytorch(
                pytorch_routing,
                name=f"routing_scaling_e{num_experts}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        return results

    def _benchmark_dispatch_scaling(self, num_experts: int) -> Dict[str, BenchmarkResult]:
        """Benchmark dispatch with specific expert count."""
        # Fixed size for scaling analysis
        batch, seq, dim, top_k = 4, 256, 512, 2
        results = {}

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)
        gate_weight_np = np.random.randn(dim, num_experts).astype(np.float32) * 0.02

        # MLX benchmark
        x_mlx = mx.array(x_np)
        gate_weight_mlx = mx.array(gate_weight_np)
        mx.eval(x_mlx, gate_weight_mlx)

        def mlx_dispatch():
            x_flat = x_mlx.reshape(-1, dim)
            logits = x_flat @ gate_weight_mlx
            sorted_indices = mx.argsort(-logits, axis=-1)
            expert_indices = sorted_indices[..., :top_k]
            return expert_indices

        results["mlx"] = self._benchmark_mlx(
            mlx_dispatch,
            name=f"dispatch_scaling_e{num_experts}_mlx",
        )

        return results
