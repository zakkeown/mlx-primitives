"""Activation parity benchmarks."""

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


class ActivationParityBenchmarks:
    """Multi-framework activation benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all activation parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            # GLU variants
            results[f"swiglu_{size}"] = self._benchmark_to_list(self.benchmark_swiglu(size))
            results[f"geglu_{size}"] = self._benchmark_to_list(self.benchmark_geglu(size))
            results[f"reglu_{size}"] = self._benchmark_to_list(self.benchmark_reglu(size))

            # GELU variants
            results[f"gelu_exact_{size}"] = self._benchmark_to_list(self.benchmark_gelu_exact(size))
            results[f"gelu_approx_{size}"] = self._benchmark_to_list(self.benchmark_gelu_approx(size))
            results[f"quick_gelu_{size}"] = self._benchmark_to_list(self.benchmark_quick_gelu(size))

            # SiLU/Swish
            results[f"silu_{size}"] = self._benchmark_to_list(self.benchmark_silu(size))
            results[f"swish_{size}"] = self._benchmark_to_list(self.benchmark_swish(size))

            # Other activations
            results[f"mish_{size}"] = self._benchmark_to_list(self.benchmark_mish(size))
            results[f"squared_relu_{size}"] = self._benchmark_to_list(self.benchmark_squared_relu(size))
            results[f"hard_swish_{size}"] = self._benchmark_to_list(self.benchmark_hard_swish(size))
            results[f"hard_sigmoid_{size}"] = self._benchmark_to_list(self.benchmark_hard_sigmoid(size))

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

        # Timed iterations
        times = []
        for _ in range(iterations):
            mx.synchronize()
            start = time.perf_counter()
            result = fn()
            if isinstance(result, mx.array):
                mx.eval(result)
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

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name=name,
            mean_time=np.mean(times),
            std_time=np.std(times) if len(times) > 1 else 0.0,
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            metadata={"framework": "jax"},
        )

    # =========================================================================
    # GLU Variants
    # =========================================================================

    def benchmark_swiglu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark SwiGLU across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import SwiGLU

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        hidden_dim = dim * 4
        results = {}

        # MLX benchmark
        swiglu = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: swiglu(x_mlx),
            name=f"swiglu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")
            w1 = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w_gate = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w2 = torch.randn(hidden_dim, dim, device="mps") * 0.02

            def pytorch_swiglu():
                gate = F.silu(x_torch @ w_gate)
                up = x_torch @ w1
                return (gate * up) @ w2

            result = self._benchmark_pytorch(
                pytorch_swiglu,
                name=f"swiglu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))
            w1_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w_gate_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w2_jax = jnp.array(np.random.randn(hidden_dim, dim).astype(np.float32)) * 0.02

            @jax.jit
            def jax_swiglu():
                gate = jnn.silu(x_jax @ w_gate_jax)
                up = x_jax @ w1_jax
                return (gate * up) @ w2_jax

            result = self._benchmark_jax(
                jax_swiglu,
                name=f"swiglu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_geglu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark GeGLU across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import GeGLU

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        hidden_dim = dim * 4
        results = {}

        # MLX benchmark
        geglu = GeGLU(dim, hidden_dim)
        mx.eval(geglu.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: geglu(x_mlx),
            name=f"geglu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")
            w1 = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w_gate = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w2 = torch.randn(hidden_dim, dim, device="mps") * 0.02

            def pytorch_geglu():
                gate = F.gelu(x_torch @ w_gate)
                up = x_torch @ w1
                return (gate * up) @ w2

            result = self._benchmark_pytorch(
                pytorch_geglu,
                name=f"geglu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))
            w1_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w_gate_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w2_jax = jnp.array(np.random.randn(hidden_dim, dim).astype(np.float32)) * 0.02

            @jax.jit
            def jax_geglu():
                gate = jnn.gelu(x_jax @ w_gate_jax)
                up = x_jax @ w1_jax
                return (gate * up) @ w2_jax

            result = self._benchmark_jax(
                jax_geglu,
                name=f"geglu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_reglu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark ReGLU across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import ReGLU

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        hidden_dim = dim * 4
        results = {}

        # MLX benchmark
        reglu = ReGLU(dim, hidden_dim)
        mx.eval(reglu.parameters())
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: reglu(x_mlx),
            name=f"reglu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")
            w1 = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w_gate = torch.randn(dim, hidden_dim, device="mps") * 0.02
            w2 = torch.randn(hidden_dim, dim, device="mps") * 0.02

            def pytorch_reglu():
                gate = F.relu(x_torch @ w_gate)
                up = x_torch @ w1
                return (gate * up) @ w2

            result = self._benchmark_pytorch(
                pytorch_reglu,
                name=f"reglu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))
            w1_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w_gate_jax = jnp.array(np.random.randn(dim, hidden_dim).astype(np.float32)) * 0.02
            w2_jax = jnp.array(np.random.randn(hidden_dim, dim).astype(np.float32)) * 0.02

            @jax.jit
            def jax_reglu():
                gate = jnn.relu(x_jax @ w_gate_jax)
                up = x_jax @ w1_jax
                return (gate * up) @ w2_jax

            result = self._benchmark_jax(
                jax_reglu,
                name=f"reglu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # GELU Variants
    # =========================================================================

    def benchmark_gelu_exact(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark exact GELU across MLX, PyTorch MPS, and JAX."""
        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: nn.gelu(x_mlx),
            name=f"gelu_exact_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.gelu(x_torch, approximate='none'),
                name=f"gelu_exact_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_gelu_exact():
                return jnn.gelu(x_jax, approximate=False)

            result = self._benchmark_jax(
                jax_gelu_exact,
                name=f"gelu_exact_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_gelu_approx(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark approximate GELU (tanh) across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import gelu_tanh

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: gelu_tanh(x_mlx),
            name=f"gelu_approx_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.gelu(x_torch, approximate='tanh'),
                name=f"gelu_approx_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_gelu_approx():
                return jnn.gelu(x_jax, approximate=True)

            result = self._benchmark_jax(
                jax_gelu_approx,
                name=f"gelu_approx_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_quick_gelu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark QuickGELU across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import quick_gelu

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: quick_gelu(x_mlx),
            name=f"quick_gelu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            def pytorch_quick_gelu():
                return x_torch * torch.sigmoid(1.702 * x_torch)

            result = self._benchmark_pytorch(
                pytorch_quick_gelu,
                name=f"quick_gelu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_quick_gelu():
                return x_jax * jnn.sigmoid(1.702 * x_jax)

            result = self._benchmark_jax(
                jax_quick_gelu,
                name=f"quick_gelu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # SiLU / Swish Variants
    # =========================================================================

    def benchmark_silu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark SiLU across MLX, PyTorch MPS, and JAX."""
        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: nn.silu(x_mlx),
            name=f"silu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.silu(x_torch),
                name=f"silu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_silu():
                return jnn.silu(x_jax)

            result = self._benchmark_jax(
                jax_silu,
                name=f"silu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_swish(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Swish (with beta=1.5) across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import swish

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        beta = 1.5
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: swish(x_mlx, beta=beta),
            name=f"swish_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            def pytorch_swish():
                return x_torch * torch.sigmoid(beta * x_torch)

            result = self._benchmark_pytorch(
                pytorch_swish,
                name=f"swish_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_swish():
                return x_jax * jnn.sigmoid(beta * x_jax)

            result = self._benchmark_jax(
                jax_swish,
                name=f"swish_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Other Activations
    # =========================================================================

    def benchmark_mish(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Mish across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import mish

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: mish(x_mlx),
            name=f"mish_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.mish(x_torch),
                name=f"mish_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_mish():
                softplus = jnp.log1p(jnp.exp(x_jax))
                return x_jax * jnp.tanh(softplus)

            result = self._benchmark_jax(
                jax_mish,
                name=f"mish_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_squared_relu(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Squared ReLU across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import squared_relu

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: squared_relu(x_mlx),
            name=f"squared_relu_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            def pytorch_squared_relu():
                return F.relu(x_torch) ** 2

            result = self._benchmark_pytorch(
                pytorch_squared_relu,
                name=f"squared_relu_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_squared_relu():
                return jnn.relu(x_jax) ** 2

            result = self._benchmark_jax(
                jax_squared_relu,
                name=f"squared_relu_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_hard_swish(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Hard Swish across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import hard_swish

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: hard_swish(x_mlx),
            name=f"hard_swish_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.hardswish(x_torch),
                name=f"hard_swish_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_hard_swish():
                return x_jax * jnp.clip(x_jax + 3, 0, 6) / 6

            result = self._benchmark_jax(
                jax_hard_swish,
                name=f"hard_swish_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_hard_sigmoid(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark Hard Sigmoid across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.layers import hard_sigmoid

        config = self.sizes.get_config("activation", size)
        batch, seq, dim = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((batch, seq, dim))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: hard_sigmoid(x_mlx),
            name=f"hard_sigmoid_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(batch, seq, dim, device="mps")

            result = self._benchmark_pytorch(
                lambda: F.hardsigmoid(x_torch),
                name=f"hard_sigmoid_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(batch, seq, dim).astype(np.float32))

            @jax.jit
            def jax_hard_sigmoid():
                return jnp.clip(x_jax + 3, 0, 6) / 6

            result = self._benchmark_jax(
                jax_hard_sigmoid,
                name=f"hard_sigmoid_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results
