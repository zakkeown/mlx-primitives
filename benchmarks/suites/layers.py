"""Layer benchmark suite for normalization, activations, pooling, and embeddings."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from benchmarks.config import BenchmarkConfig, LayerSizes
from benchmarks.utils import BenchmarkResult, benchmark_fn, benchmark_backward


class LayerBenchmarks:
    """Benchmark suite for layer operations."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        sizes: Optional[LayerSizes] = None,
    ):
        """Initialize layer benchmarks.

        Args:
            config: Benchmark configuration.
            sizes: Size configurations for layer benchmarks.
        """
        self.config = config or BenchmarkConfig()
        self.sizes = sizes or LayerSizes()

    def run_all(self) -> list[BenchmarkResult]:
        """Run all layer benchmarks.

        Returns:
            List of benchmark results.
        """
        results = []
        results.extend(self.run_normalization_benchmarks())
        results.extend(self.run_ffn_benchmarks())  # SwiGLU, GeGLU (full FFN layers)
        results.extend(self.run_activation_benchmarks())  # Mish, GELU (element-wise)
        results.extend(self.run_pooling_benchmarks())
        results.extend(self.run_embedding_benchmarks())
        return results

    def run_normalization_benchmarks(self) -> list[BenchmarkResult]:
        """Run normalization layer benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for hidden_dim in self.sizes.hidden_dims[:4]:
                seq_len = 512

                # RMSNorm
                result = self._benchmark_rmsnorm(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # GroupNorm
                for num_groups in self.sizes.num_groups[:2]:
                    if hidden_dim % num_groups == 0:
                        result = self._benchmark_groupnorm(batch_size, seq_len, hidden_dim, num_groups)
                        if result:
                            results.append(result)

                # InstanceNorm
                result = self._benchmark_instancenorm(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

        return results

    def run_ffn_benchmarks(self) -> list[BenchmarkResult]:
        """Run FFN (feed-forward network) layer benchmarks.

        SwiGLU and GeGLU are full FFN layers with 3 linear projections,
        not simple element-wise activations. They have O(seq * hidden * 4*hidden)
        compute cost from matrix multiplications.
        """
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for hidden_dim in self.sizes.hidden_dims[:4]:
                seq_len = 512

                # SwiGLU FFN (3 linear projections: W_gate, W_up, W_down)
                result = self._benchmark_swiglu(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # GeGLU FFN (3 linear projections: W_gate, W_up, W_down)
                result = self._benchmark_geglu(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

        return results

    def run_activation_benchmarks(self) -> list[BenchmarkResult]:
        """Run element-wise activation function benchmarks.

        These are pure element-wise operations with O(n) compute cost,
        as opposed to FFN layers which have O(n * d) matrix multiplication cost.
        """
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for hidden_dim in self.sizes.hidden_dims[:4]:
                seq_len = 512

                # Mish (element-wise)
                result = self._benchmark_mish(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # GELU variants (element-wise)
                result = self._benchmark_gelu_tanh(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # SiLU (element-wise)
                result = self._benchmark_silu(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # QuickGELU (element-wise)
                result = self._benchmark_quick_gelu(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # Squared ReLU (element-wise)
                result = self._benchmark_squared_relu(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # Swish (element-wise)
                result = self._benchmark_swish(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # Hard Swish (element-wise)
                result = self._benchmark_hard_swish(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # Hard Sigmoid (element-wise)
                result = self._benchmark_hard_sigmoid(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

        return results

    def run_pooling_benchmarks(self) -> list[BenchmarkResult]:
        """Run pooling layer benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for hidden_dim in self.sizes.hidden_dims[:3]:
                seq_len = 512

                # AdaptiveAvgPool1d
                result = self._benchmark_adaptive_avgpool1d(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

                # GeM pooling
                result = self._benchmark_gem(batch_size, seq_len, hidden_dim)
                if result:
                    results.append(result)

        return results

    def run_embedding_benchmarks(self) -> list[BenchmarkResult]:
        """Run embedding layer benchmarks."""
        results = []

        for batch_size in self.sizes.batch_sizes[:3]:
            for seq_len in self.sizes.seq_lengths[:3]:
                for vocab_size in self.sizes.vocab_sizes[:2]:
                    hidden_dim = 512

                    # Sinusoidal embedding
                    result = self._benchmark_sinusoidal(batch_size, seq_len, hidden_dim)
                    if result:
                        results.append(result)

                    # Rotary embedding
                    result = self._benchmark_rotary(batch_size, seq_len, hidden_dim)
                    if result:
                        results.append(result)

        return results

    def _benchmark_rmsnorm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark RMSNorm layer."""
        try:
            from mlx_primitives.layers import RMSNorm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = RMSNorm(hidden_dim)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"rmsnorm_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "normalization",
            "layer": "RMSNorm",
        }
        return result

    def _benchmark_groupnorm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_groups: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark GroupNorm layer."""
        try:
            from mlx_primitives.layers import GroupNorm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = GroupNorm(num_groups, hidden_dim)
        # GroupNorm expects NCHW format: (batch, channels, height, width)
        # We use hidden_dim as channels, with spatial dimensions sqrt(seq_len) x sqrt(seq_len)
        spatial = int(seq_len ** 0.5)
        x = mx.random.normal((batch_size, hidden_dim, spatial, spatial))

        def fn():
            return layer(x)

        name = f"groupnorm_b{batch_size}_c{hidden_dim}_s{spatial}x{spatial}_g{num_groups}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "num_channels": hidden_dim,
            "spatial": spatial,
            "num_groups": num_groups,
            "type": "normalization",
            "layer": "GroupNorm",
        }
        return result

    def _benchmark_instancenorm(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark InstanceNorm layer."""
        try:
            from mlx_primitives.layers import InstanceNorm
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = InstanceNorm(hidden_dim)
        # InstanceNorm expects NCHW format: (batch, channels, height, width)
        spatial = int(seq_len ** 0.5)
        x = mx.random.normal((batch_size, hidden_dim, spatial, spatial))

        def fn():
            return layer(x)

        name = f"instancenorm_b{batch_size}_c{hidden_dim}_s{spatial}x{spatial}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "num_channels": hidden_dim,
            "spatial": spatial,
            "type": "normalization",
            "layer": "InstanceNorm",
        }
        return result

    def _benchmark_swiglu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark SwiGLU activation."""
        try:
            from mlx_primitives.layers import SwiGLU
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = SwiGLU(hidden_dim, hidden_dim * 4)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"swiglu_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "intermediate_dim": hidden_dim * 4,
            "type": "ffn_layer",
            "layer": "SwiGLU",
        }
        return result

    def _benchmark_geglu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark GeGLU activation."""
        try:
            from mlx_primitives.layers import GeGLU
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = GeGLU(hidden_dim, hidden_dim * 4)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"geglu_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "intermediate_dim": hidden_dim * 4,
            "type": "ffn_layer",
            "layer": "GeGLU",
        }
        return result

    def _benchmark_mish(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Mish activation."""
        try:
            from mlx_primitives.layers import mish
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return mish(x)

        name = f"mish_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "Mish",
        }
        return result

    def _benchmark_gelu_tanh(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark GELU tanh approximation."""
        try:
            from mlx_primitives.layers import gelu_tanh
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return gelu_tanh(x)

        name = f"gelu_tanh_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "GELUTanh",
        }
        return result

    def _benchmark_silu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark SiLU activation."""
        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return nn.silu(x)

        name = f"silu_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "SiLU",
        }
        return result

    def _benchmark_quick_gelu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark QuickGELU activation."""
        try:
            from mlx_primitives.layers import quick_gelu
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return quick_gelu(x)

        name = f"quick_gelu_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "QuickGELU",
        }
        return result

    def _benchmark_squared_relu(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Squared ReLU activation."""
        try:
            from mlx_primitives.layers import squared_relu
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return squared_relu(x)

        name = f"squared_relu_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "SquaredReLU",
        }
        return result

    def _benchmark_swish(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Swish activation."""
        try:
            from mlx_primitives.layers import swish
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return swish(x, beta=1.0)

        name = f"swish_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "Swish",
        }
        return result

    def _benchmark_hard_swish(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Hard Swish activation."""
        try:
            from mlx_primitives.layers import hard_swish
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return hard_swish(x)

        name = f"hard_swish_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "HardSwish",
        }
        return result

    def _benchmark_hard_sigmoid(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark Hard Sigmoid activation."""
        try:
            from mlx_primitives.layers import hard_sigmoid
        except ImportError:
            return None

        mx.random.seed(self.config.seed)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return hard_sigmoid(x)

        name = f"hard_sigmoid_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "activation",
            "layer": "HardSigmoid",
        }
        return result

    def _benchmark_adaptive_avgpool1d(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark AdaptiveAvgPool1d."""
        try:
            from mlx_primitives.layers import AdaptiveAvgPool1d
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = AdaptiveAvgPool1d(output_size=1)
        x = mx.random.normal((batch_size, seq_len, hidden_dim))

        def fn():
            return layer(x)

        name = f"adaptive_avgpool1d_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "pooling",
            "layer": "AdaptiveAvgPool1d",
        }
        return result

    def _benchmark_gem(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark GeM (Generalized Mean) pooling."""
        try:
            from mlx_primitives.layers import GeM
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = GeM()
        # GeM expects NCHW format: (batch, channels, height, width)
        spatial = int(seq_len ** 0.5)
        x = mx.random.normal((batch_size, hidden_dim, spatial, spatial))
        x = mx.abs(x) + 1e-6  # GeM requires positive values

        def fn():
            return layer(x)

        name = f"gem_b{batch_size}_c{hidden_dim}_s{spatial}x{spatial}"
        result = benchmark_fn(
            fn,
            iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            name=name,
        )
        result.metadata = {
            "batch_size": batch_size,
            "num_channels": hidden_dim,
            "spatial": spatial,
            "type": "pooling",
            "layer": "GeM",
        }
        return result

    def _benchmark_sinusoidal(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark SinusoidalEmbedding."""
        try:
            from mlx_primitives.layers import SinusoidalEmbedding
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        layer = SinusoidalEmbedding(hidden_dim)
        positions = mx.arange(seq_len)
        positions = mx.broadcast_to(positions, (batch_size, seq_len))

        def fn():
            return layer(positions)

        name = f"sinusoidal_emb_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "embedding",
            "layer": "SinusoidalEmbedding",
        }
        return result

    def _benchmark_rotary(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
    ) -> Optional[BenchmarkResult]:
        """Benchmark RotaryEmbedding."""
        try:
            from mlx_primitives.layers import RotaryEmbedding
        except ImportError:
            return None

        mx.random.seed(self.config.seed)

        head_dim = 64
        num_heads = 8
        layer = RotaryEmbedding(head_dim)
        # RotaryEmbedding expects (batch, heads, seq_len, head_dim)
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        def fn():
            return layer(q, k)

        name = f"rotary_emb_b{batch_size}_s{seq_len}_d{hidden_dim}"
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
            "type": "embedding",
            "layer": "RotaryEmbedding",
        }
        return result
