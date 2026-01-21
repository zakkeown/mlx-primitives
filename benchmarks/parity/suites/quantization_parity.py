"""Quantization parity benchmarks."""

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
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


class QuantizationParityBenchmarks:
    """Multi-framework quantization benchmarks for parity comparison."""

    def __init__(self, config: Optional[ParityBenchmarkConfig] = None):
        self.config = config or ParityBenchmarkConfig()
        self.sizes = DEFAULT_SIZES

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

        # Clear JAX JIT cache to prevent memory accumulation
        jax.clear_caches()

        return benchmark_result

    # =========================================================================
    # Main Runner
    # =========================================================================

    def run_all(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all quantization parity benchmarks across frameworks."""
        results = {}

        for size in ["tiny", "small", "medium", "large"]:
            # INT8 operations
            results[f"int8_quantize_{size}"] = self._benchmark_to_list(
                self.benchmark_int8_quantize(size)
            )
            results[f"int8_dequantize_{size}"] = self._benchmark_to_list(
                self.benchmark_int8_dequantize(size)
            )
            results[f"int8_linear_{size}"] = self._benchmark_to_list(
                self.benchmark_int8_linear(size)
            )

            # INT4 operations
            results[f"int4_quantize_{size}"] = self._benchmark_to_list(
                self.benchmark_int4_quantize(size)
            )
            results[f"int4_dequantize_{size}"] = self._benchmark_to_list(
                self.benchmark_int4_dequantize(size)
            )
            results[f"int4_linear_{size}"] = self._benchmark_to_list(
                self.benchmark_int4_linear(size)
            )

            # Comparison benchmark
            results[f"vs_fp32_{size}"] = self._benchmark_to_list(
                self.benchmark_vs_fp32(size)
            )

        return results

    # =========================================================================
    # INT8 Quantization Benchmarks
    # =========================================================================

    def benchmark_int8_quantize(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT8 quantization across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced import quantize_tensor

        config = self.sizes.get_config("quantization", size)
        m, n, k = config  # Matrix dimensions
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((m, n))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: quantize_tensor(x_mlx, num_bits=8),
            name=f"int8_quantize_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(m, n, device="mps")

            def pytorch_int8_quantize():
                # Compute scale for per-tensor symmetric quantization
                x_cpu = x_torch.cpu()
                x_absmax = torch.max(torch.abs(x_cpu))
                scale = (x_absmax / 127.0).item()
                # Quantize using torch.quantize_per_tensor
                x_q = torch.quantize_per_tensor(x_cpu, scale, 0, torch.qint8)
                return x_q

            result = self._benchmark_pytorch(
                pytorch_int8_quantize,
                name=f"int8_quantize_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(m, n).astype(np.float32))

            @jax.jit
            def jax_int8_quantize():
                # Symmetric quantization: scale = max(|x|) / 127
                x_absmax = jnp.max(jnp.abs(x_jax))
                scale = x_absmax / 127.0
                x_q = jnp.round(x_jax / scale)
                x_q = jnp.clip(x_q, -128, 127).astype(jnp.int8)
                return x_q, scale

            result = self._benchmark_jax(
                jax_int8_quantize,
                name=f"int8_quantize_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_int8_dequantize(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT8 dequantization across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        config = self.sizes.get_config("quantization", size)
        m, n, k = config
        results = {}

        # MLX benchmark - prepare quantized data
        x_mlx = mx.random.normal((m, n))
        mx.eval(x_mlx)
        x_q_mlx, scale_mlx, zp_mlx = quantize_tensor(x_mlx, num_bits=8)
        mx.eval(x_q_mlx, scale_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: dequantize_tensor(x_q_mlx, scale_mlx, zp_mlx),
            name=f"int8_dequantize_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(m, n, device="cpu")  # quantize on CPU
            x_absmax = torch.max(torch.abs(x_torch))
            scale_torch = (x_absmax / 127.0).item()
            x_q_torch = torch.quantize_per_tensor(x_torch, scale_torch, 0, torch.qint8)

            def pytorch_int8_dequantize():
                return x_q_torch.dequantize().to("mps")

            result = self._benchmark_pytorch(
                pytorch_int8_dequantize,
                name=f"int8_dequantize_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_np = np.random.randn(m, n).astype(np.float32)
            x_absmax = np.max(np.abs(x_np))
            scale_np = x_absmax / 127.0
            x_q_np = np.round(x_np / scale_np).clip(-128, 127).astype(np.int8)
            x_q_jax = jnp.array(x_q_np)
            scale_jax = jnp.array(scale_np)

            @jax.jit
            def jax_int8_dequantize():
                return x_q_jax.astype(jnp.float32) * scale_jax

            result = self._benchmark_jax(
                jax_int8_dequantize,
                name=f"int8_dequantize_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # INT4 Quantization Benchmarks
    # =========================================================================

    def benchmark_int4_quantize(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT4 quantization across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced import quantize_tensor

        config = self.sizes.get_config("quantization", size)
        m, n, k = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((m, n))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: quantize_tensor(x_mlx, num_bits=4),
            name=f"int4_quantize_{size}_mlx",
        )

        # PyTorch MPS benchmark - manual INT4 (PyTorch lacks native INT4)
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(m, n, device="mps")

            def pytorch_int4_quantize():
                # Manual INT4: scale = max(|x|) / 7, quantize to [-8, 7]
                x_absmax = torch.max(torch.abs(x_torch))
                scale = x_absmax / 7.0
                x_q = torch.round(x_torch / scale)
                x_q = torch.clamp(x_q, -8, 7).to(torch.int8)
                return x_q, scale

            result = self._benchmark_pytorch(
                pytorch_int4_quantize,
                name=f"int4_quantize_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_jax = jnp.array(np.random.randn(m, n).astype(np.float32))

            @jax.jit
            def jax_int4_quantize():
                # INT4 symmetric: scale = max(|x|) / 7, quantize to [-8, 7]
                x_absmax = jnp.max(jnp.abs(x_jax))
                scale = x_absmax / 7.0
                x_q = jnp.round(x_jax / scale)
                x_q = jnp.clip(x_q, -8, 7).astype(jnp.int8)
                return x_q, scale

            result = self._benchmark_jax(
                jax_int4_quantize,
                name=f"int4_quantize_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_int4_dequantize(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT4 dequantization across MLX, PyTorch MPS, and JAX."""
        from mlx_primitives.advanced import quantize_tensor, dequantize_tensor

        config = self.sizes.get_config("quantization", size)
        m, n, k = config
        results = {}

        # MLX benchmark
        x_mlx = mx.random.normal((m, n))
        mx.eval(x_mlx)
        x_q_mlx, scale_mlx, zp_mlx = quantize_tensor(x_mlx, num_bits=4)
        mx.eval(x_q_mlx, scale_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: dequantize_tensor(x_q_mlx, scale_mlx, zp_mlx),
            name=f"int4_dequantize_{size}_mlx",
        )

        # PyTorch MPS benchmark
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            x_torch = torch.randn(m, n, device="mps")
            x_absmax = torch.max(torch.abs(x_torch))
            scale_torch = x_absmax / 7.0
            x_q_torch = torch.round(x_torch / scale_torch).clamp(-8, 7).to(torch.int8)

            def pytorch_int4_dequantize():
                return x_q_torch.float() * scale_torch

            result = self._benchmark_pytorch(
                pytorch_int4_dequantize,
                name=f"int4_dequantize_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            x_np = np.random.randn(m, n).astype(np.float32)
            x_absmax = np.max(np.abs(x_np))
            scale_np = x_absmax / 7.0
            x_q_np = np.round(x_np / scale_np).clip(-8, 7).astype(np.int8)
            x_q_jax = jnp.array(x_q_np)
            scale_jax = jnp.array(scale_np)

            @jax.jit
            def jax_int4_dequantize():
                return x_q_jax.astype(jnp.float32) * scale_jax

            result = self._benchmark_jax(
                jax_int4_dequantize,
                name=f"int4_dequantize_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # Quantized Linear Layer Benchmarks
    # =========================================================================

    def benchmark_int8_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT8 quantized linear layer across frameworks."""
        from mlx_primitives.advanced import QuantizedLinear

        config = self.sizes.get_config("quantization", size)
        m, n, k = config  # Use as (batch*seq, in_features, out_features)
        batch_seq = m
        in_features = n
        out_features = k
        results = {}

        # MLX benchmark
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=8)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())
        x_mlx = mx.random.normal((batch_seq, in_features))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: layer_mlx(x_mlx),
            name=f"int8_linear_{size}_mlx",
        )

        # PyTorch MPS benchmark - using dynamic quantization pattern
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            # Create a linear layer and simulate INT8 quantized forward
            weight_torch = torch.randn(out_features, in_features, device="cpu") * 0.02
            bias_torch = torch.zeros(out_features, device="mps")
            x_torch = torch.randn(batch_seq, in_features, device="mps")

            # Pre-quantize weights (symmetric INT8)
            w_absmax = torch.max(torch.abs(weight_torch))
            w_scale = (w_absmax / 127.0).item()
            w_q = torch.quantize_per_tensor(weight_torch, w_scale, 0, torch.qint8)

            def pytorch_int8_linear():
                # Dequantize and compute on MPS
                w_deq = w_q.dequantize().to("mps")
                return F.linear(x_torch, w_deq, bias_torch)

            result = self._benchmark_pytorch(
                pytorch_int8_linear,
                name=f"int8_linear_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
            bias_np = np.zeros(out_features, dtype=np.float32)
            x_np = np.random.randn(batch_seq, in_features).astype(np.float32)

            # Pre-quantize weights
            w_absmax = np.max(np.abs(weight_np))
            w_scale = w_absmax / 127.0
            w_q_np = np.round(weight_np / w_scale).clip(-128, 127).astype(np.int8)
            w_q_jax = jnp.array(w_q_np)
            w_scale_jax = jnp.array(w_scale)
            bias_jax = jnp.array(bias_np)
            x_jax = jnp.array(x_np)

            @jax.jit
            def jax_int8_linear():
                # Dequantize and compute
                w_deq = w_q_jax.astype(jnp.float32) * w_scale_jax
                return x_jax @ w_deq.T + bias_jax

            result = self._benchmark_jax(
                jax_int8_linear,
                name=f"int8_linear_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    def benchmark_int4_linear(self, size: str) -> Dict[str, BenchmarkResult]:
        """Benchmark INT4 quantized linear layer across frameworks."""
        from mlx_primitives.advanced import QuantizedLinear

        config = self.sizes.get_config("quantization", size)
        m, n, k = config
        batch_seq = m
        in_features = n
        out_features = k
        results = {}

        # MLX benchmark - use QuantizedLinear with num_bits=4
        layer_mlx = QuantizedLinear(in_features, out_features, num_bits=4)
        layer_mlx.quantize_weights()
        mx.eval(layer_mlx.parameters())
        x_mlx = mx.random.normal((batch_seq, in_features))
        mx.eval(x_mlx)

        results["mlx"] = self._benchmark_mlx(
            lambda: layer_mlx(x_mlx),
            name=f"int4_linear_{size}_mlx",
        )

        # PyTorch MPS benchmark - manual INT4 linear
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            weight_torch = torch.randn(out_features, in_features, device="mps") * 0.02
            bias_torch = torch.zeros(out_features, device="mps")
            x_torch = torch.randn(batch_seq, in_features, device="mps")

            # Pre-quantize weights to INT4
            w_absmax = torch.max(torch.abs(weight_torch))
            w_scale = w_absmax / 7.0
            w_q_torch = torch.round(weight_torch / w_scale).clamp(-8, 7).to(torch.int8)

            def pytorch_int4_linear():
                # Dequantize and compute
                w_deq = w_q_torch.float() * w_scale
                return F.linear(x_torch, w_deq, bias_torch)

            result = self._benchmark_pytorch(
                pytorch_int4_linear,
                name=f"int4_linear_{size}_pytorch_mps",
            )
            if result:
                results["pytorch_mps"] = result

        # JAX benchmark
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
            bias_np = np.zeros(out_features, dtype=np.float32)
            x_np = np.random.randn(batch_seq, in_features).astype(np.float32)

            # Pre-quantize weights to INT4
            w_absmax = np.max(np.abs(weight_np))
            w_scale = w_absmax / 7.0
            w_q_np = np.round(weight_np / w_scale).clip(-8, 7).astype(np.int8)
            w_q_jax = jnp.array(w_q_np)
            w_scale_jax = jnp.array(w_scale)
            bias_jax = jnp.array(bias_np)
            x_jax = jnp.array(x_np)

            @jax.jit
            def jax_int4_linear():
                w_deq = w_q_jax.astype(jnp.float32) * w_scale_jax
                return x_jax @ w_deq.T + bias_jax

            result = self._benchmark_jax(
                jax_int4_linear,
                name=f"int4_linear_{size}_jax",
            )
            if result:
                results["jax"] = result

        return results

    # =========================================================================
    # FP32 vs Quantized Comparison
    # =========================================================================

    def benchmark_vs_fp32(self, size: str) -> Dict[str, BenchmarkResult]:
        """Compare quantized vs FP32 linear performance across frameworks."""
        from mlx_primitives.advanced import QuantizedLinear

        config = self.sizes.get_config("quantization", size)
        m, n, k = config
        batch_seq = m
        in_features = n
        out_features = k
        results = {}

        # =================================================================
        # MLX: FP32, INT8, and INT4 comparison
        # =================================================================
        x_mlx = mx.random.normal((batch_seq, in_features))
        mx.eval(x_mlx)

        # FP32 baseline
        fp32_layer = nn.Linear(in_features, out_features)
        mx.eval(fp32_layer.parameters())
        results["mlx_fp32"] = self._benchmark_mlx(
            lambda: fp32_layer(x_mlx),
            name=f"vs_fp32_{size}_mlx_fp32",
        )

        # INT8
        int8_layer = QuantizedLinear(in_features, out_features, num_bits=8)
        int8_layer.quantize_weights()
        mx.eval(int8_layer.parameters())
        results["mlx_int8"] = self._benchmark_mlx(
            lambda: int8_layer(x_mlx),
            name=f"vs_fp32_{size}_mlx_int8",
        )

        # INT4
        int4_layer = QuantizedLinear(in_features, out_features, num_bits=4)
        int4_layer.quantize_weights()
        mx.eval(int4_layer.parameters())
        results["mlx_int4"] = self._benchmark_mlx(
            lambda: int4_layer(x_mlx),
            name=f"vs_fp32_{size}_mlx_int4",
        )

        # =================================================================
        # PyTorch MPS: FP32, INT8, INT4 comparison
        # =================================================================
        if HAS_PYTORCH and HAS_MPS and self.config.include_pytorch:
            weight_torch = torch.randn(out_features, in_features, device="mps") * 0.02
            bias_torch = torch.zeros(out_features, device="mps")
            x_torch = torch.randn(batch_seq, in_features, device="mps")

            # FP32 baseline
            def pytorch_fp32_linear():
                return F.linear(x_torch, weight_torch, bias_torch)

            result = self._benchmark_pytorch(
                pytorch_fp32_linear,
                name=f"vs_fp32_{size}_pytorch_fp32",
            )
            if result:
                results["pytorch_fp32"] = result

            # INT8 (pre-quantized)
            w_cpu = weight_torch.cpu()
            w_absmax_8 = torch.max(torch.abs(w_cpu))
            w_scale_8 = (w_absmax_8 / 127.0).item()
            w_q8 = torch.quantize_per_tensor(w_cpu, w_scale_8, 0, torch.qint8)

            def pytorch_int8_linear():
                w_deq = w_q8.dequantize().to("mps")
                return F.linear(x_torch, w_deq, bias_torch)

            result = self._benchmark_pytorch(
                pytorch_int8_linear,
                name=f"vs_fp32_{size}_pytorch_int8",
            )
            if result:
                results["pytorch_int8"] = result

            # INT4 (manual)
            w_absmax_4 = torch.max(torch.abs(weight_torch))
            w_scale_4 = w_absmax_4 / 7.0
            w_q4 = torch.round(weight_torch / w_scale_4).clamp(-8, 7).to(torch.int8)

            def pytorch_int4_linear():
                w_deq = w_q4.float() * w_scale_4
                return F.linear(x_torch, w_deq, bias_torch)

            result = self._benchmark_pytorch(
                pytorch_int4_linear,
                name=f"vs_fp32_{size}_pytorch_int4",
            )
            if result:
                results["pytorch_int4"] = result

        # =================================================================
        # JAX: FP32, INT8, INT4 comparison
        # =================================================================
        if HAS_JAX and self.config.include_jax:
            np.random.seed(42)
            weight_np = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
            bias_np = np.zeros(out_features, dtype=np.float32)
            x_np = np.random.randn(batch_seq, in_features).astype(np.float32)

            weight_jax = jnp.array(weight_np)
            bias_jax = jnp.array(bias_np)
            x_jax = jnp.array(x_np)

            # FP32 baseline
            @jax.jit
            def jax_fp32_linear():
                return x_jax @ weight_jax.T + bias_jax

            result = self._benchmark_jax(
                jax_fp32_linear,
                name=f"vs_fp32_{size}_jax_fp32",
            )
            if result:
                results["jax_fp32"] = result

            # INT8
            w_absmax_8 = np.max(np.abs(weight_np))
            w_scale_8 = w_absmax_8 / 127.0
            w_q8_np = np.round(weight_np / w_scale_8).clip(-128, 127).astype(np.int8)
            w_q8_jax = jnp.array(w_q8_np)
            w_scale_8_jax = jnp.array(w_scale_8)

            @jax.jit
            def jax_int8_linear():
                w_deq = w_q8_jax.astype(jnp.float32) * w_scale_8_jax
                return x_jax @ w_deq.T + bias_jax

            result = self._benchmark_jax(
                jax_int8_linear,
                name=f"vs_fp32_{size}_jax_int8",
            )
            if result:
                results["jax_int8"] = result

            # INT4
            w_absmax_4 = np.max(np.abs(weight_np))
            w_scale_4 = w_absmax_4 / 7.0
            w_q4_np = np.round(weight_np / w_scale_4).clip(-8, 7).astype(np.int8)
            w_q4_jax = jnp.array(w_q4_np)
            w_scale_4_jax = jnp.array(w_scale_4)

            @jax.jit
            def jax_int4_linear():
                w_deq = w_q4_jax.astype(jnp.float32) * w_scale_4_jax
                return x_jax @ w_deq.T + bias_jax

            result = self._benchmark_jax(
                jax_int4_linear,
                name=f"vs_fp32_{size}_jax_int4",
            )
            if result:
                results["jax_int4"] = result

        return results
