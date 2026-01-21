"""Extended PyTorch MPS baseline benchmarks for comprehensive parity comparison.

This module extends PyTorchMPSBenchmarks with 50+ additional operations
for comprehensive performance comparison against MLX implementations.
"""

from typing import Any, Dict, List, Optional, Tuple

from benchmarks.baselines.pytorch_mps import PyTorchMPSBenchmarks, PyTorchBenchmarkResult

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None
    np = None


class PyTorchMPSExtendedBenchmarks(PyTorchMPSBenchmarks):
    """Extended PyTorch MPS benchmarks covering all 50+ operations."""

    # ========== Attention Operations ==========

    def benchmark_sliding_window_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark sliding window attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        # Create sliding window mask
        i = torch.arange(seq_length, device=self.device)[:, None]
        j = torch.arange(seq_length, device=self.device)[None, :]
        mask = (torch.abs(i - j) <= window_size).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        def fn():
            with torch.no_grad():
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale + mask
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_sliding_window_b{batch_size}_s{seq_length}_h{num_heads}_w{window_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_chunked_cross_attention(
        self,
        batch_size: int,
        q_seq_length: int,
        kv_seq_length: int,
        num_heads: int,
        head_dim: int,
        chunk_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark chunked cross attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, q_seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, kv_seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, kv_seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        def fn():
            with torch.no_grad():
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_chunked_cross_b{batch_size}_q{q_seq_length}_kv{kv_seq_length}_h{num_heads}_c{chunk_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gqa(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark grouped query attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_kv_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_kv_heads, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)
        repeats = num_heads // num_kv_heads

        def fn():
            with torch.no_grad():
                k_expanded = k.repeat_interleave(repeats, dim=1)
                v_expanded = v.repeat_interleave(repeats, dim=1)
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k_expanded) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v_expanded)

        name = f"pytorch_gqa_b{batch_size}_s{seq_length}_h{num_heads}_kv{num_kv_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_mqa(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark multi-query attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, 1, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, 1, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        def fn():
            with torch.no_grad():
                scores = torch.einsum("bhqd,bckd->bhqk", q, k) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bckd->bhqd", weights, v)

        name = f"pytorch_mqa_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_sparse_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        sparsity_pattern: str,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark sparse attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        i = torch.arange(seq_length, device=self.device)[:, None]
        j = torch.arange(seq_length, device=self.device)[None, :]
        if sparsity_pattern == "local":
            window = seq_length // 4
            mask = torch.abs(i - j) <= window
        elif sparsity_pattern == "strided":
            stride = 4
            mask = ((i - j) % stride == 0) | (torch.abs(i - j) <= 2)
        else:
            mask = torch.rand(seq_length, seq_length, device=self.device) < 0.25

        mask_float = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, 0.0)

        def fn():
            with torch.no_grad():
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale + mask_float
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_sparse_attn_b{batch_size}_s{seq_length}_h{num_heads}_{sparsity_pattern}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_linear_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark linear attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)

        def fn():
            with torch.no_grad():
                q_prime = F.elu(q) + 1
                k_prime = F.elu(k) + 1
                kv = torch.einsum("bhkd,bhkv->bhdv", k_prime, v)
                output = torch.einsum("bhqd,bhdv->bhqv", q_prime, kv)
                k_sum = k_prime.sum(dim=2, keepdim=True)
                normalizer = torch.einsum("bhqd,bhkd->bhq", q_prime, k_sum).unsqueeze(-1)
                return output / (normalizer + 1e-6)

        name = f"pytorch_linear_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_alibi_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark ALiBi attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32)
        scale = torch.tensor(1.0 / (head_dim ** 0.5), device=self.device, dtype=torch.float32)

        def get_slopes(n):
            import math
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [float(start * (start ** i)) for i in range(n)]

        slopes = torch.tensor(get_slopes(num_heads), device=self.device, dtype=torch.float32)
        positions = torch.arange(seq_length, device=self.device, dtype=torch.float32)
        distances = positions[:, None] - positions[None, :]
        alibi_bias = slopes[:, None, None] * distances[None, :, :]

        def fn():
            with torch.no_grad():
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale + alibi_bias
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_alibi_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_quantized_kv_cache_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        bits: int = 8,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark attention with quantized KV cache."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k_q = torch.randint(-127, 128, (batch_size, num_heads, seq_length, head_dim), dtype=torch.int8, device=self.device)
        v_q = torch.randint(-127, 128, (batch_size, num_heads, seq_length, head_dim), dtype=torch.int8, device=self.device)
        k_scale = torch.tensor(0.02, device=self.device)
        v_scale = torch.tensor(0.02, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        def fn():
            with torch.no_grad():
                k = k_q.float() * k_scale
                v = v_q.float() * v_scale
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_quantized_kv_attn_b{batch_size}_s{seq_length}_h{num_heads}_bits{bits}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_rope_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark RoPE-integrated attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        half_dim = head_dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, device=self.device).float() / half_dim)
        positions = torch.arange(seq_length, device=self.device).float()
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        def fn():
            with torch.no_grad():
                q1, q2 = q.chunk(2, dim=-1)
                k1, k2 = k.chunk(2, dim=-1)
                q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
                k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
                scores = torch.einsum("bhqd,bhkd->bhqk", q_rot, k_rot) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_rope_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_attention_backward(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark attention backward pass."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float32, requires_grad=True)
        scale = torch.tensor(1.0 / (head_dim ** 0.5), dtype=torch.float32, device=self.device)

        def fn():
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
            weights = F.softmax(scores, dim=-1)
            output = torch.einsum("bhqk,bhkd->bhqd", weights, v)
            loss = output.sum()
            loss.backward(retain_graph=True)
            return q.grad, k.grad, v.grad

        name = f"pytorch_attention_backward_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Activation Operations ==========

    def benchmark_swiglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark SwiGLU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        W_gate = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02
        W_up = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                gate = F.silu(x @ W_gate)
                up = x @ W_up
                return gate * up

        name = f"pytorch_swiglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_geglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark GeGLU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        W_gate = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02
        W_up = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                gate = F.gelu(x @ W_gate, approximate="none")
                up = x @ W_up
                return gate * up

        name = f"pytorch_geglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_reglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark ReGLU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        W_gate = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02
        W_up = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                gate = F.relu(x @ W_gate)
                up = x @ W_up
                return gate * up

        name = f"pytorch_reglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_quick_gelu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark QuickGELU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return x * torch.sigmoid(1.702 * x)

        name = f"pytorch_quick_gelu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gelu_tanh(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark GELU with tanh approximation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.gelu(x, approximate="tanh")

        name = f"pytorch_gelu_tanh_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_mish(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark Mish activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.mish(x)

        name = f"pytorch_mish_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_squared_relu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark Squared ReLU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.relu(x) ** 2

        name = f"pytorch_squared_relu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_hard_swish(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark HardSwish activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.hardswish(x)

        name = f"pytorch_hard_swish_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_hard_sigmoid(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark HardSigmoid activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                return F.hardsigmoid(x)

        name = f"pytorch_hard_sigmoid_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Normalization Operations ==========

    def benchmark_rmsnorm(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark RMSNorm."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        weight = torch.ones(hidden_dim, device=self.device)
        eps = 1e-6

        def fn():
            with torch.no_grad():
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
                return (x / rms) * weight

        name = f"pytorch_rmsnorm_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_groupnorm(
        self,
        batch_size: int,
        channels: int,
        spatial_size: Tuple[int, ...],
        num_groups: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark GroupNorm."""
        torch.manual_seed(42)
        # For 2D spatial: (batch, channels, height, width)
        shape = (batch_size, channels) + spatial_size
        x = torch.randn(*shape, device=self.device)
        weight = torch.ones(channels, device=self.device)
        bias = torch.zeros(channels, device=self.device)

        def fn():
            with torch.no_grad():
                return F.group_norm(x, num_groups, weight, bias)

        spatial_str = "x".join(str(s) for s in spatial_size)
        name = f"pytorch_groupnorm_b{batch_size}_c{channels}_s{spatial_str}_g{num_groups}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_instancenorm(
        self,
        batch_size: int,
        channels: int,
        spatial_size: Tuple[int, ...],
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark InstanceNorm."""
        torch.manual_seed(42)
        # (batch, channels, *spatial)
        shape = (batch_size, channels) + spatial_size
        x = torch.randn(*shape, device=self.device)
        weight = torch.ones(channels, device=self.device)
        bias = torch.zeros(channels, device=self.device)

        # Use appropriate instance norm based on spatial dimensions
        if len(spatial_size) == 1:
            norm_fn = F.instance_norm
        elif len(spatial_size) == 2:
            norm_fn = F.instance_norm
        else:
            norm_fn = F.instance_norm

        def fn():
            with torch.no_grad():
                return norm_fn(x, weight=weight, bias=bias)

        spatial_str = "x".join(str(s) for s in spatial_size)
        name = f"pytorch_instancenorm_b{batch_size}_c{channels}_s{spatial_str}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adalayernorm(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark AdaLayerNorm (Adaptive Layer Normalization)."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        # Conditioning vector (e.g., from timestep embedding)
        cond = torch.randn(batch_size, hidden_dim, device=self.device)
        # Linear projections for scale and shift
        scale_proj = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02
        shift_proj = torch.randn(hidden_dim, hidden_dim, device=self.device) * 0.02
        eps = 1e-5

        def fn():
            with torch.no_grad():
                # Layer norm
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                x_norm = (x - mean) / torch.sqrt(var + eps)

                # Adaptive scale and shift from conditioning
                scale = cond @ scale_proj  # (batch, hidden_dim)
                shift = cond @ shift_proj  # (batch, hidden_dim)

                # Broadcast over sequence dimension
                scale = scale.unsqueeze(1)  # (batch, 1, hidden_dim)
                shift = shift.unsqueeze(1)  # (batch, 1, hidden_dim)

                return x_norm * (1 + scale) + shift

        name = f"pytorch_adalayernorm_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Fused Operations ==========

    def benchmark_fused_rmsnorm_linear(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        output_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark fused RMSNorm + Linear."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        norm_weight = torch.ones(hidden_dim, device=self.device)
        linear_weight = torch.randn(hidden_dim, output_dim, device=self.device) * 0.02
        linear_bias = torch.zeros(output_dim, device=self.device)
        eps = 1e-6

        def fn():
            with torch.no_grad():
                # Fused RMSNorm + Linear
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
                x_norm = (x / rms) * norm_weight
                return x_norm @ linear_weight + linear_bias

        name = f"pytorch_fused_rmsnorm_linear_b{batch_size}_s{seq_length}_d{hidden_dim}_o{output_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_swiglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark fused SwiGLU (Linear -> SiLU gate -> Linear)."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        intermediate_dim = hidden_dim * 4
        W_gate = torch.randn(hidden_dim, intermediate_dim, device=self.device) * 0.02
        W_up = torch.randn(hidden_dim, intermediate_dim, device=self.device) * 0.02
        W_down = torch.randn(intermediate_dim, hidden_dim, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                # Fused SwiGLU: down(silu(gate(x)) * up(x))
                gate = F.silu(x @ W_gate)
                up = x @ W_up
                return (gate * up) @ W_down

        name = f"pytorch_fused_swiglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_geglu(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark fused GeGLU (Linear -> GELU gate -> Linear)."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        intermediate_dim = hidden_dim * 4
        W_gate = torch.randn(hidden_dim, intermediate_dim, device=self.device) * 0.02
        W_up = torch.randn(hidden_dim, intermediate_dim, device=self.device) * 0.02
        W_down = torch.randn(intermediate_dim, hidden_dim, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                # Fused GeGLU: down(gelu(gate(x)) * up(x))
                gate = F.gelu(x @ W_gate, approximate="none")
                up = x @ W_up
                return (gate * up) @ W_down

        name = f"pytorch_fused_geglu_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_fused_rope_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark fused RoPE + Attention."""
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        # Precompute RoPE
        half_dim = head_dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, device=self.device).float() / half_dim)
        positions = torch.arange(seq_length, device=self.device).float()
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        def fn():
            with torch.no_grad():
                # Apply RoPE to Q and K
                q1, q2 = q.chunk(2, dim=-1)
                k1, k2 = k.chunk(2, dim=-1)
                q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
                k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
                # Attention
                scores = torch.einsum("bhqd,bhkd->bhqk", q_rot, k_rot) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_fused_rope_attn_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Quantization Operations ==========

    def benchmark_int8_quantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT8 quantization."""
        torch.manual_seed(42)
        x = torch.randn(m, n, device=self.device)

        def fn():
            with torch.no_grad():
                scale = torch.max(torch.abs(x)) / 127.0
                x_q = torch.round(x / scale).to(torch.int8)
                return x_q, scale

        name = f"pytorch_int8_quantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int8_dequantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT8 dequantization."""
        torch.manual_seed(42)
        x_q = torch.randint(-127, 128, (m, n), dtype=torch.int8, device=self.device)
        scale = torch.tensor(0.05, device=self.device)

        def fn():
            with torch.no_grad():
                return x_q.float() * scale

        name = f"pytorch_int8_dequantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_quantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT4 quantization."""
        torch.manual_seed(42)
        x = torch.randn(m, n, device=self.device)

        def fn():
            with torch.no_grad():
                scale = torch.max(torch.abs(x)) / 7.0
                x_q = torch.round(x / scale)
                x_q = torch.clamp(x_q, -8, 7).to(torch.int8)
                return x_q, scale

        name = f"pytorch_int4_quantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_dequantize(
        self,
        m: int,
        n: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT4 dequantization."""
        torch.manual_seed(42)
        x_q = torch.randint(-8, 8, (m, n), dtype=torch.int8, device=self.device)
        scale = torch.tensor(0.1, device=self.device)

        def fn():
            with torch.no_grad():
                return x_q.float() * scale

        name = f"pytorch_int4_dequantize_{m}x{n}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int8_linear(
        self,
        batch_size: int,
        seq_length: int,
        in_features: int,
        out_features: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT8 linear layer."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, in_features, device=self.device)
        weight_q = torch.randint(-127, 128, (out_features, in_features), dtype=torch.int8, device=self.device)
        scale = torch.tensor(0.02, device=self.device)
        bias = torch.randn(out_features, device=self.device) * 0.01

        def fn():
            with torch.no_grad():
                weight = weight_q.float() * scale
                return x @ weight.T + bias

        name = f"pytorch_int8_linear_b{batch_size}_s{seq_length}_{in_features}x{out_features}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_int4_linear(
        self,
        batch_size: int,
        seq_length: int,
        in_features: int,
        out_features: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark INT4 linear layer."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, in_features, device=self.device)
        weight_q = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8, device=self.device)
        scale = torch.tensor(0.1, device=self.device)
        bias = torch.randn(out_features, device=self.device) * 0.01

        def fn():
            with torch.no_grad():
                weight = weight_q.float() * scale
                return x @ weight.T + bias

        name = f"pytorch_int4_linear_b{batch_size}_s{seq_length}_{in_features}x{out_features}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Primitive Operations ==========

    def benchmark_associative_scan_add(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark associative scan with addition (cumsum)."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, dim, device=self.device)

        def fn():
            with torch.no_grad():
                return torch.cumsum(x, dim=1)

        name = f"pytorch_assoc_scan_add_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_associative_scan_mul(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark associative scan with multiplication (cumprod)."""
        torch.manual_seed(42)
        # Use smaller values to avoid numerical issues with cumprod
        x = torch.rand(batch_size, seq_length, dim, device=self.device) * 0.2 + 0.9

        def fn():
            with torch.no_grad():
                return torch.cumprod(x, dim=1)

        name = f"pytorch_assoc_scan_mul_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_associative_scan_ssm(
        self,
        batch_size: int,
        seq_length: int,
        state_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark associative scan for SSM (state space model).

        SSM recurrence: h_t = A_t * h_{t-1} + B_t * x_t
        PyTorch doesn't have native associative scan, so we use sequential loop.
        """
        torch.manual_seed(42)
        # A coefficients (decay factors, should be < 1 for stability)
        A = torch.rand(batch_size, seq_length, state_dim, device=self.device) * 0.09 + 0.9
        # B * x term (input contribution)
        Bx = torch.randn(batch_size, seq_length, state_dim, device=self.device) * 0.1

        def fn():
            with torch.no_grad():
                # Sequential scan (no native associative scan in PyTorch)
                h = torch.zeros(batch_size, state_dim, device=self.device)
                outputs = []
                for t in range(seq_length):
                    h = A[:, t, :] * h + Bx[:, t, :]
                    outputs.append(h)
                return torch.stack(outputs, dim=1)

        name = f"pytorch_assoc_scan_ssm_b{batch_size}_s{seq_length}_d{state_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_scan(
        self,
        batch_size: int,
        seq_length: int,
        d_model: int,
        d_state: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark selective scan (Mamba-style).

        Mamba selective scan: h_t = A_t * h_{t-1} + B_t * x_t, y_t = C_t * h_t
        Where A, B, C are input-dependent (selective).
        """
        torch.manual_seed(42)
        # Input
        x = torch.randn(batch_size, seq_length, d_model, device=self.device)
        # Selective parameters (input-dependent)
        delta = F.softplus(torch.randn(batch_size, seq_length, d_model, device=self.device))
        A = -torch.exp(torch.randn(d_model, d_state, device=self.device))  # Negative for stability
        B = torch.randn(batch_size, seq_length, d_state, device=self.device) * 0.1
        C = torch.randn(batch_size, seq_length, d_state, device=self.device) * 0.1

        def fn():
            with torch.no_grad():
                # Discretize: A_bar = exp(delta * A)
                A_bar = torch.exp(torch.einsum("bld,dn->bldn", delta, A))  # (batch, seq, d_model, d_state)
                B_bar = torch.einsum("bld,bln->bldn", delta, B)  # (batch, seq, d_model, d_state)

                # Sequential scan
                h = torch.zeros(batch_size, d_model, d_state, device=self.device)
                outputs = []
                for t in range(seq_length):
                    h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]
                    y_t = (h * C[:, t, None, :]).sum(dim=-1)  # (batch, d_model)
                    outputs.append(y_t)

                return torch.stack(outputs, dim=1)  # (batch, seq, d_model)

        name = f"pytorch_selective_scan_b{batch_size}_s{seq_length}_d{d_model}_n{d_state}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_gather(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        num_indices: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark selective gather (torch.gather)."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, dim, device=self.device)
        # Random indices to gather
        indices = torch.randint(0, seq_length, (batch_size, num_indices, dim), device=self.device)

        def fn():
            with torch.no_grad():
                return torch.gather(x, 1, indices)

        name = f"pytorch_selective_gather_b{batch_size}_s{seq_length}_d{dim}_n{num_indices}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_selective_scatter_add(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        num_indices: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark selective scatter add (torch.scatter_add)."""
        torch.manual_seed(42)
        # Source values to scatter
        values = torch.randn(batch_size, num_indices, dim, device=self.device)
        # Target indices for scatter (where to add values)
        indices = torch.randint(0, seq_length, (batch_size, num_indices, dim), device=self.device)

        def fn():
            with torch.no_grad():
                # Initialize output
                output = torch.zeros(batch_size, seq_length, dim, device=self.device)
                # Scatter add
                return output.scatter_add(1, indices, values)

        name = f"pytorch_selective_scatter_add_b{batch_size}_s{seq_length}_d{dim}_n{num_indices}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== MoE Operations ==========

    def benchmark_topk_routing(
        self,
        batch_size: int,
        seq_length: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int = 512,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark top-k routing.

        Implements top-k expert routing:
        1. Linear projection to get router logits
        2. torch.topk to select top-k experts
        3. F.softmax over selected logits for gate weights
        """
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        gate_weight = torch.randn(hidden_dim, num_experts, device=self.device)

        def fn():
            with torch.no_grad():
                # Compute router logits
                logits = x @ gate_weight  # (batch, seq, num_experts)

                # Get top-k experts
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)

                # Softmax over selected experts
                gate_weights = F.softmax(topk_values, dim=-1)

                return gate_weights, topk_indices, logits

        name = f"pytorch_topk_routing_b{batch_size}_s{seq_length}_e{num_experts}_k{top_k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_expert_dispatch(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int = None,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark expert dispatch.

        Implements full MoE forward pass:
        1. Route tokens to top-k experts
        2. Dispatch tokens to expert MLPs
        3. Combine weighted expert outputs
        """
        torch.manual_seed(42)

        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim * 4

        n_tokens = batch_size * seq_length
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)

        # Router weights
        gate_weight = torch.randn(hidden_dim, num_experts, device=self.device)

        # Expert weights (batched): w1 projects up, w2 projects down
        expert_w1 = torch.randn(num_experts, hidden_dim, expert_hidden_dim, device=self.device)
        expert_w2 = torch.randn(num_experts, expert_hidden_dim, hidden_dim, device=self.device)

        def fn():
            with torch.no_grad():
                # Flatten input
                x_flat = x.reshape(n_tokens, hidden_dim)

                # Routing
                logits = x_flat @ gate_weight  # (n_tokens, num_experts)
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                gate_weights = F.softmax(topk_values, dim=-1)  # (n_tokens, top_k)

                # Initialize output
                output = torch.zeros_like(x_flat)

                # Dispatch to each expert
                for e in range(num_experts):
                    # Find which tokens (and which top_k slot) go to this expert
                    expert_mask = (topk_indices == e)  # (n_tokens, top_k)
                    weights_for_expert = torch.where(expert_mask, gate_weights, torch.zeros_like(gate_weights))
                    token_weights = weights_for_expert.sum(dim=-1)  # (n_tokens,)

                    # Get non-zero mask
                    routed_mask = token_weights > 0
                    if not routed_mask.any():
                        continue

                    # Gather routed tokens
                    x_expert = x_flat[routed_mask]  # (n_routed, hidden_dim)
                    w_expert = token_weights[routed_mask]  # (n_routed,)

                    # Expert MLP: x -> w1 -> silu -> w2
                    hidden = x_expert @ expert_w1[e]  # (n_routed, expert_hidden_dim)
                    hidden = F.silu(hidden)
                    expert_out = hidden @ expert_w2[e]  # (n_routed, hidden_dim)

                    # Weighted scatter-add back
                    output[routed_mask] += expert_out * w_expert.unsqueeze(-1)

                return output.reshape(batch_size, seq_length, hidden_dim)

        name = f"pytorch_expert_dispatch_b{batch_size}_s{seq_length}_e{num_experts}_k{top_k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_load_balancing_loss(
        self,
        batch_size: int,
        seq_length: int,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: int = 512,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark load balancing loss computation.

        Implements GShard-style auxiliary load balancing loss:
        aux_loss = num_experts * sum(expert_fraction * mean_router_prob)

        Where:
        - expert_fraction: fraction of tokens routed to each expert
        - mean_router_prob: mean softmax probability for each expert
        """
        torch.manual_seed(42)

        # Generate router logits and expert indices
        router_logits = torch.randn(batch_size, seq_length, num_experts, device=self.device)
        expert_indices = torch.randint(0, num_experts, (batch_size, seq_length, top_k), device=self.device)

        def fn():
            with torch.no_grad():
                # Routing probabilities
                router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, num_experts)

                # Count tokens per expert
                total_tokens = batch_size * seq_length * top_k
                expert_counts = torch.zeros(num_experts, device=self.device)
                for e in range(num_experts):
                    expert_counts[e] = (expert_indices == e).sum().float()

                expert_fraction = expert_counts / total_tokens

                # Mean routing probability per expert
                mean_prob = router_probs.mean(dim=(0, 1))  # (num_experts,)

                # GShard auxiliary loss
                aux_loss = num_experts * (expert_fraction * mean_prob).sum()
                return aux_loss

        name = f"pytorch_load_balancing_loss_b{batch_size}_s{seq_length}_e{num_experts}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Pooling Operations ==========

    def benchmark_adaptive_avg_pool1d(
        self,
        batch_size: int,
        channels: int,
        length: int,
        output_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark adaptive average pooling 1D."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, channels, length, device=self.device)

        def fn():
            with torch.no_grad():
                return F.adaptive_avg_pool1d(x, output_size)

        name = f"pytorch_adaptive_avg_pool1d_b{batch_size}_c{channels}_l{length}_o{output_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_avg_pool2d(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        output_size: Tuple[int, int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark adaptive average pooling 2D."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, channels, height, width, device=self.device)

        def fn():
            with torch.no_grad():
                return F.adaptive_avg_pool2d(x, output_size)

        name = f"pytorch_adaptive_avg_pool2d_b{batch_size}_c{channels}_h{height}_w{width}_o{output_size[0]}x{output_size[1]}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_max_pool1d(
        self,
        batch_size: int,
        channels: int,
        length: int,
        output_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark adaptive max pooling 1D."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, channels, length, device=self.device)

        def fn():
            with torch.no_grad():
                return F.adaptive_max_pool1d(x, output_size)

        name = f"pytorch_adaptive_max_pool1d_b{batch_size}_c{channels}_l{length}_o{output_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_adaptive_max_pool2d(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        output_size: Tuple[int, int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark adaptive max pooling 2D."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, channels, height, width, device=self.device)

        def fn():
            with torch.no_grad():
                return F.adaptive_max_pool2d(x, output_size)

        name = f"pytorch_adaptive_max_pool2d_b{batch_size}_c{channels}_h{height}_w{width}_o{output_size[0]}x{output_size[1]}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_global_attention_pooling(
        self,
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark global attention pooling."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        attn_weight = torch.randn(hidden_dim, 1, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                # Compute attention scores
                scores = x @ attn_weight  # (batch, seq_length, 1)
                weights = F.softmax(scores, dim=1)  # (batch, seq_length, 1)
                # Weighted sum
                return (x * weights).sum(dim=1)  # (batch, hidden_dim)

        name = f"pytorch_global_attn_pool_b{batch_size}_s{seq_length}_d{hidden_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_gem(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        p: float = 3.0,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark Generalized Mean (GeM) pooling."""
        torch.manual_seed(42)
        # Ensure positive values for power operation
        x = torch.abs(torch.randn(batch_size, channels, height, width, device=self.device)) + 1e-6

        def fn():
            with torch.no_grad():
                # GeM pooling: (mean(x^p))^(1/p)
                x_pow = torch.pow(x, p)
                mean_pow = F.adaptive_avg_pool2d(x_pow, 1)
                return torch.pow(mean_pow, 1.0 / p).squeeze(-1).squeeze(-1)

        name = f"pytorch_gem_b{batch_size}_c{channels}_h{height}_w{width}_p{p}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_spatial_pyramid_pooling(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        levels: List[int],
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark Spatial Pyramid Pooling."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, channels, height, width, device=self.device)

        def fn():
            with torch.no_grad():
                outputs = []
                for level in levels:
                    pooled = F.adaptive_avg_pool2d(x, (level, level))
                    pooled_flat = pooled.view(batch_size, channels, -1)
                    outputs.append(pooled_flat)
                return torch.cat(outputs, dim=-1)

        levels_str = "_".join(str(l) for l in levels)
        name = f"pytorch_spp_b{batch_size}_c{channels}_h{height}_w{width}_levels{levels_str}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Embedding Operations ==========

    def benchmark_sinusoidal_embedding(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark sinusoidal positional embedding."""
        positions = torch.arange(seq_length, device=self.device, dtype=torch.float32)
        log_10000 = torch.tensor(9.210340371976184, dtype=torch.float32, device=self.device)  # log(10000)

        def fn():
            with torch.no_grad():
                half_dim = dim // 2
                freqs = torch.exp(
                    -log_10000 * torch.arange(half_dim, device=self.device, dtype=torch.float32) / half_dim
                )
                angles = positions[:, None] * freqs[None, :]
                sin_emb = torch.sin(angles)
                cos_emb = torch.cos(angles)
                return torch.cat([sin_emb, cos_emb], dim=-1)

        name = f"pytorch_sinusoidal_emb_b{batch_size}_s{seq_length}_d{dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_learned_positional_embedding(
        self,
        batch_size: int,
        seq_length: int,
        dim: int,
        max_length: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark learned positional embedding."""
        torch.manual_seed(42)
        embedding_table = torch.randn(max_length, dim, device=self.device) * 0.02
        positions = torch.arange(seq_length, device=self.device)

        def fn():
            with torch.no_grad():
                return embedding_table[positions]

        name = f"pytorch_learned_pos_emb_b{batch_size}_s{seq_length}_d{dim}_max{max_length}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_rotary_embedding(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark rotary positional embedding."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, num_heads, head_dim, device=self.device)

        # Precompute cos and sin for RoPE
        half_dim = head_dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, device=self.device).float() / half_dim)
        positions = torch.arange(seq_length, device=self.device).float()
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        def fn():
            with torch.no_grad():
                x1, x2 = x.chunk(2, dim=-1)
                cos_exp = cos[None, :, None, :]
                sin_exp = sin[None, :, None, :]
                return torch.cat([
                    x1 * cos_exp - x2 * sin_exp,
                    x1 * sin_exp + x2 * cos_exp
                ], dim=-1)

        name = f"pytorch_rope_b{batch_size}_s{seq_length}_h{num_heads}_d{head_dim}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_alibi_embedding(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark ALiBi embedding."""
        import math

        # Compute ALiBi slopes
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [float(start * (ratio ** i)) for i in range(n)]

            if math.log2(n) % 1 == 0:
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** int(math.floor(math.log2(n)))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                return slopes + extra_slopes

        slopes = torch.tensor(get_slopes(num_heads), device=self.device, dtype=torch.float32)

        def fn():
            with torch.no_grad():
                positions = torch.arange(seq_length, device=self.device, dtype=torch.float32)
                distances = positions[:, None] - positions[None, :]
                bias = slopes[:, None, None] * distances[None, :, :]
                return bias

        name = f"pytorch_alibi_b{batch_size}_s{seq_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_relative_positional_embedding(
        self,
        batch_size: int,
        q_length: int,
        k_length: int,
        num_heads: int,
        dim: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark relative positional embedding."""
        torch.manual_seed(42)
        num_buckets = 32
        max_distance = 128
        embedding_table = torch.randn(num_buckets, num_heads, device=self.device) * 0.02

        def fn():
            with torch.no_grad():
                q_pos = torch.arange(q_length, device=self.device)[:, None]
                k_pos = torch.arange(k_length, device=self.device)[None, :]
                relative_pos = k_pos - q_pos

                # T5-style bucketing
                is_small = torch.abs(relative_pos) < (num_buckets // 2)
                small_bucket = relative_pos + num_buckets // 2

                max_exact = num_buckets // 2
                relative_pos_abs = torch.abs(relative_pos).float()
                relative_pos_if_large = max_exact + (
                    torch.log(relative_pos_abs / max_exact + 1e-6) /
                    np.log(max_distance / max_exact) *
                    (num_buckets - max_exact)
                ).long()
                relative_pos_if_large = torch.minimum(
                    relative_pos_if_large,
                    torch.tensor(num_buckets - 1, device=self.device)
                )

                bucket = torch.where(is_small, small_bucket, relative_pos_if_large)
                bucket = torch.clamp(bucket, 0, num_buckets - 1)

                return embedding_table[bucket]

        name = f"pytorch_rel_pos_emb_b{batch_size}_q{q_length}_k{k_length}_h{num_heads}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Cache Operations ==========

    def benchmark_paged_attention(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark paged attention with block-based KV cache."""
        torch.manual_seed(42)
        num_blocks = (seq_length + block_size - 1) // block_size

        # Query for current position
        q = torch.randn(batch_size, num_heads, 1, head_dim, device=self.device)
        # Paged KV cache: (num_blocks, batch_size, num_heads, block_size, head_dim)
        k_cache = torch.randn(num_blocks, batch_size, num_heads, block_size, head_dim, device=self.device)
        v_cache = torch.randn(num_blocks, batch_size, num_heads, block_size, head_dim, device=self.device)
        # Block table: which blocks are used for each sequence
        block_table = torch.arange(num_blocks, device=self.device)
        scale = 1.0 / (head_dim ** 0.5)

        def fn():
            with torch.no_grad():
                # Gather K and V from paged cache
                k = k_cache[block_table]  # (num_blocks, batch, heads, block_size, dim)
                v = v_cache[block_table]
                # Reshape to (batch, heads, seq, dim)
                k = k.permute(1, 2, 0, 3, 4).reshape(batch_size, num_heads, -1, head_dim)
                v = v.permute(1, 2, 0, 3, 4).reshape(batch_size, num_heads, -1, head_dim)
                # Standard attention
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
                weights = F.softmax(scores, dim=-1)
                return torch.einsum("bhqk,bhkd->bhqd", weights, v)

        name = f"pytorch_paged_attn_b{batch_size}_s{seq_length}_h{num_heads}_blk{block_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_block_allocation(
        self,
        num_sequences: int,
        max_blocks: int,
        block_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark block allocation (bitmap management simulation)."""
        torch.manual_seed(42)
        # Free block bitmap (1 = free, 0 = used)
        free_blocks = (torch.rand(max_blocks, device=self.device) > 0.5).int()
        # Number of blocks to allocate per sequence
        blocks_needed = torch.randint(1, 5, (num_sequences,), device=self.device)

        def fn():
            with torch.no_grad():
                # Find free block indices
                free_indices = torch.where(free_blocks == 1)[0]

                # Allocate blocks to sequences (simplified)
                allocations = []
                offset = 0
                for i in range(num_sequences):
                    n_blocks = blocks_needed[i].item()
                    end_idx = min(offset + n_blocks, len(free_indices))
                    allocated = free_indices[offset:end_idx]
                    allocations.append(allocated)
                    offset = end_idx

                # Update free block bitmap
                all_allocated = torch.cat(allocations) if allocations else torch.tensor([], device=self.device, dtype=torch.long)
                new_free = free_blocks.clone()
                if len(all_allocated) > 0:
                    new_free[all_allocated] = 0
                return new_free, allocations

        name = f"pytorch_block_alloc_nseq{num_sequences}_maxblk{max_blocks}_blksz{block_size}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_cache_eviction(
        self,
        cache_size: int,
        num_accesses: int,
        policy: str,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark cache eviction policies (LRU/LFU simulation)."""
        torch.manual_seed(42)
        # Access timestamps for LRU (higher = more recent)
        timestamps = torch.randint(0, num_accesses, (cache_size,), device=self.device)
        # Access counts for LFU
        access_counts = torch.randint(1, 100, (cache_size,), device=self.device)
        # Number of entries to evict
        num_evict = cache_size // 4

        def fn():
            with torch.no_grad():
                if policy == "lru":
                    # Evict entries with lowest timestamps (least recently used)
                    _, evict_indices = torch.topk(-timestamps, num_evict)
                else:  # lfu
                    # Evict entries with lowest access counts (least frequently used)
                    _, evict_indices = torch.topk(-access_counts, num_evict)

                # Create eviction mask
                evict_mask = torch.zeros(cache_size, dtype=torch.bool, device=self.device)
                evict_mask[evict_indices] = True
                return evict_indices, evict_mask

        name = f"pytorch_cache_eviction_sz{cache_size}_acc{num_accesses}_{policy}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_speculative_verification(
        self,
        batch_size: int,
        draft_length: int,
        num_heads: int,
        head_dim: int,
        vocab_size: int = 32000,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark speculative decoding verification."""
        torch.manual_seed(42)
        # Draft token probabilities
        draft_probs = torch.rand(batch_size, draft_length, vocab_size, device=self.device)
        draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
        # Target model probabilities
        target_probs = torch.rand(batch_size, draft_length, vocab_size, device=self.device)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
        # Draft tokens
        draft_tokens = torch.randint(0, vocab_size, (batch_size, draft_length), device=self.device)

        def fn():
            with torch.no_grad():
                # Get probabilities for draft tokens
                batch_idx = torch.arange(batch_size, device=self.device)[:, None]
                seq_idx = torch.arange(draft_length, device=self.device)[None, :]
                p_draft = draft_probs[batch_idx, seq_idx, draft_tokens]
                p_target = target_probs[batch_idx, seq_idx, draft_tokens]

                # Compute acceptance probability: min(1, p_target / p_draft)
                accept_prob = torch.minimum(torch.ones_like(p_draft), p_target / (p_draft + 1e-10))

                # Random values for acceptance decision
                rand_vals = torch.rand_like(accept_prob)
                accepted = rand_vals < accept_prob

                # Find first rejection position
                rejection_mask = ~accepted
                first_rejection = rejection_mask.int().argmax(dim=-1)
                all_accepted = ~rejection_mask.any(dim=-1)
                num_accepted = torch.where(all_accepted, torch.tensor(draft_length, device=self.device), first_rejection)

                return num_accepted, accepted

        name = f"pytorch_spec_verify_b{batch_size}_draft{draft_length}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Generation Operations ==========

    def benchmark_temperature_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        temperature: float,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark temperature sampling.

        Args:
            batch_size: Batch size.
            vocab_size: Vocabulary size.
            temperature: Temperature for sampling.
            warmup: Warmup iterations.
            iterations: Benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=self.device)

        def fn():
            with torch.no_grad():
                if temperature == 1.0 or temperature == 0.0:
                    return logits
                return logits / temperature

        name = f"pytorch_temperature_b{batch_size}_v{vocab_size}_t{temperature}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_top_k_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        k: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark top-k sampling.

        Args:
            batch_size: Batch size.
            vocab_size: Vocabulary size.
            k: Number of top tokens to keep.
            warmup: Warmup iterations.
            iterations: Benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=self.device)

        def fn():
            with torch.no_grad():
                if k <= 0 or k >= vocab_size:
                    return logits
                sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
                threshold = sorted_logits[:, k - 1 : k]
                return torch.where(
                    logits >= threshold,
                    logits,
                    torch.tensor(float("-inf"), device=self.device),
                )

        name = f"pytorch_top_k_b{batch_size}_v{vocab_size}_k{k}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    def benchmark_top_p_sampling(
        self,
        batch_size: int,
        vocab_size: int,
        p: float,
        warmup: int = 10,
        iterations: int = 100,
    ) -> PyTorchBenchmarkResult:
        """Benchmark top-p (nucleus) sampling.

        Args:
            batch_size: Batch size.
            vocab_size: Vocabulary size.
            p: Cumulative probability threshold.
            warmup: Warmup iterations.
            iterations: Benchmark iterations.

        Returns:
            Benchmark result.
        """
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=self.device)

        def fn():
            with torch.no_grad():
                if p >= 1.0:
                    return logits

                sorted_indices = torch.argsort(logits, dim=-1, descending=True)
                sorted_logits = torch.gather(logits, -1, sorted_indices)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                shifted = torch.cat(
                    [
                        torch.zeros_like(cumulative_probs[:, :1]),
                        cumulative_probs[:, :-1],
                    ],
                    dim=-1,
                )
                mask = shifted < p
                mask[:, 0] = True

                filtered = torch.where(
                    mask,
                    sorted_logits,
                    torch.tensor(float("-inf"), device=self.device),
                )
                inverse = torch.argsort(sorted_indices, dim=-1)
                return torch.gather(filtered, -1, inverse)

        name = f"pytorch_top_p_b{batch_size}_v{vocab_size}_p{p}"
        return self._benchmark(fn, iterations=iterations, warmup_iterations=warmup, name=name)

    # ========== Utility Methods ==========

    def run_all_benchmarks(
        self,
        size: str = "small",
    ) -> Dict[str, PyTorchBenchmarkResult]:
        """Run all extended benchmarks.

        Args:
            size: Size configuration ("tiny", "small", "medium", "large").

        Returns:
            Dictionary mapping benchmark names to results.
        """
        # Size configurations
        configs = {
            "tiny": {"batch_size": 1, "seq_length": 64, "hidden_dim": 128, "num_heads": 4, "head_dim": 32},
            "small": {"batch_size": 2, "seq_length": 256, "hidden_dim": 256, "num_heads": 8, "head_dim": 32},
            "medium": {"batch_size": 4, "seq_length": 512, "hidden_dim": 512, "num_heads": 8, "head_dim": 64},
            "large": {"batch_size": 8, "seq_length": 1024, "hidden_dim": 1024, "num_heads": 16, "head_dim": 64},
        }
        cfg = configs.get(size, configs["small"])

        results = {}
        categories = ["attention", "activation", "normalization", "fused", "quantization",
                      "primitive", "moe", "pooling", "embedding", "cache", "generation"]

        for category in categories:
            try:
                category_results = self.run_category_benchmarks(category, size)
                results.update(category_results)
            except Exception as e:
                print(f"Error running {category} benchmarks: {e}")

        return results

    def run_category_benchmarks(
        self,
        category: str,
        size: str = "small",
    ) -> Dict[str, PyTorchBenchmarkResult]:
        """Run benchmarks for a specific category.

        Args:
            category: Category name (e.g., "attention", "activation").
            size: Size configuration.

        Returns:
            Dictionary mapping benchmark names to results.
        """
        # Size configurations
        configs = {
            "tiny": {"batch_size": 1, "seq_length": 64, "hidden_dim": 128, "num_heads": 4, "head_dim": 32},
            "small": {"batch_size": 2, "seq_length": 256, "hidden_dim": 256, "num_heads": 8, "head_dim": 32},
            "medium": {"batch_size": 4, "seq_length": 512, "hidden_dim": 512, "num_heads": 8, "head_dim": 64},
            "large": {"batch_size": 8, "seq_length": 1024, "hidden_dim": 1024, "num_heads": 16, "head_dim": 64},
        }
        cfg = configs.get(size, configs["small"])
        b, s, d, h, hd = cfg["batch_size"], cfg["seq_length"], cfg["hidden_dim"], cfg["num_heads"], cfg["head_dim"]

        results = {}

        if category == "attention":
            results["sliding_window_attention"] = self.benchmark_sliding_window_attention(b, s, h, hd, s // 4)
            results["gqa"] = self.benchmark_gqa(b, s, h, h // 2, hd)
            results["mqa"] = self.benchmark_mqa(b, s, h, hd)
            results["linear_attention"] = self.benchmark_linear_attention(b, s, h, hd)
            results["alibi_attention"] = self.benchmark_alibi_attention(b, s, h, hd)
            results["rope_attention"] = self.benchmark_rope_attention(b, s, h, hd)
            results["attention_backward"] = self.benchmark_attention_backward(b, s // 4, h, hd)

        elif category == "activation":
            results["swiglu"] = self.benchmark_swiglu(b, s, d)
            results["geglu"] = self.benchmark_geglu(b, s, d)
            results["reglu"] = self.benchmark_reglu(b, s, d)
            results["quick_gelu"] = self.benchmark_quick_gelu(b, s, d)
            results["mish"] = self.benchmark_mish(b, s, d)
            results["hard_swish"] = self.benchmark_hard_swish(b, s, d)

        elif category == "normalization":
            results["rmsnorm"] = self.benchmark_rmsnorm(b, s, d)
            results["groupnorm"] = self.benchmark_groupnorm(b, 64, (32, 32), 8)
            results["instancenorm"] = self.benchmark_instancenorm(b, 64, (32, 32))
            results["adalayernorm"] = self.benchmark_adalayernorm(b, s, d)

        elif category == "fused":
            results["fused_rmsnorm_linear"] = self.benchmark_fused_rmsnorm_linear(b, s, d, d)
            results["fused_swiglu"] = self.benchmark_fused_swiglu(b, s, d)
            results["fused_geglu"] = self.benchmark_fused_geglu(b, s, d)
            results["fused_rope_attention"] = self.benchmark_fused_rope_attention(b, s, h, hd)

        elif category == "quantization":
            results["int8_quantize"] = self.benchmark_int8_quantize(d, d)
            results["int8_dequantize"] = self.benchmark_int8_dequantize(d, d)
            results["int4_quantize"] = self.benchmark_int4_quantize(d, d)
            results["int4_dequantize"] = self.benchmark_int4_dequantize(d, d)
            results["int8_linear"] = self.benchmark_int8_linear(b, s, d, d)
            results["int4_linear"] = self.benchmark_int4_linear(b, s, d, d)

        elif category == "primitive":
            results["assoc_scan_add"] = self.benchmark_associative_scan_add(b, s, d)
            results["assoc_scan_mul"] = self.benchmark_associative_scan_mul(b, s, d)
            results["assoc_scan_ssm"] = self.benchmark_associative_scan_ssm(b, s // 4, 16)
            results["selective_scan"] = self.benchmark_selective_scan(b, s // 8, d, 16)
            results["selective_gather"] = self.benchmark_selective_gather(b, s, d, s // 4)
            results["selective_scatter_add"] = self.benchmark_selective_scatter_add(b, s, d, s // 4)

        elif category == "moe":
            results["topk_routing"] = self.benchmark_topk_routing(b, s, 8, 2, d)
            results["expert_dispatch"] = self.benchmark_expert_dispatch(b, s // 4, d, 4, 2)
            results["load_balancing_loss"] = self.benchmark_load_balancing_loss(b, s, 8, 2, d)

        elif category == "pooling":
            results["adaptive_avg_pool1d"] = self.benchmark_adaptive_avg_pool1d(b, 64, 256, 16)
            results["adaptive_avg_pool2d"] = self.benchmark_adaptive_avg_pool2d(b, 64, 64, 64, (8, 8))
            results["global_attention_pooling"] = self.benchmark_global_attention_pooling(b, s, d)
            results["gem"] = self.benchmark_gem(b, 64, 32, 32)
            results["spp"] = self.benchmark_spatial_pyramid_pooling(b, 64, 32, 32, [1, 2, 4])

        elif category == "embedding":
            results["sinusoidal_embedding"] = self.benchmark_sinusoidal_embedding(b, s, d)
            results["learned_positional_embedding"] = self.benchmark_learned_positional_embedding(b, s, d, 2048)
            results["rotary_embedding"] = self.benchmark_rotary_embedding(b, s, h, hd)
            results["alibi_embedding"] = self.benchmark_alibi_embedding(b, s, h)
            results["relative_positional_embedding"] = self.benchmark_relative_positional_embedding(b, s, s, h, hd)

        elif category == "cache":
            results["paged_attention"] = self.benchmark_paged_attention(b, s, h, hd, 16)
            results["block_allocation"] = self.benchmark_block_allocation(16, 128, 16)
            results["cache_eviction_lru"] = self.benchmark_cache_eviction(256, 1000, "lru")
            results["cache_eviction_lfu"] = self.benchmark_cache_eviction(256, 1000, "lfu")
            results["speculative_verification"] = self.benchmark_speculative_verification(b, 8, h, hd)

        elif category == "generation":
            results["temperature_sampling"] = self.benchmark_temperature_sampling(b, 32000, 0.7)
            results["top_k_sampling"] = self.benchmark_top_k_sampling(b, 32000, 50)
            results["top_p_sampling"] = self.benchmark_top_p_sampling(b, 32000, 0.9)

        return results
