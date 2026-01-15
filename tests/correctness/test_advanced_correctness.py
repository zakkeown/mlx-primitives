"""Correctness tests for advanced primitives.

Tests for MoE, SSM, Quantization, and KV Cache implementations.
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.advanced import (
    # MoE
    TopKRouter,
    ExpertChoiceRouter,
    Expert,
    MoELayer,
    load_balancing_loss,
    router_z_loss,
    # SSM
    selective_scan,
    MambaBlock,
    S4Layer,
    Mamba,
    # KV Cache
    KVCache,
    SlidingWindowCache,
    PagedKVCache,
    RotatingKVCache,
    CompressedKVCache,
    # Quantization
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    DynamicQuantizer,
)


# ============================================================================
# Reference Implementations
# ============================================================================


def naive_top_k_routing(logits: mx.array, k: int) -> tuple:
    """Reference implementation of top-k routing.

    Args:
        logits: Router logits [batch, seq_len, num_experts]
        k: Number of experts to route to

    Returns:
        (indices, weights) tuple
    """
    # Softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)

    # Get top-k
    # Note: MLX doesn't have topk, so we use argsort
    sorted_indices = mx.argsort(-probs, axis=-1)  # Descending
    top_k_indices = sorted_indices[..., :k]

    # Gather top-k probabilities
    batch, seq_len, _ = logits.shape
    weights = []
    for b in range(batch):
        batch_weights = []
        for s in range(seq_len):
            w = probs[b, s, top_k_indices[b, s]]
            batch_weights.append(w)
        weights.append(mx.stack(batch_weights))
    weights = mx.stack(weights)

    # Renormalize weights
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)

    return top_k_indices, weights


def naive_load_balancing_loss(
    router_probs: mx.array,
    expert_mask: mx.array,
    num_experts: int
) -> mx.array:
    """Reference load balancing loss.

    Encourages uniform expert utilization.
    """
    # Fraction of tokens routed to each expert
    # expert_mask: [batch, seq_len, num_experts] one-hot
    tokens_per_expert = mx.mean(expert_mask, axis=(0, 1))  # [num_experts]

    # Average routing probability to each expert
    prob_per_expert = mx.mean(router_probs, axis=(0, 1))  # [num_experts]

    # Loss is dot product (high when both are concentrated)
    return num_experts * mx.sum(tokens_per_expert * prob_per_expert)


def naive_quantize_int8(x: mx.array) -> tuple:
    """Reference int8 quantization."""
    # Compute scale
    x_max = mx.max(mx.abs(x))
    scale = x_max / 127.0

    # Quantize
    x_int = mx.round(x / scale)
    x_int = mx.clip(x_int, -128, 127).astype(mx.int8)

    return x_int, scale


def naive_dequantize_int8(x_int: mx.array, scale: float) -> mx.array:
    """Reference int8 dequantization."""
    return x_int.astype(mx.float32) * scale


def naive_selective_scan(
    x: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    delta: mx.array
) -> mx.array:
    """Reference selective scan implementation.

    Implements the SSM recurrence: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t

    Args:
        x: Input [batch, seq_len, d_inner]
        A: State transition [d_inner, d_state]
        B: Input projection [batch, seq_len, d_state]
        C: Output projection [batch, seq_len, d_state]
        delta: Time step [batch, seq_len, d_inner]

    Returns:
        Output [batch, seq_len, d_inner]
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A and B
    # A_bar = exp(delta * A)
    # B_bar = (exp(delta * A) - I) * A^{-1} * B ≈ delta * B for small delta

    outputs = []
    h = mx.zeros((batch, d_inner, d_state))

    for t in range(seq_len):
        # Get time step
        dt = delta[:, t:t+1, :]  # [batch, 1, d_inner]

        # Discretized A: exp(dt * A)
        A_bar = mx.exp(dt[..., None] * A[None, None, :, :])  # [batch, 1, d_inner, d_state]
        A_bar = A_bar.squeeze(1)  # [batch, d_inner, d_state]

        # Discretized B: dt * B
        B_t = B[:, t, :]  # [batch, d_state]
        B_bar = dt.squeeze(1)[..., None] * B_t[:, None, :]  # [batch, d_inner, d_state]

        # State update: h = A_bar * h + B_bar * x
        x_t = x[:, t, :, None]  # [batch, d_inner, 1]
        h = A_bar * h + B_bar * x_t[..., 0]

        # Output: y = C * h
        C_t = C[:, t, :]  # [batch, d_state]
        y_t = mx.sum(h * C_t[:, None, :], axis=-1)  # [batch, d_inner]

        outputs.append(y_t)

    return mx.stack(outputs, axis=1)


# ============================================================================
# MoE Correctness Tests
# ============================================================================


class TestTopKRouterCorrectness:
    """Correctness tests for TopKRouter."""

    def test_top_k_selection(self):
        """Test top-k correctly selects highest probability experts."""
        mx.random.seed(42)
        num_experts = 8
        k = 2

        router = TopKRouter(dims=64, num_experts=num_experts, top_k=k)

        x = mx.random.normal((2, 16, 64))
        mx.eval(x)

        weights, indices, _ = router(x)
        mx.eval(indices, weights)

        # Verify indices are in valid range
        assert mx.all(indices >= 0)
        assert mx.all(indices < num_experts)

        # Verify k experts selected
        assert indices.shape[-1] == k

        # Verify weights sum to 1
        weight_sums = mx.sum(weights, axis=-1)
        assert mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-5)

    def test_top_k_vs_naive(self):
        """Compare top-k routing against naive implementation."""
        mx.random.seed(42)
        num_experts = 4
        k = 2

        router = TopKRouter(dims=32, num_experts=num_experts, top_k=k)

        x = mx.random.normal((1, 8, 32))
        mx.eval(x)

        # Get router logits
        logits = router.gate(x)
        mx.eval(logits)

        # Our implementation
        our_weights, our_indices, _ = router(x)
        mx.eval(our_indices, our_weights)

        # Naive implementation
        naive_indices, naive_weights = naive_top_k_routing(logits, k)
        mx.eval(naive_indices, naive_weights)

        # Indices should match
        assert mx.all(our_indices == naive_indices), "Top-k indices don't match"

        # Weights should be close
        assert mx.allclose(our_weights, naive_weights, atol=1e-4), \
            f"Max weight diff: {float(mx.max(mx.abs(our_weights - naive_weights)))}"


class TestLoadBalancingLossCorrectness:
    """Correctness tests for load balancing loss."""

    def test_balanced_routing_low_loss(self):
        """Test that balanced routing produces low loss."""
        num_experts = 4
        batch_size = 2
        seq_len = 100
        top_k = 1

        # Create uniform router logits (balanced routing)
        router_logits = mx.zeros((batch_size, seq_len, num_experts))
        mx.eval(router_logits)

        # Each token goes to different expert uniformly
        expert_indices = (mx.arange(seq_len) % num_experts).reshape(1, seq_len, 1)
        expert_indices = mx.broadcast_to(expert_indices, (batch_size, seq_len, top_k))
        mx.eval(expert_indices)

        loss = load_balancing_loss(router_logits, expert_indices, num_experts)
        mx.eval(loss)

        # Balanced routing should have low loss
        assert float(loss) < 2.0, f"Balanced routing has high loss: {float(loss)}"

    def test_unbalanced_routing_high_loss(self):
        """Test that unbalanced routing produces high loss."""
        num_experts = 4
        batch_size = 2
        seq_len = 100
        top_k = 1

        # All logits favor expert 0
        router_logits = mx.zeros((batch_size, seq_len, num_experts))
        router_logits = router_logits.at[..., 0].add(10.0)  # Strong preference for expert 0
        mx.eval(router_logits)

        # All tokens to one expert
        expert_indices = mx.zeros((batch_size, seq_len, top_k), dtype=mx.int32)
        mx.eval(expert_indices)

        loss = load_balancing_loss(router_logits, expert_indices, num_experts)
        mx.eval(loss)

        # Unbalanced should have high loss (close to num_experts)
        assert float(loss) > 2.0, f"Unbalanced routing has low loss: {float(loss)}"


class TestMoELayerCorrectness:
    """Correctness tests for MoE layer."""

    def test_moe_output_shape(self):
        """Test MoE produces correct output shape."""
        mx.random.seed(42)

        moe = MoELayer(
            dims=64,
            hidden_dims=128,
            num_experts=4,
            top_k=2
        )

        x = mx.random.normal((2, 16, 64))
        mx.eval(x)

        output, aux = moe(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_moe_combines_experts(self):
        """Test MoE correctly combines expert outputs."""
        mx.random.seed(42)

        moe = MoELayer(
            dims=32,
            hidden_dims=64,
            num_experts=4,
            top_k=2
        )

        x = mx.random.normal((1, 4, 32))
        mx.eval(x)

        output, aux = moe(x)
        mx.eval(output)

        # Output should be non-trivial combination
        assert float(mx.std(output)) > 0, "MoE output has zero variance"
        assert mx.all(mx.isfinite(output)), "MoE output has non-finite values"


# ============================================================================
# SSM Correctness Tests
# ============================================================================


class TestSelectiveScanCorrectness:
    """Correctness tests for selective scan."""

    def test_scan_output_shape(self):
        """Test selective scan produces correct output shape."""
        mx.random.seed(42)
        batch = 2
        seq_len = 32
        d_inner = 64
        d_state = 16

        x = mx.random.normal((batch, seq_len, d_inner))
        A = mx.random.normal((d_inner, d_state)) * 0.1
        B = mx.random.normal((batch, seq_len, d_state))
        C = mx.random.normal((batch, seq_len, d_state))
        delta = mx.abs(mx.random.normal((batch, seq_len, d_inner))) + 0.1

        mx.eval(x, A, B, C, delta)

        # Note: selective_scan signature is (x, delta, A, B, C, D=None)
        output = selective_scan(x, delta, A, B, C)
        mx.eval(output)

        assert output.shape == x.shape

    def test_scan_causality(self):
        """Test selective scan is causal (future doesn't affect past)."""
        mx.random.seed(42)
        batch = 1
        seq_len = 16
        d_inner = 8
        d_state = 4

        x = mx.random.normal((batch, seq_len, d_inner))
        A = mx.random.normal((d_inner, d_state)) * 0.1
        B = mx.random.normal((batch, seq_len, d_state))
        C = mx.random.normal((batch, seq_len, d_state))
        delta = mx.abs(mx.random.normal((batch, seq_len, d_inner))) + 0.1

        mx.eval(x, A, B, C, delta)

        # Get output for full sequence (note: selective_scan signature is (x, delta, A, B, C))
        full_output = selective_scan(x, delta, A, B, C)
        mx.eval(full_output)

        # Get output for truncated sequence
        half_len = seq_len // 2
        trunc_output = selective_scan(
            x[:, :half_len],
            delta[:, :half_len],
            A,
            B[:, :half_len],
            C[:, :half_len]
        )
        mx.eval(trunc_output)

        # First half should be identical (causal)
        assert mx.allclose(
            full_output[:, :half_len],
            trunc_output,
            atol=1e-5
        ), "Selective scan is not causal"


class TestMambaBlockCorrectness:
    """Correctness tests for MambaBlock."""

    def test_mamba_block_output_shape(self):
        """Test MambaBlock produces correct output shape."""
        mx.random.seed(42)

        block = MambaBlock(dims=64, d_state=16, d_conv=4, expand=2)

        x = mx.random.normal((2, 32, 64))
        mx.eval(x)

        output = block(x)
        mx.eval(output)

        assert output.shape == x.shape

    def test_mamba_block_is_causal(self):
        """Test MambaBlock is causal."""
        mx.random.seed(42)

        block = MambaBlock(dims=32, d_state=8, d_conv=4, expand=2)

        x = mx.random.normal((1, 16, 32))
        mx.eval(x)

        # Full output
        full_output = block(x)
        mx.eval(full_output)

        # Modified future shouldn't affect past
        x_modified = mx.concatenate([
            x[:, :8],
            mx.random.normal((1, 8, 32))
        ], axis=1)
        mx.eval(x_modified)

        modified_output = block(x_modified)
        mx.eval(modified_output)

        # First half should be identical
        assert mx.allclose(
            full_output[:, :8],
            modified_output[:, :8],
            atol=1e-5
        ), "MambaBlock is not causal"


# ============================================================================
# KV Cache Correctness Tests
# ============================================================================


class TestKVCacheCorrectness:
    """Correctness tests for KV Cache implementations."""

    def test_basic_cache_append(self):
        """Test basic cache correctly appends new KV pairs."""
        cache = KVCache(
            num_layers=1,
            max_batch_size=2,
            max_seq_len=100,
            num_heads=4,
            head_dim=32
        )

        # Add first batch of KV
        k1 = mx.random.normal((2, 4, 10, 32))
        v1 = mx.random.normal((2, 4, 10, 32))
        mx.eval(k1, v1)

        full_k1, full_v1 = cache.update(0, k1, v1)
        mx.eval(full_k1, full_v1)

        assert full_k1.shape == (2, 4, 10, 32)

        # Add second batch
        k2 = mx.random.normal((2, 4, 5, 32))
        v2 = mx.random.normal((2, 4, 5, 32))
        mx.eval(k2, v2)

        full_k2, full_v2 = cache.update(0, k2, v2)
        mx.eval(full_k2, full_v2)

        # Should have concatenated
        assert full_k2.shape == (2, 4, 15, 32)

        # First part should match original
        assert mx.allclose(full_k2[:, :, :10, :], k1, atol=1e-6)

    def test_sliding_window_cache_limits_size(self):
        """Test sliding window cache limits sequence length."""
        window_size = 20

        cache = SlidingWindowCache(
            num_layers=1,
            max_batch_size=1,
            window_size=window_size,
            num_heads=4,
            head_dim=32
        )

        # Add more than window_size tokens
        for _ in range(5):
            k = mx.random.normal((1, 4, 10, 32))
            v = mx.random.normal((1, 4, 10, 32))
            mx.eval(k, v)

            full_k, full_v = cache.update(0, k, v)
            mx.eval(full_k, full_v)

        # Should be limited to window_size
        assert full_k.shape[2] == window_size

    def test_rotating_cache_rotation(self):
        """Test rotating cache correctly rotates old entries."""
        max_len = 10

        cache = RotatingKVCache(
            num_layers=1,
            max_batch_size=1,
            buffer_size=max_len,
            num_heads=2,
            head_dim=16
        )

        # Fill cache
        k_init = mx.arange(max_len).reshape(1, 1, max_len, 1)
        k_init = mx.broadcast_to(k_init, (1, 2, max_len, 16)).astype(mx.float32)
        v_init = mx.array(k_init)
        mx.eval(k_init, v_init)

        cache.update(0, k_init, v_init)

        # Add new token
        k_new = mx.ones((1, 2, 1, 16)) * 100
        v_new = mx.array(k_new)
        mx.eval(k_new, v_new)

        full_k, full_v = cache.update(0, k_new, v_new)
        mx.eval(full_k, full_v)

        # Should still be max_len, oldest entry replaced
        assert full_k.shape[2] == max_len


# ============================================================================
# Quantization Correctness Tests
# ============================================================================


class TestQuantizationCorrectness:
    """Correctness tests for quantization utilities."""

    def test_int8_quantize_dequantize(self):
        """Test int8 quantization roundtrip."""
        mx.random.seed(42)

        x = mx.random.normal((64, 64))
        mx.eval(x)

        # Quantize
        x_quant, scale, zero_point = quantize_tensor(x, num_bits=8)
        mx.eval(x_quant, scale, zero_point)

        # Dequantize
        x_restored = dequantize_tensor(x_quant, scale, zero_point)
        mx.eval(x_restored)

        # Should be close to original
        max_error = float(mx.max(mx.abs(x - x_restored)))
        relative_error = max_error / float(mx.max(mx.abs(x)))

        assert relative_error < 0.05, f"Quantization error too high: {relative_error}"

    def test_int4_quantize_dequantize(self):
        """Test int4 quantization roundtrip."""
        mx.random.seed(42)

        x = mx.random.normal((64, 64))
        mx.eval(x)

        # Quantize to 4 bits
        x_quant, scale, zero_point = quantize_tensor(x, num_bits=4)
        mx.eval(x_quant, scale, zero_point)

        # Dequantize
        x_restored = dequantize_tensor(x_quant, scale, zero_point)
        mx.eval(x_restored)

        # 4-bit has more error than 8-bit
        max_error = float(mx.max(mx.abs(x - x_restored)))
        relative_error = max_error / float(mx.max(mx.abs(x)))

        assert relative_error < 0.2, f"4-bit quantization error too high: {relative_error}"

    def test_quantized_linear_correctness(self):
        """Test QuantizedLinear produces reasonable outputs."""
        mx.random.seed(42)

        # Create quantized linear and quantize its weights
        quantized = QuantizedLinear(64, 128, num_bits=8)

        x = mx.random.normal((2, 16, 64))
        mx.eval(x)

        # Get output before quantization (uses float weights)
        float_out = quantized(x)
        mx.eval(float_out)

        # Quantize weights
        quantized.quantize_weights()

        # Get output after quantization
        quant_out = quantized(x)
        mx.eval(quant_out)

        # Should be similar (allowing for quantization error)
        correlation = mx.sum(float_out * quant_out) / (
            mx.sqrt(mx.sum(float_out ** 2)) * mx.sqrt(mx.sum(quant_out ** 2))
        )
        mx.eval(correlation)

        assert float(correlation) > 0.95, f"Quantized output poorly correlated: {float(correlation)}"


class TestDynamicQuantizationCorrectness:
    """Correctness tests for dynamic quantization."""

    def test_dynamic_quantizer_quantize_dequantize(self):
        """Test dynamic quantizer produces reasonable roundtrip."""
        quantizer = DynamicQuantizer(num_bits=8)

        x = mx.random.normal((32, 64))
        mx.eval(x)

        # Quantize-dequantize should preserve signal
        x_restored = quantizer.quantize_dequantize(x)
        mx.eval(x_restored)

        # Should be close to original
        max_error = float(mx.max(mx.abs(x - x_restored)))
        relative_error = max_error / float(mx.max(mx.abs(x)))

        assert relative_error < 0.1, f"Quantization error too high: {relative_error}"

    def test_dynamic_quantizer_respects_bits(self):
        """Test dynamic quantizer uses specified bits."""
        for num_bits in [4, 8]:
            quantizer = DynamicQuantizer(num_bits=num_bits)

            x = mx.random.normal((32, 64))
            mx.eval(x)

            x_quant, scale, zero_point = quantizer.quantize(x)
            mx.eval(x_quant)

            # Check values are in valid range for bits
            max_val = 2 ** (num_bits - 1) - 1
            min_val = -(2 ** (num_bits - 1))

            assert float(mx.max(x_quant)) <= max_val
            assert float(mx.min(x_quant)) >= min_val


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestAdvancedNumericalStability:
    """Test numerical stability of advanced primitives."""

    def test_moe_stability_with_large_logits(self):
        """Test MoE routing is stable with large logits."""
        mx.random.seed(42)

        router = TopKRouter(dims=64, num_experts=8, top_k=2)

        # Large input values
        x = mx.random.normal((2, 16, 64)) * 100
        mx.eval(x)

        weights, indices, _ = router(x)
        mx.eval(indices, weights)

        # Weights should still sum to 1 and be finite
        assert mx.all(mx.isfinite(weights))
        weight_sums = mx.sum(weights, axis=-1)
        assert mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-4)

    def test_mamba_stability_long_sequence(self):
        """Test Mamba is stable with long sequences."""
        mx.random.seed(42)

        block = MambaBlock(dims=64, d_state=16, d_conv=4, expand=2)

        x = mx.random.normal((1, 512, 64))
        mx.eval(x)

        output = block(x)
        mx.eval(output)

        assert mx.all(mx.isfinite(output)), "Mamba unstable with long sequence"

    def test_quantization_stability_extreme_values(self):
        """Test quantization handles extreme values."""
        # Very large values
        x_large = mx.random.normal((32, 32)) * 1000
        mx.eval(x_large)

        x_quant, scale, zero_point = quantize_tensor(x_large, num_bits=8)
        x_restored = dequantize_tensor(x_quant, scale, zero_point)
        mx.eval(x_restored)

        assert mx.all(mx.isfinite(x_restored))

        # Very small values
        x_small = mx.random.normal((32, 32)) * 0.001
        mx.eval(x_small)

        x_quant, scale, zero_point = quantize_tensor(x_small, num_bits=8)
        x_restored = dequantize_tensor(x_quant, scale, zero_point)
        mx.eval(x_restored)

        assert mx.all(mx.isfinite(x_restored))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
