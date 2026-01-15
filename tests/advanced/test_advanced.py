"""Tests for advanced primitives."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_primitives.advanced import (
    # MoE
    TopKRouter,
    ExpertChoiceRouter,
    Expert,
    MoELayer,
    SwitchMoE,
    load_balancing_loss,
    router_z_loss,
    # SSM
    selective_scan,
    MambaBlock,
    S4Layer,
    Mamba,
    H3Layer,
    H3Block,
    H3,
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
    CalibrationCollector,
    Int4Linear,
    QLoRALinear,
    GPTQLinear,
    AWQLinear,
)


# ============================================================================
# MoE Tests
# ============================================================================


class TestTopKRouter:
    """Tests for TopKRouter."""

    def test_router_output_shape(self):
        """Test router output shapes."""
        router = TopKRouter(dims=64, num_experts=8, top_k=2)
        x = mx.random.normal((2, 10, 64))

        gate_weights, expert_indices, router_logits = router(x)

        assert gate_weights.shape == (2, 10, 2)
        assert expert_indices.shape == (2, 10, 2)
        assert router_logits.shape == (2, 10, 8)

    def test_gate_weights_sum_to_one(self):
        """Test that gate weights sum to 1."""
        router = TopKRouter(dims=64, num_experts=8, top_k=2)
        x = mx.random.normal((2, 10, 64))

        gate_weights, _, _ = router(x)

        sums = mx.sum(gate_weights, axis=-1)
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)

    def test_expert_indices_valid(self):
        """Test expert indices are valid."""
        router = TopKRouter(dims=64, num_experts=8, top_k=2)
        x = mx.random.normal((2, 10, 64))

        _, expert_indices, _ = router(x)

        assert mx.all(expert_indices >= 0)
        assert mx.all(expert_indices < 8)


class TestExpert:
    """Tests for Expert module."""

    def test_expert_output_shape(self):
        """Test expert output shape."""
        expert = Expert(dims=64, hidden_dims=256)
        x = mx.random.normal((2, 10, 64))

        y = expert(x)

        assert y.shape == x.shape


class TestMoELayer:
    """Tests for MoELayer."""

    def test_moe_output_shape(self):
        """Test MoE layer output shape."""
        moe = MoELayer(
            dims=64,
            hidden_dims=128,
            num_experts=4,
            top_k=2,
        )
        x = mx.random.normal((2, 10, 64))

        output, aux_loss = moe(x)

        assert output.shape == x.shape
        assert aux_loss.shape == ()

    def test_moe_returns_aux_loss(self):
        """Test that MoE returns non-negative aux loss."""
        moe = MoELayer(dims=64, hidden_dims=128, num_experts=4, top_k=2)
        x = mx.random.normal((2, 10, 64))

        _, aux_loss = moe(x)

        assert float(aux_loss) >= 0


class TestSwitchMoE:
    """Tests for SwitchMoE."""

    def test_switch_moe_shape(self):
        """Test Switch MoE output shape."""
        moe = SwitchMoE(dims=64, hidden_dims=128, num_experts=4)
        x = mx.random.normal((2, 10, 64))

        output, aux_loss = moe(x)

        assert output.shape == x.shape


class TestMoELosses:
    """Tests for MoE loss functions."""

    def test_router_z_loss(self):
        """Test router z-loss computation."""
        logits = mx.random.normal((2, 10, 8))

        loss = router_z_loss(logits)

        assert loss.shape == ()
        assert float(loss) >= 0


# ============================================================================
# SSM Tests
# ============================================================================


class TestMambaBlock:
    """Tests for MambaBlock."""

    def test_mamba_block_shape(self):
        """Test Mamba block output shape."""
        block = MambaBlock(dims=64, d_state=16, d_conv=4, expand=2)
        x = mx.random.normal((2, 20, 64))

        y = block(x)

        assert y.shape == x.shape

    def test_mamba_block_different_seq_lens(self):
        """Test Mamba block with different sequence lengths."""
        block = MambaBlock(dims=64, d_state=16)

        for seq_len in [10, 50, 100]:
            x = mx.random.normal((1, seq_len, 64))
            y = block(x)
            assert y.shape == x.shape


class TestS4Layer:
    """Tests for S4Layer."""

    def test_s4_shape(self):
        """Test S4 layer output shape."""
        layer = S4Layer(dims=64, d_state=32)
        x = mx.random.normal((2, 20, 64))

        y = layer(x)

        assert y.shape == x.shape

    def test_s4_bidirectional(self):
        """Test bidirectional S4."""
        layer = S4Layer(dims=64, d_state=32, bidirectional=True)
        x = mx.random.normal((2, 20, 64))

        y = layer(x)

        assert y.shape == x.shape


class TestMamba:
    """Tests for full Mamba model."""

    def test_mamba_model_shape(self):
        """Test Mamba model output shape."""
        model = Mamba(dims=64, n_layers=2, d_state=16)
        x = mx.random.normal((2, 20, 64))

        y = model(x)

        assert y.shape == x.shape

    def test_mamba_lm_head(self):
        """Test Mamba with language model head."""
        model = Mamba(dims=64, n_layers=2, vocab_size=1000)
        x = mx.random.randint(0, 1000, (2, 20))

        y = model(x)

        assert y.shape == (2, 20, 1000)


# ============================================================================
# KV Cache Tests
# ============================================================================


class TestKVCache:
    """Tests for KVCache."""

    def test_cache_basic(self):
        """Test basic cache operations."""
        cache = KVCache(
            num_layers=2,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
        )

        # Initially empty
        k, v = cache.get(layer_idx=0)
        assert k.shape[2] == 0

        # Add some tokens
        new_k = mx.random.normal((1, 4, 10, 32))
        new_v = mx.random.normal((1, 4, 10, 32))

        full_k, full_v = cache.update(layer_idx=0, new_k=new_k, new_v=new_v)

        assert full_k.shape == (1, 4, 10, 32)
        assert cache.seq_len == 10

    def test_cache_incremental(self):
        """Test incremental cache updates."""
        cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
        )

        # First update
        k1 = mx.random.normal((1, 4, 5, 32))
        v1 = mx.random.normal((1, 4, 5, 32))
        cache.update(0, k1, v1)

        # Second update
        k2 = mx.random.normal((1, 4, 3, 32))
        v2 = mx.random.normal((1, 4, 3, 32))
        full_k, full_v = cache.update(0, k2, v2)

        assert full_k.shape == (1, 4, 8, 32)
        assert cache.seq_len == 8

    def test_cache_reset(self):
        """Test cache reset."""
        cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
        )

        # Add tokens
        k = mx.random.normal((1, 4, 10, 32))
        v = mx.random.normal((1, 4, 10, 32))
        cache.update(0, k, v)

        # Reset
        cache.reset()

        assert cache.seq_len == 0


class TestSlidingWindowCache:
    """Tests for SlidingWindowCache."""

    def test_sliding_window(self):
        """Test sliding window behavior."""
        cache = SlidingWindowCache(
            num_layers=1,
            max_batch_size=1,
            window_size=10,
            num_heads=4,
            head_dim=32,
        )

        # Add more than window size
        for _ in range(15):
            k = mx.random.normal((1, 4, 1, 32))
            v = mx.random.normal((1, 4, 1, 32))
            cache.update(0, k, v)

        # Should only keep window_size tokens
        assert cache.current_len == 10

    def test_position_offset(self):
        """Test position offset calculation."""
        cache = SlidingWindowCache(
            num_layers=1,
            max_batch_size=1,
            window_size=10,
            num_heads=4,
            head_dim=32,
        )

        # Add 15 tokens
        for _ in range(15):
            k = mx.random.normal((1, 4, 1, 32))
            v = mx.random.normal((1, 4, 1, 32))
            cache.update(0, k, v)

        assert cache.position_offset == 5


class TestPagedKVCache:
    """Tests for PagedKVCache."""

    def test_allocate_sequence(self):
        """Test sequence allocation."""
        cache = PagedKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            block_size=16,
            num_blocks=100,
        )

        seq_id = cache.allocate_sequence(max_len=64)

        assert seq_id in cache.block_tables
        assert len(cache.block_tables[seq_id]) == 4  # 64 / 16 = 4 blocks

    def test_free_sequence(self):
        """Test sequence freeing."""
        cache = PagedKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            block_size=16,
            num_blocks=100,
        )

        initial_free = cache.num_free_blocks
        seq_id = cache.allocate_sequence(max_len=64)

        assert cache.num_free_blocks == initial_free - 4

        cache.free_sequence(seq_id)

        assert cache.num_free_blocks == initial_free


class TestRotatingKVCache:
    """Tests for RotatingKVCache."""

    def test_rotating_buffer(self):
        """Test rotating buffer behavior."""
        cache = RotatingKVCache(
            num_layers=1,
            max_batch_size=1,
            buffer_size=10,
            num_heads=4,
            head_dim=32,
        )

        # Add more than buffer size
        for _ in range(15):
            k = mx.random.normal((1, 4, 1, 32))
            v = mx.random.normal((1, 4, 1, 32))
            cache.update(0, k, v)

        k, v = cache.get(0)
        assert k.shape[2] == 10  # Buffer size


# ============================================================================
# Quantization Tests
# ============================================================================


class TestQuantization:
    """Tests for quantization functions."""

    def test_quantize_dequantize(self):
        """Test quantize then dequantize preserves values approximately."""
        x = mx.random.normal((64, 64))

        x_q, scale, zp = quantize_tensor(x, num_bits=8, symmetric=True)
        x_rec = dequantize_tensor(x_q, scale, zp)

        # Should be close to original
        error = mx.mean(mx.abs(x - x_rec))
        assert float(error) < 0.1

    def test_quantize_4bit(self):
        """Test 4-bit quantization."""
        x = mx.random.normal((64, 64))

        x_q, scale, zp = quantize_tensor(x, num_bits=4, symmetric=True)

        # Check values are in 4-bit range
        assert mx.all(x_q >= -8)
        assert mx.all(x_q <= 7)

    def test_per_channel_quantize(self):
        """Test per-channel quantization."""
        x = mx.random.normal((64, 64))

        x_q, scale, zp = quantize_tensor(
            x, num_bits=8, per_channel=True, symmetric=True
        )

        # Scale should have one value per output channel
        assert scale.shape[0] == 64


class TestQuantizedLinear:
    """Tests for QuantizedLinear."""

    def test_quantized_linear_shape(self):
        """Test quantized linear output shape."""
        layer = QuantizedLinear(64, 128, num_bits=8)
        layer.quantize_weights()

        x = mx.random.normal((2, 10, 64))
        y = layer(x)

        assert y.shape == (2, 10, 128)

    def test_from_linear(self):
        """Test creating from existing linear."""
        linear = nn.Linear(64, 128)
        q_linear = QuantizedLinear.from_linear(linear, num_bits=8)

        x = mx.random.normal((2, 10, 64))

        y1 = linear(x)
        y2 = q_linear(x)

        # Should be close
        error = mx.mean(mx.abs(y1 - y2))
        assert float(error) < 1.0


class TestDynamicQuantizer:
    """Tests for DynamicQuantizer."""

    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        quantizer = DynamicQuantizer(num_bits=8)
        x = mx.random.normal((64, 64))

        x_q, scale, zp = quantizer.quantize(x)

        assert x_q.dtype == mx.int8

    def test_quantize_dequantize(self):
        """Test simulated quantization."""
        quantizer = DynamicQuantizer(num_bits=8)
        x = mx.random.normal((64, 64))

        x_qd = quantizer.quantize_dequantize(x)

        assert x_qd.dtype == x.dtype
        assert x_qd.shape == x.shape


class TestCalibrationCollector:
    """Tests for CalibrationCollector."""

    def test_calibration(self):
        """Test calibration statistics collection."""
        collector = CalibrationCollector(method="minmax")

        # Observe some activations
        for _ in range(10):
            x = mx.random.normal((32, 64))
            collector.observe(x)

        scale, zp = collector.compute_scale(num_bits=8)

        assert scale > 0


class TestInt4Linear:
    """Tests for Int4Linear."""

    def test_int4_shape(self):
        """Test int4 linear output shape."""
        layer = Int4Linear(64, 128, group_size=32)

        # Initialize with random weights
        weight = mx.random.normal((128, 64))
        layer.pack_weights(weight)

        x = mx.random.normal((2, 10, 64))
        y = layer(x)

        assert y.shape == (2, 10, 128)


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark
class TestAdvancedBenchmarks:
    """Benchmark tests for advanced primitives."""

    def test_moe_benchmark(self, benchmark):
        """Benchmark MoE layer."""
        moe = MoELayer(dims=256, hidden_dims=512, num_experts=8, top_k=2)
        x = mx.random.normal((1, 32, 256))

        def run_moe():
            output, _ = moe(x)
            mx.eval(output)
            return output

        benchmark(run_moe)

    def test_mamba_benchmark(self, benchmark):
        """Benchmark Mamba block."""
        block = MambaBlock(dims=256, d_state=16)
        x = mx.random.normal((1, 64, 256))

        def run_mamba():
            y = block(x)
            mx.eval(y)
            return y

        benchmark(run_mamba)

    def test_kv_cache_benchmark(self, benchmark):
        """Benchmark KV cache updates."""
        cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=2048,
            num_heads=32,
            head_dim=128,
        )

        k = mx.random.normal((1, 32, 1, 128))
        v = mx.random.normal((1, 32, 1, 128))

        def update_cache():
            cache.reset()
            for _ in range(100):
                cache.update(0, k, v)

        benchmark(update_cache)


# ============================================================================
# H3 (Hungry Hungry Hippos) Tests
# ============================================================================


class TestH3Layer:
    """Tests for H3Layer."""

    def test_h3_layer_shape(self):
        """Test H3 layer output shape."""
        layer = H3Layer(dims=64, d_state=16)
        x = mx.random.normal((2, 20, 64))

        y = layer(x)

        assert y.shape == x.shape

    def test_h3_layer_different_seq_lens(self):
        """Test H3 layer with different sequence lengths."""
        layer = H3Layer(dims=64, d_state=16)

        for seq_len in [10, 50, 100]:
            x = mx.random.normal((1, seq_len, 64))
            y = layer(x)
            assert y.shape == x.shape


class TestH3Block:
    """Tests for H3Block."""

    def test_h3_block_shape(self):
        """Test H3 block output shape."""
        block = H3Block(dims=64, d_state=16)
        x = mx.random.normal((2, 20, 64))

        y = block(x)

        assert y.shape == x.shape


class TestH3:
    """Tests for full H3 model."""

    def test_h3_model_shape(self):
        """Test H3 model output shape."""
        model = H3(dims=64, n_layers=2, d_state=16)
        x = mx.random.normal((2, 20, 64))

        y = model(x)

        assert y.shape == x.shape

    def test_h3_lm_head(self):
        """Test H3 with language model head."""
        model = H3(dims=64, n_layers=2, vocab_size=1000, d_state=16)
        x = mx.random.randint(0, 1000, (2, 20))

        y = model(x)

        assert y.shape == (2, 20, 1000)


# ============================================================================
# CompressedKVCache Tests
# ============================================================================


class TestCompressedKVCache:
    """Tests for CompressedKVCache."""

    def test_quantized_compression(self):
        """Test quantized KV cache compression."""
        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
            compression='quantize',
            bits=8,
        )

        # Add tokens
        k = mx.random.normal((1, 4, 10, 32))
        v = mx.random.normal((1, 4, 10, 32))

        full_k, full_v = cache.update(0, k, v)

        assert full_k.shape == (1, 4, 10, 32)

    def test_pruned_compression(self):
        """Test pruned KV cache compression."""
        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
            compression='prune',
            keep_ratio=0.5,
        )

        # Add tokens
        k = mx.random.normal((1, 4, 20, 32))
        v = mx.random.normal((1, 4, 20, 32))

        full_k, full_v = cache.update(0, k, v)

        # Should keep at most keep_ratio of original
        assert full_k.shape[2] <= 20

    def test_clustered_compression(self):
        """Test clustered KV cache compression."""
        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=100,
            num_heads=4,
            head_dim=32,
            compression='cluster',
        )

        # Add tokens
        k = mx.random.normal((1, 4, 10, 32))
        v = mx.random.normal((1, 4, 10, 32))

        full_k, full_v = cache.update(0, k, v)

        assert full_k.shape[2] > 0


# ============================================================================
# QLoRA Tests
# ============================================================================


class TestQLoRALinear:
    """Tests for QLoRALinear."""

    def test_qlora_shape(self):
        """Test QLoRA linear output shape."""
        layer = QLoRALinear(
            in_features=64,
            out_features=128,
            rank=8,
            bits=4,
        )

        x = mx.random.normal((2, 10, 64))
        y = layer(x)

        assert y.shape == (2, 10, 128)

    def test_qlora_lora_only_trainable(self):
        """Test that only LoRA weights are trainable."""
        layer = QLoRALinear(
            in_features=64,
            out_features=128,
            rank=8,
            bits=4,
        )

        # Check that LoRA matrices exist
        assert hasattr(layer, 'lora_A')
        assert hasattr(layer, 'lora_B')


# ============================================================================
# GPTQ Tests
# ============================================================================


class TestGPTQLinear:
    """Tests for GPTQLinear."""

    def test_gptq_shape(self):
        """Test GPTQ linear output shape."""
        layer = GPTQLinear(
            in_features=64,
            out_features=128,
            bits=4,
            group_size=32,
        )

        x = mx.random.normal((2, 10, 64))
        y = layer(x)

        assert y.shape == (2, 10, 128)

    def test_gptq_quantize_weights(self):
        """Test GPTQ weight quantization."""
        layer = GPTQLinear(
            in_features=64,
            out_features=128,
            bits=4,
            group_size=32,
        )

        # Initialize with random weights
        weight = mx.random.normal((128, 64))
        layer.quantize(weight)

        x = mx.random.normal((1, 5, 64))
        y = layer(x)

        assert y.shape == (1, 5, 128)


# ============================================================================
# AWQ Tests
# ============================================================================


class TestAWQLinear:
    """Tests for AWQLinear."""

    def test_awq_shape(self):
        """Test AWQ linear output shape."""
        layer = AWQLinear(
            in_features=64,
            out_features=128,
            bits=4,
            group_size=32,
        )

        x = mx.random.normal((2, 10, 64))
        y = layer(x)

        assert y.shape == (2, 10, 128)

    def test_awq_quantize(self):
        """Test AWQ quantization with calibration."""
        layer = AWQLinear(
            in_features=64,
            out_features=128,
            bits=4,
            group_size=32,
        )

        # Calibration data
        weight = mx.random.normal((128, 64))
        act_scales = mx.random.uniform(shape=(64,)) + 0.1

        layer.quantize(weight, act_scales)

        x = mx.random.normal((1, 5, 64))
        y = layer(x)

        assert y.shape == (1, 5, 128)
