"""End-to-end integration tests for MLX Primitives.

These tests verify that multiple modules work correctly together
in realistic pipeline configurations.

Test Scenarios:
- Attention + MoE pipeline (transformer with mixture of experts)
- Stacked SSM blocks with normalization
- Quantized attention pipeline
- Multi-precision pipelines

To run tests:
    pytest tests/integration/test_integration.py -v
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class TestAttentionMoEIntegration:
    """Test attention + MoE pipeline integration."""

    def test_attention_with_moe_forward(self):
        """Test attention output feeding into MoE layer."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.advanced.moe import MoELayer

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8

        # Generate Q, K, V
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch, seq_len, num_heads, head_dim))

        # Attention
        attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
        attn_out = attn_out.reshape(batch, seq_len, dims)
        mx.eval(attn_out)

        # MoE
        moe = MoELayer(
            dims=dims,
            hidden_dims=dims * 4,
            num_experts=4,
            top_k=2,
        )
        moe_out = moe(attn_out)
        mx.eval(moe_out)

        assert moe_out.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(moe_out)), "Output contains NaN"

    def test_attention_moe_backward(self):
        """Test gradient flow through attention->MoE pipeline."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.advanced.moe import MoELayer

        batch, seq_len, dims = 2, 32, 128
        num_heads, head_dim = 4, dims // 4

        moe = MoELayer(dims=dims, hidden_dims=dims * 4, num_experts=4, top_k=2)

        def loss_fn(x):
            # Attention forward
            q = x.reshape(batch, seq_len, num_heads, head_dim)
            k = q
            v = q
            attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
            attn_out = attn_out.reshape(batch, seq_len, dims)

            # MoE forward
            moe_out = moe(attn_out)
            return mx.sum(moe_out)

        x = mx.random.normal((batch, seq_len, dims))
        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)), "Gradients contain NaN"
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"


class TestSSMPipelineIntegration:
    """Test SSM pipeline integration."""

    def test_mamba_block_stacked(self):
        """Test multiple Mamba blocks with residual connections."""
        from mlx_primitives.advanced.ssm import MambaBlock
        from mlx_primitives.layers.normalization import RMSNorm

        batch, seq_len, dims = 2, 128, 256
        num_layers = 4

        # Build stacked Mamba model
        layers = []
        for _ in range(num_layers):
            layers.append(RMSNorm(dims))
            layers.append(MambaBlock(dims=dims, d_state=16))

        x = mx.random.normal((batch, seq_len, dims))

        # Forward with residuals
        hidden = x
        for i in range(0, len(layers), 2):
            norm = layers[i]
            mamba = layers[i + 1]
            hidden = hidden + mamba(norm(hidden))
        mx.eval(hidden)

        assert hidden.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(hidden)), "Output contains NaN"

    def test_mamba_with_attention_hybrid(self):
        """Test Mamba + Attention hybrid architecture."""
        from mlx_primitives.advanced.ssm import MambaBlock
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.layers.normalization import RMSNorm

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8

        mamba = MambaBlock(dims=dims, d_state=16)
        norm1 = RMSNorm(dims)
        norm2 = RMSNorm(dims)

        x = mx.random.normal((batch, seq_len, dims))

        # Mamba block
        hidden = x + mamba(norm1(x))

        # Attention block
        q = hidden.reshape(batch, seq_len, num_heads, head_dim)
        k, v = q, q
        attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
        attn_out = attn_out.reshape(batch, seq_len, dims)
        hidden = hidden + attn_out

        mx.eval(hidden)

        assert hidden.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(hidden)), "Output contains NaN"


class TestQuantizationInference:
    """Test quantized inference pipelines."""

    def test_quantized_linear_pipeline(self):
        """Test inference with quantized linear layers."""
        from mlx_primitives.quantization.int8 import quantize_weight, quantized_matmul

        batch, seq_len, dims = 2, 64, 256
        hidden_dims = dims * 4

        # Create and quantize weights
        weight1 = mx.random.normal((hidden_dims, dims))
        weight2 = mx.random.normal((dims, hidden_dims))

        w1_quant, w1_scale = quantize_weight(weight1)
        w2_quant, w2_scale = quantize_weight(weight2)

        x = mx.random.normal((batch, seq_len, dims))

        # Quantized forward pass
        h = quantized_matmul(x, w1_quant, w1_scale)
        h = mx.maximum(0, h)  # ReLU
        out = quantized_matmul(h, w2_quant, w2_scale)
        mx.eval(out)

        assert out.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(out)), "Output contains NaN"

    def test_attention_with_quantized_projections(self):
        """Test attention with INT8 weight quantization."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.quantization.int8 import quantize_weight, quantized_matmul

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8

        # Quantize projection weights
        wq = mx.random.normal((dims, dims))
        wk = mx.random.normal((dims, dims))
        wv = mx.random.normal((dims, dims))
        wo = mx.random.normal((dims, dims))

        wq_quant, wq_scale = quantize_weight(wq)
        wk_quant, wk_scale = quantize_weight(wk)
        wv_quant, wv_scale = quantize_weight(wv)
        wo_quant, wo_scale = quantize_weight(wo)

        x = mx.random.normal((batch, seq_len, dims))

        # Quantized projections
        q = quantized_matmul(x, wq_quant, wq_scale).reshape(batch, seq_len, num_heads, head_dim)
        k = quantized_matmul(x, wk_quant, wk_scale).reshape(batch, seq_len, num_heads, head_dim)
        v = quantized_matmul(x, wv_quant, wv_scale).reshape(batch, seq_len, num_heads, head_dim)

        # Full precision attention
        attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
        attn_out = attn_out.reshape(batch, seq_len, dims)

        # Quantized output projection
        out = quantized_matmul(attn_out, wo_quant, wo_scale)
        mx.eval(out)

        assert out.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(out)), "Output contains NaN"


class TestMultiPrecisionPipeline:
    """Test multi-precision pipeline integration."""

    def test_fp16_attention_fp32_ffn(self):
        """Test mixed precision: FP16 attention + FP32 FFN."""
        from mlx_primitives.attention import flash_attention_forward

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8
        hidden_dims = dims * 4

        # FP32 inputs
        x = mx.random.normal((batch, seq_len, dims))

        # FP16 attention
        q = x.reshape(batch, seq_len, num_heads, head_dim).astype(mx.float16)
        k, v = q, q
        attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
        attn_out = attn_out.reshape(batch, seq_len, dims).astype(mx.float32)

        # FP32 residual + norm (simulated)
        hidden = x + attn_out
        hidden = hidden / mx.sqrt(mx.mean(hidden ** 2, axis=-1, keepdims=True) + 1e-6)

        # FP32 FFN
        w1 = mx.random.normal((hidden_dims, dims))
        w2 = mx.random.normal((dims, hidden_dims))
        ffn_out = mx.maximum(0, hidden @ w1.T) @ w2.T

        out = hidden + ffn_out
        mx.eval(out)

        assert out.shape == (batch, seq_len, dims)
        assert out.dtype == mx.float32
        assert not mx.any(mx.isnan(out)), "Output contains NaN"

    def test_precision_system_with_attention(self):
        """Test precision context manager with attention."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.precision import PrecisionContext

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8

        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k, v = q, q

        # FP32 attention
        with PrecisionContext(dtype=mx.float32):
            out_fp32 = flash_attention_forward(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=1.0 / np.sqrt(head_dim),
            )
        mx.eval(out_fp32)

        # FP16 attention
        with PrecisionContext(dtype=mx.float16):
            out_fp16 = flash_attention_forward(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                scale=1.0 / np.sqrt(head_dim),
            )
        mx.eval(out_fp16)

        # Results should be similar
        out_fp16_as_fp32 = out_fp16.astype(mx.float32)
        diff = mx.abs(out_fp32 - out_fp16_as_fp32)
        max_diff = float(mx.max(diff))

        assert max_diff < 0.1, f"FP16/FP32 difference too large: {max_diff}"


class TestEndToEndTransformer:
    """Test end-to-end transformer-like architectures."""

    def test_transformer_block(self):
        """Test complete transformer block: attention + ffn + norms."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.layers.normalization import RMSNorm
        from mlx_primitives.layers.activations import SwiGLU

        batch, seq_len, dims = 2, 64, 256
        num_heads, head_dim = 8, dims // 8
        hidden_dims = dims * 4

        # Layers
        norm1 = RMSNorm(dims)
        norm2 = RMSNorm(dims)
        swiglu = SwiGLU(dims, hidden_dims)
        w_out = mx.random.normal((dims, dims))

        x = mx.random.normal((batch, seq_len, dims))

        # Attention sub-block
        h = norm1(x)
        q = h.reshape(batch, seq_len, num_heads, head_dim)
        k, v = q, q
        attn_out = flash_attention_forward(q, k, v, scale=1.0 / np.sqrt(head_dim))
        attn_out = attn_out.reshape(batch, seq_len, dims) @ w_out.T
        x = x + attn_out

        # FFN sub-block
        h = norm2(x)
        ffn_out = swiglu(h)
        x = x + ffn_out

        mx.eval(x)

        assert x.shape == (batch, seq_len, dims)
        assert not mx.any(mx.isnan(x)), "Output contains NaN"

    def test_transformer_block_gradient_flow(self):
        """Verify gradient flow through complete transformer block."""
        from mlx_primitives.attention import flash_attention_forward
        from mlx_primitives.layers.normalization import RMSNorm
        from mlx_primitives.layers.activations import SwiGLU

        batch, seq_len, dims = 2, 32, 128
        num_heads, head_dim = 4, dims // 4
        hidden_dims = dims * 4

        norm1 = RMSNorm(dims)
        norm2 = RMSNorm(dims)
        swiglu = SwiGLU(dims, hidden_dims)
        w_out = mx.random.normal((dims, dims))

        def loss_fn(x):
            # Attention sub-block
            h = norm1(x)
            q = h.reshape(batch, seq_len, num_heads, head_dim)
            attn_out = flash_attention_forward(q, q, q, scale=1.0 / np.sqrt(head_dim))
            attn_out = attn_out.reshape(batch, seq_len, dims) @ w_out.T
            x = x + attn_out

            # FFN sub-block
            h = norm2(x)
            ffn_out = swiglu(h)
            x = x + ffn_out

            return mx.sum(x)

        x = mx.random.normal((batch, seq_len, dims))
        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)), "Gradients contain NaN"
        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are all zero"
