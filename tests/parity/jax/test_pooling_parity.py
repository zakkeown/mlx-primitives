"""JAX Metal parity tests for pooling operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn
    from tests.reference_jax_extended import (
        jax_adaptive_avg_pool1d,
        jax_adaptive_avg_pool2d,
        jax_adaptive_max_pool1d,
        jax_adaptive_max_pool2d,
        jax_global_attention_pooling,
        jax_gem,
        jax_spatial_pyramid_pooling,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestAdaptiveAvgPool1dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [1, 4, 16])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, output_size, dtype, skip_without_jax):
        """Test AdaptiveAvgPool1d forward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveAvgPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]  # Use width as 1D length

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # Convert to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        x_typed = np.array(jnp.array(x_np).astype(jax_dtype).astype(jnp.float32))

        # MLX AdaptiveAvgPool1d
        pool_mlx = AdaptiveAvgPool1d(output_size=output_size)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference with dtype-converted input
        jax_out = jax_adaptive_avg_pool1d(x_typed, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool1d forward mismatch (JAX) [{size}, output={output_size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("output_size", [1, 4])
    def test_backward_parity(self, size, output_size, skip_without_jax):
        """Test AdaptiveAvgPool1d backward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveAvgPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX gradient
        pool_mlx = AdaptiveAvgPool1d(output_size=output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient - pure JAX implementation for differentiability
        def jax_adaptive_avg_pool1d_pure(x, out_size):
            input_size = x.shape[-1]
            outputs = []
            for i in range(out_size):
                start = (i * input_size) // out_size
                end = ((i + 1) * input_size + out_size - 1) // out_size
                pooled = jnp.mean(x[..., start:end], axis=-1, keepdims=True)
                outputs.append(pooled)
            return jnp.concatenate(outputs, axis=-1)

        def jax_loss_fn(x):
            return jnp.sum(jax_adaptive_avg_pool1d_pure(x, output_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        # Use looser tolerance for gradients (10x forward tolerance)
        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"AdaptiveAvgPool1d backward mismatch (JAX) [{size}, output={output_size}]"
        )


class TestAdaptiveAvgPool2dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4), (7, 7)])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, output_size, dtype, skip_without_jax):
        """Test AdaptiveAvgPool2d forward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveAvgPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # Convert to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        x_typed = np.array(jnp.array(x_np).astype(jax_dtype).astype(jnp.float32))

        # MLX AdaptiveAvgPool2d
        pool_mlx = AdaptiveAvgPool2d(output_size=output_size)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference with dtype-converted input
        jax_out = jax_adaptive_avg_pool2d(x_typed, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveAvgPool2d forward mismatch (JAX) [{size}, output={output_size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4)])
    def test_backward_parity(self, size, output_size, skip_without_jax):
        """Test AdaptiveAvgPool2d backward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveAvgPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX gradient
        pool_mlx = AdaptiveAvgPool2d(output_size=output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient - pure JAX implementation
        def jax_adaptive_avg_pool2d_pure(x, out_size):
            _, _, in_h, in_w = x.shape
            out_h, out_w = out_size
            if out_h == 1 and out_w == 1:
                return jnp.mean(x, axis=(2, 3), keepdims=True)
            outputs = []
            for i in range(out_h):
                row_outputs = []
                start_h = (i * in_h) // out_h
                end_h = ((i + 1) * in_h + out_h - 1) // out_h
                for j in range(out_w):
                    start_w = (j * in_w) // out_w
                    end_w = ((j + 1) * in_w + out_w - 1) // out_w
                    pooled = jnp.mean(x[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
                    row_outputs.append(pooled)
                outputs.append(jnp.concatenate(row_outputs, axis=3))
            return jnp.concatenate(outputs, axis=2)

        def jax_loss_fn(x):
            return jnp.sum(jax_adaptive_avg_pool2d_pure(x, output_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool2d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"AdaptiveAvgPool2d backward mismatch (JAX) [{size}, output={output_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_global_pooling(self, skip_without_jax):
        """Test global pooling special case (1, 1)."""
        from mlx_primitives.layers import AdaptiveAvgPool2d

        batch, channels, height, width = 2, 64, 32, 32
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        pool_mlx = AdaptiveAvgPool2d((1, 1))
        x_mlx = mx.array(x_np)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference - global average pooling
        jax_out = jnp.mean(jnp.array(x_np), axis=(2, 3), keepdims=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="Global avg pooling mismatch (JAX)"
        )


class TestAdaptiveMaxPool1dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [1, 4, 16])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, output_size, dtype, skip_without_jax):
        """Test AdaptiveMaxPool1d forward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveMaxPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX AdaptiveMaxPool1d
        pool_mlx = AdaptiveMaxPool1d(output_size=output_size)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_adaptive_max_pool1d(x_np, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool1d forward mismatch (JAX) [{size}, output={output_size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("output_size", [1, 4])
    def test_backward_parity(self, size, output_size, skip_without_jax):
        """Test AdaptiveMaxPool1d backward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveMaxPool1d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        length = config["width"]

        # Use unique values to avoid max pool tie-breaking ambiguity
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)
        # Add small perturbation to ensure unique max values
        x_np = x_np + np.arange(length).reshape(1, 1, -1) * 1e-6

        # MLX gradient
        pool_mlx = AdaptiveMaxPool1d(output_size=output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient - pure JAX implementation
        def jax_adaptive_max_pool1d_pure(x, out_size):
            input_size = x.shape[-1]
            outputs = []
            for i in range(out_size):
                start = (i * input_size) // out_size
                end = ((i + 1) * input_size + out_size - 1) // out_size
                pooled = jnp.max(x[..., start:end], axis=-1, keepdims=True)
                outputs.append(pooled)
            return jnp.concatenate(outputs, axis=-1)

        def jax_loss_fn(x):
            return jnp.sum(jax_adaptive_max_pool1d_pure(x, output_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"AdaptiveMaxPool1d backward mismatch (JAX) [{size}, output={output_size}]"
        )


class TestAdaptiveMaxPool2dParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4), (7, 7)])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, output_size, dtype, skip_without_jax):
        """Test AdaptiveMaxPool2d forward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveMaxPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX AdaptiveMaxPool2d
        pool_mlx = AdaptiveMaxPool2d(output_size=output_size)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_adaptive_max_pool2d(x_np, output_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool2d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"AdaptiveMaxPool2d forward mismatch (JAX) [{size}, output={output_size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("output_size", [(1, 1), (4, 4)])
    def test_backward_parity(self, size, output_size, skip_without_jax):
        """Test AdaptiveMaxPool2d backward pass parity with JAX."""
        from mlx_primitives.layers import AdaptiveMaxPool2d

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        # Use unique values to avoid max pool tie-breaking ambiguity
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)
        # Add small perturbation to ensure unique max values
        idx = np.arange(height * width).reshape(1, 1, height, width) * 1e-6
        x_np = x_np + idx

        # MLX gradient
        pool_mlx = AdaptiveMaxPool2d(output_size=output_size)

        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient - pure JAX implementation
        def jax_adaptive_max_pool2d_pure(x, out_size):
            _, _, in_h, in_w = x.shape
            out_h, out_w = out_size
            if out_h == 1 and out_w == 1:
                return jnp.max(x, axis=(2, 3), keepdims=True)
            outputs = []
            for i in range(out_h):
                row_outputs = []
                start_h = (i * in_h) // out_h
                end_h = ((i + 1) * in_h + out_h - 1) // out_h
                for j in range(out_w):
                    start_w = (j * in_w) // out_w
                    end_w = ((j + 1) * in_w + out_w - 1) // out_w
                    pooled = jnp.max(x[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
                    row_outputs.append(pooled)
                outputs.append(jnp.concatenate(row_outputs, axis=3))
            return jnp.concatenate(outputs, axis=2)

        def jax_loss_fn(x):
            return jnp.sum(jax_adaptive_max_pool2d_pure(x, output_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool2d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"AdaptiveMaxPool2d backward mismatch (JAX) [{size}, output={output_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_global_pooling(self, skip_without_jax):
        """Test global max pooling special case (1, 1)."""
        from mlx_primitives.layers import AdaptiveMaxPool2d

        batch, channels, height, width = 2, 64, 32, 32
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        pool_mlx = AdaptiveMaxPool2d((1, 1))
        x_mlx = mx.array(x_np)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference - global max pooling
        jax_out = jnp.max(jnp.array(x_np), axis=(2, 3), keepdims=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=1e-5, atol=1e-6,
            err_msg="Global max pooling mismatch (JAX)"
        )


class TestGlobalAttentionPoolingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test GlobalAttentionPooling forward pass parity with JAX."""
        from mlx_primitives.layers import GlobalAttentionPooling

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["hidden"]
        hidden = dim // 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # Convert to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        x_typed = np.array(jnp.array(x_np).astype(jax_dtype).astype(jnp.float32))

        # MLX GlobalAttentionPooling
        pool_mlx = GlobalAttentionPooling(dims=dim, hidden_dims=hidden)
        mx.eval(pool_mlx.parameters())

        # Get weights - attention module uses nn.Sequential with Linear layers
        # Linear layers have bias by default
        W1_np = np.array(pool_mlx.attention.layers[0].weight)  # (hidden_dims, dims)
        b1_np = np.array(pool_mlx.attention.layers[0].bias)    # (hidden_dims,)
        W2_np = np.array(pool_mlx.attention.layers[2].weight)  # (1, hidden_dims)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference - compute same way as MLX
        # MLX Linear does: x @ weight.T + bias
        x_j = jnp.array(x_typed, dtype=jnp.float32)
        W1_j = jnp.array(W1_np, dtype=jnp.float32)
        b1_j = jnp.array(b1_np, dtype=jnp.float32)
        W2_j = jnp.array(W2_np, dtype=jnp.float32)

        # Step 1: h = tanh(x @ W1.T + b1)
        h = jnp.tanh(x_j @ W1_j.T + b1_j)
        # Step 2: scores = h @ W2.T (no bias for layers[2])
        scores = h @ W2_j.T
        # Step 3: softmax over seq dimension
        scores = scores.squeeze(-1)  # (batch, seq)
        weights = jnn.softmax(scores, axis=1)
        # Step 4: weighted sum
        jax_out = np.array(jnp.sum(x_j * weights[..., None], axis=1))

        rtol, atol = get_tolerance("pooling", "global_attention_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"GlobalAttentionPooling forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test GlobalAttentionPooling backward pass parity with JAX (input gradient only)."""
        from mlx_primitives.layers import GlobalAttentionPooling

        config = SIZE_CONFIGS[size]["normalization"]
        batch = config["batch"]
        seq = config["seq"]
        dim = config["hidden"]
        hidden = dim // 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX GlobalAttentionPooling
        pool_mlx = GlobalAttentionPooling(dims=dim, hidden_dims=hidden)
        mx.eval(pool_mlx.parameters())

        # Get weights for JAX reference
        W1_np = np.array(pool_mlx.attention.layers[0].weight)
        b1_np = np.array(pool_mlx.attention.layers[0].bias)
        W2_np = np.array(pool_mlx.attention.layers[2].weight)

        # MLX gradient w.r.t. input
        def mlx_loss_fn(x):
            return mx.sum(pool_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient w.r.t. input (with same weights)
        W1_j = jnp.array(W1_np)
        b1_j = jnp.array(b1_np)
        W2_j = jnp.array(W2_np)

        def jax_loss_fn(x):
            h = jnp.tanh(x @ W1_j.T + b1_j)
            scores = (h @ W2_j.T).squeeze(-1)
            weights = jnn.softmax(scores, axis=1)
            out = jnp.sum(x * weights[..., None], axis=1)
            return jnp.sum(out)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "global_attention_pooling", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"GlobalAttentionPooling backward mismatch (JAX) [{size}]"
        )


class TestGeMParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, p, dtype, skip_without_jax):
        """Test GeM pooling forward pass parity with JAX."""
        from mlx_primitives.layers import GeM

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        # Use positive values for GeM (clipped internally)
        x_np = np.abs(np.random.randn(batch, channels, height, width)).astype(np.float32) + 0.1

        # MLX GeM
        gem_mlx = GeM(p=p)
        mx.eval(gem_mlx.parameters())

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = gem_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_gem(x_np, p=p, eps=gem_mlx.eps)

        rtol, atol = get_tolerance("pooling", "gem", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"GeM forward mismatch (JAX) [{size}, p={p}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    @pytest.mark.parametrize("p", [2.0, 3.0])  # Skip p=1 for gradient stability
    def test_backward_parity(self, size, p, skip_without_jax):
        """Test GeM pooling backward pass parity with JAX."""
        from mlx_primitives.layers import GeM

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        # Use positive values with offset to avoid gradient issues near zero
        x_np = np.abs(np.random.randn(batch, channels, height, width)).astype(np.float32) + 0.5

        gem_mlx = GeM(p=p)
        mx.eval(gem_mlx.parameters())
        eps = gem_mlx.eps

        # MLX gradient
        def mlx_loss_fn(x):
            return mx.sum(gem_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient
        def jax_loss_fn(x):
            x_clipped = jnp.clip(x, eps, None)
            x_pow = jnp.power(x_clipped, p)
            pooled = jnp.mean(x_pow, axis=(2, 3), keepdims=True)
            out = jnp.power(pooled, 1.0 / p)
            return jnp.sum(out)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "gem", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"GeM backward mismatch (JAX) [{size}, p={p}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_p_equals_1(self, skip_without_jax):
        """Test GeM with p=1 (equivalent to average pooling)."""
        from mlx_primitives.layers import GeM

        batch, channels, height, width = 2, 64, 8, 8
        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + 1e-6

        gem_mlx = GeM(p=1.0, eps=1e-6, learnable=False)
        x_mlx = mx.array(x_np)
        mlx_out = gem_mlx(x_mlx)
        mx.eval(mlx_out)

        # p=1 should be equivalent to average pooling
        jax_out = jnp.mean(jnp.array(x_np), axis=(2, 3), keepdims=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=1e-4, atol=1e-5,
            err_msg="GeM p=1 should equal average pooling (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_large_p(self, skip_without_jax):
        """Test GeM with large p (approaches max pooling)."""
        from mlx_primitives.layers import GeM

        batch, channels, height, width = 2, 64, 8, 8
        np.random.seed(42)
        x_np = np.abs(np.random.randn(batch, channels, height, width).astype(np.float32)) + 1e-6

        gem_mlx = GeM(p=10.0, eps=1e-6, learnable=False)
        x_mlx = mx.array(x_np)
        mlx_out = gem_mlx(x_mlx)
        mx.eval(mlx_out)

        # Large p approaches max pooling
        jax_max = jnp.max(jnp.array(x_np), axis=(2, 3), keepdims=True)

        # GeM with large p should be close to max but not exact
        mlx_np = _to_numpy(mlx_out)
        jax_np = np.array(jax_max)

        # GeM with large p should be <= max (and close to it)
        assert np.all(mlx_np <= jax_np * 1.01), "GeM large p should be close to max"


# =============================================================================
# 1D Pooling Tests
# =============================================================================

# Size configs for 1D pooling
POOLING_1D_SIZE_CONFIGS = {
    "tiny": {"batch": 2, "channels": 16, "length": 32},
    "small": {"batch": 4, "channels": 32, "length": 64},
    "medium": {"batch": 8, "channels": 64, "length": 128},
    "large": {"batch": 16, "channels": 128, "length": 256},
}


def _numpy_avg_pool1d(x, kernel_size, stride=None):
    """NumPy reference for 1D average pooling."""
    if stride is None:
        stride = kernel_size
    batch, channels, length = x.shape
    out_length = (length - kernel_size) // stride + 1
    output = np.zeros((batch, channels, out_length), dtype=x.dtype)
    for i in range(out_length):
        start = i * stride
        end = start + kernel_size
        output[:, :, i] = np.mean(x[:, :, start:end], axis=2)
    return output


def _numpy_max_pool1d(x, kernel_size, stride=None):
    """NumPy reference for 1D max pooling."""
    if stride is None:
        stride = kernel_size
    batch, channels, length = x.shape
    out_length = (length - kernel_size) // stride + 1
    output = np.zeros((batch, channels, out_length), dtype=x.dtype)
    for i in range(out_length):
        start = i * stride
        end = start + kernel_size
        output[:, :, i] = np.max(x[:, :, start:end], axis=2)
    return output


class TestAvgPool1dParity:
    """Tests for AvgPool1d parity with JAX/NumPy reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32"])
    @pytest.mark.parametrize("kernel_size", [2, 3, 5])
    def test_forward_parity(self, size, dtype, kernel_size, skip_without_jax):
        """Test AvgPool1d forward pass matches reference."""
        from mlx_primitives.layers.pooling import AvgPool1d

        config = POOLING_1D_SIZE_CONFIGS[size]
        batch, channels, length = config["batch"], config["channels"], config["length"]

        if kernel_size > length:
            pytest.skip(f"kernel_size={kernel_size} > length={length}")

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = AvgPool1d(kernel_size=kernel_size)
        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference
        ref_out = _numpy_avg_pool1d(x_np, kernel_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"AvgPool1d mismatch [{size}, {dtype}, kernel={kernel_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_various_strides(self, stride, skip_without_jax):
        """Test AvgPool1d with different strides."""
        from mlx_primitives.layers.pooling import AvgPool1d

        batch, channels, length = 4, 64, 128
        kernel_size = 3

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = AvgPool1d(kernel_size=kernel_size, stride=stride)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference
        ref_out = _numpy_avg_pool1d(x_np, kernel_size, stride)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"AvgPool1d stride mismatch [stride={stride}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test AvgPool1d backward pass gradients flow correctly."""
        from mlx_primitives.layers.pooling import AvgPool1d

        config = POOLING_1D_SIZE_CONFIGS[size]
        batch, channels, length = config["batch"], config["channels"], config["length"]
        kernel_size = 3

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = AvgPool1d(kernel_size=kernel_size)

        def mlx_loss_fn(x):
            return mx.sum(pool(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient using pure implementation
        def jax_avg_pool1d_pure(x, kernel_size):
            length = x.shape[-1]
            out_length = (length - kernel_size) // kernel_size + 1
            outputs = []
            for i in range(out_length):
                start = i * kernel_size
                end = start + kernel_size
                pooled = jnp.mean(x[..., start:end], axis=-1, keepdims=True)
                outputs.append(pooled)
            return jnp.concatenate(outputs, axis=-1)

        def jax_loss_fn(x):
            return jnp.sum(jax_avg_pool1d_pure(x, kernel_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"AvgPool1d backward mismatch [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.parametrize("padding", [0, 1, 2])
    def test_with_padding(self, padding, skip_without_jax):
        """Test AvgPool1d with padding."""
        from mlx_primitives.layers.pooling import AvgPool1d

        batch, channels, length = 4, 64, 128
        kernel_size = 5

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = AvgPool1d(kernel_size=kernel_size, padding=padding)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference with padding
        x_padded = np.pad(x_np, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        ref_out = _numpy_avg_pool1d(x_padded, kernel_size)

        rtol, atol = get_tolerance("pooling", "adaptive_avg_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"AvgPool1d with padding={padding} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_kernel_equals_input(self, skip_without_jax):
        """Test AvgPool1d when kernel_size equals input length (global pooling)."""
        from mlx_primitives.layers.pooling import AvgPool1d

        batch, channels, length = 4, 64, 32
        kernel_size = length

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = AvgPool1d(kernel_size=kernel_size)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # Global average pooling reference
        ref_out = np.mean(x_np, axis=2, keepdims=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=1e-5, atol=1e-6,
            err_msg="AvgPool1d global pooling mismatch (JAX)"
        )


class TestMaxPool1dParity:
    """Tests for MaxPool1d parity with NumPy reference."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32"])
    @pytest.mark.parametrize("kernel_size", [2, 3, 5])
    def test_forward_parity(self, size, dtype, kernel_size, skip_without_jax):
        """Test MaxPool1d forward pass matches reference."""
        from mlx_primitives.layers.pooling import MaxPool1d

        config = POOLING_1D_SIZE_CONFIGS[size]
        batch, channels, length = config["batch"], config["channels"], config["length"]

        if kernel_size > length:
            pytest.skip(f"kernel_size={kernel_size} > length={length}")

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = MaxPool1d(kernel_size=kernel_size)
        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference
        ref_out = _numpy_max_pool1d(x_np, kernel_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"MaxPool1d mismatch [{size}, {dtype}, kernel={kernel_size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_various_strides(self, stride, skip_without_jax):
        """Test MaxPool1d with different strides."""
        from mlx_primitives.layers.pooling import MaxPool1d

        batch, channels, length = 4, 64, 128
        kernel_size = 3

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = MaxPool1d(kernel_size=kernel_size, stride=stride)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference
        ref_out = _numpy_max_pool1d(x_np, kernel_size, stride)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"MaxPool1d stride mismatch [stride={stride}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test MaxPool1d backward pass gradients flow correctly."""
        from mlx_primitives.layers.pooling import MaxPool1d

        config = POOLING_1D_SIZE_CONFIGS[size]
        batch, channels, length = config["batch"], config["channels"], config["length"]
        kernel_size = 3

        # Use unique values to avoid max pool tie-breaking ambiguity
        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)
        x_np = x_np + np.arange(length).reshape(1, 1, -1) * 1e-6

        pool = MaxPool1d(kernel_size=kernel_size)

        def mlx_loss_fn(x):
            return mx.sum(pool(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient using pure implementation
        def jax_max_pool1d_pure(x, kernel_size):
            length = x.shape[-1]
            out_length = (length - kernel_size) // kernel_size + 1
            outputs = []
            for i in range(out_length):
                start = i * kernel_size
                end = start + kernel_size
                pooled = jnp.max(x[..., start:end], axis=-1, keepdims=True)
                outputs.append(pooled)
            return jnp.concatenate(outputs, axis=-1)

        def jax_loss_fn(x):
            return jnp.sum(jax_max_pool1d_pure(x, kernel_size))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"MaxPool1d backward mismatch [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.parametrize("padding", [0, 1, 2])
    def test_with_padding(self, padding, skip_without_jax):
        """Test MaxPool1d with padding."""
        from mlx_primitives.layers.pooling import MaxPool1d

        batch, channels, length = 4, 64, 128
        kernel_size = 5

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = MaxPool1d(kernel_size=kernel_size, padding=padding)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference with padding (use -inf for max pooling)
        x_padded = np.pad(x_np, ((0, 0), (0, 0), (padding, padding)),
                         mode='constant', constant_values=-np.inf)
        ref_out = _numpy_max_pool1d(x_padded, kernel_size)

        rtol, atol = get_tolerance("pooling", "adaptive_max_pool1d", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=rtol, atol=atol,
            err_msg=f"MaxPool1d with padding={padding} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_kernel_equals_input(self, skip_without_jax):
        """Test MaxPool1d when kernel_size equals input length (global max pooling)."""
        from mlx_primitives.layers.pooling import MaxPool1d

        batch, channels, length = 4, 64, 32
        kernel_size = length

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        pool = MaxPool1d(kernel_size=kernel_size)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # Global max pooling reference
        ref_out = np.max(x_np, axis=2, keepdims=True)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=1e-5, atol=1e-6,
            err_msg="MaxPool1d global pooling mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_with_negative_values(self, skip_without_jax):
        """Test MaxPool1d with all negative values."""
        from mlx_primitives.layers.pooling import MaxPool1d

        batch, channels, length = 2, 32, 64
        kernel_size = 3

        np.random.seed(42)
        x_np = -np.abs(np.random.randn(batch, channels, length)).astype(np.float32)

        pool = MaxPool1d(kernel_size=kernel_size)
        x_mlx = mx.array(x_np)
        mlx_out = pool(x_mlx)
        mx.eval(mlx_out)

        # NumPy reference
        ref_out = _numpy_max_pool1d(x_np, kernel_size)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), ref_out,
            rtol=1e-5, atol=1e-6,
            err_msg="MaxPool1d with negative values mismatch (JAX)"
        )


# =============================================================================
# 2D Pooling Tests
# =============================================================================

class TestSpatialPyramidPoolingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    @pytest.mark.parametrize("levels", [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]])
    def test_forward_parity(self, size, dtype, levels, skip_without_jax):
        """Test SpatialPyramidPooling forward pass parity with JAX."""
        from mlx_primitives.layers import SpatialPyramidPooling

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # Convert to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        x_typed = np.array(jnp.array(x_np).astype(jax_dtype).astype(jnp.float32))

        # MLX SpatialPyramidPooling
        spp_mlx = SpatialPyramidPooling(output_sizes=levels)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = spp_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference with dtype-converted input
        jax_out = jax_spatial_pyramid_pooling(x_typed, levels)

        rtol, atol = get_tolerance("pooling", "spatial_pyramid_pooling", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"SpatialPyramidPooling forward mismatch (JAX) [{size}, {dtype}, levels={levels}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test SpatialPyramidPooling backward pass parity with JAX."""
        from mlx_primitives.layers import SpatialPyramidPooling

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        levels = [1, 2, 4]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX gradient
        spp_mlx = SpatialPyramidPooling(output_sizes=levels)

        def mlx_loss_fn(x):
            return mx.sum(spp_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX gradient - pure JAX implementation
        def jax_adaptive_avg_pool2d_pure(x, out_size):
            _, _, in_h, in_w = x.shape
            out_h, out_w = out_size
            if out_h == 1 and out_w == 1:
                return jnp.mean(x, axis=(2, 3), keepdims=True)
            outputs = []
            for i in range(out_h):
                row_outputs = []
                start_h = (i * in_h) // out_h
                end_h = ((i + 1) * in_h + out_h - 1) // out_h
                for j in range(out_w):
                    start_w = (j * in_w) // out_w
                    end_w = ((j + 1) * in_w + out_w - 1) // out_w
                    pooled = jnp.mean(x[:, :, start_h:end_h, start_w:end_w], axis=(2, 3), keepdims=True)
                    row_outputs.append(pooled)
                outputs.append(jnp.concatenate(row_outputs, axis=3))
            return jnp.concatenate(outputs, axis=2)

        def jax_spp_pure(x, pyramid_levels):
            batch_size, chans, _, _ = x.shape
            outputs = []
            for level in pyramid_levels:
                pooled = jax_adaptive_avg_pool2d_pure(x, (level, level))
                pooled = pooled.reshape(batch_size, -1)
                outputs.append(pooled)
            return jnp.concatenate(outputs, axis=1)

        def jax_loss_fn(x):
            return jnp.sum(jax_spp_pure(x, levels))

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("pooling", "spatial_pyramid_pooling", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol * 10, atol=atol * 10,
            err_msg=f"SpatialPyramidPooling backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_output_shape(self, skip_without_jax):
        """Test SPP output shape matches expected."""
        from mlx_primitives.layers import SpatialPyramidPooling

        batch, channels, height, width = 2, 256, 13, 13
        levels = [1, 2, 4]

        # Expected output: channels * (1 + 4 + 16) = channels * 21
        expected_features = channels * sum(level**2 for level in levels)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        spp_mlx = SpatialPyramidPooling(output_sizes=levels)
        x_mlx = mx.array(x_np)
        mlx_out = spp_mlx(x_mlx)
        mx.eval(mlx_out)

        assert mlx_out.shape == (batch, expected_features), \
            f"SPP output shape mismatch: expected {(batch, expected_features)}, got {mlx_out.shape}"
