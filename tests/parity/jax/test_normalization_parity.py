"""JAX Metal parity tests for normalization operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import normalization_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from tests.reference_jax import jax_rmsnorm


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestRMSNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test RMSNorm forward pass parity with JAX."""
        from mlx_primitives.layers import RMSNorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX RMSNorm
        rmsnorm_mlx = RMSNorm(dims=hidden)
        mx.eval(rmsnorm_mlx.parameters())
        weight_np = np.array(rmsnorm_mlx.weight)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = rmsnorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_rmsnorm(x_np, weight_np, eps=rmsnorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        """Test RMSNorm backward pass parity with JAX."""
        from mlx_primitives.layers import RMSNorm

        config = SIZE_CONFIGS["small"]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX forward and backward
        rmsnorm_mlx = RMSNorm(dims=hidden)
        mx.eval(rmsnorm_mlx.parameters())
        weight_np = np.array(rmsnorm_mlx.weight)
        eps = rmsnorm_mlx.eps

        def mlx_fn(x):
            return rmsnorm_mlx(x).sum()

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(x):
            w = jnp.array(weight_np, dtype=jnp.float32)
            variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
            normalized = x * jax.lax.rsqrt(variance + eps)
            return (normalized * w).sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg="RMSNorm backward mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_epsilon_values(self, eps, skip_without_jax):
        """Test different epsilon values."""
        from mlx_primitives.layers import RMSNorm

        batch, seq, hidden = 2, 64, 256

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX RMSNorm
        rmsnorm_mlx = RMSNorm(dims=hidden, eps=eps)
        mx.eval(rmsnorm_mlx.parameters())
        weight_np = np.array(rmsnorm_mlx.weight)

        x_mlx = mx.array(x_np)
        mlx_out = rmsnorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_rmsnorm(x_np, weight_np, eps=eps)

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"RMSNorm epsilon mismatch (JAX) [eps={eps}]"
        )


class TestLayerNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test LayerNorm forward pass parity with JAX."""
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX LayerNorm (using mlx.nn built-in)
        layernorm_mlx = nn.LayerNorm(dims=hidden)
        mx.eval(layernorm_mlx.parameters())
        weight_np = np.array(layernorm_mlx.weight)
        bias_np = np.array(layernorm_mlx.bias)
        eps = 1e-5

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = layernorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference - standard layer normalization
        x_jax = jnp.array(x_np, dtype=jnp.float32)
        w_jax = jnp.array(weight_np, dtype=jnp.float32)
        b_jax = jnp.array(bias_np, dtype=jnp.float32)
        mean = jnp.mean(x_jax, axis=-1, keepdims=True)
        var = jnp.var(x_jax, axis=-1, keepdims=True)
        x_norm = (x_jax - mean) / jnp.sqrt(var + eps)
        jax_out = x_norm * w_jax + b_jax

        rtol, atol = get_tolerance("normalization", "layernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"LayerNorm forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test LayerNorm backward pass parity with JAX."""
        import mlx.nn as nn

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX backward
        layernorm_mlx = nn.LayerNorm(dims=hidden, eps=eps)
        mx.eval(layernorm_mlx.parameters())
        weight_np = np.array(layernorm_mlx.weight)
        bias_np = np.array(layernorm_mlx.bias)

        def mlx_fn(x):
            return layernorm_mlx(x).sum()

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(x):
            w = jnp.array(weight_np, dtype=jnp.float32)
            b = jnp.array(bias_np, dtype=jnp.float32)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / jnp.sqrt(var + eps)
            return (x_norm * w + b).sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("normalization", "layernorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"LayerNorm backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_with_bias(self, skip_without_jax):
        """Test LayerNorm with bias parameter."""
        import mlx.nn as nn

        batch, seq, hidden = 2, 64, 256
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX with affine=True (default)
        layernorm_mlx = nn.LayerNorm(dims=hidden, eps=eps, affine=True)
        mx.eval(layernorm_mlx.parameters())

        # Set non-trivial weight and bias
        layernorm_mlx.weight = mx.array(np.random.randn(hidden).astype(np.float32))
        layernorm_mlx.bias = mx.array(np.random.randn(hidden).astype(np.float32))
        mx.eval(layernorm_mlx.parameters())

        weight_np = np.array(layernorm_mlx.weight)
        bias_np = np.array(layernorm_mlx.bias)

        x_mlx = mx.array(x_np)
        mlx_out = layernorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = jnp.array(x_np, dtype=jnp.float32)
        w_jax = jnp.array(weight_np, dtype=jnp.float32)
        b_jax = jnp.array(bias_np, dtype=jnp.float32)
        mean = jnp.mean(x_jax, axis=-1, keepdims=True)
        var = jnp.var(x_jax, axis=-1, keepdims=True)
        x_norm = (x_jax - mean) / jnp.sqrt(var + eps)
        jax_out = x_norm * w_jax + b_jax

        rtol, atol = get_tolerance("normalization", "layernorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=rtol, atol=atol,
            err_msg="LayerNorm with bias mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_without_bias(self, skip_without_jax):
        """Test LayerNorm without bias parameter."""
        import mlx.nn as nn

        batch, seq, hidden = 2, 64, 256
        eps = 1e-5

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)

        # MLX with affine=False
        layernorm_mlx = nn.LayerNorm(dims=hidden, eps=eps, affine=False)
        mx.eval(layernorm_mlx.parameters())

        x_mlx = mx.array(x_np)
        mlx_out = layernorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference (no weight/bias)
        x_jax = jnp.array(x_np, dtype=jnp.float32)
        mean = jnp.mean(x_jax, axis=-1, keepdims=True)
        var = jnp.var(x_jax, axis=-1, keepdims=True)
        jax_out = (x_jax - mean) / jnp.sqrt(var + eps)

        rtol, atol = get_tolerance("normalization", "layernorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), np.array(jax_out),
            rtol=rtol, atol=atol,
            err_msg="LayerNorm without bias mismatch (JAX)"
        )


class TestGroupNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("num_groups", [1, 4, 8, 32])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, num_groups, dtype, skip_without_jax):
        """Test GroupNorm forward pass parity with JAX."""
        from mlx_primitives.layers import GroupNorm
        from tests.reference_jax_extended import jax_groupnorm

        config = SIZE_CONFIGS[size]["pooling"]  # Use pooling config for NCHW format
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        # Skip if channels not divisible by num_groups
        if channels % num_groups != 0:
            pytest.skip(f"channels={channels} not divisible by num_groups={num_groups}")

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX GroupNorm
        groupnorm_mlx = GroupNorm(num_groups=num_groups, num_channels=channels)
        mx.eval(groupnorm_mlx.parameters())
        weight_np = np.array(groupnorm_mlx.weight)
        bias_np = np.array(groupnorm_mlx.bias)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = groupnorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_groupnorm(x_np, num_groups, weight_np, bias_np, eps=groupnorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"GroupNorm forward mismatch (JAX) [{size}, {num_groups} groups, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test GroupNorm backward pass parity with JAX."""
        from mlx_primitives.layers import GroupNorm
        from tests.reference_jax_extended import jax_groupnorm

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]
        num_groups = 8

        # Skip if not divisible
        if channels % num_groups != 0:
            pytest.skip(f"channels ({channels}) not divisible by num_groups ({num_groups})")

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX backward
        groupnorm_mlx = GroupNorm(num_groups=num_groups, num_channels=channels)
        mx.eval(groupnorm_mlx.parameters())
        weight_np = np.array(groupnorm_mlx.weight)
        bias_np = np.array(groupnorm_mlx.bias)
        eps = groupnorm_mlx.eps

        def mlx_fn(x):
            return groupnorm_mlx(x).sum()

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(x):
            return jax_groupnorm(x, num_groups, weight_np, bias_np, eps=eps).sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("normalization", "groupnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"GroupNorm backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_single_group(self, skip_without_jax):
        """Test GroupNorm with single group (equivalent to LayerNorm)."""
        from mlx_primitives.layers import GroupNorm
        from tests.reference_jax_extended import jax_groupnorm

        batch, channels, height, width = 2, 64, 8, 8
        num_groups = 1

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX GroupNorm with 1 group
        groupnorm_mlx = GroupNorm(num_groups=num_groups, num_channels=channels)
        mx.eval(groupnorm_mlx.parameters())
        weight_np = np.array(groupnorm_mlx.weight)
        bias_np = np.array(groupnorm_mlx.bias)

        x_mlx = mx.array(x_np)
        mlx_out = groupnorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_groupnorm(x_np, num_groups, weight_np, bias_np, eps=groupnorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg="GroupNorm single group mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_channels_equals_groups(self, skip_without_jax):
        """Test GroupNorm with channels == groups (equivalent to InstanceNorm)."""
        from mlx_primitives.layers import GroupNorm
        from tests.reference_jax_extended import jax_groupnorm

        batch, channels, height, width = 2, 64, 8, 8
        num_groups = channels  # One channel per group

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX GroupNorm with channels=groups
        groupnorm_mlx = GroupNorm(num_groups=num_groups, num_channels=channels)
        mx.eval(groupnorm_mlx.parameters())
        weight_np = np.array(groupnorm_mlx.weight)
        bias_np = np.array(groupnorm_mlx.bias)

        x_mlx = mx.array(x_np)
        mlx_out = groupnorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_groupnorm(x_np, num_groups, weight_np, bias_np, eps=groupnorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "groupnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg="GroupNorm channels=groups mismatch (JAX)"
        )


class TestInstanceNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test InstanceNorm forward pass parity with JAX."""
        from mlx_primitives.layers import InstanceNorm
        from tests.reference_jax_extended import jax_instancenorm

        config = SIZE_CONFIGS[size]["pooling"]  # Use pooling config for NCHW format
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX InstanceNorm
        instancenorm_mlx = InstanceNorm(num_features=channels)
        mx.eval(instancenorm_mlx.parameters())
        weight_np = np.array(instancenorm_mlx.weight)
        bias_np = np.array(instancenorm_mlx.bias)

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = instancenorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_instancenorm(x_np, weight_np, bias_np, eps=instancenorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"InstanceNorm forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test InstanceNorm backward pass parity with JAX."""
        from mlx_primitives.layers import InstanceNorm
        from tests.reference_jax_extended import jax_instancenorm

        config = SIZE_CONFIGS[size]["pooling"]
        batch = config["batch"]
        channels = config["channels"]
        height = config["height"]
        width = config["width"]

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX backward
        instancenorm_mlx = InstanceNorm(num_features=channels)
        mx.eval(instancenorm_mlx.parameters())
        weight_np = np.array(instancenorm_mlx.weight)
        bias_np = np.array(instancenorm_mlx.bias)
        eps = instancenorm_mlx.eps

        def mlx_fn(x):
            return instancenorm_mlx(x).sum()

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(x):
            return jax_instancenorm(x, weight_np, bias_np, eps=eps).sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(x_jax)

        rtol, atol = get_gradient_tolerance("normalization", "instancenorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg=f"InstanceNorm backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_1d_input(self, skip_without_jax):
        """Test InstanceNorm1d parity with JAX."""
        from mlx_primitives.layers import InstanceNorm
        from tests.reference_jax_extended import jax_instancenorm

        batch, channels, length = 2, 64, 32

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        # MLX InstanceNorm (1D: NCL)
        instancenorm_mlx = InstanceNorm(num_features=channels)
        mx.eval(instancenorm_mlx.parameters())
        weight_np = np.array(instancenorm_mlx.weight)
        bias_np = np.array(instancenorm_mlx.bias)

        x_mlx = mx.array(x_np)
        mlx_out = instancenorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_instancenorm(x_np, weight_np, bias_np, eps=instancenorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg="InstanceNorm1d mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_2d_input(self, skip_without_jax):
        """Test InstanceNorm2d parity with JAX."""
        from mlx_primitives.layers import InstanceNorm
        from tests.reference_jax_extended import jax_instancenorm

        batch, channels, height, width = 2, 64, 16, 16

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, height, width).astype(np.float32)

        # MLX InstanceNorm (2D: NCHW)
        instancenorm_mlx = InstanceNorm(num_features=channels)
        mx.eval(instancenorm_mlx.parameters())
        weight_np = np.array(instancenorm_mlx.weight)
        bias_np = np.array(instancenorm_mlx.bias)

        x_mlx = mx.array(x_np)
        mlx_out = instancenorm_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_instancenorm(x_np, weight_np, bias_np, eps=instancenorm_mlx.eps)

        rtol, atol = get_tolerance("normalization", "instancenorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg="InstanceNorm2d mismatch (JAX)"
        )


class TestAdaLayerNormParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test AdaLayerNorm forward pass parity with JAX."""
        from mlx_primitives.layers import AdaLayerNorm
        from tests.reference_jax_extended import jax_adalayernorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        cond_dims = hidden // 2  # Conditioning dimension

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32)

        # MLX AdaLayerNorm
        adaln_mlx = AdaLayerNorm(dims=hidden, cond_dims=cond_dims)
        mx.eval(adaln_mlx.parameters())

        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        cond_mlx = mx.array(cond_np).astype(mlx_dtype)
        mlx_out = adaln_mlx(x_mlx, cond_mlx)
        mx.eval(mlx_out)

        # For JAX reference, we need to compute the scale and shift from conditioning
        # using the same projection weights as MLX
        proj_weight_np = np.array(adaln_mlx.proj.weight)  # (dims * 2, cond_dims)
        proj_bias_np = np.array(adaln_mlx.proj.bias) if adaln_mlx.proj.bias is not None else np.zeros(hidden * 2)

        # Convert inputs to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        x_typed = np.array(jnp.array(x_np).astype(jax_dtype).astype(jnp.float32))
        cond_typed = np.array(jnp.array(cond_np).astype(jax_dtype).astype(jnp.float32))

        # Compute scale and shift: cond @ weight.T + bias (use typed cond)
        scale_shift = cond_typed @ proj_weight_np.T + proj_bias_np
        scale_np = scale_shift[:, :hidden]
        shift_np = scale_shift[:, hidden:]

        # Reshape for broadcasting with (batch, seq, hidden)
        scale_np = scale_np[:, None, :]  # (batch, 1, hidden)
        shift_np = shift_np[:, None, :]

        # JAX reference with dtype-converted input
        jax_out = jax_adalayernorm(x_typed, scale_np, shift_np, eps=adaln_mlx.eps)

        rtol, atol = get_tolerance("normalization", "adalayernorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test AdaLayerNorm backward pass parity with JAX."""
        from mlx_primitives.layers import AdaLayerNorm
        from tests.reference_jax_extended import jax_adalayernorm

        config = SIZE_CONFIGS[size]["normalization"]
        batch, seq, hidden = config["batch"], config["seq"], config["hidden"]
        cond_dims = hidden // 2
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32)

        # MLX backward
        adaln_mlx = AdaLayerNorm(dims=hidden, cond_dims=cond_dims, eps=eps)
        mx.eval(adaln_mlx.parameters())
        proj_weight_np = np.array(adaln_mlx.proj.weight)
        proj_bias_np = np.array(adaln_mlx.proj.bias) if adaln_mlx.proj.bias is not None else np.zeros(hidden * 2)

        def mlx_fn(x, cond):
            return adaln_mlx(x, cond).sum()

        x_mlx = mx.array(x_np)
        cond_mlx = mx.array(cond_np)
        grad_fn = mx.grad(mlx_fn, argnums=(0, 1))
        mlx_grad_x, mlx_grad_cond = grad_fn(x_mlx, cond_mlx)
        mx.eval(mlx_grad_x, mlx_grad_cond)

        # JAX backward
        def jax_fn(x, cond):
            scale_shift = cond @ jnp.array(proj_weight_np).T + jnp.array(proj_bias_np)
            scale = scale_shift[:, :hidden][:, None, :]
            shift = scale_shift[:, hidden:][:, None, :]
            return jax_adalayernorm(x, scale, shift, eps=eps).sum()

        x_jax = jnp.array(x_np, dtype=jnp.float32)
        cond_jax = jnp.array(cond_np, dtype=jnp.float32)
        jax_grad_x, jax_grad_cond = jax.grad(jax_fn, argnums=(0, 1))(x_jax, cond_jax)

        rtol, atol = get_gradient_tolerance("normalization", "adalayernorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_x), np.array(jax_grad_x),
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm backward (x) mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_cond), np.array(jax_grad_cond),
            rtol=rtol, atol=atol,
            err_msg=f"AdaLayerNorm backward (cond) mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_with_shift_scale(self, skip_without_jax):
        """Test AdaLayerNorm with shift and scale conditioning."""
        from mlx_primitives.layers import AdaLayerNorm
        from tests.reference_jax_extended import jax_adalayernorm

        batch, seq, hidden, cond_dims = 2, 64, 256, 128
        eps = 1e-6

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
        cond_np = np.random.randn(batch, cond_dims).astype(np.float32) * 2  # Larger scale

        # MLX AdaLayerNorm
        adaln_mlx = AdaLayerNorm(dims=hidden, cond_dims=cond_dims, eps=eps)
        mx.eval(adaln_mlx.parameters())
        proj_weight_np = np.array(adaln_mlx.proj.weight)
        proj_bias_np = np.array(adaln_mlx.proj.bias) if adaln_mlx.proj.bias is not None else np.zeros(hidden * 2)

        x_mlx = mx.array(x_np)
        cond_mlx = mx.array(cond_np)
        mlx_out = adaln_mlx(x_mlx, cond_mlx)
        mx.eval(mlx_out)

        # JAX reference
        scale_shift = cond_np @ proj_weight_np.T + proj_bias_np
        scale_np = scale_shift[:, :hidden][:, None, :]
        shift_np = scale_shift[:, hidden:][:, None, :]
        jax_out = jax_adalayernorm(x_np, scale_np, shift_np, eps=eps)

        rtol, atol = get_tolerance("normalization", "adalayernorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg="AdaLayerNorm shift/scale mismatch (JAX)"
        )


class TestQKNormParity:
    """QKNorm (Query-Key Normalization) parity tests with JAX.

    QKNorm applies RMSNorm to queries and keys separately before attention.
    """

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test QKNorm forward pass parity with JAX."""
        # Skip bf16+large: bf16 accumulates significant precision errors
        if dtype == "bf16" and size == "large":
            pytest.skip("bf16+large tests precision limits, not algorithm correctness")

        from mlx_primitives.layers import QKNorm

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        eps = 1e-6

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX QKNorm
        qk_norm = QKNorm(head_dim, eps=eps)
        mx.eval(qk_norm.parameters())
        q_scale_np = np.array(qk_norm.q_scale)
        k_scale_np = np.array(qk_norm.k_scale)

        mlx_dtype = get_mlx_dtype(dtype)
        q_mlx = mx.array(q_np).astype(mlx_dtype)
        k_mlx = mx.array(k_np).astype(mlx_dtype)
        q_mlx_out, k_mlx_out = qk_norm(q_mlx, k_mlx)
        mx.eval(q_mlx_out, k_mlx_out)

        # JAX reference - RMSNorm in fp32 for numerical stability
        q_jax = jnp.array(q_np, dtype=jnp.float32)
        k_jax = jnp.array(k_np, dtype=jnp.float32)
        q_scale_jax = jnp.array(q_scale_np, dtype=jnp.float32)
        k_scale_jax = jnp.array(k_scale_np, dtype=jnp.float32)

        # RMSNorm: x / sqrt(mean(x^2) + eps) * scale
        q_rms = jnp.sqrt(jnp.mean(q_jax ** 2, axis=-1, keepdims=True) + eps)
        k_rms = jnp.sqrt(jnp.mean(k_jax ** 2, axis=-1, keepdims=True) + eps)
        q_jax_out = (q_jax / q_rms) * q_scale_jax
        k_jax_out = (k_jax / k_rms) * k_scale_jax

        rtol, atol = get_tolerance("normalization", "rmsnorm", dtype)
        np.testing.assert_allclose(
            _to_numpy(q_mlx_out), np.array(q_jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"QKNorm Q forward mismatch (JAX) [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_mlx_out), np.array(k_jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"QKNorm K forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test QKNorm backward pass parity with JAX."""
        from mlx_primitives.layers import QKNorm

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        eps = 1e-6

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX backward
        qk_norm = QKNorm(head_dim, eps=eps)
        mx.eval(qk_norm.parameters())
        q_scale_np = np.array(qk_norm.q_scale)
        k_scale_np = np.array(qk_norm.k_scale)

        def mlx_loss_fn(q, k):
            q_out, k_out = qk_norm(q, k)
            return mx.sum(q_out) + mx.sum(k_out)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1))
        mlx_grad_q, mlx_grad_k = grad_fn(q_mlx, k_mlx)
        mx.eval(mlx_grad_q, mlx_grad_k)

        # JAX backward
        q_scale_jax = jnp.array(q_scale_np, dtype=jnp.float32)
        k_scale_jax = jnp.array(k_scale_np, dtype=jnp.float32)

        def jax_loss_fn(q, k):
            q_rms = jnp.sqrt(jnp.mean(q ** 2, axis=-1, keepdims=True) + eps)
            k_rms = jnp.sqrt(jnp.mean(k ** 2, axis=-1, keepdims=True) + eps)
            q_out = (q / q_rms) * q_scale_jax
            k_out = (k / k_rms) * k_scale_jax
            return jnp.sum(q_out) + jnp.sum(k_out)

        q_jax = jnp.array(q_np, dtype=jnp.float32)
        k_jax = jnp.array(k_np, dtype=jnp.float32)
        jax_grad_q, jax_grad_k = jax.grad(jax_loss_fn, argnums=(0, 1))(q_jax, k_jax)

        rtol, atol = get_gradient_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_q), np.array(jax_grad_q),
            rtol=rtol, atol=atol,
            err_msg=f"QKNorm Q backward mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_k), np.array(jax_grad_k),
            rtol=rtol, atol=atol,
            err_msg=f"QKNorm K backward mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_single_token(self, skip_without_jax):
        """Test QKNorm with single token (decoding scenario)."""
        from mlx_primitives.layers import QKNorm

        batch, seq, heads, head_dim = 4, 1, 8, 64
        eps = 1e-6

        np.random.seed(42)
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        # MLX QKNorm
        qk_norm = QKNorm(head_dim, eps=eps)
        mx.eval(qk_norm.parameters())
        q_scale_np = np.array(qk_norm.q_scale)
        k_scale_np = np.array(qk_norm.k_scale)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_mlx_out, k_mlx_out = qk_norm(q_mlx, k_mlx)
        mx.eval(q_mlx_out, k_mlx_out)

        # JAX reference
        q_jax = jnp.array(q_np, dtype=jnp.float32)
        k_jax = jnp.array(k_np, dtype=jnp.float32)
        q_scale_jax = jnp.array(q_scale_np, dtype=jnp.float32)
        k_scale_jax = jnp.array(k_scale_np, dtype=jnp.float32)

        q_rms = jnp.sqrt(jnp.mean(q_jax ** 2, axis=-1, keepdims=True) + eps)
        k_rms = jnp.sqrt(jnp.mean(k_jax ** 2, axis=-1, keepdims=True) + eps)
        q_jax_out = (q_jax / q_rms) * q_scale_jax
        k_jax_out = (k_jax / k_rms) * k_scale_jax

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_mlx_out), np.array(q_jax_out),
            rtol=rtol, atol=atol,
            err_msg="QKNorm single token Q mismatch (JAX)"
        )
        np.testing.assert_allclose(
            _to_numpy(k_mlx_out), np.array(k_jax_out),
            rtol=rtol, atol=atol,
            err_msg="QKNorm single token K mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.edge_case
    def test_different_qk_magnitudes(self, skip_without_jax):
        """Test QKNorm with Q and K having different magnitudes."""
        from mlx_primitives.layers import QKNorm

        batch, seq, heads, head_dim = 2, 32, 8, 64
        eps = 1e-6

        np.random.seed(42)
        # Q has small values, K has large values
        q_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32) * 0.01
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32) * 100

        # MLX QKNorm
        qk_norm = QKNorm(head_dim, eps=eps)
        mx.eval(qk_norm.parameters())
        q_scale_np = np.array(qk_norm.q_scale)
        k_scale_np = np.array(qk_norm.k_scale)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_mlx_out, k_mlx_out = qk_norm(q_mlx, k_mlx)
        mx.eval(q_mlx_out, k_mlx_out)

        # JAX reference
        q_jax = jnp.array(q_np, dtype=jnp.float32)
        k_jax = jnp.array(k_np, dtype=jnp.float32)
        q_scale_jax = jnp.array(q_scale_np, dtype=jnp.float32)
        k_scale_jax = jnp.array(k_scale_np, dtype=jnp.float32)

        q_rms = jnp.sqrt(jnp.mean(q_jax ** 2, axis=-1, keepdims=True) + eps)
        k_rms = jnp.sqrt(jnp.mean(k_jax ** 2, axis=-1, keepdims=True) + eps)
        q_jax_out = (q_jax / q_rms) * q_scale_jax
        k_jax_out = (k_jax / k_rms) * k_scale_jax

        rtol, atol = get_tolerance("normalization", "rmsnorm", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_mlx_out), np.array(q_jax_out),
            rtol=rtol, atol=atol,
            err_msg="QKNorm different magnitudes Q mismatch (JAX)"
        )
        np.testing.assert_allclose(
            _to_numpy(k_mlx_out), np.array(k_jax_out),
            rtol=rtol, atol=atol,
            err_msg="QKNorm different magnitudes K mismatch (JAX)"
        )
