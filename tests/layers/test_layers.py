"""Tests for layer modules."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
import numpy as np

from mlx_primitives.layers import (
    # Normalization
    RMSNorm,
    GroupNorm,
    InstanceNorm,
    AdaLayerNorm,
    QKNorm,
    rms_norm,
    group_norm,
    # Activations
    SwiGLU,
    GeGLU,
    ReGLU,
    FusedSwiGLU,
    Mish,
    mish,
    GELUTanh,
    gelu_tanh,
    SquaredReLU,
    squared_relu,
    QuickGELU,
    quick_gelu,
    Swish,
    swish,
    HardSwish,
    hard_swish,
    HardSigmoid,
    hard_sigmoid,
    get_activation,
    # Pooling
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    GlobalAttentionPooling,
    GeM,
    SpatialPyramidPooling,
    AvgPool1d,
    MaxPool1d,
)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_basic(self):
        """Test basic RMSNorm forward pass."""
        norm = RMSNorm(dims=64)
        x = mx.random.normal((2, 16, 64))
        y = norm(x)
        assert y.shape == x.shape

    def test_output_scale(self):
        """Test that RMSNorm normalizes correctly."""
        norm = RMSNorm(dims=64, eps=1e-6)
        x = mx.random.normal((2, 16, 64)) * 10  # Large values

        y = norm(x)
        mx.eval(y)

        # Check that RMS is approximately 1 after normalization
        # (before weight multiplication)
        rms = mx.sqrt(mx.mean(y * y, axis=-1))
        mx.eval(rms)
        # Weight is 1, so RMS should be close to 1
        np.testing.assert_allclose(np.array(rms), 1.0, rtol=0.1)

    def test_gradient(self):
        """Test gradient flow through RMSNorm."""
        norm = RMSNorm(dims=32)

        def forward(x):
            return mx.sum(norm(x))

        x = mx.random.normal((1, 8, 32))
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape

    def test_functional(self):
        """Test functional rms_norm."""
        weight = mx.ones((64,))
        x = mx.random.normal((2, 16, 64))
        y = rms_norm(x, weight, eps=1e-6)
        assert y.shape == x.shape


class TestGroupNorm:
    """Tests for GroupNorm."""

    def test_basic(self):
        """Test basic GroupNorm forward pass."""
        norm = GroupNorm(num_groups=4, num_channels=64)
        x = mx.random.normal((2, 64, 16, 16))  # NCHW
        y = norm(x)
        assert y.shape == x.shape

    def test_groups_divisibility(self):
        """Test that channels must be divisible by groups."""
        with pytest.raises(ValueError):
            GroupNorm(num_groups=5, num_channels=64)  # 64 not divisible by 5

    def test_1d_input(self):
        """Test GroupNorm with 1D spatial input."""
        norm = GroupNorm(num_groups=4, num_channels=64)
        x = mx.random.normal((2, 64, 100))  # (N, C, L)
        y = norm(x)
        assert y.shape == x.shape

    def test_without_affine(self):
        """Test GroupNorm without learnable parameters."""
        norm = GroupNorm(num_groups=4, num_channels=64, affine=False)
        x = mx.random.normal((2, 64, 16, 16))
        y = norm(x)
        assert y.shape == x.shape

    def test_functional(self):
        """Test functional group_norm."""
        x = mx.random.normal((2, 64, 16, 16))
        weight = mx.ones((64,))
        bias = mx.zeros((64,))
        y = group_norm(x, num_groups=4, weight=weight, bias=bias)
        assert y.shape == x.shape


class TestInstanceNorm:
    """Tests for InstanceNorm."""

    def test_basic(self):
        """Test basic InstanceNorm forward pass."""
        norm = InstanceNorm(num_features=64)
        x = mx.random.normal((2, 64, 16, 16))
        y = norm(x)
        assert y.shape == x.shape

    def test_without_affine(self):
        """Test InstanceNorm without learnable parameters."""
        norm = InstanceNorm(num_features=64, affine=False)
        x = mx.random.normal((2, 64, 16, 16))
        y = norm(x)
        assert y.shape == x.shape


class TestAdaLayerNorm:
    """Tests for Adaptive Layer Normalization."""

    def test_basic(self):
        """Test basic AdaLayerNorm forward pass."""
        norm = AdaLayerNorm(dims=64, cond_dims=128)
        x = mx.random.normal((2, 16, 64))
        cond = mx.random.normal((2, 128))
        y = norm(x, cond)
        assert y.shape == x.shape

    def test_2d_input(self):
        """Test AdaLayerNorm with 2D input."""
        norm = AdaLayerNorm(dims=64, cond_dims=128)
        x = mx.random.normal((2, 64))
        cond = mx.random.normal((2, 128))
        y = norm(x, cond)
        assert y.shape == x.shape


class TestQKNorm:
    """Tests for Query-Key Normalization."""

    def test_basic(self):
        """Test basic QKNorm forward pass."""
        qk_norm = QKNorm(head_dim=64)
        q = mx.random.normal((2, 16, 8, 64))
        k = mx.random.normal((2, 16, 8, 64))
        q_norm, k_norm = qk_norm(q, k)
        assert q_norm.shape == q.shape
        assert k_norm.shape == k.shape


class TestSwiGLU:
    """Tests for SwiGLU and GLU variants."""

    def test_swiglu_basic(self):
        """Test basic SwiGLU forward pass."""
        swiglu = SwiGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))
        y = swiglu(x)
        assert y.shape == x.shape

    def test_swiglu_custom_out(self):
        """Test SwiGLU with custom output dimension."""
        swiglu = SwiGLU(in_features=64, hidden_features=128, out_features=32)
        x = mx.random.normal((2, 16, 64))
        y = swiglu(x)
        assert y.shape == (2, 16, 32)

    def test_fused_swiglu(self):
        """Test FusedSwiGLU."""
        swiglu = FusedSwiGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))
        y = swiglu(x)
        assert y.shape == x.shape

    def test_geglu(self):
        """Test GeGLU."""
        geglu = GeGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))
        y = geglu(x)
        assert y.shape == x.shape

    def test_reglu(self):
        """Test ReGLU."""
        reglu = ReGLU(in_features=64, hidden_features=128)
        x = mx.random.normal((2, 16, 64))
        y = reglu(x)
        assert y.shape == x.shape

    def test_swiglu_gradient(self):
        """Test gradient flow through SwiGLU."""
        swiglu = SwiGLU(in_features=32, hidden_features=64)

        def forward(x):
            return mx.sum(swiglu(x))

        x = mx.random.normal((1, 8, 32))
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape


class TestActivationFunctions:
    """Tests for individual activation functions."""

    def test_mish(self):
        """Test Mish activation."""
        x = mx.random.normal((2, 16, 64))
        y = mish(x)
        assert y.shape == x.shape

        # Module version
        mish_mod = Mish()
        y2 = mish_mod(x)
        mx.eval(y, y2)
        np.testing.assert_allclose(np.array(y), np.array(y2))

    def test_gelu_tanh(self):
        """Test GELU tanh approximation."""
        x = mx.random.normal((2, 16, 64))
        y = gelu_tanh(x)
        assert y.shape == x.shape

        # Should be close to exact GELU
        y_exact = nn.gelu(x)
        mx.eval(y, y_exact)
        np.testing.assert_allclose(
            np.array(y), np.array(y_exact), rtol=0.01, atol=0.01
        )

    def test_squared_relu(self):
        """Test Squared ReLU."""
        x = mx.array([[-1.0, 0.0, 1.0, 2.0]])
        y = squared_relu(x)
        mx.eval(y)
        expected = mx.array([[0.0, 0.0, 1.0, 4.0]])
        np.testing.assert_allclose(np.array(y), np.array(expected))

    def test_quick_gelu(self):
        """Test Quick GELU."""
        x = mx.random.normal((2, 16, 64))
        y = quick_gelu(x)
        assert y.shape == x.shape

    def test_swish(self):
        """Test Swish with different beta."""
        x = mx.random.normal((2, 16, 64))

        # beta=1 should be same as SiLU
        y1 = swish(x, beta=1.0)
        y_silu = nn.silu(x)
        mx.eval(y1, y_silu)
        np.testing.assert_allclose(np.array(y1), np.array(y_silu))

    def test_hard_swish(self):
        """Test Hard Swish."""
        x = mx.random.normal((2, 16, 64))
        y = hard_swish(x)
        assert y.shape == x.shape

    def test_hard_sigmoid(self):
        """Test Hard Sigmoid."""
        x = mx.array([[-10.0, -3.0, 0.0, 3.0, 10.0]])
        y = hard_sigmoid(x)
        mx.eval(y)
        # Should be clipped to [0, 1]
        assert float(mx.min(y)) >= 0.0
        assert float(mx.max(y)) <= 1.0

    def test_get_activation(self):
        """Test activation registry."""
        relu_fn = get_activation("relu")
        gelu_fn = get_activation("gelu")

        x = mx.random.normal((2, 16))
        y_relu = relu_fn(x)
        y_gelu = gelu_fn(x)

        assert y_relu.shape == x.shape
        assert y_gelu.shape == x.shape

    def test_get_activation_unknown(self):
        """Test that unknown activation raises error."""
        with pytest.raises(ValueError):
            get_activation("unknown_activation")


class TestAdaptivePooling:
    """Tests for adaptive pooling layers."""

    def test_adaptive_avg_pool1d(self):
        """Test AdaptiveAvgPool1d."""
        pool = AdaptiveAvgPool1d(output_size=8)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        assert y.shape == (2, 64, 8)

    def test_adaptive_avg_pool1d_same_size(self):
        """Test AdaptiveAvgPool1d when output equals input."""
        pool = AdaptiveAvgPool1d(output_size=100)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        mx.eval(x, y)
        np.testing.assert_allclose(np.array(y), np.array(x))

    def test_adaptive_avg_pool2d(self):
        """Test AdaptiveAvgPool2d."""
        pool = AdaptiveAvgPool2d(output_size=(7, 7))
        x = mx.random.normal((2, 256, 224, 224))
        y = pool(x)
        assert y.shape == (2, 256, 7, 7)

    def test_adaptive_avg_pool2d_global(self):
        """Test AdaptiveAvgPool2d for global pooling."""
        pool = AdaptiveAvgPool2d(output_size=1)
        x = mx.random.normal((2, 256, 14, 14))
        y = pool(x)
        assert y.shape == (2, 256, 1, 1)

        # Should equal mean over spatial dims
        y_mean = mx.mean(x, axis=(2, 3), keepdims=True)
        mx.eval(y, y_mean)
        np.testing.assert_allclose(np.array(y), np.array(y_mean))

    def test_adaptive_max_pool1d(self):
        """Test AdaptiveMaxPool1d."""
        pool = AdaptiveMaxPool1d(output_size=8)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        assert y.shape == (2, 64, 8)

    def test_adaptive_max_pool2d(self):
        """Test AdaptiveMaxPool2d."""
        pool = AdaptiveMaxPool2d(output_size=(7, 7))
        x = mx.random.normal((2, 256, 14, 14))
        y = pool(x)
        assert y.shape == (2, 256, 7, 7)


class TestGlobalAttentionPooling:
    """Tests for GlobalAttentionPooling."""

    def test_basic(self):
        """Test basic forward pass."""
        pool = GlobalAttentionPooling(dims=64)
        x = mx.random.normal((2, 100, 64))
        y = pool(x)
        assert y.shape == (2, 64)

    def test_with_mask(self):
        """Test with attention mask."""
        pool = GlobalAttentionPooling(dims=64)
        x = mx.random.normal((2, 100, 64))
        # Create mask: True for first 50, False for rest
        mask = mx.concatenate([
            mx.ones((2, 50), dtype=mx.bool_),
            mx.zeros((2, 50), dtype=mx.bool_)
        ], axis=1)

        y = pool(x, mask=mask)
        assert y.shape == (2, 64)

    def test_gradient(self):
        """Test gradient flow."""
        pool = GlobalAttentionPooling(dims=32)

        def forward(x):
            return mx.sum(pool(x))

        x = mx.random.normal((1, 16, 32))
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape


class TestGeM:
    """Tests for Generalized Mean Pooling."""

    def test_basic(self):
        """Test basic GeM forward pass."""
        pool = GeM(p=3.0)
        x = mx.random.normal((2, 256, 7, 7))
        x = mx.abs(x) + 0.1  # GeM needs positive values
        y = pool(x)
        assert y.shape == (2, 256, 1, 1)

    def test_learnable_p(self):
        """Test GeM with learnable p parameter."""
        pool = GeM(p=3.0, learnable=True)
        assert hasattr(pool, "p")

    def test_fixed_p(self):
        """Test GeM with fixed p parameter."""
        pool = GeM(p=3.0, learnable=False)
        x = mx.abs(mx.random.normal((2, 256, 7, 7))) + 0.1
        y = pool(x)
        assert y.shape == (2, 256, 1, 1)


class TestSpatialPyramidPooling:
    """Tests for Spatial Pyramid Pooling."""

    def test_basic(self):
        """Test basic SPP forward pass."""
        spp = SpatialPyramidPooling(output_sizes=[1, 2, 4])
        x = mx.random.normal((2, 256, 13, 13))
        y = spp(x)
        # Output: 256 * (1 + 4 + 16) = 256 * 21 = 5376
        assert y.shape == (2, 256 * (1 + 4 + 16))


class TestAvgMaxPool1d:
    """Tests for 1D pooling layers."""

    def test_avg_pool1d(self):
        """Test AvgPool1d."""
        pool = AvgPool1d(kernel_size=3, stride=2)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        expected_len = (100 - 3) // 2 + 1
        assert y.shape == (2, 64, expected_len)

    def test_max_pool1d(self):
        """Test MaxPool1d."""
        pool = MaxPool1d(kernel_size=3, stride=2)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        expected_len = (100 - 3) // 2 + 1
        assert y.shape == (2, 64, expected_len)

    def test_avg_pool1d_with_padding(self):
        """Test AvgPool1d with padding."""
        pool = AvgPool1d(kernel_size=3, stride=1, padding=1)
        x = mx.random.normal((2, 64, 100))
        y = pool(x)
        assert y.shape == (2, 64, 100)


class TestLayerGradients:
    """Tests for gradient computation through layers."""

    def test_rmsnorm_gradient(self):
        """Test RMSNorm gradients."""
        norm = RMSNorm(dims=64)

        def forward(x):
            return mx.sum(norm(x) ** 2)

        x = mx.random.normal((2, 16, 64))
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape
        assert not mx.any(mx.isnan(grad))

    def test_groupnorm_gradient(self):
        """Test GroupNorm gradients."""
        norm = GroupNorm(num_groups=4, num_channels=64)

        def forward(x):
            return mx.sum(norm(x) ** 2)

        x = mx.random.normal((2, 64, 8, 8))
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape

    def test_gem_gradient(self):
        """Test GeM gradients."""
        pool = GeM(p=3.0)

        def forward(x):
            return mx.sum(pool(x))

        x = mx.abs(mx.random.normal((2, 64, 7, 7))) + 0.1
        loss, grad = mx.value_and_grad(forward)(x)
        assert grad.shape == x.shape


try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestLayerBenchmarks:
    """Benchmark tests for layers."""

    def test_rmsnorm_benchmark(self, benchmark):
        """Benchmark RMSNorm."""
        norm = RMSNorm(dims=768)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            y = norm(x)
            mx.eval(y)
            return y

        benchmark(run)

    def test_swiglu_benchmark(self, benchmark):
        """Benchmark SwiGLU."""
        swiglu = SwiGLU(in_features=768, hidden_features=2048)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            y = swiglu(x)
            mx.eval(y)
            return y

        benchmark(run)

    def test_fused_swiglu_benchmark(self, benchmark):
        """Benchmark FusedSwiGLU."""
        swiglu = FusedSwiGLU(in_features=768, hidden_features=2048)
        x = mx.random.normal((4, 512, 768))
        mx.eval(x)

        def run():
            y = swiglu(x)
            mx.eval(y)
            return y

        benchmark(run)

    def test_groupnorm_benchmark(self, benchmark):
        """Benchmark GroupNorm."""
        norm = GroupNorm(num_groups=32, num_channels=256)
        x = mx.random.normal((4, 256, 32, 32))
        mx.eval(x)

        def run():
            y = norm(x)
            mx.eval(y)
            return y

        benchmark(run)
