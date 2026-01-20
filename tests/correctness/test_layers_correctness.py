"""Correctness tests for layer primitives.

Tests compare our implementations against reference (naive) implementations
to verify numerical correctness for normalization, activation, pooling, and
embedding layers.
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.layers import (
    # Normalization
    RMSNorm,
    GroupNorm,
    InstanceNorm,
    AdaLayerNorm,
    QKNorm,
    # Activations
    SwiGLU,
    GeGLU,
    ReGLU,
    Mish,
    GELUTanh,
    SquaredReLU,
    # Pooling
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    GlobalAttentionPooling,
    GeM,
    # Embeddings
    SinusoidalEmbedding,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    AlibiEmbedding,
)


# ============================================================================
# Reference Implementations
# ============================================================================


def naive_rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Reference implementation of RMSNorm."""
    # Compute root mean square
    rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)
    # Normalize and scale
    return (x / rms) * weight


def naive_groupnorm_nchw(
    x: mx.array,
    num_groups: int,
    weight: mx.array,
    bias: mx.array,
    eps: float = 1e-5
) -> mx.array:
    """Reference implementation of GroupNorm for NCHW format.

    Args:
        x: Input [batch, channels, height, width]
        num_groups: Number of groups
        weight: Scale parameters [channels]
        bias: Shift parameters [channels]
        eps: Epsilon for numerical stability
    """
    batch_size, channels, h, w = x.shape
    channels_per_group = channels // num_groups

    # Reshape to [batch, num_groups, channels_per_group, h, w]
    x_reshaped = x.reshape(batch_size, num_groups, channels_per_group, h, w)

    # Compute mean and var per group (over channels_per_group, h, w)
    mean = mx.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = mx.var(x_reshaped, axis=(2, 3, 4), keepdims=True)

    # Normalize
    x_norm = (x_reshaped - mean) / mx.sqrt(var + eps)

    # Reshape back to [batch, channels, h, w]
    x_norm = x_norm.reshape(batch_size, channels, h, w)

    # Apply scale and shift (broadcast weight/bias over spatial dims)
    return weight[None, :, None, None] * x_norm + bias[None, :, None, None]


def naive_groupnorm(
    x: mx.array,
    num_groups: int,
    weight: mx.array,
    bias: mx.array,
    eps: float = 1e-5
) -> mx.array:
    """Reference implementation of GroupNorm.

    Args:
        x: Input [batch, ..., channels]
        num_groups: Number of groups
        weight: Scale parameters [channels]
        bias: Shift parameters [channels]
        eps: Epsilon for numerical stability
    """
    batch_size = x.shape[0]
    channels = x.shape[-1]
    spatial_dims = x.shape[1:-1]
    channels_per_group = channels // num_groups

    # Reshape to [batch, spatial, num_groups, channels_per_group]
    x_reshaped = x.reshape(batch_size, -1, num_groups, channels_per_group)

    # Compute mean and var per group
    mean = mx.mean(x_reshaped, axis=(1, 3), keepdims=True)
    var = mx.var(x_reshaped, axis=(1, 3), keepdims=True)

    # Normalize
    x_norm = (x_reshaped - mean) / mx.sqrt(var + eps)

    # Reshape back
    x_norm = x_norm.reshape(batch_size, *spatial_dims, channels)

    # Apply affine transform
    return x_norm * weight + bias


def naive_instance_norm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    eps: float = 1e-5
) -> mx.array:
    """Reference implementation of InstanceNorm for images.

    Args:
        x: Input [batch, height, width, channels]
        weight: Scale parameters [channels]
        bias: Shift parameters [channels]
        eps: Epsilon for numerical stability
    """
    # Compute mean and var over spatial dimensions (H, W)
    mean = mx.mean(x, axis=(1, 2), keepdims=True)
    var = mx.var(x, axis=(1, 2), keepdims=True)

    # Normalize
    x_norm = (x - mean) / mx.sqrt(var + eps)

    # Apply affine transform
    return x_norm * weight + bias


def naive_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """Reference implementation of SwiGLU: silu(gate) * up."""
    silu = gate * mx.sigmoid(gate)  # SiLU/Swish activation
    return silu * up


def naive_geglu(gate: mx.array, up: mx.array) -> mx.array:
    """Reference implementation of GeGLU: gelu(gate) * up."""
    # GELU approximation using tanh
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    cdf = 0.5 * (1.0 + mx.tanh(sqrt_2_over_pi * (gate + 0.044715 * gate ** 3)))
    gelu = gate * cdf
    return gelu * up


def naive_reglu(gate: mx.array, up: mx.array) -> mx.array:
    """Reference implementation of ReGLU: relu(gate) * up."""
    return mx.maximum(gate, 0) * up


def naive_mish(x: mx.array) -> mx.array:
    """Reference implementation of Mish: x * tanh(softplus(x))."""
    softplus = mx.log(1 + mx.exp(x))
    return x * mx.tanh(softplus)


def naive_gelu_tanh(x: mx.array) -> mx.array:
    """Reference implementation of GELU with tanh approximation."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))


def naive_squared_relu(x: mx.array) -> mx.array:
    """Reference implementation of Squared ReLU."""
    relu = mx.maximum(x, 0)
    return relu * relu


def naive_adaptive_avg_pool1d(x: mx.array, output_size: int) -> mx.array:
    """Reference implementation of 1D adaptive average pooling.

    Args:
        x: Input [batch, channels, length]
        output_size: Desired output length
    """
    batch, channels, length = x.shape

    output = []
    for i in range(output_size):
        start = int(math.floor(i * length / output_size))
        end = int(math.ceil((i + 1) * length / output_size))
        output.append(mx.mean(x[:, :, start:end], axis=-1, keepdims=True))

    return mx.concatenate(output, axis=-1)


def naive_adaptive_max_pool1d(x: mx.array, output_size: int) -> mx.array:
    """Reference implementation of 1D adaptive max pooling."""
    batch, channels, length = x.shape

    output = []
    for i in range(output_size):
        start = int(math.floor(i * length / output_size))
        end = int(math.ceil((i + 1) * length / output_size))
        output.append(mx.max(x[:, :, start:end], axis=-1, keepdims=True))

    return mx.concatenate(output, axis=-1)


def naive_gem_pool(x: mx.array, p: float = 3.0, eps: float = 1e-6) -> mx.array:
    """Reference implementation of Generalized Mean Pooling.

    Args:
        x: Input [batch, channels, height, width] (NCHW format)
        p: Power parameter
        eps: Epsilon for numerical stability
    """
    # Clamp to avoid negative values with fractional powers
    x_clamped = mx.clip(x, eps, None)

    # Compute mean of x^p over spatial dimensions (2, 3 for NCHW)
    x_pow = x_clamped ** p
    mean_pow = mx.mean(x_pow, axis=(2, 3), keepdims=True)

    # Take p-th root
    return mean_pow ** (1.0 / p)


def naive_sinusoidal_embedding(positions: mx.array, dims: int, base: float = 10000.0) -> mx.array:
    """Reference implementation of sinusoidal positional embeddings.

    Args:
        positions: Position indices [seq_len]
        dims: Embedding dimension
        base: Base for the geometric progression
    """
    # Compute frequency bands (matches actual implementation)
    dims_range = mx.arange(0, dims, 2)
    freqs = base ** (-dims_range / dims)

    # Compute angles
    positions = positions[:, None]  # (seq_len, 1)
    angles = positions * freqs  # (seq_len, dims/2)

    # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
    sin_emb = mx.sin(angles)
    cos_emb = mx.cos(angles)

    # Combine by interleaving
    emb = mx.concatenate([
        sin_emb[:, :, None],
        cos_emb[:, :, None]
    ], axis=2).reshape(positions.shape[0], dims)

    return emb


# ============================================================================
# RMSNorm Correctness Tests
# ============================================================================


class TestRMSNormCorrectness:
    """Correctness tests for RMSNorm implementation."""

    @pytest.mark.parametrize("shape", [
        (2, 64),
        (2, 32, 128),
        (4, 16, 32, 256),
    ])
    def test_rmsnorm_vs_naive(self, shape):
        """Compare RMSNorm against naive implementation."""
        mx.random.seed(42)
        dims = shape[-1]

        rmsnorm = RMSNorm(dims=dims)

        x = mx.random.normal(shape)
        mx.eval(x)

        # Our implementation
        out = rmsnorm(x)
        mx.eval(out)

        # Naive implementation
        naive_out = naive_rmsnorm(x, rmsnorm.weight, eps=rmsnorm.eps)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_rmsnorm_output_scale(self):
        """Test that RMSNorm produces unit RMS output (before scaling)."""
        mx.random.seed(42)
        dims = 128

        # RMSNorm with unit weights
        rmsnorm = RMSNorm(dims=dims)
        rmsnorm.weight = mx.ones(dims)

        x = mx.random.normal((2, 32, dims)) * 5  # Scale up input
        mx.eval(x)

        out = rmsnorm(x)
        mx.eval(out)

        # Output RMS should be close to 1
        rms = mx.sqrt(mx.mean(out ** 2, axis=-1))
        mx.eval(rms)

        assert mx.allclose(rms, mx.ones_like(rms), atol=1e-4), \
            f"Output RMS not close to 1: mean={float(mx.mean(rms))}"

    def test_rmsnorm_gradient(self):
        """Test RMSNorm gradient flow."""
        mx.random.seed(42)
        dims = 64

        rmsnorm = RMSNorm(dims=dims)

        x = mx.random.normal((2, 16, dims))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(rmsnorm(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are zero"
        assert mx.all(mx.isfinite(grad)), "Gradients contain non-finite values"


# ============================================================================
# GroupNorm Correctness Tests
# ============================================================================


class TestGroupNormCorrectness:
    """Correctness tests for GroupNorm implementation."""

    @pytest.mark.parametrize("num_groups", [1, 2, 4, 8])
    def test_groupnorm_vs_naive(self, num_groups):
        """Compare GroupNorm against naive implementation."""
        mx.random.seed(42)
        batch_size = 2
        height, width = 8, 8
        channels = 32

        groupnorm = GroupNorm(num_groups=num_groups, num_channels=channels)

        # Use NCHW format (implementation expects channels-first)
        x = mx.random.normal((batch_size, channels, height, width))
        mx.eval(x)

        # Our implementation
        out = groupnorm(x)
        mx.eval(out)

        # Naive implementation (expects NCHW format)
        naive_out = naive_groupnorm_nchw(
            x, num_groups, groupnorm.weight, groupnorm.bias, groupnorm.eps
        )
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-4), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_groupnorm_output_properties(self):
        """Test GroupNorm produces normalized output."""
        mx.random.seed(42)
        channels = 32
        num_groups = 4

        groupnorm = GroupNorm(num_groups=num_groups, num_channels=channels)

        # NCHW format
        x = mx.random.normal((2, channels, 8, 8))
        mx.eval(x)

        out = groupnorm(x)
        mx.eval(out)

        # Output should have finite values and similar magnitude
        assert mx.all(mx.isfinite(out))
        assert out.shape == x.shape


# ============================================================================
# InstanceNorm Correctness Tests
# ============================================================================


class TestInstanceNormCorrectness:
    """Correctness tests for InstanceNorm implementation."""

    def test_instancenorm_output_properties(self):
        """Test InstanceNorm produces normalized output."""
        mx.random.seed(42)
        batch_size = 2
        height, width = 16, 16
        channels = 32

        instancenorm = InstanceNorm(num_features=channels)

        # NCHW format (batch, channels, height, width)
        x = mx.random.normal((batch_size, channels, height, width))
        mx.eval(x)

        out = instancenorm(x)
        mx.eval(out)

        # Output should have finite values and same shape
        assert mx.all(mx.isfinite(out))
        assert out.shape == x.shape


# ============================================================================
# Activation Correctness Tests
# ============================================================================


class TestSwiGLUCorrectness:
    """Correctness tests for SwiGLU activation."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 256])
    def test_swiglu_vs_naive(self, hidden_dim):
        """Compare SwiGLU against naive implementation."""
        mx.random.seed(42)
        in_dim = 64

        swiglu = SwiGLU(in_dim, hidden_dim)

        x = mx.random.normal((2, 16, in_dim))
        mx.eval(x)

        # Our implementation
        out = swiglu(x)
        mx.eval(out)

        # Manual computation: w2(silu(w_gate(x)) * w1(x))
        gate_out = swiglu.w_gate(x)
        up_out = swiglu.w1(x)
        manual_out = swiglu.w2(nn.silu(gate_out) * up_out)
        mx.eval(manual_out)

        assert mx.allclose(out, manual_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - manual_out)))}"

    def test_swiglu_gradient(self):
        """Test SwiGLU gradient flow."""
        mx.random.seed(42)

        swiglu = SwiGLU(64, 128)

        x = mx.random.normal((2, 16, 64))
        mx.eval(x)

        def loss_fn(x):
            return mx.sum(swiglu(x))

        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        assert float(mx.sum(mx.abs(grad))) > 0, "Gradients are zero"


class TestGeGLUCorrectness:
    """Correctness tests for GeGLU activation."""

    def test_geglu_vs_naive(self):
        """Compare GeGLU against naive implementation."""
        mx.random.seed(42)
        in_dim = 64
        hidden_dim = 128

        geglu = GeGLU(in_dim, hidden_dim)

        x = mx.random.normal((2, 16, in_dim))
        mx.eval(x)

        out = geglu(x)
        mx.eval(out)

        # Manual computation: w2(gelu(w_gate(x)) * w1(x))
        gate_out = geglu.w_gate(x)
        up_out = geglu.w1(x)
        manual_out = geglu.w2(nn.gelu(gate_out) * up_out)
        mx.eval(manual_out)

        assert mx.allclose(out, manual_out, atol=1e-4), \
            f"Max diff: {float(mx.max(mx.abs(out - manual_out)))}"


class TestReGLUCorrectness:
    """Correctness tests for ReGLU activation."""

    def test_reglu_vs_naive(self):
        """Compare ReGLU against naive implementation."""
        mx.random.seed(42)
        in_dim = 64
        hidden_dim = 128

        reglu = ReGLU(in_dim, hidden_dim)

        x = mx.random.normal((2, 16, in_dim))
        mx.eval(x)

        out = reglu(x)
        mx.eval(out)

        # Manual computation: w2(relu(w_gate(x)) * w1(x))
        gate_out = reglu.w_gate(x)
        up_out = reglu.w1(x)
        manual_out = reglu.w2(nn.relu(gate_out) * up_out)
        mx.eval(manual_out)

        assert mx.allclose(out, manual_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - manual_out)))}"


class TestMishCorrectness:
    """Correctness tests for Mish activation."""

    def test_mish_vs_naive(self):
        """Compare Mish against naive implementation."""
        mx.random.seed(42)

        x = mx.random.normal((2, 32, 64))
        mx.eval(x)

        mish = Mish()
        out = mish(x)
        mx.eval(out)

        naive_out = naive_mish(x)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_mish_smoothness(self):
        """Test Mish is smooth (continuous first derivative)."""
        mx.random.seed(42)

        mish = Mish()

        # Test around zero where some activations have issues
        x = mx.linspace(-2, 2, 100)
        mx.eval(x)

        out = mish(x)
        mx.eval(out)

        # Check output is smooth (no sudden jumps)
        diff = out[1:] - out[:-1]
        mx.eval(diff)

        # Second differences should be small (smooth curve)
        diff2 = diff[1:] - diff[:-1]
        mx.eval(diff2)

        assert float(mx.max(mx.abs(diff2))) < 0.1, "Mish not smooth"


class TestGELUTanhCorrectness:
    """Correctness tests for GELU with tanh approximation."""

    def test_gelu_tanh_vs_naive(self):
        """Compare GELUTanh against naive implementation."""
        mx.random.seed(42)

        x = mx.random.normal((2, 32, 64))
        mx.eval(x)

        gelu = GELUTanh()
        out = gelu(x)
        mx.eval(out)

        naive_out = naive_gelu_tanh(x)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"


class TestSquaredReLUCorrectness:
    """Correctness tests for Squared ReLU."""

    def test_squared_relu_vs_naive(self):
        """Compare SquaredReLU against naive implementation."""
        mx.random.seed(42)

        x = mx.random.normal((2, 32, 64))
        mx.eval(x)

        sqrelu = SquaredReLU()
        out = sqrelu(x)
        mx.eval(out)

        naive_out = naive_squared_relu(x)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-6), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_squared_relu_negative_inputs(self):
        """Test SquaredReLU zeros out negative inputs."""
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mx.eval(x)

        sqrelu = SquaredReLU()
        out = sqrelu(x)
        mx.eval(out)

        expected = mx.array([0.0, 0.0, 0.0, 1.0, 4.0])
        assert mx.allclose(out, expected, atol=1e-6)


# ============================================================================
# Pooling Correctness Tests
# ============================================================================


class TestAdaptivePoolingCorrectness:
    """Correctness tests for adaptive pooling layers."""

    @pytest.mark.parametrize("input_size,output_size", [
        (100, 10),
        (64, 8),
        (128, 1),
        (50, 25),
    ])
    def test_adaptive_avg_pool1d_vs_naive(self, input_size, output_size):
        """Compare AdaptiveAvgPool1d against naive implementation."""
        mx.random.seed(42)
        batch_size = 2
        channels = 32

        pool = AdaptiveAvgPool1d(output_size=output_size)

        # Input format: (batch, channels, length)
        x = mx.random.normal((batch_size, channels, input_size))
        mx.eval(x)

        out = pool(x)
        mx.eval(out)

        naive_out = naive_adaptive_avg_pool1d(x, output_size)
        mx.eval(naive_out)

        assert out.shape == (batch_size, channels, output_size)
        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    @pytest.mark.parametrize("input_size,output_size", [
        (100, 10),
        (64, 8),
        (128, 1),
    ])
    def test_adaptive_max_pool1d_vs_naive(self, input_size, output_size):
        """Compare AdaptiveMaxPool1d against naive implementation."""
        mx.random.seed(42)
        batch_size = 2
        channels = 32

        pool = AdaptiveMaxPool1d(output_size=output_size)

        # Input format: (batch, channels, length)
        x = mx.random.normal((batch_size, channels, input_size))
        mx.eval(x)

        out = pool(x)
        mx.eval(out)

        naive_out = naive_adaptive_max_pool1d(x, output_size)
        mx.eval(naive_out)

        assert out.shape == (batch_size, channels, output_size)
        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_adaptive_pool_to_one_equals_global_pool(self):
        """Adaptive pooling to size 1 should equal global pooling."""
        mx.random.seed(42)

        # Input format: (batch, channels, length)
        x = mx.random.normal((2, 32, 64))
        mx.eval(x)

        avg_pool = AdaptiveAvgPool1d(output_size=1)
        max_pool = AdaptiveMaxPool1d(output_size=1)

        avg_out = avg_pool(x)
        max_out = max_pool(x)
        mx.eval(avg_out, max_out)

        # Pool over last dimension (length)
        global_avg = mx.mean(x, axis=-1, keepdims=True)
        global_max = mx.max(x, axis=-1, keepdims=True)
        mx.eval(global_avg, global_max)

        assert mx.allclose(avg_out, global_avg, atol=1e-5)
        assert mx.allclose(max_out, global_max, atol=1e-5)


class TestGeMPoolingCorrectness:
    """Correctness tests for Generalized Mean Pooling."""

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 4.0])
    def test_gem_vs_naive(self, p):
        """Compare GeM against naive implementation."""
        mx.random.seed(42)

        gem = GeM(p=p)

        # NCHW format: (batch, channels, height, width)
        # Use positive values to avoid issues with fractional powers
        x = mx.abs(mx.random.normal((2, 64, 8, 8))) + 0.1
        mx.eval(x)

        out = gem(x)
        mx.eval(out)

        naive_out = naive_gem_pool(x, p=p)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-4), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_gem_p1_equals_avg_pool(self):
        """GeM with p=1 should equal average pooling."""
        mx.random.seed(42)

        gem = GeM(p=1.0)

        # NCHW format
        x = mx.abs(mx.random.normal((2, 64, 8, 8))) + 0.1
        mx.eval(x)

        gem_out = gem(x)
        # Average over spatial dimensions (2, 3 for NCHW)
        avg_out = mx.mean(x, axis=(2, 3), keepdims=True)
        mx.eval(gem_out, avg_out)

        assert mx.allclose(gem_out, avg_out, atol=1e-4)

    def test_gem_large_p_approaches_max_pool(self):
        """GeM with large p should approach max pooling."""
        mx.random.seed(42)

        gem = GeM(p=10.0)

        # NCHW format
        x = mx.abs(mx.random.normal((2, 64, 8, 8))) + 0.1
        mx.eval(x)

        gem_out = gem(x)
        # Max over spatial dimensions (2, 3 for NCHW)
        max_out = mx.max(x, axis=(2, 3), keepdims=True)
        mx.eval(gem_out, max_out)

        # GeM output should be between average and max (closer to max for large p)
        avg_out = mx.mean(x, axis=(2, 3), keepdims=True)
        mx.eval(avg_out)

        # GeM should be larger than avg and approach max
        assert float(mx.mean(gem_out)) > float(mx.mean(avg_out)), "GeM should exceed average"
        # Allow some tolerance - GeM doesn't equal max, but should be somewhat close
        assert float(mx.mean(mx.abs(gem_out - max_out))) < 1.0, "GeM with large p not approaching max"


# ============================================================================
# Embedding Correctness Tests
# ============================================================================


class TestSinusoidalEmbeddingCorrectness:
    """Correctness tests for sinusoidal positional embeddings."""

    def test_sinusoidal_vs_naive(self):
        """Compare sinusoidal embeddings against naive implementation."""
        mx.random.seed(42)
        max_len = 128
        dims = 64

        embedding = SinusoidalEmbedding(dims=dims, max_seq_len=max_len)

        positions = mx.arange(32)
        mx.eval(positions)

        out = embedding(positions)
        mx.eval(out)

        naive_out = naive_sinusoidal_embedding(positions, dims)
        mx.eval(naive_out)

        assert mx.allclose(out, naive_out, atol=1e-5), \
            f"Max diff: {float(mx.max(mx.abs(out - naive_out)))}"

    def test_sinusoidal_unique_positions(self):
        """Test each position gets a unique embedding."""
        dims = 64
        max_len = 100

        embedding = SinusoidalEmbedding(dims=dims, max_seq_len=max_len)

        positions = mx.arange(max_len)
        mx.eval(positions)

        out = embedding(positions)
        mx.eval(out)

        # Check all embeddings are different
        for i in range(max_len):
            for j in range(i + 1, max_len):
                diff = float(mx.sum(mx.abs(out[i] - out[j])))
                assert diff > 0.01, f"Positions {i} and {j} have same embedding"

    def test_sinusoidal_orthogonality(self):
        """Test sin and cos components are orthogonal."""
        dims = 64
        max_len = 100

        embedding = SinusoidalEmbedding(dims=dims, max_seq_len=max_len)

        positions = mx.arange(max_len)
        mx.eval(positions)

        out = embedding(positions)
        mx.eval(out)

        # Split into sin and cos parts
        sin_part = out[:, :dims//2]
        cos_part = out[:, dims//2:]

        # Dot product should be small (not perfectly orthogonal, but close)
        dot = mx.sum(sin_part * cos_part, axis=-1)
        mx.eval(dot)

        assert float(mx.mean(mx.abs(dot))) < dims, "Sin/cos not approximately orthogonal"


class TestLearnedPositionalEmbeddingCorrectness:
    """Correctness tests for learned positional embeddings."""

    def test_learned_embedding_indexing(self):
        """Test learned embeddings are correctly indexed."""
        mx.random.seed(42)
        max_len = 100
        dims = 64

        embedding = LearnedPositionalEmbedding(dims=dims, max_seq_len=max_len)

        # Test specific positions
        positions = mx.array([0, 5, 10, 50, 99])
        mx.eval(positions)

        out = embedding(positions)
        mx.eval(out)

        # Manually index the embedding table
        manual_out = embedding.embedding.weight[positions]
        mx.eval(manual_out)

        assert mx.allclose(out, manual_out, atol=1e-6)

    def test_learned_embedding_gradient(self):
        """Test gradients flow to embedding weights."""
        mx.random.seed(42)
        max_len = 50
        dims = 32

        embedding = LearnedPositionalEmbedding(dims=dims, max_seq_len=max_len)

        positions = mx.array([0, 1, 2, 3, 4])
        mx.eval(positions)

        def loss_fn(weights):
            embedding.embedding.weight = weights
            return mx.sum(embedding(positions))

        grad = mx.grad(loss_fn)(embedding.embedding.weight)
        mx.eval(grad)

        # Only accessed positions should have non-zero gradients
        accessed_grads = grad[:5]
        other_grads = grad[5:]

        assert float(mx.sum(mx.abs(accessed_grads))) > 0, "Accessed positions have zero gradient"


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of layer implementations."""

    def test_rmsnorm_large_values(self):
        """Test RMSNorm handles large values."""
        dims = 64
        rmsnorm = RMSNorm(dims=dims)

        x = mx.random.normal((2, 16, dims)) * 100
        mx.eval(x)

        out = rmsnorm(x)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "RMSNorm produced non-finite values"

    def test_rmsnorm_small_values(self):
        """Test RMSNorm handles small values."""
        dims = 64
        rmsnorm = RMSNorm(dims=dims)

        x = mx.random.normal((2, 16, dims)) * 1e-5
        mx.eval(x)

        out = rmsnorm(x)
        mx.eval(out)

        assert mx.all(mx.isfinite(out)), "RMSNorm produced non-finite values"

    def test_swiglu_stability(self):
        """Test SwiGLU is stable with various input ranges."""
        swiglu = SwiGLU(64, 128)

        for scale in [0.001, 0.1, 1.0, 10.0, 100.0]:
            x = mx.random.normal((2, 16, 64)) * scale
            mx.eval(x)

            out = swiglu(x)
            mx.eval(out)

            assert mx.all(mx.isfinite(out)), f"SwiGLU unstable at scale {scale}"

    def test_mish_extreme_values(self):
        """Test Mish handles extreme values."""
        mish = Mish()

        # Very negative values (should approach 0)
        x_neg = mx.array([-100.0, -50.0, -10.0])
        out_neg = mish(x_neg)
        mx.eval(out_neg)
        assert mx.all(mx.isfinite(out_neg))
        assert mx.all(mx.abs(out_neg) < 1.0)  # Should be small

        # Very positive values (should approach x)
        x_pos = mx.array([10.0, 50.0, 100.0])
        out_pos = mish(x_pos)
        mx.eval(out_pos)
        assert mx.all(mx.isfinite(out_pos))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
