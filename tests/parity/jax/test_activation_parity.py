"""JAX Metal parity tests for activation functions."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


# =============================================================================
# GLU Variants
# =============================================================================

class TestSwiGLUParity:
    """SwiGLU activation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test SwiGLU forward pass parity with JAX."""
        from mlx_primitives.layers import SwiGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX SwiGLU
        swiglu_mlx = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu_mlx.parameters())

        # Get weights for JAX
        w1_np = np.array(swiglu_mlx.w1.weight)
        w2_np = np.array(swiglu_mlx.w2.weight)
        w_gate_np = np.array(swiglu_mlx.w_gate.weight)

        # MLX forward
        mlx_dtype = get_mlx_dtype(dtype)
        x_mlx = mx.array(x_np).astype(mlx_dtype)
        mlx_out = swiglu_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_dtype = get_jax_dtype(dtype)
        x_jax = jnp.array(x_np).astype(jax_dtype)
        # MLX Linear stores weight as (out, in), need to transpose for x @ W
        w1_jax = jnp.array(w1_np.T).astype(jax_dtype)
        w2_jax = jnp.array(w2_np.T).astype(jax_dtype)
        w_gate_jax = jnp.array(w_gate_np.T).astype(jax_dtype)

        gate = jnn.silu(x_jax @ w_gate_jax)
        up = x_jax @ w1_jax
        jax_out = (gate * up) @ w2_jax

        rtol, atol = get_tolerance("activations", "swiglu", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"SwiGLU forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        """Test SwiGLU backward pass parity with JAX."""
        from mlx_primitives.layers import SwiGLU

        config = SIZE_CONFIGS["small"]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX forward and backward
        swiglu_mlx = SwiGLU(dim, hidden_dim)
        mx.eval(swiglu_mlx.parameters())

        w1_np = np.array(swiglu_mlx.w1.weight)
        w2_np = np.array(swiglu_mlx.w2.weight)
        w_gate_np = np.array(swiglu_mlx.w_gate.weight)

        def mlx_loss_fn(x):
            return mx.sum(swiglu_mlx(x))

        x_mlx = mx.array(x_np)
        mlx_grad = mx.grad(mlx_loss_fn)(x_mlx)
        mx.eval(mlx_grad)

        # JAX backward
        w1_jax = jnp.array(w1_np.T)
        w2_jax = jnp.array(w2_np.T)
        w_gate_jax = jnp.array(w_gate_np.T)

        def jax_loss_fn(x):
            gate = jnn.silu(x @ w_gate_jax)
            up = x @ w1_jax
            out = (gate * up) @ w2_jax
            return jnp.sum(out)

        x_jax = jnp.array(x_np)
        jax_grad = jax.grad(jax_loss_fn)(x_jax)

        rtol, atol = get_tolerance("activations", "swiglu", "fp32")
        # Gradient tolerance is 20x looser
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), _to_numpy(jax_grad),
            rtol=rtol * 20, atol=atol * 20,
            err_msg="SwiGLU backward mismatch (JAX)"
        )


class TestGeGLUParity:
    """GeGLU activation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test GeGLU forward pass parity with JAX."""
        from mlx_primitives.layers import GeGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX GeGLU
        geglu_mlx = GeGLU(dim, hidden_dim)
        mx.eval(geglu_mlx.parameters())

        w1_np = np.array(geglu_mlx.w1.weight)
        w2_np = np.array(geglu_mlx.w2.weight)
        w_gate_np = np.array(geglu_mlx.w_gate.weight)

        x_mlx = mx.array(x_np)
        mlx_out = geglu_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = jnp.array(x_np)
        w1_jax = jnp.array(w1_np.T)
        w2_jax = jnp.array(w2_np.T)
        w_gate_jax = jnp.array(w_gate_np.T)

        gate = jnn.gelu(x_jax @ w_gate_jax)
        up = x_jax @ w1_jax
        jax_out = (gate * up) @ w2_jax

        rtol, atol = get_tolerance("activations", "geglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GeGLU forward mismatch (JAX) [{size}]"
        )


class TestReGLUParity:
    """ReGLU activation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test ReGLU forward pass parity with JAX."""
        from mlx_primitives.layers import ReGLU

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        hidden_dim = dim * 4

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX ReGLU
        reglu_mlx = ReGLU(dim, hidden_dim)
        mx.eval(reglu_mlx.parameters())

        w1_np = np.array(reglu_mlx.w1.weight)
        w2_np = np.array(reglu_mlx.w2.weight)
        w_gate_np = np.array(reglu_mlx.w_gate.weight)

        x_mlx = mx.array(x_np)
        mlx_out = reglu_mlx(x_mlx)
        mx.eval(mlx_out)

        # JAX reference
        x_jax = jnp.array(x_np)
        w1_jax = jnp.array(w1_np.T)
        w2_jax = jnp.array(w2_np.T)
        w_gate_jax = jnp.array(w_gate_np.T)

        gate = jnn.relu(x_jax @ w_gate_jax)
        up = x_jax @ w1_jax
        jax_out = (gate * up) @ w2_jax

        rtol, atol = get_tolerance("activations", "reglu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"ReGLU forward mismatch (JAX) [{size}]"
        )


# =============================================================================
# GELU Variants
# =============================================================================

class TestGELUExactParity:
    """GELU (exact) parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test exact GELU forward pass parity with JAX."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = nn.gelu(x_mlx)
        mx.eval(mlx_out)

        # JAX
        x_jax = jnp.array(x_np)
        jax_out = jnn.gelu(x_jax, approximate=False)

        rtol, atol = get_tolerance("activations", "gelu_exact", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU exact forward mismatch (JAX) [{size}]"
        )


class TestGELUApproximateParity:
    """GELU (approximate/tanh) parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test approximate GELU forward pass parity with JAX."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = gelu_tanh(x_mlx)
        mx.eval(mlx_out)

        # JAX
        x_jax = jnp.array(x_np)
        jax_out = jnn.gelu(x_jax, approximate=True)

        rtol, atol = get_tolerance("activations", "gelu_approx", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU approximate forward mismatch (JAX) [{size}]"
        )


class TestSiLUParity:
    """SiLU parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test SiLU forward pass parity with JAX."""
        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = nn.silu(x_mlx)
        mx.eval(mlx_out)

        # JAX
        x_jax = jnp.array(x_np)
        jax_out = jnn.silu(x_jax)

        rtol, atol = get_tolerance("activations", "silu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"SiLU forward mismatch (JAX) [{size}]"
        )


class TestQuickGELUParity:
    """QuickGELU parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test QuickGELU forward pass parity with JAX."""
        from mlx_primitives.layers import quick_gelu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = quick_gelu(x_mlx)
        mx.eval(mlx_out)

        # JAX: x * sigmoid(1.702 * x)
        x_jax = jnp.array(x_np)
        jax_out = x_jax * jnn.sigmoid(1.702 * x_jax)

        rtol, atol = get_tolerance("activations", "quick_gelu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"QuickGELU forward mismatch (JAX) [{size}]"
        )


class TestGELUTanhParity:
    """GELU tanh approximation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test GELU tanh forward pass parity with JAX."""
        from mlx_primitives.layers import gelu_tanh

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = gelu_tanh(x_mlx)
        mx.eval(mlx_out)

        # JAX
        x_jax = jnp.array(x_np)
        jax_out = jnn.gelu(x_jax, approximate=True)

        rtol, atol = get_tolerance("activations", "gelu_tanh", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"GELU tanh forward mismatch (JAX) [{size}]"
        )


# =============================================================================
# Other Activations
# =============================================================================

class TestMishParity:
    """Mish activation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test Mish forward pass parity with JAX."""
        from mlx_primitives.layers import mish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = mish(x_mlx)
        mx.eval(mlx_out)

        # JAX: x * tanh(softplus(x)) = x * tanh(log1p(exp(x)))
        x_jax = jnp.array(x_np)
        softplus = jnp.log1p(jnp.exp(x_jax))
        jax_out = x_jax * jnp.tanh(softplus)

        rtol, atol = get_tolerance("activations", "mish", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Mish forward mismatch (JAX) [{size}]"
        )


class TestSquaredReLUParity:
    """Squared ReLU parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test Squared ReLU forward pass parity with JAX."""
        from mlx_primitives.layers import squared_relu

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = squared_relu(x_mlx)
        mx.eval(mlx_out)

        # JAX: relu(x)^2
        x_jax = jnp.array(x_np)
        jax_out = jnn.relu(x_jax) ** 2

        rtol, atol = get_tolerance("activations", "squared_relu", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Squared ReLU forward mismatch (JAX) [{size}]"
        )


class TestSwishParity:
    """Swish activation parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test Swish forward pass parity with JAX."""
        from mlx_primitives.layers import swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]
        beta = 1.5  # Test with non-default beta

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = swish(x_mlx, beta=beta)
        mx.eval(mlx_out)

        # JAX: x * sigmoid(beta * x)
        x_jax = jnp.array(x_np)
        jax_out = x_jax * jnn.sigmoid(beta * x_jax)

        rtol, atol = get_tolerance("activations", "swish", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Swish forward mismatch (JAX) [{size}]"
        )


class TestHardSwishParity:
    """Hard Swish parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test Hard Swish forward pass parity with JAX."""
        from mlx_primitives.layers import hard_swish

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = hard_swish(x_mlx)
        mx.eval(mlx_out)

        # JAX: x * clip(x + 3, 0, 6) / 6
        x_jax = jnp.array(x_np)
        jax_out = x_jax * jnp.clip(x_jax + 3, 0, 6) / 6

        rtol, atol = get_tolerance("activations", "hard_swish", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Swish forward mismatch (JAX) [{size}]"
        )


class TestHardSigmoidParity:
    """Hard Sigmoid parity tests against JAX."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test Hard Sigmoid forward pass parity with JAX."""
        from mlx_primitives.layers import hard_sigmoid

        config = SIZE_CONFIGS[size]["activation"]
        batch, seq, dim = config["batch"], config["seq"], config["dim"]

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, dim).astype(np.float32)

        # MLX
        x_mlx = mx.array(x_np)
        mlx_out = hard_sigmoid(x_mlx)
        mx.eval(mlx_out)

        # JAX: clip(x + 3, 0, 6) / 6
        x_jax = jnp.array(x_np)
        jax_out = jnp.clip(x_jax + 3, 0, 6) / 6

        rtol, atol = get_tolerance("activations", "hard_sigmoid", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(jax_out),
            rtol=rtol, atol=atol,
            err_msg=f"Hard Sigmoid forward mismatch (JAX) [{size}]"
        )
