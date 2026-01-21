"""JAX-specific fixtures for parity tests."""

import pytest
import numpy as np

# Import shared fixtures
from tests.parity.conftest import (
    HAS_JAX,
    skip_without_jax,
    skip_without_jax_metal,
    jax_device,
    jax_to_numpy,
    numpy_to_jax,
    get_jax_dtype,
)

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import nn as jnn


# =============================================================================
# JAX-Specific Fixtures
# =============================================================================

@pytest.fixture
def jax_key():
    """Get a JAX PRNG key for reproducible tests."""
    if not HAS_JAX:
        pytest.skip("JAX not available")
    return jax.random.PRNGKey(42)


@pytest.fixture
def jax_dtype(dtype_config):
    """Convert dtype config to JAX dtype."""
    if not HAS_JAX:
        return None
    return get_jax_dtype(dtype_config)


@pytest.fixture
def numpy_to_jax_arr(jax_device):
    """Convert numpy array to JAX array on device."""
    def convert(arr, dtype=None):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        a = jnp.array(arr)
        if dtype is not None:
            a = a.astype(dtype)
        if jax_device is not None:
            a = jax.device_put(a, jax_device)
        return a
    return convert


@pytest.fixture
def jax_arr_to_numpy():
    """Convert JAX array to numpy array."""
    def convert(a):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        return np.array(a)
    return convert


# =============================================================================
# Reference Implementation Fixtures
# =============================================================================

@pytest.fixture
def jax_attention_ref():
    """JAX scaled dot-product attention reference."""
    def attention(q, k, v, scale=None, causal=False):
        if not HAS_JAX:
            raise ImportError("JAX not available")

        if scale is None:
            scale = 1.0 / jnp.sqrt(q.shape[-1])

        # q, k, v: (batch, heads, seq, dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        if causal:
            seq_len = q.shape[2]
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            scores = jnp.where(mask, scores, -1e9)

        weights = jnn.softmax(scores, axis=-1)
        return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

    return attention


@pytest.fixture
def jax_gelu_ref():
    """JAX GELU reference."""
    def gelu(x, approximate=False):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        return jnn.gelu(x, approximate=approximate)
    return gelu


@pytest.fixture
def jax_silu_ref():
    """JAX SiLU reference."""
    def silu(x):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        return jnn.silu(x)
    return silu


@pytest.fixture
def jax_layer_norm_ref():
    """JAX LayerNorm reference."""
    def layer_norm(x, weight=None, bias=None, eps=1e-5):
        if not HAS_JAX:
            raise ImportError("JAX not available")

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + eps)

        if weight is not None:
            normalized = normalized * weight
        if bias is not None:
            normalized = normalized + bias

        return normalized
    return layer_norm


@pytest.fixture
def jax_softmax_ref():
    """JAX softmax reference."""
    def softmax(x, axis=-1):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        return jnn.softmax(x, axis=axis)
    return softmax


# =============================================================================
# Test Configuration Fixtures
# =============================================================================

@pytest.fixture
def attention_configs():
    """Standard attention test configurations."""
    return {
        "tiny": {"batch": 1, "seq": 64, "heads": 4, "head_dim": 32},
        "small": {"batch": 2, "seq": 256, "heads": 8, "head_dim": 64},
        "medium": {"batch": 4, "seq": 1024, "heads": 16, "head_dim": 64},
        "large": {"batch": 8, "seq": 4096, "heads": 32, "head_dim": 128},
    }


@pytest.fixture
def activation_configs():
    """Standard activation test configurations."""
    return {
        "tiny": {"batch": 1, "seq": 64, "dim": 256},
        "small": {"batch": 4, "seq": 256, "dim": 1024},
        "medium": {"batch": 8, "seq": 1024, "dim": 2048},
        "large": {"batch": 16, "seq": 4096, "dim": 4096},
    }


@pytest.fixture
def normalization_configs():
    """Standard normalization test configurations."""
    return {
        "tiny": {"batch": 1, "seq": 64, "hidden": 256},
        "small": {"batch": 4, "seq": 256, "hidden": 1024},
        "medium": {"batch": 8, "seq": 1024, "hidden": 2048},
        "large": {"batch": 16, "seq": 4096, "hidden": 4096},
    }
