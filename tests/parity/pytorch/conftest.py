"""PyTorch-specific fixtures for parity tests."""

import pytest
import numpy as np

# Import shared fixtures
from tests.parity.conftest import (
    HAS_PYTORCH,
    HAS_MPS,
    skip_without_pytorch,
    skip_without_mps,
    pytorch_device,
    pytorch_to_numpy,
    numpy_to_pytorch,
    get_pytorch_dtype,
)

if HAS_PYTORCH:
    import torch
    import torch.nn.functional as F


# =============================================================================
# PyTorch-Specific Fixtures
# =============================================================================

@pytest.fixture
def torch_device():
    """Get the torch device for tests (MPS if available, else CPU)."""
    if not HAS_PYTORCH:
        pytest.skip("PyTorch not available")
    return torch.device("mps" if HAS_MPS else "cpu")


@pytest.fixture
def torch_dtype(dtype_config):
    """Convert dtype config to torch dtype."""
    if not HAS_PYTORCH:
        return None
    return get_pytorch_dtype(dtype_config)


@pytest.fixture
def numpy_to_torch(torch_device):
    """Convert numpy array to torch tensor on device."""
    def convert(arr, dtype=None, requires_grad=False):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        t = torch.from_numpy(arr)
        if dtype is not None:
            t = t.to(dtype)
        t = t.to(torch_device)
        if requires_grad:
            t = t.requires_grad_(True)
        return t
    return convert


@pytest.fixture
def torch_to_numpy():
    """Convert torch tensor to numpy array."""
    def convert(t):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        return t.detach().cpu().numpy()
    return convert


# =============================================================================
# Reference Implementation Fixtures
# =============================================================================

@pytest.fixture
def torch_attention_ref():
    """PyTorch scaled dot-product attention reference."""
    def attention(q, k, v, scale=None, causal=False):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        with torch.no_grad():
            return F.scaled_dot_product_attention(
                q, k, v,
                scale=scale,
                is_causal=causal,
            )
    return attention


@pytest.fixture
def torch_gelu_ref():
    """PyTorch GELU reference."""
    def gelu(x, approximate="none"):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        with torch.no_grad():
            return F.gelu(x, approximate=approximate)
    return gelu


@pytest.fixture
def torch_silu_ref():
    """PyTorch SiLU reference."""
    def silu(x):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        with torch.no_grad():
            return F.silu(x)
    return silu


@pytest.fixture
def torch_layer_norm_ref():
    """PyTorch LayerNorm reference."""
    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        with torch.no_grad():
            return F.layer_norm(x, normalized_shape, weight, bias, eps)
    return layer_norm


@pytest.fixture
def torch_softmax_ref():
    """PyTorch softmax reference."""
    def softmax(x, dim=-1):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        with torch.no_grad():
            return F.softmax(x, dim=dim)
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
