"""Shared pytest fixtures and configuration for parity tests."""

import pytest
import numpy as np

import mlx.core as mx

# Framework availability checks
try:
    import torch
    HAS_PYTORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_PYTORCH = False
    HAS_MPS = False
    torch = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


def pytest_configure(config):
    """Register custom markers for parity tests."""
    # Framework markers
    config.addinivalue_line("markers", "parity: parity tests against external frameworks")
    config.addinivalue_line("markers", "parity_pytorch: PyTorch parity tests")
    config.addinivalue_line("markers", "parity_jax: JAX parity tests")

    # Test type markers
    config.addinivalue_line("markers", "forward_parity: forward pass parity tests")
    config.addinivalue_line("markers", "backward_parity: gradient/backward pass parity tests")
    config.addinivalue_line("markers", "edge_case: edge case tests")

    # Size markers
    config.addinivalue_line("markers", "size_tiny: tiny input size tests")
    config.addinivalue_line("markers", "size_small: small input size tests")
    config.addinivalue_line("markers", "size_medium: medium input size tests")
    config.addinivalue_line("markers", "size_large: large input size tests")


# =============================================================================
# Framework Availability Fixtures
# =============================================================================

@pytest.fixture
def pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return HAS_PYTORCH


@pytest.fixture
def jax_available() -> bool:
    """Check if JAX is available."""
    return HAS_JAX


@pytest.fixture
def skip_without_pytorch():
    """Skip test if PyTorch is not available."""
    if not HAS_PYTORCH:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_without_jax():
    """Skip test if JAX is not available."""
    if not HAS_JAX:
        pytest.skip("JAX not available")


@pytest.fixture
def skip_without_mps():
    """Skip test if PyTorch MPS backend is not available."""
    if not HAS_PYTORCH or not HAS_MPS:
        pytest.skip("PyTorch MPS not available")


@pytest.fixture
def skip_without_jax_metal():
    """Skip test if JAX Metal backend is not available."""
    if not HAS_JAX:
        pytest.skip("JAX not available")
    # Check for Metal/GPU device
    devices = jax.devices()
    has_metal = any("gpu" in str(d).lower() or "metal" in str(d).lower() for d in devices)
    if not has_metal:
        pytest.skip("JAX Metal not available")


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def pytorch_device():
    """Get the PyTorch device to use (MPS if available, else CPU)."""
    if not HAS_PYTORCH:
        return None
    return torch.device("mps" if HAS_MPS else "cpu")


@pytest.fixture
def jax_device():
    """Get the JAX device to use."""
    if not HAS_JAX:
        return None
    devices = jax.devices()
    # Prefer GPU/Metal if available
    for d in devices:
        if "gpu" in str(d).lower() or "metal" in str(d).lower():
            return d
    return devices[0] if devices else None


# =============================================================================
# Size and Dtype Fixtures
# =============================================================================

@pytest.fixture(params=["tiny", "small", "medium", "large"])
def size_config(request):
    """Parameterized fixture for test sizes."""
    return request.param


@pytest.fixture(params=["fp32", "fp16", "bf16"])
def dtype_config(request):
    """Parameterized fixture for data types."""
    return request.param


@pytest.fixture(params=["fp32"])
def dtype_config_fp32_only(request):
    """Parameterized fixture for fp32 only (for gradient tests)."""
    return request.param


# =============================================================================
# Random Seed Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Auto-fixture to set random seeds for reproducibility."""
    np.random.seed(42)
    mx.random.seed(42)

    if HAS_PYTORCH:
        torch.manual_seed(42)
        if HAS_MPS:
            torch.mps.manual_seed(42)

    if HAS_JAX:
        # JAX uses explicit PRNGKey, handled in tests
        pass

    yield


# =============================================================================
# Conversion Utilities
# =============================================================================

@pytest.fixture
def mlx_to_numpy():
    """Converter from MLX arrays to NumPy."""
    def convert(x):
        if isinstance(x, mx.array):
            return np.array(x)
        return x
    return convert


@pytest.fixture
def numpy_to_mlx():
    """Converter from NumPy arrays to MLX."""
    def convert(x, dtype=None):
        if isinstance(x, np.ndarray):
            arr = mx.array(x)
            if dtype is not None:
                arr = arr.astype(dtype)
            return arr
        return x
    return convert


@pytest.fixture
def pytorch_to_numpy():
    """Converter from PyTorch tensors to NumPy."""
    def convert(x):
        if HAS_PYTORCH and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    return convert


@pytest.fixture
def numpy_to_pytorch():
    """Converter from NumPy arrays to PyTorch tensors."""
    def convert(x, device=None):
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if device is not None:
                t = t.to(device)
            return t
        return x
    return convert


@pytest.fixture
def jax_to_numpy():
    """Converter from JAX arrays to NumPy."""
    def convert(x):
        if HAS_JAX and isinstance(x, jnp.ndarray):
            return np.array(x)
        return x
    return convert


@pytest.fixture
def numpy_to_jax():
    """Converter from NumPy arrays to JAX arrays."""
    def convert(x, device=None):
        if not HAS_JAX:
            raise ImportError("JAX not available")
        if isinstance(x, np.ndarray):
            arr = jnp.array(x)
            if device is not None:
                arr = jax.device_put(arr, device)
            return arr
        return x
    return convert


# =============================================================================
# Dtype Conversion Utilities
# =============================================================================

def get_numpy_dtype(dtype_str: str) -> np.dtype:
    """Convert dtype string to numpy dtype."""
    mapping = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,  # NumPy doesn't support bf16, use fp32
    }
    return mapping.get(dtype_str, np.float32)


def get_mlx_dtype(dtype_str: str):
    """Convert dtype string to MLX dtype."""
    mapping = {
        "fp32": mx.float32,
        "fp16": mx.float16,
        "bf16": mx.bfloat16,
    }
    return mapping.get(dtype_str, mx.float32)


def get_pytorch_dtype(dtype_str: str):
    """Convert dtype string to PyTorch dtype."""
    if not HAS_PYTORCH:
        return None
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


def get_jax_dtype(dtype_str: str):
    """Convert dtype string to JAX dtype."""
    if not HAS_JAX:
        return None
    mapping = {
        "fp32": jnp.float32,
        "fp16": jnp.float16,
        "bf16": jnp.bfloat16,
    }
    return mapping.get(dtype_str, jnp.float32)


@pytest.fixture
def dtype_converter():
    """Fixture providing dtype conversion utilities."""
    return {
        "numpy": get_numpy_dtype,
        "mlx": get_mlx_dtype,
        "pytorch": get_pytorch_dtype,
        "jax": get_jax_dtype,
    }
