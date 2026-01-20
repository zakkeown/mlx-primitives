"""Baseline implementations for benchmark comparison."""

from benchmarks.baselines.mlx_native import (
    naive_attention,
    naive_cumsum,
    naive_layer_norm,
    naive_rms_norm,
    naive_silu,
    naive_gelu,
)

# Conditionally import PyTorch and JAX baselines
try:
    from benchmarks.baselines.pytorch_mps import (
        PyTorchMPSBenchmarks,
        pytorch_available,
    )
except ImportError:
    PyTorchMPSBenchmarks = None
    pytorch_available = lambda: False

try:
    from benchmarks.baselines.jax_metal import (
        JAXMetalBenchmarks,
        jax_available,
    )
except ImportError:
    JAXMetalBenchmarks = None
    jax_available = lambda: False

__all__ = [
    # MLX naive implementations
    "naive_attention",
    "naive_cumsum",
    "naive_layer_norm",
    "naive_rms_norm",
    "naive_silu",
    "naive_gelu",
    # External baselines
    "PyTorchMPSBenchmarks",
    "pytorch_available",
    "JAXMetalBenchmarks",
    "jax_available",
]
