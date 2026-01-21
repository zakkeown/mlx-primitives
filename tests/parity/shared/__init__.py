"""Shared utilities for parity testing."""

from tests.parity.shared.base_parity_test import (
    ParityTestCase,
    ParityTestConfig,
    ForwardParityMixin,
    BackwardParityMixin,
)
from tests.parity.shared.input_generators import (
    SIZE_CONFIGS,
    EDGE_CASE_CONFIGS,
    attention_inputs,
    activation_inputs,
    normalization_inputs,
    quantization_inputs,
    moe_inputs,
    pooling_inputs,
    embedding_inputs,
    scan_inputs,
    cache_inputs,
    sampling_inputs,
)
from tests.parity.shared.tolerance_config import TOLERANCES, get_tolerance
from tests.parity.shared.gradient_utils import (
    compute_mlx_gradients,
    compute_pytorch_gradients,
    compute_jax_gradients,
    compare_gradients,
    numerical_gradient,
)

__all__ = [
    "ParityTestCase",
    "ParityTestConfig",
    "ForwardParityMixin",
    "BackwardParityMixin",
    "SIZE_CONFIGS",
    "EDGE_CASE_CONFIGS",
    "attention_inputs",
    "activation_inputs",
    "normalization_inputs",
    "quantization_inputs",
    "moe_inputs",
    "pooling_inputs",
    "embedding_inputs",
    "scan_inputs",
    "cache_inputs",
    "sampling_inputs",
    "TOLERANCES",
    "get_tolerance",
    "compute_mlx_gradients",
    "compute_pytorch_gradients",
    "compute_jax_gradients",
    "compare_gradients",
    "numerical_gradient",
]
