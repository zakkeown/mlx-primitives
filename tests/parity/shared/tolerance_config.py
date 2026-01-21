"""Tolerance configuration for parity tests.

Each operation has specific tolerances for different dtypes, accounting for
numerical precision differences between frameworks and floating point formats.
"""

from typing import Dict, Tuple


# =============================================================================
# Tolerance Definitions
# =============================================================================

# Format: {category: {operation: (rtol_fp32, atol_fp32, rtol_fp16, atol_fp16, rtol_bf16, atol_bf16)}}

TOLERANCES: Dict[str, Dict[str, Tuple[float, float, float, float, float, float]]] = {
    # =========================================================================
    # Attention Tolerances
    # =========================================================================
    "attention": {
        "flash_attention": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "flash_attention_causal": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "sliding_window": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "chunked_cross": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "gqa": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "mqa": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "sparse": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "linear": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "alibi": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "quantized_kv": (1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1),  # Wider for quantized
        "rope": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "layout_bhsd": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "layout_bshd": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
    },

    # =========================================================================
    # Activation Tolerances
    # =========================================================================
    "activations": {
        "swiglu": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "geglu": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "reglu": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "gelu_exact": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "gelu_approx": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),  # Approximation has larger error
        "silu": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "quick_gelu": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),
        "gelu_tanh": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),
        "mish": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "squared_relu": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "swish": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "hard_swish": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "hard_sigmoid": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
    },

    # =========================================================================
    # Normalization Tolerances
    # bf16 needs wider tolerances due to accumulated precision errors in
    # variance/mean calculations, especially for large tensors
    # =========================================================================
    "normalization": {
        "rmsnorm": (1e-4, 1e-5, 1e-3, 1e-3, 5e-2, 5e-2),
        "layernorm": (1e-4, 1e-5, 1e-3, 1e-3, 5e-2, 5e-2),
        "groupnorm": (1e-4, 1e-5, 1e-3, 1e-3, 5e-2, 5e-2),
        "instancenorm": (1e-4, 1e-5, 1e-3, 1e-3, 5e-2, 5e-2),
        # AdaLayerNorm chains multiple operations, needs even wider bf16 tolerance
        "adalayernorm": (1e-4, 1e-5, 1e-3, 1e-3, 1e-1, 1e-1),
    },

    # =========================================================================
    # Fused Operation Tolerances
    # =========================================================================
    "fused_ops": {
        "fused_rmsnorm_linear": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "fused_swiglu": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "fused_geglu": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "fused_rope_attention": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
    },

    # =========================================================================
    # Quantization Tolerances (wider due to quantization error)
    # =========================================================================
    "quantization": {
        "int8_quantize": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int8_dequantize": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int4_quantize": (5e-2, 5e-2, 1e-1, 1e-1, 2e-1, 2e-1),
        "int4_dequantize": (5e-2, 5e-2, 1e-1, 1e-1, 2e-1, 2e-1),
        "int8_linear": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int4_linear": (5e-2, 5e-2, 1e-1, 1e-1, 2e-1, 2e-1),
    },

    # =========================================================================
    # Primitive Tolerances
    # =========================================================================
    "primitives": {
        "associative_scan_add": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "associative_scan_mul": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "associative_scan_ssm": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "selective_scan": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "selective_gather": (1e-6, 1e-7, 1e-5, 1e-5, 1e-4, 1e-4),
        "selective_scatter_add": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
    },

    # =========================================================================
    # MoE Tolerances
    # =========================================================================
    "moe": {
        "topk_routing": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "expert_dispatch": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "load_balancing_loss": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
    },

    # =========================================================================
    # Pooling Tolerances
    # =========================================================================
    "pooling": {
        "adaptive_avg_pool1d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
        "adaptive_avg_pool2d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
        "adaptive_max_pool1d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
        "adaptive_max_pool2d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
        "global_attention_pooling": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "gem": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "spatial_pyramid_pooling": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
    },

    # =========================================================================
    # Embedding Tolerances
    # =========================================================================
    "embeddings": {
        "sinusoidal": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "learned_positional": (1e-6, 1e-7, 1e-5, 1e-5, 1e-4, 1e-4),
        "rotary": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "alibi": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "relative_positional": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
    },

    # =========================================================================
    # Cache Tolerances
    # =========================================================================
    "cache": {
        "paged_attention": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "block_allocation": (0, 0, 0, 0, 0, 0),  # Exact integer operations
        "eviction_lru": (0, 0, 0, 0, 0, 0),  # Exact
        "eviction_fifo": (0, 0, 0, 0, 0, 0),  # Exact
        "speculative_verification": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
    },

    # =========================================================================
    # Generation/Sampling Tolerances
    # =========================================================================
    "generation": {
        "temperature_sampling": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "top_k_sampling": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
        "top_p_sampling": (1e-5, 1e-6, 1e-4, 1e-4, 1e-3, 1e-3),
    },
}


# =============================================================================
# Tolerance Access Functions
# =============================================================================

def get_tolerance(
    category: str,
    operation: str,
    dtype: str,
) -> Tuple[float, float]:
    """Get tolerance for a specific operation and dtype.

    Args:
        category: Operation category (attention, activations, etc.).
        operation: Operation name within the category.
        dtype: Data type (fp32, fp16, bf16).

    Returns:
        Tuple of (rtol, atol) for the specified operation and dtype.
    """
    # Default tolerances if category/operation not found
    default_tolerances = {
        "fp32": (1e-4, 1e-5),
        "fp16": (1e-3, 1e-3),
        "bf16": (1e-2, 1e-2),
    }

    category_tolerances = TOLERANCES.get(category, {})
    operation_tolerances = category_tolerances.get(operation)

    if operation_tolerances is None:
        return default_tolerances.get(dtype, (1e-4, 1e-4))

    # Extract tolerances based on dtype
    rtol_fp32, atol_fp32, rtol_fp16, atol_fp16, rtol_bf16, atol_bf16 = operation_tolerances

    dtype_mapping = {
        "fp32": (rtol_fp32, atol_fp32),
        "fp16": (rtol_fp16, atol_fp16),
        "bf16": (rtol_bf16, atol_bf16),
    }

    return dtype_mapping.get(dtype, (rtol_fp32, atol_fp32))


def get_all_tolerances(category: str, operation: str) -> Dict[str, Tuple[float, float]]:
    """Get tolerances for all dtypes for a specific operation.

    Args:
        category: Operation category.
        operation: Operation name.

    Returns:
        Dictionary mapping dtype to (rtol, atol) tuples.
    """
    return {
        "fp32": get_tolerance(category, operation, "fp32"),
        "fp16": get_tolerance(category, operation, "fp16"),
        "bf16": get_tolerance(category, operation, "bf16"),
    }


def get_gradient_tolerance(
    category: str,
    operation: str,
    dtype: str = "fp32",
) -> Tuple[float, float]:
    """Get tolerance for gradient comparison.

    Gradient tolerances are typically looser than forward pass tolerances
    due to accumulated numerical errors in backpropagation.

    Args:
        category: Operation category.
        operation: Operation name.
        dtype: Data type.

    Returns:
        Tuple of (rtol, atol) for gradient comparison.
    """
    rtol, atol = get_tolerance(category, operation, dtype)
    # Gradient tolerances are 20x looser due to accumulated numerical errors
    # in backpropagation through multiple operations
    return (rtol * 20, atol * 20)


# =============================================================================
# Tolerance Assertion Helpers
# =============================================================================

def assert_close(
    actual,
    expected,
    category: str,
    operation: str,
    dtype: str,
    msg: str = "",
) -> None:
    """Assert that actual and expected are close within operation tolerances.

    Args:
        actual: Actual array (numpy or MLX).
        expected: Expected array (numpy).
        category: Operation category.
        operation: Operation name.
        dtype: Data type.
        msg: Optional error message.
    """
    import numpy as np
    import mlx.core as mx

    if isinstance(actual, mx.array):
        actual = np.array(actual)

    rtol, atol = get_tolerance(category, operation, dtype)
    np.testing.assert_allclose(
        actual, expected,
        rtol=rtol, atol=atol,
        err_msg=f"{msg} [category={category}, op={operation}, dtype={dtype}]"
    )


def assert_gradient_close(
    actual,
    expected,
    category: str,
    operation: str,
    dtype: str = "fp32",
    msg: str = "",
) -> None:
    """Assert that gradient arrays are close within operation tolerances.

    Args:
        actual: Actual gradient array.
        expected: Expected gradient array.
        category: Operation category.
        operation: Operation name.
        dtype: Data type.
        msg: Optional error message.
    """
    import numpy as np
    import mlx.core as mx

    if isinstance(actual, mx.array):
        actual = np.array(actual)

    rtol, atol = get_gradient_tolerance(category, operation, dtype)
    np.testing.assert_allclose(
        actual, expected,
        rtol=rtol, atol=atol,
        err_msg=f"Gradient {msg} [category={category}, op={operation}, dtype={dtype}]"
    )
