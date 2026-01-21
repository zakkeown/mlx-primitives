"""Tolerance configuration for parity tests.

Each operation has specific tolerances for different dtypes, accounting for
numerical precision differences between frameworks and floating point formats.

TOLERANCE RATIONALE & EXTERNAL RESEARCH
=======================================

Floating Point Precision Standards:
- fp32: Machine epsilon = 2^-23 ≈ 1.2e-7, practical rtol = 1e-5 to 1e-4
- fp16: Machine epsilon = 2^-10 ≈ 9.8e-4, practical rtol = 1e-3
- bf16: 7-bit mantissa (vs 10-bit fp16), practical rtol = 1e-2

External References:
1. PyTorch Numerical Accuracy:
   https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
   - Recommends rtol=1e-5, atol=1e-3 for fp16 comparisons
   - Notes bitwise differences expected across platforms

2. Flash Attention Precision ("Is Flash Attention Stable?" - arXiv:2405.02803):
   https://arxiv.org/html/2405.02803v1
   - BF16 Flash Attention has ~10x more numeric deviation than baseline
   - Deviation scales with 1/sqrt(mantissa bits)
   - Industry practice: 1% tolerance for bf16 attention

3. Triton GPU Kernels (Issue #5283):
   https://github.com/triton-lang/triton/issues/5283
   - fp16 matmul commonly uses atol=1e-2

4. BFloat16 Format (Wikipedia):
   https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
   - 7-bit mantissa = ~100x less precise than fp32
   - Same exponent range as fp32 (good for gradients)

Audit History:
- 2026-01: Tightened SSM tolerances from 15% to 2% (passed)
- 2026-01: Reduced gradient multiplier from 20x to 10x
- 2026-01: Tightened INT4 bf16 from 20% to 10%
- 2026-01: Tightened selective_scan bf16 from 10% to 5%
- 2026-01: Fixed incorrectly tight activation tolerances
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
        # Quantized KV cache: INT8 quantization introduces ~0.5-1% error inherently.
        # bf16 at 10% accounts for quantization + bf16 accumulation error.
        "quantized_kv": (1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1),
        "rope": (1e-4, 1e-5, 2e-3, 2e-3, 1e-2, 1e-2),
        # NTK-aware and YaRN variants compute cache internally with different precision
        # Looser tolerances account for accumulated error in scaled frequency computation
        "ntk_rope": (5e-4, 5e-4, 3e-3, 3e-3, 2e-2, 2e-2),
        "yarn_rope": (5e-4, 5e-4, 3e-3, 3e-3, 2e-2, 2e-2),
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
        # GELU involves erf() which differs between MLX and PyTorch implementations.
        # fp16: 1e-3 tolerance is standard for fp16 precision (~2e-3 actual diff).
        "gelu_exact": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
        "gelu_approx": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),  # Approximation has larger error
        # SiLU = x * sigmoid(x). fp16/bf16 differ due to sigmoid implementations.
        "silu": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
        "quick_gelu": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),
        "gelu_tanh": (1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2),
        "mish": (1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "squared_relu": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
        # Swish variants involve sigmoid. fp16/bf16 adjusted to standard precision.
        "swish": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
        "hard_swish": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
        "hard_sigmoid": (1e-5, 1e-6, 1e-3, 1e-3, 1e-2, 1e-2),
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
        # AdaLayerNorm chains multiple operations. bf16 tightened from 10% to 5%
        # per audit. Chained ops cause error accumulation but 10% was excessive.
        "adalayernorm": (1e-4, 1e-5, 1e-3, 1e-3, 5e-2, 5e-2),
    },

    # =========================================================================
    # Fused Operation Tolerances
    # Wider tolerances for fp16/bf16 due to:
    # - MLX kernels compute in float32 internally, then convert to output dtype
    # - PyTorch MPS computes in native fp16/bf16
    # - This causes accumulated numerical differences at larger tensor sizes
    # =========================================================================
    "fused_ops": {
        "fused_rmsnorm_linear": (1e-4, 1e-5, 1e-2, 5e-3, 5e-2, 2e-2),
        "fused_swiglu": (1e-4, 1e-5, 1e-2, 5e-3, 5e-2, 2e-2),
        "fused_geglu": (1e-4, 1e-5, 1e-2, 5e-3, 5e-2, 2e-2),
        "fused_rope_attention": (1e-4, 1e-5, 1e-2, 5e-3, 5e-2, 2e-2),
    },

    # =========================================================================
    # Quantization Tolerances (wider due to quantization error)
    # =========================================================================
    "quantization": {
        "int8_quantize": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int8_dequantize": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        # INT4 bf16 tightened from 20% to 10% per audit. 4-bit quantization has
        # inherent error but 20% was excessive. See: typical INT4 error is ~5-10%.
        "int4_quantize": (5e-2, 5e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int4_dequantize": (5e-2, 5e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int8_linear": (1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1),
        "int4_linear": (5e-2, 5e-2, 1e-1, 1e-1, 1e-1, 1e-1),
    },

    # =========================================================================
    # Primitive Tolerances
    # Parallel prefix sum algorithms have O(log n * eps) error accumulation
    # For seq=1024, this gives ~2e-5 absolute error for fp32
    # Lower precision (fp16/bf16) has larger error due to reduced mantissa bits
    # and accumulated errors in the parallel recurrence
    # =========================================================================
    "primitives": {
        "associative_scan_add": (1e-5, 2e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "associative_scan_mul": (1e-4, 2e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        "associative_scan_ssm": (1e-4, 2e-5, 1e-3, 1e-3, 1e-2, 1e-2),
        # Selective scan chains multiple parallel operations (discretization + SSM scan)
        # fp16/bf16 accumulate more error due to the multi-stage computation.
        # bf16 tightened from 10% to 5% per audit.
        "selective_scan": (1e-4, 5e-5, 5e-3, 5e-3, 5e-2, 5e-2),
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
        "avg_pool1d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
        "max_pool1d": (1e-5, 1e-6, 1e-3, 1e-4, 1e-2, 1e-3),
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

    # =========================================================================
    # SSM (State Space Model) Tolerances
    # Sequential scan operations accumulate error over sequence length.
    # For seq=N, expected error accumulation is O(sqrt(N) * eps) for stable scans.
    # Mamba selective scan chains discretization + parallel scan + output projection.
    # =========================================================================
    "ssm": {
        # Selective scan: core SSM operation with discretization
        # fp32: tight tolerance, fp16/bf16: wider due to accumulated error in recurrence
        "selective_scan": (1e-4, 5e-5, 5e-3, 5e-3, 5e-2, 5e-2),
        # MambaBlock: full block including conv, projections, and SSM
        # Slightly wider than selective_scan due to additional operations
        "mamba_block": (1e-4, 1e-4, 5e-3, 5e-3, 5e-2, 5e-2),
        # S4Layer: structured state space with HiPPO initialization
        # Uses sequential scan internally
        "s4_layer": (1e-4, 5e-5, 5e-3, 5e-3, 5e-2, 5e-2),
        # H3Layer: hybrid attention-SSM with multiplicative interaction
        "h3_layer": (1e-4, 5e-5, 5e-3, 5e-3, 5e-2, 5e-2),
    },

    # =========================================================================
    # Training Utilities Tolerances
    # Pure arithmetic operations, so tolerances are very tight.
    # =========================================================================
    "training": {
        # EMA: shadow = decay * shadow + (1 - decay) * current
        # Pure arithmetic, should be exact
        "ema": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        "ema_warmup": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        # Gradient clipping: involves sqrt for norm computation
        "clip_grad_norm": (1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6),
        "clip_grad_value": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        "gradient_norm": (1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6),
        # SWA: running average formula
        "swa": (1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6),
        # Lookahead: interpolation formula
        "lookahead": (1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6),
    },

    # =========================================================================
    # KV Cache Variants Tolerances
    # Most are exact array operations, quantized variants have wider tolerances.
    # =========================================================================
    "kv_cache": {
        # Basic KV cache: exact array operations
        "basic": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        # Sliding window: exact array operations
        "sliding_window": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        # Rotating cache: exact array operations
        "rotating": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
        # Compressed (8-bit): quantization introduces ~1% error
        "compressed_8bit": (1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2),
        # Compressed (4-bit): quantization introduces ~5-10% error
        "compressed_4bit": (5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2),
        # Pruned cache: exact for kept entries
        "pruned": (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7),
    },

    # =========================================================================
    # Learning Rate Scheduler Tolerances
    # Schedulers are pure Python math operations, so tolerances are extremely tight.
    # Differences should only come from floating point rounding in trig/exp functions.
    # =========================================================================
    "schedulers": {
        # Cosine annealing: uses cos(), very precise
        "cosine_annealing": (1e-6, 1e-8, 1e-6, 1e-8, 1e-6, 1e-8),
        # OneCycleLR: uses cos() for annealing, multi-phase
        "one_cycle": (1e-5, 1e-8, 1e-5, 1e-8, 1e-5, 1e-8),
        # Polynomial decay: uses power function
        "polynomial_decay": (1e-5, 1e-8, 1e-5, 1e-8, 1e-5, 1e-8),
        # MultiStep: discrete steps, should be exact
        "multi_step": (1e-10, 1e-12, 1e-10, 1e-12, 1e-10, 1e-12),
        # Exponential decay: uses exp(), very precise
        "exponential_decay": (1e-6, 1e-8, 1e-6, 1e-8, 1e-6, 1e-8),
        # Linear warmup: simple linear interpolation
        "linear_warmup": (1e-10, 1e-12, 1e-10, 1e-12, 1e-10, 1e-12),
        # Inverse sqrt: uses sqrt(), very precise
        "inverse_sqrt": (1e-6, 1e-8, 1e-6, 1e-8, 1e-6, 1e-8),
        # Warmup cosine: combines linear + cosine
        "warmup_cosine": (1e-6, 1e-8, 1e-6, 1e-8, 1e-6, 1e-8),
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
    # Gradient tolerances are 10x looser due to accumulated numerical errors
    # in backpropagation through multiple operations.
    # Reduced from 20x per audit: 20x was potentially masking gradient bugs.
    # See: PyTorch testing typically uses 5-10x for gradients.
    return (rtol * 10, atol * 10)


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
