"""Tolerance configurations and standard test shapes for golden file generation."""

from typing import Dict
from .base import ToleranceConfig


# =============================================================================
# Tolerance Configurations by Operation Category
# =============================================================================

TOLERANCE_CONFIGS: Dict[str, ToleranceConfig] = {
    # -------------------------------------------------------------------------
    # Attention Operations
    # -------------------------------------------------------------------------
    "attention_standard": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
        max_diff_fp32=1e-4,
        max_diff_fp16=5e-3,
    ),
    "attention_linear": ToleranceConfig(
        # Linear attention approximations have higher error
        rtol_fp32=1e-4,
        atol_fp32=1e-5,
        rtol_fp16=5e-3,
        atol_fp16=1e-3,
        max_diff_fp32=5e-4,
        max_diff_fp16=1e-2,
    ),
    "attention_sparse": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "rope": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "alibi": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    # -------------------------------------------------------------------------
    # Normalization Operations
    # -------------------------------------------------------------------------
    "normalization": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "normalization_adaptive": ToleranceConfig(
        # AdaLayerNorm has conditioning which can amplify errors
        rtol_fp32=1e-5,
        atol_fp32=1e-5,
        rtol_fp16=1e-3,
        atol_fp16=5e-4,
    ),
    # -------------------------------------------------------------------------
    # Activation Functions
    # -------------------------------------------------------------------------
    "activations_exact": ToleranceConfig(
        # Exact activations (ReLU, etc.) should match very closely
        rtol_fp32=1e-6,
        atol_fp32=1e-7,
        rtol_fp16=1e-4,
        atol_fp16=1e-5,
    ),
    "activations_transcendental": ToleranceConfig(
        # GELU, Mish, SiLU use transcendental functions
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "activations_glu": ToleranceConfig(
        # GLU variants involve products and transcendentals
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    # -------------------------------------------------------------------------
    # Pooling Operations
    # -------------------------------------------------------------------------
    "pooling": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "pooling_attention": ToleranceConfig(
        # Attention pooling has softmax which can accumulate error
        rtol_fp32=1e-5,
        atol_fp32=1e-5,
        rtol_fp16=1e-3,
        atol_fp16=5e-4,
    ),
    # -------------------------------------------------------------------------
    # Element-wise Operations (general)
    # -------------------------------------------------------------------------
    "elementwise": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    # -------------------------------------------------------------------------
    # Embedding Operations
    # -------------------------------------------------------------------------
    "embeddings_sinusoidal": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "embeddings_rotary": ToleranceConfig(
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    # -------------------------------------------------------------------------
    # State Space Models (SSM)
    # -------------------------------------------------------------------------
    "ssm": ToleranceConfig(
        # Sequential operations accumulate error over time
        rtol_fp32=1e-4,
        atol_fp32=1e-5,
        rtol_fp16=5e-3,
        atol_fp16=1e-3,
        max_diff_fp32=1e-3,
        max_diff_fp16=5e-2,
    ),
    "ssm_scan": ToleranceConfig(
        # Selective scan has long sequential dependencies
        rtol_fp32=1e-4,
        atol_fp32=1e-4,
        rtol_fp16=5e-3,
        atol_fp16=5e-3,
        max_diff_fp32=5e-3,
        max_diff_fp16=0.1,
    ),
    # -------------------------------------------------------------------------
    # Mixture of Experts (MoE)
    # -------------------------------------------------------------------------
    "moe": ToleranceConfig(
        # General MoE tolerance
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "moe_continuous": ToleranceConfig(
        # Continuous outputs from MoE layers
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    "moe_routing": ToleranceConfig(
        # Routing decisions can have discrete effects
        rtol_fp32=1e-4,
        atol_fp32=1e-5,
        rtol_fp16=5e-3,
        atol_fp16=1e-3,
    ),
    "moe_loss": ToleranceConfig(
        # Load balancing and router losses
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
    # -------------------------------------------------------------------------
    # Quantization
    # -------------------------------------------------------------------------
    "quantization_int8": ToleranceConfig(
        # INT8 quantization is inherently lossy
        rtol_fp32=1e-2,
        atol_fp32=1e-3,
        rtol_fp16=5e-2,
        atol_fp16=5e-3,
        max_diff_fp32=0.1,
        max_diff_fp16=0.2,
    ),
    "quantization_int4": ToleranceConfig(
        # INT4 is more lossy than INT8
        rtol_fp32=5e-2,
        atol_fp32=1e-2,
        rtol_fp16=0.1,
        atol_fp16=5e-2,
        max_diff_fp32=0.5,
        max_diff_fp16=1.0,
    ),
    "quantization_gptq": ToleranceConfig(
        # GPTQ uses calibration data for better accuracy
        rtol_fp32=2e-2,
        atol_fp32=5e-3,
        rtol_fp16=5e-2,
        atol_fp16=1e-2,
    ),
    "quantization_awq": ToleranceConfig(
        # AWQ preserves salient weights
        rtol_fp32=2e-2,
        atol_fp32=5e-3,
        rtol_fp16=5e-2,
        atol_fp16=1e-2,
    ),
    # -------------------------------------------------------------------------
    # Training Utilities
    # -------------------------------------------------------------------------
    "schedulers": ToleranceConfig(
        # LR schedulers should match exactly
        rtol_fp32=1e-6,
        atol_fp32=1e-7,
        rtol_fp16=1e-4,
        atol_fp16=1e-5,
    ),
    "ema": ToleranceConfig(
        # EMA involves multiplicative updates
        rtol_fp32=1e-5,
        atol_fp32=1e-6,
        rtol_fp16=1e-3,
        atol_fp16=1e-4,
    ),
}


# =============================================================================
# Standard Test Shapes
# =============================================================================

STANDARD_SHAPES = {
    "tiny": {
        "batch": 1,
        "seq": 16,
        "heads": 2,
        "num_kv_heads": 1,
        "head_dim": 32,
        "dims": 64,
        "hidden": 256,
    },
    "small": {
        "batch": 2,
        "seq": 64,
        "heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "dims": 512,
        "hidden": 2048,
    },
    "medium": {
        "batch": 4,
        "seq": 128,
        "heads": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "dims": 1024,
        "hidden": 4096,
    },
    "large": {
        "batch": 2,
        "seq": 256,
        "heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "dims": 2048,
        "hidden": 8192,
    },
}


# =============================================================================
# Edge Case Configurations
# =============================================================================

EDGE_CASES = [
    {
        "name": "zero_input",
        "input_type": "zeros",
        "description": "All zero inputs for numerical stability",
    },
    {
        "name": "small_input",
        "input_type": "small",
        "scale": 1e-7,
        "description": "Very small inputs for underflow handling",
    },
    {
        "name": "large_input",
        "input_type": "large",
        "scale": 1e4,
        "description": "Large inputs for overflow handling",
    },
    {
        "name": "negative_input",
        "input_type": "negative",
        "description": "All negative inputs for sign handling",
    },
    {
        "name": "mixed_sign",
        "input_type": "random",
        "description": "Random mixed positive/negative inputs",
    },
    {
        "name": "long_sequence",
        "seq_len": 4096,
        "description": "Long sequence for memory/accumulation testing",
    },
    {
        "name": "single_element",
        "batch": 1,
        "seq": 1,
        "description": "Single element boundary condition",
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_tolerance(category: str) -> ToleranceConfig:
    """Get tolerance config for a category, with fallback to defaults."""
    return TOLERANCE_CONFIGS.get(category, ToleranceConfig())


def get_shape(size: str) -> dict:
    """Get standard shape configuration by size name."""
    return STANDARD_SHAPES.get(size, STANDARD_SHAPES["small"])
