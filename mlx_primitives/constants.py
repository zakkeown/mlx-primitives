"""Centralized numerical constants for MLX Primitives.

This module provides documented constants used throughout the codebase
to ensure consistency and traceability.
"""

import math

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# FP16-safe epsilon for division and normalization operations
# FP16 machine epsilon is ~1e-3, so 1e-6 provides safety margin
EPSILON_FP16_SAFE = 1e-6

# Standard epsilon for FP32 normalization (RMSNorm, LayerNorm)
EPSILON_NORM = 1e-5


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

# GELU approximation constants
# From: "Gaussian Error Linear Units (GELUs)" - Hendrycks & Gimpel, 2016
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
GELU_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/π)
GELU_TANH_COEFF = 0.044715  # Empirical coefficient for tanh approximation

# Quick GELU constant (less accurate but faster)
# QuickGELU(x) = x * sigmoid(1.702 * x)
QUICK_GELU_COEFF = 1.702


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

# Default base frequency for RoPE
# From: "RoFormer" - Su et al., 2021
# Standard value used in LLaMA, Mistral, and most modern LLMs
ROPE_BASE_FREQUENCY = 10000.0


# =============================================================================
# ATTENTION
# =============================================================================

# Mask value for causal attention (large negative to become ~0 after softmax)
ATTENTION_MASK_VALUE = -1e9


# =============================================================================
# QUANTIZATION
# =============================================================================

# INT8 quantization range
QUANT_INT8_MIN = -128
QUANT_INT8_MAX = 127
QUANT_INT8_RANGE = 255

# INT4 quantization range
QUANT_INT4_MIN = 0
QUANT_INT4_MAX = 15
QUANT_INT4_RANGE = 15


# =============================================================================
# HARDWARE
# =============================================================================

# Default L2 cache size assumption (MB) for M3 Pro
# Used for optimal block size calculations
DEFAULT_L2_CACHE_MB = 24.0

# Minimum sequence length for Metal kernel dispatch
MIN_SEQ_FOR_METAL = 32
