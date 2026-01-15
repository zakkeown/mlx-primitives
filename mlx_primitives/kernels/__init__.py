"""Custom Metal kernels for MLX.

This module provides optimized Metal kernel implementations for
performance-critical operations. The kernels use `mx.fast.metal_kernel`
to compile and execute custom Metal shaders on Apple Silicon GPUs.

Available Kernels:
    - fast_rmsnorm: RMSNorm with 1.88x speedup at large sizes
    - fast_rmsnorm_residual: Fused RMSNorm(x + residual) with 1.58x speedup
    - fast_swiglu: SwiGLU activation (silu(gate) * up) with 1.25x speedup
    - fast_geglu: GeGLU activation (gelu(gate) * up) with 6.1x speedup
    - fast_rope: Rotary Position Embeddings with fused rotation
    - fast_rope_qk: Fused RoPE for Q and K (up to 4.3x speedup)
    - fast_layernorm: LayerNorm with mean/var fusion (up to 3.5x speedup)
    - fast_bias_gelu: Fused bias + GELU (up to 3.1x speedup)
    - fast_bias_silu: Fused bias + SiLU (up to 1.35x speedup)
    - fast_add_scale: Fused add + scale (up to 1.6x speedup)

Usage:
    from mlx_primitives.kernels import fast_rmsnorm, fast_swiglu, rmsnorm
    from mlx_primitives.kernels import fast_rope, precompute_rope_cache
    from mlx_primitives.kernels import fast_layernorm, fast_bias_gelu

    # Direct Metal kernel (fastest, requires 3D input)
    out = fast_rmsnorm(x, weight)  # x: (batch, seq, hidden)
    out = fast_swiglu(gate, up)

    # RoPE kernel
    cos, sin = precompute_rope_cache(seq_len=2048, head_dim=64)
    q_rot = fast_rope(q, cos, sin)  # q: (batch, seq, heads, head_dim)
    q_rot, k_rot = fast_rope_qk(q, k, cos, sin)

    # LayerNorm kernel
    out = fast_layernorm(x, gamma, beta)

    # Fused bias + activation (common in MLPs)
    out = fast_bias_gelu(linear_out, bias)
    out = fast_bias_silu(linear_out, bias)

    # Auto-fallback wrapper (handles 2D/3D, falls back if Metal unavailable)
    out = rmsnorm(x, weight, use_metal=True)

Performance Notes:
    Metal kernels show overhead at small sizes due to kernel launch cost.
    Speedups increase with tensor size. At (8, 4096, 2048):
    - RMSNorm: 1.88x, GeGLU: 6.1x, SwiGLU: 1.25x, LayerNorm: 3.5x
    - RoPE Q+K: 4.3x, Bias+GELU: 3.1x
"""

from mlx_primitives.kernels.rmsnorm import (
    fast_rmsnorm,
    fast_rmsnorm_residual,
    rmsnorm,
)
from mlx_primitives.kernels.swiglu import (
    fast_swiglu,
    fast_geglu,
    swiglu,
    geglu,
)
from mlx_primitives.kernels.rope import (
    fast_rope,
    fast_rope_qk,
    rope,
    rope_qk,
    precompute_rope_cache,
)
from mlx_primitives.kernels.layernorm import (
    fast_layernorm,
    layernorm,
)
from mlx_primitives.kernels.fused_ops import (
    fast_bias_gelu,
    fast_bias_silu,
    fast_add_scale,
    bias_gelu,
    bias_silu,
    add_scale,
)
from mlx_primitives.kernels.fused_rope_attention import (
    fast_fused_rope_attention,
    fast_fused_rope_attention_tiled,
    fused_rope_attention,
    FusedRoPEFlashAttention,
)
from mlx_primitives.kernels.hardware_info import (
    get_hardware_info,
    get_chip_family,
    get_max_threadgroup_memory,
    AppleSiliconInfo,
)
from mlx_primitives.kernels.block_config import (
    get_optimal_block_config,
    get_block_config_info,
    validate_block_config,
    warmup_block_configs,
    BlockConfig,
)
from mlx_primitives.kernels.gqa_optimized import (
    fast_gqa_attention,
    gqa_attention,
    gqa_attention_reference,
    OptimizedGQA,
)

__all__ = [
    # RMSNorm
    "fast_rmsnorm",
    "fast_rmsnorm_residual",
    "rmsnorm",
    # GLU variants
    "fast_swiglu",
    "fast_geglu",
    "swiglu",
    "geglu",
    # RoPE
    "fast_rope",
    "fast_rope_qk",
    "rope",
    "rope_qk",
    "precompute_rope_cache",
    # LayerNorm
    "fast_layernorm",
    "layernorm",
    # Fused ops
    "fast_bias_gelu",
    "fast_bias_silu",
    "fast_add_scale",
    "bias_gelu",
    "bias_silu",
    "add_scale",
    # Fused RoPE + Attention
    "fast_fused_rope_attention",
    "fast_fused_rope_attention_tiled",
    "fused_rope_attention",
    "FusedRoPEFlashAttention",
    # Hardware detection
    "get_hardware_info",
    "get_chip_family",
    "get_max_threadgroup_memory",
    "AppleSiliconInfo",
    # Block configuration
    "get_optimal_block_config",
    "get_block_config_info",
    "validate_block_config",
    "warmup_block_configs",
    "BlockConfig",
    # Optimized GQA
    "fast_gqa_attention",
    "gqa_attention",
    "gqa_attention_reference",
    "OptimizedGQA",
]
