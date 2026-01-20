"""NumPy-only reference implementations for validation.

This module provides pure NumPy implementations of all operations,
enabling validation without requiring PyTorch.
"""

from .activations import (
    gelu,
    gelu_tanh,
    quick_gelu,
    silu,
    swiglu,
    geglu,
    reglu,
    mish,
    squared_relu,
    hard_swish,
    hard_sigmoid,
)

from .attention import (
    scaled_dot_product_attention,
    linear_attention,
    performer_attention,
    cosformer_attention,
)

from .normalization import (
    rmsnorm,
    groupnorm,
    instancenorm,
    ada_layernorm,
    qknorm,
)

from .embeddings import (
    sinusoidal_embedding,
    rotary_embedding,
    alibi_slopes,
    alibi_bias,
)

from .pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    gem_pool,
)

from .ssm import (
    selective_scan,
    mamba_block,
)

from .moe import (
    topk_routing,
    load_balancing_loss,
    router_z_loss,
)

from .quantization import (
    quantize,
    dequantize,
)

from .training import (
    cosine_annealing_lr,
    warmup_cosine_lr,
    polynomial_decay_lr,
    multistep_lr,
    inverse_sqrt_lr,
    ema_update,
)

__all__ = [
    # Activations
    "gelu",
    "gelu_tanh",
    "quick_gelu",
    "silu",
    "swiglu",
    "geglu",
    "reglu",
    "mish",
    "squared_relu",
    "hard_swish",
    "hard_sigmoid",
    # Attention
    "scaled_dot_product_attention",
    "linear_attention",
    "performer_attention",
    "cosformer_attention",
    # Normalization
    "rmsnorm",
    "groupnorm",
    "instancenorm",
    "ada_layernorm",
    "qknorm",
    # Embeddings
    "sinusoidal_embedding",
    "rotary_embedding",
    "alibi_slopes",
    "alibi_bias",
    # Pooling
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "gem_pool",
    # SSM
    "selective_scan",
    "mamba_block",
    # MoE
    "topk_routing",
    "load_balancing_loss",
    "router_z_loss",
    # Quantization
    "quantize",
    "dequantize",
    # Training
    "cosine_annealing_lr",
    "warmup_cosine_lr",
    "polynomial_decay_lr",
    "multistep_lr",
    "inverse_sqrt_lr",
    "ema_update",
]
