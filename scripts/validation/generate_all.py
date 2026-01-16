#!/usr/bin/env python3
"""Generate all PyTorch golden files for correctness testing.

This script generates reference outputs from PyTorch implementations
and saves them as .npz files for comparison with MLX implementations.

Usage:
    # Install dependencies
    pip install torch numpy

    # Generate all golden files
    python scripts/validation/generate_all.py

    # Generate only specific category
    python scripts/validation/generate_all.py --category activations

    # List available categories
    python scripts/validation/generate_all.py --list

Output files are saved to tests/golden/<category>/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Type

# Add the validation directory to sys.path so imports work when running as script
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Check for PyTorch
try:
    import torch
    import numpy as np
except ImportError:
    print("Error: This script requires PyTorch and NumPy.")
    print("Install with: pip install torch numpy")
    sys.exit(1)

from base import GoldenGenerator

# Import generators
from generators.activations import (
    SwiGLUGenerator,
    GeGLUGenerator,
    ReGLUGenerator,
    FusedSwiGLUGenerator,
    GELUGenerator,
    GELUTanhGenerator,
    QuickGELUGenerator,
    MishGenerator,
    SquaredReLUGenerator,
    SwishGenerator,
    HardSwishGenerator,
    HardSigmoidGenerator,
)

from generators.attention import (
    SDPAGenerator,
    GQAGenerator,
    MQAGenerator,
    SlidingWindowGenerator,
    LinearAttentionGenerator,
    PerformerGenerator,
    CosFormerGenerator,
    BlockSparseGenerator,
    LongformerGenerator,
    BigBirdGenerator,
    ALiBiGenerator,
    RoPEGenerator,
    RoPENTKGenerator,
    RoPEYaRNGenerator,
)

from generators.normalization import (
    RMSNormGenerator,
    GroupNormGenerator,
    InstanceNormGenerator,
    AdaLayerNormGenerator,
    QKNormGenerator,
)

from generators.ssm import (
    SelectiveScanGenerator,
    MambaBlockGenerator,
    S4Generator,
    H3Generator,
)

from generators.pooling import (
    AdaptiveAvgPool1dGenerator,
    AdaptiveAvgPool2dGenerator,
    AdaptiveMaxPool1dGenerator,
    AdaptiveMaxPool2dGenerator,
    GeMGenerator,
    SpatialPyramidPoolingGenerator,
    GlobalAttentionPoolingGenerator,
)

from generators.embeddings import (
    SinusoidalEmbeddingGenerator,
    LearnedPositionalEmbeddingGenerator,
    RotaryEmbeddingGenerator,
    ALiBiEmbeddingGenerator,
    RelativePositionalEmbeddingGenerator,
)

from generators.moe import (
    TopKRouterGenerator,
    ExpertChoiceRouterGenerator,
    MoELayerGenerator,
    LoadBalancingLossGenerator,
    RouterZLossGenerator,
)

from generators.quantization import (
    QuantizeDequantizeGenerator,
    Int8LinearGenerator,
    Int4LinearGenerator,
    QLoRALinearGenerator,
)

from generators.training import (
    CosineAnnealingLRGenerator,
    WarmupCosineSchedulerGenerator,
    OneCycleLRGenerator,
    PolynomialDecayLRGenerator,
    MultiStepLRGenerator,
    InverseSqrtSchedulerGenerator,
    EMAGenerator,
    EMAWithWarmupGenerator,
)


# Registry of all generators by category
GENERATORS: Dict[str, List[Tuple[str, Type[GoldenGenerator]]]] = {
    "activations": [
        ("swiglu", SwiGLUGenerator),
        ("geglu", GeGLUGenerator),
        ("reglu", ReGLUGenerator),
        ("fused_swiglu", FusedSwiGLUGenerator),
        ("gelu", GELUGenerator),
        ("gelu_tanh", GELUTanhGenerator),
        ("quick_gelu", QuickGELUGenerator),
        ("mish", MishGenerator),
        ("squared_relu", SquaredReLUGenerator),
        ("swish", SwishGenerator),
        ("hard_swish", HardSwishGenerator),
        ("hard_sigmoid", HardSigmoidGenerator),
    ],
    "attention": [
        ("sdpa", SDPAGenerator),
        ("gqa", GQAGenerator),
        ("mqa", MQAGenerator),
        ("sliding_window", SlidingWindowGenerator),
        ("linear_attention", LinearAttentionGenerator),
        ("performer", PerformerGenerator),
        ("cosformer", CosFormerGenerator),
        ("block_sparse", BlockSparseGenerator),
        ("longformer", LongformerGenerator),
        ("bigbird", BigBirdGenerator),
        ("alibi", ALiBiGenerator),
        ("rope", RoPEGenerator),
        ("rope_ntk", RoPENTKGenerator),
        ("rope_yarn", RoPEYaRNGenerator),
    ],
    "normalization": [
        ("rmsnorm", RMSNormGenerator),
        ("groupnorm", GroupNormGenerator),
        ("instancenorm", InstanceNormGenerator),
        ("adalayernorm", AdaLayerNormGenerator),
        ("qknorm", QKNormGenerator),
    ],
    "ssm": [
        ("selective_scan", SelectiveScanGenerator),
        ("mamba_block", MambaBlockGenerator),
        ("s4", S4Generator),
        ("h3", H3Generator),
    ],
    "pooling": [
        ("adaptive_avg_pool1d", AdaptiveAvgPool1dGenerator),
        ("adaptive_avg_pool2d", AdaptiveAvgPool2dGenerator),
        ("adaptive_max_pool1d", AdaptiveMaxPool1dGenerator),
        ("adaptive_max_pool2d", AdaptiveMaxPool2dGenerator),
        ("gem", GeMGenerator),
        ("spp", SpatialPyramidPoolingGenerator),
        ("global_attention_pooling", GlobalAttentionPoolingGenerator),
    ],
    "embeddings": [
        ("sinusoidal_embedding", SinusoidalEmbeddingGenerator),
        ("learned_positional_embedding", LearnedPositionalEmbeddingGenerator),
        ("rotary_embedding", RotaryEmbeddingGenerator),
        ("alibi_embedding", ALiBiEmbeddingGenerator),
        ("relative_positional_embedding", RelativePositionalEmbeddingGenerator),
    ],
    "moe": [
        ("topk_router", TopKRouterGenerator),
        ("expert_choice_router", ExpertChoiceRouterGenerator),
        ("moe_layer", MoELayerGenerator),
        ("load_balancing_loss", LoadBalancingLossGenerator),
        ("router_z_loss", RouterZLossGenerator),
    ],
    "quantization": [
        ("quantize_dequantize", QuantizeDequantizeGenerator),
        ("int8_linear", Int8LinearGenerator),
        ("int4_linear", Int4LinearGenerator),
        ("qlora_linear", QLoRALinearGenerator),
    ],
    "training": [
        ("cosine_annealing_lr", CosineAnnealingLRGenerator),
        ("warmup_cosine_scheduler", WarmupCosineSchedulerGenerator),
        ("one_cycle_lr", OneCycleLRGenerator),
        ("polynomial_decay_lr", PolynomialDecayLRGenerator),
        ("multi_step_lr", MultiStepLRGenerator),
        ("inverse_sqrt_scheduler", InverseSqrtSchedulerGenerator),
        ("ema", EMAGenerator),
        ("ema_with_warmup", EMAWithWarmupGenerator),
    ],
}


def list_categories():
    """Print available categories and generators."""
    print("Available categories and generators:")
    print()
    for category, generators in GENERATORS.items():
        status = "implemented" if generators else "TODO"
        print(f"  {category} ({status}):")
        if generators:
            for name, _ in generators:
                print(f"    - {name}")
        else:
            print("    (no generators yet)")
        print()


def generate_category(
    category: str,
    output_dir: Path,
    seed: int = 42,
    operation: str = None,
) -> int:
    """Generate golden files for a category.

    Returns:
        Number of files generated
    """
    if category not in GENERATORS:
        print(f"Unknown category: {category}")
        print(f"Available: {', '.join(GENERATORS.keys())}")
        return 0

    generators = GENERATORS[category]
    if not generators:
        print(f"Category '{category}' has no generators implemented yet.")
        return 0

    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0

    for gen_name, gen_class in generators:
        if operation and gen_name != operation:
            continue

        print(f"  Generating {gen_name}...")

        try:
            generator = gen_class(output_dir=category_dir, seed=seed)
            files = generator.generate_all()
            total_files += len(files)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return total_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch golden files for MLX primitives validation"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generate only for specific category (e.g., 'activations', 'attention')",
    )
    parser.add_argument(
        "--operation",
        type=str,
        default=None,
        help="Generate only for specific operation within a category",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "tests" / "golden",
        help="Output directory for golden files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories and generators",
    )
    args = parser.parse_args()

    if args.list:
        list_categories()
        return

    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    total_files = 0

    if args.category:
        # Generate single category
        print(f"=== {args.category.upper()} ===")
        total_files = generate_category(
            args.category,
            args.output_dir,
            args.seed,
            args.operation,
        )
    else:
        # Generate all categories
        for category in GENERATORS.keys():
            if not GENERATORS[category]:
                continue

            print(f"=== {category.upper()} ===")
            files = generate_category(category, args.output_dir, args.seed)
            total_files += files
            print()

    print(f"Done! Generated {total_files} golden files.")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
