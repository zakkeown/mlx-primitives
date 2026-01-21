#!/usr/bin/env python3
"""Generate golden files for correctness testing.

This script generates reference outputs from NumPy or PyTorch implementations
and saves them as .npz files for comparison with MLX implementations.

Usage:
    # Install dependencies (NumPy is required, PyTorch is optional)
    pip install numpy scipy

    # Generate all golden files using NumPy backend (default)
    python scripts/validation/generate_all.py

    # Generate using PyTorch backend (for cross-validation)
    python scripts/validation/generate_all.py --backend pytorch

    # Generate with cross-validation between NumPy and PyTorch
    python scripts/validation/generate_all.py --backend both --validate

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

# Core dependencies
import numpy as np

# PyTorch is optional - only required for pytorch backend
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

from base import GoldenGenerator, Backend

# Import generators
from generators.activations import (
    SwiGLUGenerator,
    GeGLUGenerator,
    ReGLUGenerator,
    FusedSwiGLUGenerator,
    FusedGeGLUGenerator,
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
    FusedRoPEAttentionGenerator,
)

from generators.normalization import (
    RMSNormGenerator,
    GroupNormGenerator,
    InstanceNormGenerator,
    AdaLayerNormGenerator,
    QKNormGenerator,
    FusedRMSNormLinearGenerator,
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
        ("fused_geglu", FusedGeGLUGenerator),
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
        ("fused_rope_attention", FusedRoPEAttentionGenerator),
    ],
    "normalization": [
        ("rmsnorm", RMSNormGenerator),
        ("groupnorm", GroupNormGenerator),
        ("instancenorm", InstanceNormGenerator),
        ("adalayernorm", AdaLayerNormGenerator),
        ("qknorm", QKNormGenerator),
        ("fused_rmsnorm_linear", FusedRMSNormLinearGenerator),
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
    backend: Backend = "numpy",
    validate: bool = False,
) -> int:
    """Generate golden files for a category.

    Args:
        category: Category name (e.g., 'activations', 'attention')
        output_dir: Output directory for golden files
        seed: Random seed for reproducibility
        operation: Specific operation to generate (or None for all)
        backend: Reference implementation backend
        validate: Whether to cross-validate NumPy and PyTorch

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
            generator = gen_class(output_dir=category_dir, seed=seed, backend=backend)
            files = generator.generate_all(validate=validate)
            total_files += len(files)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return total_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden files for MLX primitives validation"
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
        "--backend",
        type=str,
        choices=["numpy", "pytorch", "both"],
        default="numpy",
        help="Reference implementation backend (default: numpy)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-validate NumPy and PyTorch outputs (requires --backend=both)",
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

    # Validate backend choice
    if args.backend in ("pytorch", "both") and not HAS_TORCH:
        if args.backend == "pytorch":
            print("Error: PyTorch is required for --backend=pytorch")
            print("Install with: pip install torch")
            sys.exit(1)
        else:
            print("Warning: PyTorch not available, falling back to NumPy only")
            args.backend = "numpy"

    if args.validate and args.backend != "both":
        print("Warning: --validate only has effect with --backend=both")

    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Backend: {args.backend}")
    if args.validate:
        print("Cross-validation: enabled")
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
            args.backend,
            args.validate,
        )
    else:
        # Generate all categories
        for category in GENERATORS.keys():
            if not GENERATORS[category]:
                continue

            print(f"=== {category.upper()} ===")
            files = generate_category(
                category,
                args.output_dir,
                args.seed,
                backend=args.backend,
                validate=args.validate,
            )
            total_files += files
            print()

    print(f"Done! Generated {total_files} golden files.")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
