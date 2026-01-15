"""Data loading and augmentation utilities for MLX.

This module provides efficient data pipelines:
- DataLoader: Batched data loading with prefetching
- Dataset: Base classes for map-style and iterable datasets
- Transforms: Vision and text augmentations
- Samplers: Custom sampling strategies
"""

# Data loading
from mlx_primitives.data.dataloader import (
    DataLoader,
    InfiniteDataLoader,
    default_collate,
)
from mlx_primitives.data.dataset import (
    Dataset,
    IterableDataset,
    TensorDataset,
    ListDataset,
    ConcatDataset,
    SubsetDataset,
    TransformDataset,
    ChainDataset,
    random_split,
)
from mlx_primitives.data.sampler import (
    Sampler,
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
    SubsetRandomSampler,
    BatchSampler,
    DistributedSampler,
)

# Vision transforms
from mlx_primitives.data.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    Resize,
    ColorJitter,
    GaussianNoise,
    CutOut,
    MixUpTransform,
    CutMixTransform,
    RandomAugment,
    TrivialAugmentWide,
    mixup,
    cutmix,
)

# Text/sequence transforms
from mlx_primitives.data.text_transforms import (
    pad_sequence,
    pad_to_length,
    create_attention_mask,
    create_causal_mask,
    RandomMask,
    SpanMask,
    TokenDropout,
    pack_sequences,
    truncate_sequences,
)

__all__ = [
    # DataLoader
    "DataLoader",
    "InfiniteDataLoader",
    "default_collate",
    # Dataset
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ListDataset",
    "ConcatDataset",
    "SubsetDataset",
    "TransformDataset",
    "ChainDataset",
    "random_split",
    # Samplers
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "WeightedRandomSampler",
    "SubsetRandomSampler",
    "BatchSampler",
    "DistributedSampler",
    # Vision transforms
    "Compose",
    "Normalize",
    "ToTensor",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "RandomCrop",
    "CenterCrop",
    "Resize",
    "ColorJitter",
    "GaussianNoise",
    "CutOut",
    "MixUpTransform",
    "CutMixTransform",
    "RandomAugment",
    "TrivialAugmentWide",
    "mixup",
    "cutmix",
    # Text transforms
    "pad_sequence",
    "pad_to_length",
    "create_attention_mask",
    "create_causal_mask",
    "RandomMask",
    "SpanMask",
    "TokenDropout",
    "pack_sequences",
    "truncate_sequences",
]
