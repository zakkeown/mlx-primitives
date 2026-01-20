"""Pooling layer generators."""

from .adaptive import (
    AdaptiveAvgPool1dGenerator,
    AdaptiveAvgPool2dGenerator,
    AdaptiveMaxPool1dGenerator,
    AdaptiveMaxPool2dGenerator,
)
from .specialized import (
    GeMGenerator,
    SpatialPyramidPoolingGenerator,
    GlobalAttentionPoolingGenerator,
)

__all__ = [
    "AdaptiveAvgPool1dGenerator",
    "AdaptiveAvgPool2dGenerator",
    "AdaptiveMaxPool1dGenerator",
    "AdaptiveMaxPool2dGenerator",
    "GeMGenerator",
    "SpatialPyramidPoolingGenerator",
    "GlobalAttentionPoolingGenerator",
]
