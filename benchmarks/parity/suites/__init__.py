"""Parity benchmark suites."""

from benchmarks.parity.suites.attention_parity import AttentionParityBenchmarks
from benchmarks.parity.suites.activation_parity import ActivationParityBenchmarks
from benchmarks.parity.suites.normalization_parity import NormalizationParityBenchmarks
from benchmarks.parity.suites.fused_ops_parity import FusedOpsParityBenchmarks
from benchmarks.parity.suites.quantization_parity import QuantizationParityBenchmarks
from benchmarks.parity.suites.primitives_parity import PrimitivesParityBenchmarks
from benchmarks.parity.suites.moe_parity import MoEParityBenchmarks
from benchmarks.parity.suites.pooling_parity import PoolingParityBenchmarks
from benchmarks.parity.suites.embeddings_parity import EmbeddingsParityBenchmarks
from benchmarks.parity.suites.cache_parity import CacheParityBenchmarks
from benchmarks.parity.suites.generation_parity import GenerationParityBenchmarks

__all__ = [
    "AttentionParityBenchmarks",
    "ActivationParityBenchmarks",
    "NormalizationParityBenchmarks",
    "FusedOpsParityBenchmarks",
    "QuantizationParityBenchmarks",
    "PrimitivesParityBenchmarks",
    "MoEParityBenchmarks",
    "PoolingParityBenchmarks",
    "EmbeddingsParityBenchmarks",
    "CacheParityBenchmarks",
    "GenerationParityBenchmarks",
]
