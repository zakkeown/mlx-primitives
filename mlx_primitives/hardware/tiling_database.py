"""Tiling configuration database with chip-specific defaults.

This module provides a database of optimal tiling configurations for
different Apple Silicon chips, operations, problem sizes, and data types.
"""

import json
import logging
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

from mlx_primitives.hardware.detection import ChipFamily, ChipTier

logger = logging.getLogger(__name__)
from mlx_primitives.hardware.tiling import (
    DataType,
    OperationType,
    ProblemSize,
    TilingConfig,
)


# Type alias for the config key
ConfigKey = tuple[OperationType, ChipFamily, ChipTier, ProblemSize, DataType]


def _make_attention_config(
    block_m: int,
    block_n: int,
    head_dim: int,
    prefetch: int = 1,
    unroll: int = 1,
) -> TilingConfig:
    """Create attention tiling config with shared memory calculation.

    Note: shared memory = 2 * block_n * (head_dim + 4) * 4 must be <= 32KB
    For head_dim=64: max block_n = 60 (use 48)
    For head_dim=128: max block_n = 31 (use 24)
    """
    padded_dim = head_dim + 4  # Bank conflict avoidance
    shared_mem = 2 * block_n * padded_dim * 4  # K + V tiles, float32
    # Cap shared memory to 32KB limit
    if shared_mem > 32768:
        # Reduce block_n to fit
        max_block_n = 32768 // (2 * padded_dim * 4)
        block_n = min(block_n, max_block_n)
        shared_mem = 2 * block_n * padded_dim * 4
    return TilingConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=head_dim,
        threads_per_threadgroup=min(block_m * 8, 1024),
        shared_memory_bytes=shared_mem,
        use_padding=True,
        padding_elements=4,
        prefetch_distance=prefetch,
        unroll_factor=unroll,
    )


def _make_scan_config(block_size: int, prefetch: int = 1) -> TilingConfig:
    """Create scan tiling config."""
    shared_mem = 2 * block_size * 4  # A and h arrays
    return TilingConfig(
        block_m=block_size,
        block_n=1,
        block_k=1,
        threads_per_threadgroup=block_size,
        num_simd_groups=block_size // 32,
        shared_memory_bytes=shared_mem,
        prefetch_distance=prefetch,
    )


def _make_matmul_config(
    block_m: int,
    block_n: int,
    block_k: int,
    prefetch: int = 1,
    unroll: int = 1,
) -> TilingConfig:
    """Create matmul tiling config."""
    shared_mem = (block_m * block_k + block_k * block_n) * 4
    return TilingConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads_per_threadgroup=256,
        shared_memory_bytes=shared_mem,
        prefetch_distance=prefetch,
        unroll_factor=unroll,
    )


def _make_quantized_config(
    block_m: int,
    block_n: int,
    block_k: int,
    vector_width: int = 8,
) -> TilingConfig:
    """Create quantized linear tiling config."""
    # INT8/INT4 have smaller memory footprint
    shared_mem = (block_m * block_k + block_k * block_n)  # 1 byte per element
    return TilingConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads_per_threadgroup=256,
        shared_memory_bytes=shared_mem,
        vector_width=vector_width,
        use_vector_loads=True,
    )


# =============================================================================
# Default Configurations
# =============================================================================
# These are embedded defaults based on Apple Silicon architecture analysis.
# User-tuned configs (from auto-tuning) take precedence over these.

_DEFAULT_CONFIGS: dict[ConfigKey, TilingConfig] = {
    # =========================================================================
    # FLASH ATTENTION - M1 Family
    # =========================================================================
    # M1 BASE - Conservative, memory-limited (68 GB/s)
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_attention_config(32, 32, 64, prefetch=1),
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(32, 32, 64, prefetch=1),
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_attention_config(32, 32, 64, prefetch=1),
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP16): _make_attention_config(64, 64, 64, prefetch=1),
    # M1 PRO/MAX - More GPU cores, higher bandwidth
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.PRO, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(48, 48, 64, prefetch=1),
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.MAX, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 48, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 48, 64, prefetch=2),
    # M1 with head_dim=128 needs smaller blocks
    # NOTE: Key includes head_dim via block_k parameter, using LARGE size to differentiate from head_dim=64
    (OperationType.FLASH_ATTENTION, ChipFamily.M1, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_attention_config(32, 24, 128, prefetch=1),

    # =========================================================================
    # FLASH ATTENTION - M2 Family
    # =========================================================================
    # M2 has improved memory controller, better cache
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_attention_config(48, 48, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP16): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.PRO, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.MAX, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2, unroll=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M2, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2, unroll=2),

    # =========================================================================
    # FLASH ATTENTION - M3 Family
    # =========================================================================
    # M3 has dynamic caching, mesh shading
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_attention_config(48, 48, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP16): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.PRO, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.MAX, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2, unroll=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M3, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3, unroll=2),

    # =========================================================================
    # FLASH ATTENTION - M4 Family
    # =========================================================================
    # M4 has enhanced memory subsystem, can use largest blocks
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_attention_config(48, 48, 64, prefetch=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.BASE, ProblemSize.HUGE, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP16): _make_attention_config(64, 64, 64, prefetch=3),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.PRO, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3, unroll=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.MAX, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3, unroll=2),
    (OperationType.FLASH_ATTENTION, ChipFamily.M4, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_attention_config(64, 64, 128, prefetch=3, unroll=2),

    # =========================================================================
    # SLIDING WINDOW ATTENTION
    # =========================================================================
    (OperationType.SLIDING_WINDOW, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(32, 32, 64),
    (OperationType.SLIDING_WINDOW, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(48, 48, 64, prefetch=2),
    (OperationType.SLIDING_WINDOW, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=2),
    (OperationType.SLIDING_WINDOW, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_attention_config(64, 64, 64, prefetch=3),

    # =========================================================================
    # SCAN OPERATIONS
    # =========================================================================
    # M1 Family
    (OperationType.SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_scan_config(256),
    (OperationType.SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512),
    (OperationType.SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024),
    (OperationType.SSM_SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_scan_config(256),
    (OperationType.SSM_SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512),
    (OperationType.SSM_SCAN, ChipFamily.M1, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024),
    # M2 Family
    (OperationType.SCAN, ChipFamily.M2, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_scan_config(256, prefetch=2),
    (OperationType.SCAN, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=2),
    (OperationType.SCAN, ChipFamily.M2, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024, prefetch=2),
    (OperationType.SSM_SCAN, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=2),
    # M3 Family
    (OperationType.SCAN, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=2),
    (OperationType.SCAN, ChipFamily.M3, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024, prefetch=2),
    (OperationType.SSM_SCAN, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=2),
    # M4 Family
    (OperationType.SCAN, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=3),
    (OperationType.SCAN, ChipFamily.M4, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024, prefetch=3),
    (OperationType.SSM_SCAN, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_scan_config(512, prefetch=3),
    (OperationType.SSM_SCAN, ChipFamily.M4, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_scan_config(1024, prefetch=3),

    # =========================================================================
    # MATMUL
    # =========================================================================
    # M1 Family
    (OperationType.MATMUL, ChipFamily.M1, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_matmul_config(32, 32, 32),
    (OperationType.MATMUL, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_matmul_config(64, 64, 32),
    (OperationType.MATMUL, ChipFamily.M1, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32),
    (OperationType.MATMUL, ChipFamily.M1, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2),
    # M2 Family
    (OperationType.MATMUL, ChipFamily.M2, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_matmul_config(32, 32, 32, prefetch=2),
    (OperationType.MATMUL, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2),
    (OperationType.MATMUL, ChipFamily.M2, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2),
    (OperationType.MATMUL, ChipFamily.M2, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2, unroll=2),
    # M3 Family
    (OperationType.MATMUL, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2),
    (OperationType.MATMUL, ChipFamily.M3, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2),
    (OperationType.MATMUL, ChipFamily.M3, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=2, unroll=2),
    # M4 Family
    (OperationType.MATMUL, ChipFamily.M4, ChipTier.BASE, ProblemSize.SMALL, DataType.FP32): _make_matmul_config(32, 32, 32, prefetch=3),
    (OperationType.MATMUL, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=3),
    (OperationType.MATMUL, ChipFamily.M4, ChipTier.BASE, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=3),
    (OperationType.MATMUL, ChipFamily.M4, ChipTier.PRO, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(64, 64, 32, prefetch=3, unroll=2),
    (OperationType.MATMUL, ChipFamily.M4, ChipTier.MAX, ProblemSize.LARGE, DataType.FP32): _make_matmul_config(128, 128, 32, prefetch=3, unroll=4),

    # =========================================================================
    # QUANTIZED LINEAR
    # =========================================================================
    # INT8 - Larger K blocks due to smaller memory footprint
    (OperationType.INT8_LINEAR, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT8): _make_quantized_config(64, 64, 64, vector_width=8),
    (OperationType.INT8_LINEAR, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT8): _make_quantized_config(64, 64, 64, vector_width=8),
    (OperationType.INT8_LINEAR, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT8): _make_quantized_config(64, 64, 64, vector_width=8),
    (OperationType.INT8_LINEAR, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT8): _make_quantized_config(64, 64, 128, vector_width=8),
    (OperationType.INT8_LINEAR, ChipFamily.M4, ChipTier.MAX, ProblemSize.LARGE, DataType.INT8): _make_quantized_config(128, 64, 128, vector_width=8),
    # INT4 - Even larger K blocks
    (OperationType.INT4_LINEAR, ChipFamily.M1, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT4): _make_quantized_config(64, 64, 128, vector_width=16),
    (OperationType.INT4_LINEAR, ChipFamily.M2, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT4): _make_quantized_config(64, 64, 128, vector_width=16),
    (OperationType.INT4_LINEAR, ChipFamily.M3, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT4): _make_quantized_config(64, 64, 128, vector_width=16),
    (OperationType.INT4_LINEAR, ChipFamily.M4, ChipTier.BASE, ProblemSize.MEDIUM, DataType.INT4): _make_quantized_config(128, 64, 128, vector_width=16),
    (OperationType.INT4_LINEAR, ChipFamily.M4, ChipTier.MAX, ProblemSize.LARGE, DataType.INT4): _make_quantized_config(128, 64, 128, vector_width=16),
}


class TilingDatabase:
    """Database for managing tiling configurations.

    Provides lookup for default configurations and caching for
    user-tuned (auto-tuned) configurations.

    Attributes:
        cache_dir: Directory for persistent cache storage.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the tiling database.

        Args:
            cache_dir: Directory for caching tuned configs.
                Defaults to ~/.mlx_primitives/tiling_cache/
        """
        self._defaults = _DEFAULT_CONFIGS
        self._cache_dir = cache_dir or Path.home() / ".mlx_primitives" / "tiling_cache"
        self._user_configs: dict[ConfigKey, TilingConfig] = {}
        self._load_user_cache()

    def _load_user_cache(self) -> None:
        """Load user-tuned configs from disk."""
        cache_file = self._cache_dir / "tuned_configs.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                loaded_count = 0
                skipped_count = 0
                for key_str, config_dict in data.items():
                    # Parse key from string representation
                    try:
                        key = self._parse_key_string(key_str)
                        config = TilingConfig(**config_dict)
                        self._user_configs[key] = config
                        loaded_count += 1
                    except (ValueError, TypeError) as e:
                        skipped_count += 1
                        logger.debug(f"Skipping invalid cache entry '{key_str}': {e}")
                if loaded_count > 0:
                    logger.debug(f"Loaded {loaded_count} tuned configs from cache")
                if skipped_count > 0:
                    logger.warning(
                        f"Skipped {skipped_count} invalid entries in tiling cache. "
                        f"Consider clearing cache with TilingDatabase.clear_user_cache()."
                    )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Tiling cache file is corrupted ({e}). "
                    f"Starting with empty cache. Your previously tuned configs have been lost. "
                    f"Cache file: {cache_file}"
                )
            except OSError as e:
                logger.warning(
                    f"Could not read tiling cache ({e}). Starting with empty cache."
                )

    def _parse_key_string(self, key_str: str) -> ConfigKey:
        """Parse a key string back to ConfigKey tuple."""
        parts = key_str.split("|")
        return (
            OperationType(parts[0]),
            ChipFamily(parts[1]),
            ChipTier(parts[2]),
            ProblemSize(parts[3]),
            DataType(parts[4]),
        )

    def _key_to_string(self, key: ConfigKey) -> str:
        """Convert ConfigKey tuple to string for JSON storage."""
        return "|".join(str(k.value) for k in key)

    def _save_user_cache(self) -> None:
        """Save user-tuned configs to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / "tuned_configs.json"

        data = {}
        for key, config in self._user_configs.items():
            key_str = self._key_to_string(key)
            data[key_str] = asdict(config)

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_config(
        self,
        operation: OperationType,
        chip_family: ChipFamily,
        chip_tier: ChipTier,
        problem_size: ProblemSize,
        dtype: DataType,
    ) -> TilingConfig:
        """Get best tiling config for the given parameters.

        Lookup order:
        1. User-tuned configs (from auto-tuning)
        2. Exact match in defaults
        3. Fallback with tier relaxation (MAX -> PRO -> BASE)
        4. Fallback with chip family (newer -> older)
        5. Generic default

        Args:
            operation: Operation type.
            chip_family: Apple Silicon family.
            chip_tier: Chip tier within family.
            problem_size: Problem size category.
            dtype: Data type.

        Returns:
            Best matching TilingConfig.
        """
        key = (operation, chip_family, chip_tier, problem_size, dtype)

        # 1. Check user-tuned configs first
        if key in self._user_configs:
            return self._user_configs[key]

        # 2. Check defaults with exact match
        if key in self._defaults:
            return self._defaults[key]

        # 3. Tier fallback: ULTRA -> MAX -> PRO -> BASE
        tier_fallback = [ChipTier.ULTRA, ChipTier.MAX, ChipTier.PRO, ChipTier.BASE]
        tier_idx = tier_fallback.index(chip_tier) if chip_tier in tier_fallback else 3
        for fallback_tier in tier_fallback[tier_idx:]:
            fallback_key = (operation, chip_family, fallback_tier, problem_size, dtype)
            if fallback_key in self._defaults:
                return self._defaults[fallback_key]

        # 4. Chip family fallback: M4 -> M3 -> M2 -> M1
        family_fallback = [ChipFamily.M4, ChipFamily.M3, ChipFamily.M2, ChipFamily.M1]
        family_idx = family_fallback.index(chip_family) if chip_family in family_fallback else 0
        for fallback_family in family_fallback[family_idx:]:
            for fallback_tier in tier_fallback:
                fallback_key = (operation, fallback_family, fallback_tier, problem_size, dtype)
                if fallback_key in self._defaults:
                    return self._defaults[fallback_key]

        # 5. Problem size fallback: try MEDIUM as default
        for fallback_family in family_fallback:
            fallback_key = (operation, fallback_family, ChipTier.BASE, ProblemSize.MEDIUM, dtype)
            if fallback_key in self._defaults:
                return self._defaults[fallback_key]

        # 6. Last resort: generic default
        return self._get_generic_default(operation, dtype)

    def _get_generic_default(
        self,
        operation: OperationType,
        dtype: DataType,
    ) -> TilingConfig:
        """Get a safe generic default for any operation."""
        if operation in (
            OperationType.ATTENTION,
            OperationType.FLASH_ATTENTION,
            OperationType.SLIDING_WINDOW,
            OperationType.CHUNKED_ATTENTION,
        ):
            return _make_attention_config(32, 32, 64)
        elif operation in (OperationType.SCAN, OperationType.SSM_SCAN):
            return _make_scan_config(256)
        elif operation in (OperationType.INT8_LINEAR, OperationType.INT4_LINEAR):
            return _make_quantized_config(64, 64, 64)
        else:
            return _make_matmul_config(64, 64, 32)

    def save_tuned_config(
        self,
        operation: OperationType,
        chip_family: ChipFamily,
        chip_tier: ChipTier,
        problem_size: ProblemSize,
        dtype: DataType,
        config: TilingConfig,
    ) -> None:
        """Save a tuned configuration.

        Args:
            operation: Operation type.
            chip_family: Apple Silicon family.
            chip_tier: Chip tier within family.
            problem_size: Problem size category.
            dtype: Data type.
            config: The tuned configuration to save.
        """
        key = (operation, chip_family, chip_tier, problem_size, dtype)
        self._user_configs[key] = config
        self._save_user_cache()

    def clear_user_cache(self) -> None:
        """Clear all user-tuned configurations."""
        self._user_configs.clear()
        cache_file = self._cache_dir / "tuned_configs.json"
        if cache_file.exists():
            cache_file.unlink()


@lru_cache(maxsize=1)
def get_tiling_database() -> TilingDatabase:
    """Get the global tiling database instance.

    Thread-safe singleton via lru_cache (thread-safe in Python 3.9+).

    Returns:
        Singleton TilingDatabase instance.
    """
    return TilingDatabase()
