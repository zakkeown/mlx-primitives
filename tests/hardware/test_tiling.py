"""Tests for tiling configuration and hardware detection."""

import pytest
import mlx.core as mx

from mlx_primitives.hardware import (
    ChipFamily,
    ChipTier,
    DataType,
    OperationType,
    ProblemSize,
    TilingConfig,
    TilingDatabase,
    classify_problem_size,
    dtype_to_enum,
    get_chip_info,
    get_optimal_config,
    get_tiling_database,
)


class TestTilingConfig:
    """Tests for TilingConfig dataclass."""

    def test_default_values(self):
        """Test default TilingConfig values."""
        config = TilingConfig(block_m=64, block_n=64)
        assert config.block_m == 64
        assert config.block_n == 64
        assert config.block_k == 32  # Default
        assert config.threads_per_threadgroup == 256
        assert config.use_padding is True
        assert config.padding_elements == 4

    def test_shared_memory_calculation(self):
        """Test shared memory calculation for tiles."""
        config = TilingConfig(block_m=64, block_n=64, block_k=64)

        # With padding: 2 * 64 * (64 + 4) * 4 = 34,816 bytes
        shared_mem = config.shared_memory_for_tiles(dtype_size=4, num_tiles=2)
        expected = 2 * 64 * (64 + 4) * 4
        assert shared_mem == expected

    def test_shared_memory_no_padding(self):
        """Test shared memory without padding."""
        config = TilingConfig(block_m=64, block_n=64, block_k=64, use_padding=False)

        # Without padding: 2 * 64 * 64 * 4 = 32,768 bytes
        shared_mem = config.shared_memory_for_tiles(dtype_size=4, num_tiles=2)
        expected = 2 * 64 * 64 * 4
        assert shared_mem == expected

    def test_validation_valid_config(self):
        """Test validation passes for valid config."""
        config = TilingConfig(
            block_m=64,
            block_n=64,
            block_k=32,
            threads_per_threadgroup=256,
            shared_memory_bytes=16384,
        )
        assert config.validate(max_shared_memory=32768) is True

    def test_validation_invalid_block_size(self):
        """Test validation fails for invalid block sizes."""
        config = TilingConfig(block_m=0, block_n=64)
        assert config.validate() is False

    def test_validation_invalid_threads(self):
        """Test validation fails for invalid thread count."""
        config = TilingConfig(
            block_m=64,
            block_n=64,
            threads_per_threadgroup=2048,  # > 1024 max
        )
        assert config.validate() is False

    def test_validation_exceeds_shared_memory(self):
        """Test validation fails when exceeding shared memory."""
        config = TilingConfig(
            block_m=64,
            block_n=64,
            shared_memory_bytes=65536,  # > 32KB
        )
        assert config.validate(max_shared_memory=32768) is False

    def test_to_kernel_config(self):
        """Test conversion to legacy KernelConfig."""
        config = TilingConfig(
            block_m=64,
            block_n=48,
            block_k=128,
            num_simd_groups=8,
            shared_memory_bytes=24576,
        )
        kernel_config = config.to_kernel_config()

        assert kernel_config.block_m == 64
        assert kernel_config.block_n == 48
        assert kernel_config.block_k == 128
        assert kernel_config.num_warps == 8
        assert kernel_config.shared_memory == 24576


class TestProblemSizeClassification:
    """Tests for problem size classification."""

    def test_attention_tiny(self):
        """Test attention classification for tiny sequences."""
        size = classify_problem_size((1, 64, 8, 64), OperationType.FLASH_ATTENTION)
        assert size == ProblemSize.TINY

    def test_attention_small(self):
        """Test attention classification for small sequences."""
        size = classify_problem_size((1, 256, 8, 64), OperationType.FLASH_ATTENTION)
        assert size == ProblemSize.SMALL

    def test_attention_medium(self):
        """Test attention classification for medium sequences."""
        size = classify_problem_size((1, 1024, 8, 64), OperationType.FLASH_ATTENTION)
        assert size == ProblemSize.MEDIUM

    def test_attention_large(self):
        """Test attention classification for large sequences."""
        size = classify_problem_size((1, 4096, 8, 64), OperationType.FLASH_ATTENTION)
        assert size == ProblemSize.LARGE

    def test_attention_huge(self):
        """Test attention classification for huge sequences."""
        size = classify_problem_size((1, 16384, 8, 64), OperationType.FLASH_ATTENTION)
        assert size == ProblemSize.HUGE

    def test_matmul_classification(self):
        """Test matmul problem size classification."""
        # Small matmul
        size = classify_problem_size((256, 256), OperationType.MATMUL)
        assert size == ProblemSize.SMALL

        # Large matmul
        size = classify_problem_size((4096, 4096), OperationType.MATMUL)
        assert size == ProblemSize.LARGE

    def test_scan_classification(self):
        """Test scan problem size classification."""
        size = classify_problem_size((4, 512, 256), OperationType.SSM_SCAN)
        assert size == ProblemSize.MEDIUM


class TestDtypeConversion:
    """Tests for dtype conversion utilities."""

    def test_float32_conversion(self):
        """Test float32 dtype conversion."""
        assert dtype_to_enum(mx.float32) == DataType.FP32

    def test_float16_conversion(self):
        """Test float16 dtype conversion."""
        assert dtype_to_enum(mx.float16) == DataType.FP16

    def test_bfloat16_conversion(self):
        """Test bfloat16 dtype conversion."""
        assert dtype_to_enum(mx.bfloat16) == DataType.BF16

    def test_int8_conversion(self):
        """Test int8 dtype conversion."""
        assert dtype_to_enum(mx.int8) == DataType.INT8


class TestTilingDatabase:
    """Tests for TilingDatabase."""

    def test_get_default_config(self):
        """Test retrieving default configurations."""
        db = TilingDatabase()

        config = db.get_config(
            operation=OperationType.FLASH_ATTENTION,
            chip_family=ChipFamily.M4,
            chip_tier=ChipTier.BASE,
            problem_size=ProblemSize.MEDIUM,
            dtype=DataType.FP32,
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        assert config.block_n > 0

    def test_tier_fallback(self):
        """Test tier fallback when exact config not found."""
        db = TilingDatabase()

        # M4 ULTRA might not have explicit config, should fall back
        config = db.get_config(
            operation=OperationType.FLASH_ATTENTION,
            chip_family=ChipFamily.M4,
            chip_tier=ChipTier.ULTRA,  # May not have explicit config
            problem_size=ProblemSize.MEDIUM,
            dtype=DataType.FP32,
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0

    def test_chip_family_fallback(self):
        """Test chip family fallback."""
        db = TilingDatabase()

        # Unknown chip should fall back to known family
        config = db.get_config(
            operation=OperationType.FLASH_ATTENTION,
            chip_family=ChipFamily.UNKNOWN,
            chip_tier=ChipTier.BASE,
            problem_size=ProblemSize.MEDIUM,
            dtype=DataType.FP32,
        )

        assert isinstance(config, TilingConfig)

    def test_save_and_retrieve_tuned_config(self):
        """Test saving and retrieving user-tuned configs."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db = TilingDatabase(cache_dir=Path(tmpdir))

            # Create custom config
            custom_config = TilingConfig(
                block_m=128,
                block_n=48,
                block_k=64,
            )

            # Save it
            db.save_tuned_config(
                operation=OperationType.FLASH_ATTENTION,
                chip_family=ChipFamily.M4,
                chip_tier=ChipTier.BASE,
                problem_size=ProblemSize.LARGE,
                dtype=DataType.FP32,
                config=custom_config,
            )

            # Retrieve it
            retrieved = db.get_config(
                operation=OperationType.FLASH_ATTENTION,
                chip_family=ChipFamily.M4,
                chip_tier=ChipTier.BASE,
                problem_size=ProblemSize.LARGE,
                dtype=DataType.FP32,
            )

            assert retrieved.block_m == 128
            assert retrieved.block_n == 48


class TestChipDetection:
    """Tests for chip detection."""

    def test_get_chip_info(self):
        """Test that chip info is retrieved."""
        chip_info = get_chip_info()

        assert chip_info.family in ChipFamily
        assert chip_info.tier in ChipTier
        assert chip_info.gpu_cores > 0
        assert chip_info.memory_gb > 0
        assert chip_info.simd_width == 32
        assert chip_info.max_threadgroup_memory == 32768
        assert chip_info.ane_tops > 0
        assert chip_info.memory_bandwidth_gbps > 0

    def test_chip_info_caching(self):
        """Test that chip info is cached."""
        info1 = get_chip_info()
        info2 = get_chip_info()

        # Should be same object due to lru_cache
        assert info1 is info2


class TestGetOptimalConfig:
    """Tests for get_optimal_config API."""

    def test_attention_config(self):
        """Test getting optimal attention config."""
        config = get_optimal_config(
            operation=OperationType.FLASH_ATTENTION,
            problem_shape=(1, 1024, 8, 64),
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        assert config.block_n > 0
        # Shared memory should fit in 32KB
        assert config.shared_memory_bytes <= 32768

    def test_matmul_config(self):
        """Test getting optimal matmul config."""
        config = get_optimal_config(
            operation=OperationType.MATMUL,
            problem_shape=(1024, 1024, 512),
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        assert config.block_n > 0
        assert config.block_k > 0

    def test_scan_config(self):
        """Test getting optimal scan config."""
        config = get_optimal_config(
            operation=OperationType.SSM_SCAN,
            problem_shape=(4, 1024, 256),
        )

        assert isinstance(config, TilingConfig)
        assert config.block_m > 0
        # Scan block should be power of 2
        assert (config.block_m & (config.block_m - 1)) == 0

    def test_config_with_dtype(self):
        """Test config with specific dtype."""
        config_fp32 = get_optimal_config(
            operation=OperationType.INT8_LINEAR,
            problem_shape=(1024, 1024, 512),
            dtype=mx.int8,
        )

        assert isinstance(config_fp32, TilingConfig)
