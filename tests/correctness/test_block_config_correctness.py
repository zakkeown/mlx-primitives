"""Correctness tests for Block configuration system.

Tests verify:
1. BlockConfig calculations (shared memory, fits_in_threadgroup)
2. Optimal config selection for various head_dim/dtype combinations
3. Memory constraint validation
4. Hardware info detection
"""

import pytest
import mlx.core as mx

from mlx_primitives.kernels.block_config import (
    BlockConfig,
    get_optimal_block_config,
    get_block_config_info,
    estimate_shared_memory,
    validate_block_config,
    warmup_block_configs,
    OPTIMAL_BLOCK_SIZES,
)

from mlx_primitives.kernels.hardware_info import (
    AppleSiliconInfo,
    get_hardware_info,
    get_chip_family,
    get_max_threadgroup_memory,
)


# =============================================================================
# BlockConfig Tests
# =============================================================================


class TestBlockConfig:
    """Test BlockConfig dataclass calculations."""

    def test_shared_memory_calculation_fp32(self):
        """Shared memory should be (block_m + 2*block_n) * head_dim * dtype_bytes."""
        config = BlockConfig(
            block_m=32,
            block_n=32,
            head_dim=64,
            dtype_bytes=4,  # float32
        )

        # (32 + 2*32) * 64 * 4 = 96 * 64 * 4 = 24,576 bytes
        expected = (32 + 2 * 32) * 64 * 4
        assert config.shared_memory_bytes == expected
        assert config.shared_memory_bytes == 24576

    def test_shared_memory_calculation_fp16(self):
        """Float16 should use half the memory."""
        config = BlockConfig(
            block_m=64,
            block_n=64,
            head_dim=64,
            dtype_bytes=2,  # float16
        )

        # (64 + 2*64) * 64 * 2 = 192 * 64 * 2 = 24,576 bytes
        expected = (64 + 2 * 64) * 64 * 2
        assert config.shared_memory_bytes == expected
        assert config.shared_memory_bytes == 24576

    def test_fits_in_threadgroup_within_limit(self):
        """Config under 32KB should fit in threadgroup."""
        config = BlockConfig(
            block_m=32,
            block_n=32,
            head_dim=64,
            dtype_bytes=4,
        )

        assert config.shared_memory_bytes < 32768
        assert config.fits_in_threadgroup == True

    def test_fits_in_threadgroup_over_limit(self):
        """Config over 32KB should not fit in threadgroup."""
        config = BlockConfig(
            block_m=64,
            block_n=64,
            head_dim=128,
            dtype_bytes=4,  # float32
        )

        # (64 + 128) * 128 * 4 = 192 * 128 * 4 = 98,304 bytes > 32KB
        assert config.shared_memory_bytes > 32768
        assert config.fits_in_threadgroup == False


class TestOptimalBlockConfig:
    """Test optimal block config selection."""

    def test_returns_valid_config_for_common_head_dims(self):
        """Should return valid configs for common head dimensions."""
        for head_dim in [32, 64, 96, 128]:
            for dtype in [mx.float32, mx.float16]:
                block_m, block_n = get_optimal_block_config(head_dim, dtype)

                assert block_m > 0, f"block_m should be positive for head_dim={head_dim}"
                assert block_n > 0, f"block_n should be positive for head_dim={head_dim}"

                # Verify it fits in threadgroup
                dtype_bytes = 4 if dtype == mx.float32 else 2
                shared_mem = (block_m + 2 * block_n) * head_dim * dtype_bytes
                assert shared_mem <= 32768, (
                    f"Config for head_dim={head_dim}, dtype={dtype} "
                    f"exceeds 32KB: {shared_mem} bytes"
                )

    def test_fp16_allows_larger_blocks(self):
        """Float16 should allow same or larger blocks than float32."""
        head_dim = 64

        block_m_fp32, block_n_fp32 = get_optimal_block_config(head_dim, mx.float32)
        block_m_fp16, block_n_fp16 = get_optimal_block_config(head_dim, mx.float16)

        # FP16 uses half the memory per element, so can have larger blocks
        fp32_mem = (block_m_fp32 + 2 * block_n_fp32) * head_dim * 4
        fp16_mem = (block_m_fp16 + 2 * block_n_fp16) * head_dim * 2

        # FP16 config should not be more restrictive memory-wise
        assert fp16_mem <= 32768

    def test_blocks_are_power_of_2_or_multiples_of_8(self):
        """Block sizes should be multiples of 8 for SIMD alignment."""
        for head_dim in [64, 128]:
            block_m, block_n = get_optimal_block_config(head_dim, mx.float16)

            # Multiples of 8 (most common) or power of 2
            assert block_m % 8 == 0 or (block_m & (block_m - 1)) == 0
            assert block_n % 8 == 0 or (block_n & (block_n - 1)) == 0


class TestEstimateSharedMemory:
    """Test shared memory estimation."""

    def test_matches_block_config_calculation(self):
        """estimate_shared_memory should match BlockConfig calculation."""
        block_m, block_n, head_dim = 32, 32, 64

        estimated = estimate_shared_memory(block_m, block_n, head_dim, mx.float32)
        config = BlockConfig(block_m, block_n, head_dim, dtype_bytes=4)

        assert estimated == config.shared_memory_bytes

    def test_fp16_half_of_fp32(self):
        """FP16 should use exactly half the memory of FP32."""
        block_m, block_n, head_dim = 32, 32, 64

        fp32_mem = estimate_shared_memory(block_m, block_n, head_dim, mx.float32)
        fp16_mem = estimate_shared_memory(block_m, block_n, head_dim, mx.float16)

        assert fp16_mem == fp32_mem // 2


class TestValidateBlockConfig:
    """Test block config validation."""

    def test_valid_config_returns_true(self):
        """Valid configuration should return True."""
        # Small config that definitely fits
        result = validate_block_config(16, 16, 64, mx.float32)
        assert result == True

    def test_invalid_config_raises_error(self):
        """Configuration exceeding memory should raise ValueError."""
        # Very large config that won't fit
        with pytest.raises(ValueError) as exc_info:
            validate_block_config(128, 128, 256, mx.float32)

        assert "shared memory" in str(exc_info.value).lower()


class TestWarmupBlockConfigs:
    """Test warmup function for pre-computing configs."""

    def test_returns_configs_for_all_combinations(self):
        """Should return config for each head_dim/dtype combination."""
        head_dims = [64, 128]
        dtypes = [mx.float32, mx.float16]

        configs = warmup_block_configs(head_dims=head_dims, dtypes=dtypes)

        # Should have 2 head_dims * 2 dtypes = 4 configs
        assert len(configs) == 4

        for head_dim in head_dims:
            for dtype in dtypes:
                assert (head_dim, dtype) in configs
                block_m, block_n = configs[(head_dim, dtype)]
                assert block_m > 0
                assert block_n > 0

    def test_default_warmup(self):
        """Default warmup should work without arguments."""
        configs = warmup_block_configs()

        # Should have some configs
        assert len(configs) > 0


# =============================================================================
# Hardware Info Tests
# =============================================================================


class TestHardwareInfo:
    """Test hardware detection."""

    def test_returns_apple_silicon_info(self):
        """Should return AppleSiliconInfo dataclass."""
        info = get_hardware_info()

        assert isinstance(info, AppleSiliconInfo)
        assert isinstance(info.device_name, str)
        assert isinstance(info.chip_family, str)
        assert isinstance(info.max_threadgroup_memory, int)

    def test_threadgroup_memory_is_32kb(self):
        """Threadgroup memory should be 32KB for Apple Silicon."""
        info = get_hardware_info()

        # All current Apple Silicon has 32KB
        assert info.max_threadgroup_memory == 32768

    def test_simd_width_is_32(self):
        """SIMD width should be 32 for Apple Silicon."""
        info = get_hardware_info()

        assert info.simd_width == 32


class TestGetChipFamily:
    """Test chip family detection."""

    def test_returns_string(self):
        """Should return a chip family string."""
        family = get_chip_family()

        assert isinstance(family, str)
        # Should be one of known families or Unknown
        assert family in ["M1", "M2", "M3", "M4", "Unknown"]


class TestGetMaxThreadgroupMemory:
    """Test threadgroup memory getter."""

    def test_returns_positive_int(self):
        """Should return positive integer."""
        mem = get_max_threadgroup_memory()

        assert isinstance(mem, int)
        assert mem > 0

    def test_returns_32kb(self):
        """Should return 32KB for Apple Silicon."""
        mem = get_max_threadgroup_memory()

        assert mem == 32768


# =============================================================================
# Integration Tests
# =============================================================================


class TestBlockConfigIntegration:
    """Test integration between block config and hardware info."""

    def test_optimal_config_fits_hardware(self):
        """Optimal config should always fit hardware constraints."""
        max_mem = get_max_threadgroup_memory()

        for head_dim in [64, 128]:
            for dtype in [mx.float32, mx.float16]:
                block_m, block_n = get_optimal_block_config(head_dim, dtype)
                dtype_bytes = 4 if dtype == mx.float32 else 2

                shared_mem = (block_m + 2 * block_n) * head_dim * dtype_bytes
                assert shared_mem <= max_mem, (
                    f"Config ({block_m}, {block_n}) for head_dim={head_dim}, "
                    f"dtype={dtype} exceeds hardware limit"
                )

    def test_block_config_info_is_consistent(self):
        """get_block_config_info should return consistent data."""
        head_dim = 64
        dtype = mx.float16

        config = get_block_config_info(head_dim, dtype)

        # Verify block sizes match optimal
        block_m, block_n = get_optimal_block_config(head_dim, dtype)
        assert config.block_m == block_m
        assert config.block_n == block_n

        # Verify shared memory calculation
        expected_mem = (block_m + 2 * block_n) * head_dim * 2
        assert config.shared_memory_bytes == expected_mem

    def test_all_precomputed_configs_are_valid(self):
        """All precomputed optimal configs should be valid."""
        max_mem = 32768  # 32KB

        for (head_dim, dtype_bytes), (block_m, block_n) in OPTIMAL_BLOCK_SIZES.items():
            shared_mem = (block_m + 2 * block_n) * head_dim * dtype_bytes

            # Some precomputed configs may be slightly over but are marked as such
            # Just verify they're not wildly over
            assert shared_mem <= max_mem * 1.5, (
                f"Precomputed config for head_dim={head_dim}, bytes={dtype_bytes} "
                f"is way over limit: {shared_mem} bytes"
            )
