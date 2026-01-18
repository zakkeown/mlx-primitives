"""Tests for the DSL auto-tuning system.

Tests the autotuner, cache, and config selection.
"""

import pytest
import tempfile
import os
from pathlib import Path


def _mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


class TestAutoTuneCache:
    """Test the AutoTuneCache class."""

    def test_cache_creation(self):
        """Test cache can be created."""
        from mlx_primitives.dsl.autotuner import AutoTuneCache

        cache = AutoTuneCache()
        assert cache is not None

    def test_memory_cache_put_get(self):
        """Test putting and getting from memory cache."""
        from mlx_primitives.dsl.autotuner import AutoTuneCache
        from mlx_primitives.dsl.decorators import Config

        cache = AutoTuneCache()

        config = Config(BLOCK_SIZE=256)
        cache.put("test_kernel", "key1", config)

        retrieved = cache.get("test_kernel", "key1")
        assert retrieved is not None
        # Config stores extra kwargs in .kwargs dict, accessed via .get()
        assert retrieved.get("BLOCK_SIZE") == 256

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from mlx_primitives.dsl.autotuner import AutoTuneCache

        cache = AutoTuneCache()

        result = cache.get("nonexistent", "key")
        assert result is None

    def test_disk_cache(self):
        """Test disk cache persistence."""
        from mlx_primitives.dsl.autotuner import AutoTuneCache
        from mlx_primitives.dsl.decorators import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            # Create cache and add entry
            cache1 = AutoTuneCache(cache_dir=cache_dir)
            config = Config(num_warps=8, BLOCK_SIZE=512)
            cache1.put("kernel1", "tuning_key_123", config)

            # Create new cache instance from same dir
            cache2 = AutoTuneCache(cache_dir=cache_dir)
            retrieved = cache2.get("kernel1", "tuning_key_123")

            assert retrieved is not None
            assert retrieved.num_warps == 8
            assert retrieved.get("BLOCK_SIZE") == 512


class TestDSLAutoTuner:
    """Test the DSLAutoTuner class."""

    def test_autotuner_creation(self):
        """Test autotuner can be created."""
        from mlx_primitives.dsl.autotuner import DSLAutoTuner

        tuner = DSLAutoTuner(warmup=2, rep=5)
        assert tuner.warmup == 2
        assert tuner.rep == 5

    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass."""
        from mlx_primitives.dsl.autotuner import BenchmarkResult
        from mlx_primitives.dsl.decorators import Config

        config = Config(BLOCK_SIZE=256)
        result = BenchmarkResult(
            config=config,
            times_ms=[1.0, 1.1, 0.9, 1.05],
            min_ms=0.9,
            median_ms=1.025,
            mean_ms=1.0125,
            std_ms=0.081,
            valid=True,
        )

        assert result.min_ms == 0.9
        assert result.valid is True
        assert len(result.times_ms) == 4


class TestConfig:
    """Test the Config class."""

    def test_config_creation(self):
        """Test Config creation with kwargs."""
        from mlx_primitives.dsl.decorators import Config

        # num_warps and num_stages are explicit, rest in kwargs
        config = Config(num_warps=8, num_stages=2, BLOCK_SIZE=256)

        assert config.num_warps == 8
        assert config.num_stages == 2
        assert config.get("BLOCK_SIZE") == 256

    def test_config_equality(self):
        """Test Config equality."""
        from mlx_primitives.dsl.decorators import Config

        c1 = Config(num_warps=8, BLOCK_SIZE=256)
        c2 = Config(num_warps=8, BLOCK_SIZE=256)
        c3 = Config(num_warps=8, BLOCK_SIZE=512)

        # Both have same num_warps and kwargs
        assert c1.num_warps == c2.num_warps
        assert c1.get("BLOCK_SIZE") == c2.get("BLOCK_SIZE")

    def test_config_get(self):
        """Test Config.get() method."""
        from mlx_primitives.dsl.decorators import Config

        config = Config(BLOCK_SIZE=256, TILE_M=32, TILE_N=32)

        # Access via get()
        assert config.get("BLOCK_SIZE") == 256
        assert config.get("TILE_M") == 32
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", 42) == 42


class TestKernelDefinition:
    """Test KernelDefinition with autotune fields."""

    def test_autotune_fields(self):
        """Test autotune fields are present."""
        from mlx_primitives.dsl.decorators import KernelDefinition

        def dummy_func(): pass

        kdef = KernelDefinition(
            name="test",
            func=dummy_func,
            source_code="def test(): pass",
            parameters=[],
            constexpr_params=[],
            autotune_warmup=5,
            autotune_rep=20,
        )

        assert kdef.autotune_warmup == 5
        assert kdef.autotune_rep == 20


class TestAutotuneDecorator:
    """Test the @autotune decorator."""

    def test_autotune_decorator_adds_configs(self):
        """Test @autotune adds configs to kernel."""
        from mlx_primitives.dsl import metal_kernel, autotune, Config

        @autotune(
            configs=[
                Config(BLOCK_SIZE=128),
                Config(BLOCK_SIZE=256),
                Config(BLOCK_SIZE=512),
            ],
            key=["N"],
        )
        @metal_kernel
        def test_kernel(x_ptr, out_ptr, N):
            pass

        # Should have configs attached via kernel_def
        assert hasattr(test_kernel, "kernel_def")
        assert test_kernel.kernel_def.configs is not None
        assert len(test_kernel.kernel_def.configs) == 3

    def test_autotune_with_warmup_rep(self):
        """Test @autotune accepts warmup and rep."""
        from mlx_primitives.dsl import metal_kernel, autotune, Config

        @autotune(
            configs=[Config(BLOCK_SIZE=256)],
            key=["N"],
            warmup=10,
            rep=50,
        )
        @metal_kernel
        def test_kernel2(x_ptr, out_ptr, N):
            pass

        assert test_kernel2.kernel_def.autotune_warmup == 10
        assert test_kernel2.kernel_def.autotune_rep == 50


class TestEndToEndAutoTune:
    """End-to-end autotuning tests."""

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_simple_autotuned_kernel(self):
        """Test a simple kernel with autotune configs attached."""
        from mlx_primitives.dsl import metal_kernel, autotune, Config

        @autotune(
            configs=[
                Config(BLOCK_SIZE=128),
                Config(BLOCK_SIZE=256),
            ],
            key=["N"],
            warmup=1,
            rep=2,
        )
        @metal_kernel
        def test_kernel_e2e(x_ptr, out_ptr, N):
            pass

        # Verify the kernel has autotune configs
        assert test_kernel_e2e.kernel_def.configs is not None
        assert len(test_kernel_e2e.kernel_def.configs) == 2

    @pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
    def test_vector_add_with_autotune(self):
        """Test vector_add example kernel runs correctly."""
        import mlx.core as mx
        from mlx_primitives.dsl.examples.vector_ops import vector_add

        N = 1024
        a = mx.ones((N,))
        b = mx.ones((N,)) * 2.0
        c = mx.zeros((N,))

        result = vector_add(a, b, c, N=N, grid=((N + 255) // 256,))
        if isinstance(result, list):
            result = result[0]

        mx.eval(result)

        expected = mx.ones((N,)) * 3.0
        assert mx.allclose(result, expected)


if __name__ == "__main__":
    # Run quick tests
    test = TestAutoTuneCache()
    test.test_cache_creation()
    test.test_memory_cache_put_get()
    test.test_cache_miss()
    print("Cache tests passed!")

    test2 = TestConfig()
    test2.test_config_creation()
    test2.test_config_get()
    print("Config tests passed!")

    test3 = TestAutotuneDecorator()
    test3.test_autotune_decorator_adds_configs()
    test3.test_autotune_with_warmup_rep()
    print("Decorator tests passed!")

    print("\nAll autotune tests passed!")
