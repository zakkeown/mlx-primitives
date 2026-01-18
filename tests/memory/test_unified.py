"""Tests for unified memory primitives."""

import gc
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.memory import (
    AccessMode,
    MemoryInfo,
    PingPongBuffer,
    StreamingDataLoader,
    StreamingTensor,
    SyncPoint,
    UnifiedView,
    create_unified_buffer,
    ensure_contiguous,
    estimate_cache_usage,
    get_memory_info,
    parallel_cpu_gpu,
    ping_pong_buffer,
    prefetch_to_gpu,
    recommend_chunk_size,
    streaming_reduce,
    zero_copy_slice,
)


class TestUnifiedView:
    """Tests for UnifiedView class."""

    def test_create_view(self):
        """Test creating a unified view."""
        tensor = mx.random.normal((100, 100))
        mx.eval(tensor)

        view = UnifiedView(tensor)
        assert view.access_mode == AccessMode.SHARED

    def test_as_numpy(self):
        """Test getting NumPy view."""
        tensor = mx.array([[1.0, 2.0], [3.0, 4.0]])
        mx.eval(tensor)

        view = UnifiedView(tensor)
        np_arr = view.as_numpy()

        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (2, 2)
        np.testing.assert_array_almost_equal(np_arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_as_mlx(self):
        """Test getting MLX array back."""
        tensor = mx.random.normal((50, 50))
        mx.eval(tensor)

        view = UnifiedView(tensor)
        mlx_arr = view.as_mlx()

        assert mlx_arr is tensor

    def test_memory_info(self):
        """Test getting memory info."""
        tensor = mx.zeros((100, 100), dtype=mx.float32)
        mx.eval(tensor)

        view = UnifiedView(tensor)
        info = view.memory_info

        assert isinstance(info, MemoryInfo)
        assert info.total_bytes == 100 * 100 * 4
        assert info.dtype_size == 4
        assert info.shape == (100, 100)
        assert info.is_contiguous is True

    def test_sync_to_cpu(self):
        """Test sync to CPU."""
        tensor = mx.random.normal((100, 100))
        view = UnifiedView(tensor)

        # Should not raise
        view.sync_to_cpu()

        # NumPy view should be refreshed
        np_arr = view.as_numpy()
        assert np_arr.shape == (100, 100)

    def test_access_modes(self):
        """Test different access modes."""
        tensor = mx.random.normal((50, 50))
        mx.eval(tensor)

        view_shared = UnifiedView(tensor, AccessMode.SHARED)
        assert view_shared.access_mode == AccessMode.SHARED

        view_gpu = UnifiedView(tensor, AccessMode.GPU_PRIMARY)
        assert view_gpu.access_mode == AccessMode.GPU_PRIMARY


class TestMemoryUtilities:
    """Tests for memory utility functions."""

    def test_create_unified_buffer(self):
        """Test creating a unified buffer."""
        buffer = create_unified_buffer((100, 100), dtype=mx.float32)

        assert buffer.shape == (100, 100)
        assert buffer.dtype == mx.float32

    def test_get_memory_info(self):
        """Test getting memory info for tensor."""
        tensor = mx.zeros((50, 100), dtype=mx.float16)
        mx.eval(tensor)

        info = get_memory_info(tensor)

        assert info.total_bytes == 50 * 100 * 2  # float16 = 2 bytes
        assert info.dtype_size == 2
        assert info.shape == (50, 100)

    def test_zero_copy_slice(self):
        """Test zero-copy slicing."""
        tensor = mx.arange(100).reshape(10, 10)
        mx.eval(tensor)

        sliced = zero_copy_slice(tensor, (slice(2, 5), slice(3, 7)))

        assert sliced.shape == (3, 4)

    def test_ensure_contiguous(self):
        """Test ensuring tensor is contiguous."""
        tensor = mx.random.normal((100, 100))
        mx.eval(tensor)

        contiguous = ensure_contiguous(tensor)
        assert contiguous.shape == tensor.shape


class TestStreamingTensor:
    """Tests for StreamingTensor."""

    def test_create_from_numpy(self):
        """Test creating streaming tensor from NumPy."""
        arr = np.random.randn(1000, 100).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)

            assert tensor.shape == (1000, 100)
            assert tensor.dtype == mx.float32
            assert tensor.size == 100000

            # Clean up references before close
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_getitem(self):
        """Test indexing into streaming tensor."""
        arr = np.arange(1000).reshape(100, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)

            # Get single row
            row = tensor[5]
            assert row.shape == (10,)
            np.testing.assert_array_equal(np.array(row), arr[5])

            # Get slice
            sliced = tensor[10:20]
            assert sliced.shape == (10, 10)
            np.testing.assert_array_equal(np.array(sliced), arr[10:20])

            # Clean up references before close
            del row, sliced
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_iter_chunks(self):
        """Test iterating over chunks."""
        arr = np.arange(1000).reshape(100, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)

            chunks = list(tensor.iter_chunks(chunk_size=25, dim=0))

            assert len(chunks) == 4
            assert chunks[0].shape == (25, 10)
            assert chunks[-1].shape == (25, 10)

            # Clean up references before close
            del chunks
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test using streaming tensor as context manager."""
        arr = np.random.randn(100, 50).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            with StreamingTensor.from_numpy(arr, path) as tensor:
                chunk = tensor[0:10]
                assert chunk.shape == (10, 50)
                del chunk  # Clean up before context exit
                gc.collect()
            # Should be closed after context
        finally:
            Path(path).unlink(missing_ok=True)


class TestStreamingDataLoader:
    """Tests for StreamingDataLoader."""

    def test_basic_iteration(self):
        """Test basic batch iteration."""
        arr = np.arange(1000).reshape(100, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)
            loader = StreamingDataLoader(tensor, batch_size=16)

            assert len(loader) == 7  # ceil(100/16)

            batches = list(loader)
            assert len(batches) == 7
            assert batches[0].shape == (16, 10)
            assert batches[-1].shape == (4, 10)  # Remainder

            # Clean up references before close
            del batches, loader
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_drop_last(self):
        """Test dropping last incomplete batch."""
        arr = np.arange(1000).reshape(100, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)
            loader = StreamingDataLoader(tensor, batch_size=16, drop_last=True)

            batches = list(loader)
            assert len(batches) == 6  # 100 // 16

            for batch in batches:
                assert batch.shape == (16, 10)

            # Clean up references before close
            del batches, loader
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)


class TestStreamingReduce:
    """Tests for streaming_reduce function."""

    def test_streaming_sum(self):
        """Test streaming sum reduction."""
        arr = np.arange(1000).reshape(100, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)

            result = streaming_reduce(tensor, op="sum", axis=0, chunk_size=25)

            expected = mx.array(arr.sum(axis=0))
            np.testing.assert_array_almost_equal(
                np.array(result), np.array(expected), decimal=3
            )

            # Clean up references before close
            del result, expected
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_streaming_mean(self):
        """Test streaming mean reduction."""
        arr = np.ones((100, 10), dtype=np.float32) * 5.0

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            tensor = StreamingTensor.from_numpy(arr, path)

            result = streaming_reduce(tensor, op="mean", axis=0, chunk_size=25)

            # Mean of all 5s should be 5
            np.testing.assert_array_almost_equal(
                np.array(result), np.ones(10) * 5.0, decimal=5
            )

            # Clean up references before close
            del result
            gc.collect()
            tensor.close()
        finally:
            Path(path).unlink(missing_ok=True)


class TestPingPongBuffer:
    """Tests for PingPongBuffer."""

    def test_create_buffer(self):
        """Test creating ping-pong buffer."""
        buffer = PingPongBuffer((64, 128), dtype=mx.float32, count=2)

        assert buffer.count == 2
        assert buffer.shape == (64, 128)

    def test_ping_pong_factory(self):
        """Test ping_pong_buffer factory function."""
        buffer = ping_pong_buffer((100, 100), count=3)

        assert buffer.count == 3
        assert buffer.shape == (100, 100)

    def test_get_for_write(self):
        """Test getting buffer for writing."""
        buffer = PingPongBuffer((64, 64), count=2)

        idx, arr = buffer.get_for_write()

        assert idx == 0
        assert arr.shape == (64, 64)

    def test_advance_write(self):
        """Test advancing write position."""
        buffer = PingPongBuffer((64, 64), count=2)

        idx1, _ = buffer.get_for_write()
        buffer.advance_write()
        idx2, _ = buffer.get_for_write()

        assert idx1 == 0
        assert idx2 == 1

    def test_wrap_around(self):
        """Test buffer position wrap-around."""
        buffer = PingPongBuffer((64, 64), count=2)

        buffer.advance_write()  # 0 -> 1
        buffer.advance_write()  # 1 -> 0 (wrap)

        idx, _ = buffer.get_for_write()
        assert idx == 0

    def test_is_full_empty(self):
        """Test full/empty status."""
        buffer = PingPongBuffer((64, 64), count=2)

        assert buffer.is_empty() is True
        assert buffer.is_full() is False

        buffer.advance_write()
        buffer.advance_write()

        assert buffer.is_full() is True


class TestSyncPoint:
    """Tests for SyncPoint."""

    def test_create_sync_point(self):
        """Test creating sync point."""
        sync = SyncPoint()
        assert sync.is_complete is True

    def test_mark_and_wait(self):
        """Test marking and waiting for completion."""
        sync = SyncPoint()
        tensor = mx.random.normal((100, 100))

        sync.mark_gpu_complete([tensor])
        success = sync.wait_gpu_complete(timeout=5.0)

        assert success is True
        assert sync.is_complete is True


class TestParallelCpuGpu:
    """Tests for parallel_cpu_gpu function."""

    def test_parallel_execution(self):
        """Test parallel CPU and GPU execution."""

        def cpu_work():
            return sum(range(1000))

        def gpu_work():
            return mx.sum(mx.arange(100))

        cpu_result, gpu_result = parallel_cpu_gpu(cpu_work, gpu_work)

        assert cpu_result == sum(range(1000))
        assert float(gpu_result) == sum(range(100))


class TestCacheEstimation:
    """Tests for cache estimation utilities."""

    def test_estimate_cache_usage(self):
        """Test cache usage estimation."""
        tensors = [
            mx.zeros((1000, 1000), dtype=mx.float32),
            mx.zeros((500, 500), dtype=mx.float32),
        ]
        mx.eval(*tensors)

        estimate = estimate_cache_usage(tensors)

        expected_bytes = 1000 * 1000 * 4 + 500 * 500 * 4
        assert estimate.total_bytes == expected_bytes
        assert isinstance(estimate.fits_in_l2, bool)

    def test_recommend_chunk_size(self):
        """Test chunk size recommendation."""
        chunk_size = recommend_chunk_size(
            total_elements=10_000_000,
            dtype=mx.float32,
            target_cache_fraction=0.5,
        )

        assert chunk_size > 0
        # Should be power of 2
        assert (chunk_size & (chunk_size - 1)) == 0


class TestPrefetch:
    """Tests for prefetch functions."""

    def test_prefetch_to_gpu(self):
        """Test prefetch to GPU."""
        tensor = mx.random.normal((100, 100))

        # Should not raise
        prefetch_to_gpu(tensor)

        # Tensor should be evaluated
        assert tensor.shape == (100, 100)
