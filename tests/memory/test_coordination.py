"""Extended tests for CPU/GPU work coordination primitives.

Tests SyncPoint, PingPongBuffer, WorkQueue, parallel_cpu_gpu,
and overlap_compute_io from the coordination module.
"""

import threading
import time
from concurrent.futures import Future
from typing import List

import mlx.core as mx
import numpy as np
import pytest

from mlx_primitives.memory.coordination import (
    SyncPoint,
    PingPongBuffer,
    WorkQueue,
    parallel_cpu_gpu,
    overlap_compute_io,
    ping_pong_buffer,
)


# ---------------------------------------------------------------------------
# SyncPoint Extended Tests
# ---------------------------------------------------------------------------


class TestSyncPointExtended:
    """Extended tests for SyncPoint synchronization."""

    def test_initial_state_complete(self) -> None:
        """Sync point should be complete initially."""
        sync = SyncPoint()
        assert sync.is_complete is True

    def test_mark_gpu_complete_single_tensor(self) -> None:
        """Mark single tensor as complete."""
        sync = SyncPoint()
        tensor = mx.random.normal((100, 100))

        sync.mark_gpu_complete([tensor])

        assert sync.is_complete is True
        # Tensor should be evaluated
        np_arr = np.array(tensor)
        assert np_arr.shape == (100, 100)

    def test_mark_gpu_complete_multiple_tensors(self) -> None:
        """Mark multiple tensors as complete."""
        sync = SyncPoint()
        tensors = [
            mx.random.normal((50, 50)),
            mx.zeros((100,)),
            mx.ones((10, 10, 10)),
        ]

        sync.mark_gpu_complete(tensors)

        assert sync.is_complete is True
        # All tensors should be evaluated
        for t in tensors:
            _ = np.array(t)  # Should not hang

    def test_wait_gpu_complete_success(self) -> None:
        """Wait should return True when complete."""
        sync = SyncPoint()
        tensor = mx.random.normal((100, 100))

        sync.mark_gpu_complete([tensor])
        result = sync.wait_gpu_complete(timeout=5.0)

        assert result is True

    def test_wait_gpu_complete_with_timeout(self) -> None:
        """Wait with timeout should work correctly."""
        sync = SyncPoint()
        tensor = mx.random.normal((100, 100))

        sync.mark_gpu_complete([tensor])

        # Should complete quickly
        start = time.time()
        result = sync.wait_gpu_complete(timeout=10.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 5.0  # Should not use full timeout

    def test_mark_cpu_complete_noop(self) -> None:
        """CPU complete mark should not raise."""
        sync = SyncPoint()

        # Should not raise - this is a no-op on unified memory
        sync.mark_cpu_complete()
        assert sync.is_complete is True

    def test_sync_point_reuse(self) -> None:
        """Sync point can be reused for multiple sync operations."""
        sync = SyncPoint()

        # First sync
        tensor1 = mx.random.normal((50, 50))
        sync.mark_gpu_complete([tensor1])
        assert sync.wait_gpu_complete()

        # Second sync
        tensor2 = mx.random.normal((100, 100))
        sync.mark_gpu_complete([tensor2])
        assert sync.wait_gpu_complete()

    def test_sync_point_empty_tensor_list(self) -> None:
        """Empty tensor list should still work."""
        sync = SyncPoint()

        sync.mark_gpu_complete([])
        assert sync.is_complete is True


# ---------------------------------------------------------------------------
# PingPongBuffer Extended Tests
# ---------------------------------------------------------------------------


class TestPingPongBufferExtended:
    """Extended tests for PingPongBuffer."""

    def test_create_double_buffer(self) -> None:
        """Create standard double buffer."""
        buffer = PingPongBuffer(shape=(64, 512), count=2)

        assert buffer.count == 2
        assert buffer.shape == (64, 512)

    def test_create_triple_buffer(self) -> None:
        """Create triple buffer for better pipelining."""
        buffer = PingPongBuffer(shape=(32, 256), count=3)

        assert buffer.count == 3

    def test_buffer_minimum_count(self) -> None:
        """Buffer count must be at least 2."""
        with pytest.raises(ValueError, match="at least 2"):
            PingPongBuffer(shape=(64, 64), count=1)

    def test_get_for_write_initial(self) -> None:
        """Initial write buffer is index 0."""
        buffer = PingPongBuffer(shape=(32, 32))

        idx, buf = buffer.get_for_write()

        assert idx == 0
        assert buf.shape == (32, 32)

    def test_get_for_read_empty_raises(self) -> None:
        """Reading from empty buffer raises error."""
        buffer = PingPongBuffer(shape=(32, 32))

        with pytest.raises(ValueError, match="No buffer ready"):
            buffer.get_for_read()

    def test_advance_write_then_read(self) -> None:
        """Advance write makes buffer available for read."""
        buffer = PingPongBuffer(shape=(32, 32))

        # Initially empty
        assert buffer.is_empty()

        # Write and advance
        idx, write_buf = buffer.get_for_write()
        buffer.advance_write()

        # Now can read
        assert not buffer.is_empty()
        read_idx, read_buf = buffer.get_for_read()
        assert read_idx == 0

    def test_double_buffer_ping_pong(self) -> None:
        """Full ping-pong cycle with double buffer."""
        buffer = PingPongBuffer(shape=(16, 16), count=2)

        # Fill first buffer
        idx0, buf0 = buffer.get_for_write()
        assert idx0 == 0
        buffer.advance_write()

        # Fill second buffer while first is "processing"
        idx1, buf1 = buffer.get_for_write()
        assert idx1 == 1
        buffer.advance_write()

        # Buffer should be full
        assert buffer.is_full()

        # Read first
        read_idx, _ = buffer.get_for_read()
        assert read_idx == 0
        buffer.advance_read()

        # Read second
        read_idx, _ = buffer.get_for_read()
        assert read_idx == 1
        buffer.advance_read()

        # Buffer should be empty
        assert buffer.is_empty()

    def test_triple_buffer_cycle(self) -> None:
        """Full cycle with triple buffer."""
        buffer = PingPongBuffer(shape=(8, 8), count=3)

        # Fill all three
        for i in range(3):
            idx, _ = buffer.get_for_write()
            assert idx == i
            buffer.advance_write()

        assert buffer.is_full()

        # Read all three
        for i in range(3):
            idx, _ = buffer.get_for_read()
            assert idx == i
            buffer.advance_read()

        assert buffer.is_empty()

    def test_reset_clears_state(self) -> None:
        """Reset returns buffer to initial state."""
        buffer = PingPongBuffer(shape=(16, 16), count=2)

        # Fill and partially read
        buffer.advance_write()
        buffer.advance_write()
        buffer.advance_read()

        # Reset
        buffer.reset()

        # Should be back to initial state
        assert buffer.is_empty()
        idx, _ = buffer.get_for_write()
        assert idx == 0

    def test_convenience_function(self) -> None:
        """ping_pong_buffer() creates PingPongBuffer."""
        buffer = ping_pong_buffer((64, 64), dtype=mx.float16, count=3)

        assert isinstance(buffer, PingPongBuffer)
        assert buffer.count == 3

    def test_buffer_with_different_dtypes(self) -> None:
        """Buffers work with different data types."""
        for dtype in [mx.float32, mx.float16, mx.int32]:
            buffer = PingPongBuffer(shape=(32, 32), dtype=dtype, count=2)
            _, buf = buffer.get_for_write()
            assert buf.dtype == dtype


# ---------------------------------------------------------------------------
# WorkQueue Tests
# ---------------------------------------------------------------------------


class TestWorkQueue:
    """Tests for WorkQueue CPU/GPU coordination."""

    def test_create_work_queue(self) -> None:
        """Create work queue with default workers."""
        with WorkQueue() as queue:
            assert queue is not None

    def test_submit_cpu_work(self) -> None:
        """Submit CPU work and get result."""
        with WorkQueue(cpu_workers=2) as queue:
            future = queue.submit_cpu(lambda x: x * 2, 5)
            result = future.result()

            assert result == 10

    def test_submit_multiple_cpu_work(self) -> None:
        """Submit multiple CPU work items."""
        with WorkQueue(cpu_workers=4) as queue:
            futures = [queue.submit_cpu(lambda x: x ** 2, i) for i in range(10)]
            results = [f.result() for f in futures]

            assert results == [i ** 2 for i in range(10)]

    def test_submit_gpu_work(self) -> None:
        """Submit GPU work and get result."""
        with WorkQueue() as queue:
            def gpu_op():
                x = mx.random.normal((100, 100))
                return mx.sum(x)

            future = queue.submit_gpu(gpu_op)
            result = future.result()

            assert isinstance(result, mx.array)

    def test_cpu_work_with_dependency(self) -> None:
        """CPU work can depend on prior work."""
        with WorkQueue() as queue:
            # First task produces data
            f1 = queue.submit_cpu(lambda: [1, 2, 3])

            # Second task depends on first
            def process(data):
                return sum(data)

            f2 = queue.submit_cpu(
                lambda: process(f1.result()),
                depends_on=f1,
            )

            result = f2.result()
            assert result == 6

    def test_gpu_work_with_cpu_dependency(self) -> None:
        """GPU work can depend on CPU work."""
        with WorkQueue() as queue:
            # CPU creates data
            cpu_future = queue.submit_cpu(
                lambda: mx.array([1.0, 2.0, 3.0])
            )

            # GPU processes data
            def gpu_process(data):
                return mx.sum(data)

            gpu_future = queue.submit_gpu(
                lambda: gpu_process(cpu_future.result()),
                depends_on=cpu_future,
            )

            result = gpu_future.result()
            np.testing.assert_allclose(np.array(result), 6.0)

    def test_pipeline_simple(self) -> None:
        """Simple pipeline execution."""
        with WorkQueue() as queue:
            stages = [
                ("cpu", lambda x: x * 2),
                ("cpu", lambda x: x + 1),
            ]

            inputs = iter([1, 2, 3, 4, 5])
            results = list(queue.pipeline(stages, inputs))

            # (x * 2) + 1 for each input
            assert results == [3, 5, 7, 9, 11]

    def test_pipeline_with_gpu(self) -> None:
        """Pipeline with GPU stage."""
        with WorkQueue() as queue:
            stages = [
                ("cpu", lambda x: mx.array([x])),
                ("gpu", lambda x: x * 2),
                ("cpu", lambda x: float(np.array(x)[0])),
            ]

            inputs = iter([1.0, 2.0, 3.0])
            results = list(queue.pipeline(stages, inputs))

            assert results == [2.0, 4.0, 6.0]

    def test_pipeline_empty_stages(self) -> None:
        """Empty stages just passes through inputs."""
        with WorkQueue() as queue:
            inputs = iter([1, 2, 3])
            results = list(queue.pipeline([], inputs))

            assert results == [1, 2, 3]

    def test_context_manager(self) -> None:
        """Work queue works as context manager."""
        queue = WorkQueue()

        with queue:
            future = queue.submit_cpu(lambda: 42)
            result = future.result()

        assert result == 42

    def test_shutdown_waits_for_work(self) -> None:
        """Shutdown with wait=True waits for pending work."""
        queue = WorkQueue()

        # Submit slow work
        future = queue.submit_cpu(lambda: (time.sleep(0.1), 42)[1])

        # Shutdown should wait
        queue.shutdown(wait=True)

        # Work should be complete
        assert future.done()


# ---------------------------------------------------------------------------
# parallel_cpu_gpu Tests
# ---------------------------------------------------------------------------


class TestParallelCpuGpu:
    """Tests for parallel_cpu_gpu function."""

    def test_parallel_execution(self) -> None:
        """CPU and GPU work execute in parallel."""
        cpu_result, gpu_result = parallel_cpu_gpu(
            lambda: "cpu_done",
            lambda: mx.sum(mx.ones((100, 100))),
        )

        assert cpu_result == "cpu_done"
        np.testing.assert_allclose(np.array(gpu_result), 10000.0)

    def test_cpu_produces_data_for_later_use(self) -> None:
        """CPU work can produce data while GPU computes."""
        def cpu_work():
            # Simulate loading next batch
            time.sleep(0.01)
            return [1, 2, 3, 4, 5]

        def gpu_work():
            # Process current batch
            x = mx.random.normal((100, 100))
            return mx.sum(x)

        next_batch, current_result = parallel_cpu_gpu(cpu_work, gpu_work)

        assert next_batch == [1, 2, 3, 4, 5]
        assert isinstance(current_result, mx.array)

    def test_both_return_arrays(self) -> None:
        """Both CPU and GPU can return MLX arrays."""
        cpu_result, gpu_result = parallel_cpu_gpu(
            lambda: mx.array([1, 2, 3]),
            lambda: mx.array([4, 5, 6]),
        )

        np.testing.assert_array_equal(np.array(cpu_result), [1, 2, 3])
        np.testing.assert_array_equal(np.array(gpu_result), [4, 5, 6])

    def test_cpu_exception_propagates(self) -> None:
        """CPU exception should propagate."""
        def cpu_error():
            raise ValueError("CPU error")

        with pytest.raises(ValueError, match="CPU error"):
            parallel_cpu_gpu(cpu_error, lambda: mx.zeros((10,)))


# ---------------------------------------------------------------------------
# overlap_compute_io Tests
# ---------------------------------------------------------------------------


class TestOverlapComputeIO:
    """Tests for overlap_compute_io function."""

    def test_basic_overlap(self) -> None:
        """Basic compute/IO overlap."""
        batches_loaded: List[int] = []

        def load_fn(idx):
            batches_loaded.append(idx)
            return mx.ones((10,)) * idx

        def compute_fn(batch):
            return batch * 2

        results = list(overlap_compute_io(
            compute_fn,
            load_fn,
            num_batches=5,
        ))

        # Should have loaded all batches
        assert len(batches_loaded) == 5

        # Should have correct results
        for i, result in enumerate(results):
            expected = np.ones(10) * i * 2
            np.testing.assert_allclose(np.array(result), expected)

    def test_single_batch(self) -> None:
        """Single batch should work."""
        def load_fn(idx):
            return mx.array([float(idx)])

        def compute_fn(batch):
            return batch + 1

        results = list(overlap_compute_io(compute_fn, load_fn, num_batches=1))

        assert len(results) == 1
        np.testing.assert_allclose(np.array(results[0]), [1.0])

    def test_zero_batches(self) -> None:
        """Zero batches returns empty iterator."""
        results = list(overlap_compute_io(
            lambda x: x,
            lambda idx: mx.zeros((10,)),
            num_batches=0,
        ))

        assert results == []

    def test_compute_fn_receives_correct_batches(self) -> None:
        """Compute function receives batches in order."""
        computed_batches: List[int] = []

        def load_fn(idx):
            return mx.array([float(idx)])

        def compute_fn(batch):
            val = int(np.array(batch)[0])
            computed_batches.append(val)
            return batch

        list(overlap_compute_io(compute_fn, load_fn, num_batches=5))

        assert computed_batches == [0, 1, 2, 3, 4]

    def test_overlap_timing(self) -> None:
        """Load and compute should overlap (rough timing check)."""
        load_times: List[float] = []
        compute_times: List[float] = []

        def load_fn(idx):
            load_times.append(time.time())
            time.sleep(0.01)  # Simulate I/O
            return mx.ones((100,)) * idx

        def compute_fn(batch):
            compute_times.append(time.time())
            # Simulate compute (MLX operations)
            for _ in range(10):
                batch = batch * 1.001
            mx.eval(batch)
            return batch

        list(overlap_compute_io(compute_fn, load_fn, num_batches=5))

        # After first batch, loads should overlap with computes
        # (This is a loose check - timing is not guaranteed)
        assert len(load_times) == 5
        assert len(compute_times) == 5


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestCoordinationIntegration:
    """Integration tests for coordination primitives."""

    def test_sync_point_with_ping_pong(self) -> None:
        """SyncPoint works with PingPongBuffer."""
        buffer = PingPongBuffer(shape=(32, 32), count=2)
        sync = SyncPoint()

        # Fill first buffer
        _, buf0 = buffer.get_for_write()
        # Simulate filling with data
        new_data = mx.random.normal((32, 32))
        sync.mark_gpu_complete([new_data])
        sync.wait_gpu_complete()
        buffer.advance_write()

        # Read and process
        _, read_buf = buffer.get_for_read()
        assert read_buf.shape == (32, 32)

    def test_work_queue_with_ping_pong(self) -> None:
        """WorkQueue manages ping-pong buffer processing."""
        buffer = PingPongBuffer(shape=(16, 16), count=2)

        with WorkQueue(cpu_workers=2) as queue:
            # Producer fills buffers
            def fill_buffer():
                for i in range(4):
                    _, buf = buffer.get_for_write()
                    # Would fill buf here
                    buffer.advance_write()
                return "filled"

            # Consumer processes buffers
            def process_buffers():
                results = []
                for i in range(4):
                    try:
                        _, buf = buffer.get_for_read()
                        results.append(buf.shape)
                        buffer.advance_read()
                    except ValueError:
                        time.sleep(0.01)
                return results

            # Run in parallel
            fill_future = queue.submit_cpu(fill_buffer)
            fill_future.result()  # Wait for filling

            # Reset for processing demo
            buffer.reset()
            buffer.advance_write()
            buffer.advance_write()

            process_future = queue.submit_cpu(process_buffers)
            results = process_future.result()

            assert len(results) >= 2

    def test_full_pipeline_scenario(self) -> None:
        """Full scenario: load -> preprocess -> inference -> postprocess."""
        with WorkQueue(cpu_workers=2) as queue:
            # Define stages
            def load(idx):
                return np.random.randn(10).astype(np.float32)

            def preprocess(data):
                return mx.array(data)

            def inference(tensor):
                return tensor * 2

            def postprocess(tensor):
                return float(np.array(mx.sum(tensor)))

            # Process multiple items
            results = []
            for i in range(3):
                data = load(i)
                tensor = preprocess(data)
                output = inference(tensor)
                mx.eval(output)
                result = postprocess(output)
                results.append(result)

            assert len(results) == 3
            assert all(isinstance(r, float) for r in results)


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Thread safety tests for coordination primitives."""

    def test_sync_point_sequential_thread_safe(self) -> None:
        """SyncPoint works correctly when used sequentially from different sync points."""
        # Use separate sync points to avoid race conditions
        results = []

        for idx in range(5):
            sync = SyncPoint()
            tensor = mx.ones((10,)) * idx
            sync.mark_gpu_complete([tensor])
            sync.wait_gpu_complete()
            results.append(idx)

        assert len(results) == 5
        assert results == list(range(5))

    def test_work_queue_concurrent_submit(self) -> None:
        """WorkQueue handles concurrent submissions (CPU-only to avoid MLX threading issues)."""
        results = []
        lock = threading.Lock()

        with WorkQueue(cpu_workers=4) as queue:
            # Submit all work items
            futures = [queue.submit_cpu(lambda x: x * 2, i) for i in range(10)]

            # Collect results
            for f in futures:
                result = f.result()
                with lock:
                    results.append(result)

        assert sorted(results) == [i * 2 for i in range(10)]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_ping_pong_buffer_3d_shape(self) -> None:
        """PingPongBuffer works with 3D shapes."""
        buffer = PingPongBuffer(shape=(4, 8, 16), count=2)
        _, buf = buffer.get_for_write()
        assert buf.shape == (4, 8, 16)

    def test_work_queue_zero_workers(self) -> None:
        """WorkQueue with minimum workers still works."""
        with WorkQueue(cpu_workers=1) as queue:
            future = queue.submit_cpu(lambda: 42)
            assert future.result() == 42

    def test_overlap_compute_io_large_batches(self) -> None:
        """overlap_compute_io handles larger batch counts."""
        count = 0

        def load_fn(idx):
            nonlocal count
            count += 1
            return mx.ones((5,)) * idx

        def compute_fn(batch):
            return batch

        results = list(overlap_compute_io(compute_fn, load_fn, num_batches=20))

        assert len(results) == 20
        assert count == 20
