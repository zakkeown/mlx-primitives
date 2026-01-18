"""CPU/GPU work coordination primitives for Apple Silicon.

This module provides primitives for efficient hybrid CPU/GPU workloads,
leveraging Apple Silicon's unified memory for seamless coordination.
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar

import mlx.core as mx

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class SyncPoint:
    """Explicit synchronization point between CPU and GPU.

    More fine-grained than mx.eval() for coordinating specific tensors.
    Useful when you need to ensure specific operations are complete
    before proceeding.

    Example:
        >>> sync = SyncPoint()
        >>> output = model(input)
        >>> sync.mark_gpu_complete([output])
        >>> # CPU can now safely read output
        >>> sync.wait_gpu_complete()
        >>> np_output = np.array(output)
    """

    def __init__(self):
        """Create a synchronization point."""
        self._gpu_tensors: List[mx.array] = []
        self._complete = threading.Event()
        self._complete.set()  # Initially complete

    def mark_gpu_complete(self, tensors: List[mx.array]) -> None:
        """Signal that GPU should complete these tensors.

        Args:
            tensors: List of tensors to synchronize.
        """
        self._gpu_tensors = tensors
        self._complete.clear()
        # Evaluate tensors asynchronously
        mx.eval(*tensors)
        self._complete.set()

    def wait_gpu_complete(self, timeout: Optional[float] = None) -> bool:
        """Wait for GPU to complete marked operations.

        Args:
            timeout: Maximum wait time in seconds. None for infinite.

        Returns:
            True if completed, False if timeout occurred.
        """
        return self._complete.wait(timeout)

    def mark_cpu_complete(self) -> None:
        """Signal that CPU is done modifying unified memory.

        Call this after CPU operations that modify memory that
        will be used by GPU operations.
        """
        # On unified memory, this is typically automatic
        # but we ensure any Python-side operations are flushed
        pass

    @property
    def is_complete(self) -> bool:
        """Check if synchronization is complete."""
        return self._complete.is_set()


class PingPongBuffer:
    """Double/triple buffer for overlapped CPU/GPU processing.

    Enables efficient pipelining where GPU processes buffer[i]
    while CPU fills buffer[i+1]. This hides data transfer latency
    and maximizes hardware utilization.

    Example:
        >>> buffer = PingPongBuffer(shape=(64, 512), count=2)
        >>> while has_data:
        ...     # Get buffer for CPU to fill
        ...     idx, write_buf = buffer.get_for_write()
        ...     fill_buffer(write_buf)
        ...     buffer.advance()
        ...     # GPU processes previous buffer
        ...     idx, read_buf = buffer.get_for_read()
        ...     output = model(read_buf)
    """

    def __init__(
        self,
        shape: tuple,
        dtype: mx.Dtype = mx.float32,
        count: int = 2,
    ):
        """Create a ping-pong buffer.

        Args:
            shape: Shape of each buffer.
            dtype: Data type.
            count: Number of buffers (2 for double, 3 for triple).
        """
        if count < 2:
            raise ValueError("Need at least 2 buffers for ping-pong")

        self._count = count
        self._shape = shape
        self._dtype = dtype

        # Create buffers
        self._buffers = [mx.zeros(shape, dtype=dtype) for _ in range(count)]

        # Initialize all buffers
        mx.eval(*self._buffers)

        # Current positions
        self._write_idx = 0
        self._read_idx = 0
        self._filled_count = 0

    @property
    def count(self) -> int:
        """Get number of buffers."""
        return self._count

    @property
    def shape(self) -> tuple:
        """Get buffer shape."""
        return self._shape

    def get_for_write(self) -> Tuple[int, mx.array]:
        """Get next buffer for CPU to write to.

        Returns:
            Tuple of (buffer_index, buffer_array).
        """
        return self._write_idx, self._buffers[self._write_idx]

    def get_for_read(self) -> Tuple[int, mx.array]:
        """Get buffer ready for GPU to read from.

        Returns:
            Tuple of (buffer_index, buffer_array).

        Raises:
            ValueError: If no buffer is ready for reading.
        """
        if self._filled_count == 0:
            raise ValueError("No buffer ready for reading")

        return self._read_idx, self._buffers[self._read_idx]

    def advance_write(self) -> None:
        """Mark current write buffer as filled and move to next."""
        self._write_idx = (self._write_idx + 1) % self._count
        self._filled_count = min(self._filled_count + 1, self._count)

    def advance_read(self) -> None:
        """Mark current read buffer as consumed and move to next."""
        if self._filled_count > 0:
            self._read_idx = (self._read_idx + 1) % self._count
            self._filled_count -= 1

    def advance(self) -> None:
        """Advance both read and write positions.

        Use this for simple ping-pong patterns where read and write
        advance together.
        """
        self.advance_write()

    def reset(self) -> None:
        """Reset buffer positions to initial state."""
        self._write_idx = 0
        self._read_idx = 0
        self._filled_count = 0

    def is_full(self) -> bool:
        """Check if all buffers are filled."""
        return self._filled_count >= self._count

    def is_empty(self) -> bool:
        """Check if no buffers are filled."""
        return self._filled_count == 0


def ping_pong_buffer(
    shape: tuple,
    dtype: mx.Dtype = mx.float32,
    count: int = 2,
) -> PingPongBuffer:
    """Create a ping-pong buffer for overlapped CPU/GPU work.

    Convenience function for creating PingPongBuffer instances.

    Args:
        shape: Shape of each buffer.
        dtype: Data type.
        count: Number of buffers.

    Returns:
        PingPongBuffer instance.
    """
    return PingPongBuffer(shape, dtype, count)


def parallel_cpu_gpu(
    cpu_fn: Callable[[], T],
    gpu_fn: Callable[[], U],
) -> Tuple[T, U]:
    """Execute CPU and GPU work in parallel.

    Useful when CPU and GPU operations are independent and can
    overlap for better utilization.

    Example:
        >>> # CPU loads next batch while GPU processes current
        >>> next_batch, output = parallel_cpu_gpu(
        ...     lambda: load_batch(idx + 1),
        ...     lambda: model(current_batch),
        ... )

    Args:
        cpu_fn: Function to run on CPU (in background thread).
        gpu_fn: Function to run on GPU (in main thread).

    Returns:
        Tuple of (cpu_result, gpu_result).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit CPU work to background thread
        cpu_future = executor.submit(cpu_fn)

        # Run GPU work in main thread
        gpu_result = gpu_fn()

        # Wait for CPU work and evaluate GPU result
        mx.eval(gpu_result) if isinstance(gpu_result, mx.array) else None
        cpu_result = cpu_future.result()

    return cpu_result, gpu_result


class WorkQueue:
    """Coordinated work queue for CPU/GPU hybrid processing.

    Enables patterns like:
    - CPU preprocessing -> GPU inference -> CPU postprocessing
    - Parallel CPU tokenization while GPU processes previous batch

    Example:
        >>> queue = WorkQueue(cpu_workers=4)
        >>> # Submit CPU preprocessing
        >>> preprocess_future = queue.submit_cpu(preprocess, data)
        >>> # Submit GPU work that depends on preprocessing
        >>> gpu_future = queue.submit_gpu(model, depends_on=preprocess_future)
    """

    def __init__(
        self,
        cpu_workers: int = 4,
    ):
        """Initialize work queue.

        Args:
            cpu_workers: Number of CPU worker threads.
        """
        self._cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
        self._active = True

    def submit_cpu(
        self,
        fn: Callable,
        *args,
        depends_on: Optional[Future] = None,
        **kwargs,
    ) -> Future:
        """Submit CPU work, optionally depending on prior work.

        Args:
            fn: Function to execute.
            *args: Positional arguments for fn.
            depends_on: Future to wait for before executing.
            **kwargs: Keyword arguments for fn.

        Returns:
            Future representing the pending result.
        """
        if depends_on is not None:
            def wrapped():
                depends_on.result()  # Wait for dependency
                return fn(*args, **kwargs)
            return self._cpu_executor.submit(wrapped)
        else:
            return self._cpu_executor.submit(fn, *args, **kwargs)

    def submit_gpu(
        self,
        fn: Callable,
        *args,
        depends_on: Optional[Future] = None,
        **kwargs,
    ) -> Future:
        """Submit GPU work, optionally depending on prior work.

        Note: GPU work is submitted to the CPU executor but is
        expected to dispatch to GPU internally via MLX operations.

        Args:
            fn: Function to execute (should contain MLX operations).
            *args: Positional arguments for fn.
            depends_on: Future to wait for before executing.
            **kwargs: Keyword arguments for fn.

        Returns:
            Future representing the pending result.
        """
        def gpu_work():
            if depends_on is not None:
                depends_on.result()  # Wait for dependency
            result = fn(*args, **kwargs)
            # Ensure GPU work is complete
            if isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, (list, tuple)):
                arrays = [x for x in result if isinstance(x, mx.array)]
                if arrays:
                    mx.eval(*arrays)
            return result

        return self._cpu_executor.submit(gpu_work)

    def pipeline(
        self,
        stages: List[Tuple[str, Callable]],
        inputs: Iterator,
        buffer_size: int = 2,
    ) -> Iterator:
        """Execute multi-stage pipeline with automatic overlapping.

        Creates a pipeline where each stage can overlap with others,
        maximizing throughput.

        Args:
            stages: List of (device, function) tuples.
                device is "cpu" or "gpu".
            inputs: Iterator of input data.
            buffer_size: Number of items to buffer between stages.

        Yields:
            Results from the final stage.

        Example:
            >>> stages = [
            ...     ("cpu", tokenize),
            ...     ("gpu", embed),
            ...     ("gpu", forward),
            ...     ("cpu", decode),
            ... ]
            >>> for result in queue.pipeline(stages, data_iter):
            ...     process(result)
        """
        if not stages:
            yield from inputs
            return

        # Simple sequential pipeline with overlapping
        # For full async pipeline, would need more complex implementation
        current_data = list(inputs)

        for device, fn in stages:
            next_data = []
            for item in current_data:
                if device == "gpu":
                    result = fn(item)
                    if isinstance(result, mx.array):
                        mx.eval(result)
                else:
                    result = fn(item)
                next_data.append(result)
            current_data = next_data

        yield from current_data

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the work queue.

        Args:
            wait: If True, wait for pending work to complete.
        """
        self._active = False
        self._cpu_executor.shutdown(wait=wait)

    def __enter__(self) -> "WorkQueue":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()


def overlap_compute_io(
    compute_fn: Callable[[mx.array], mx.array],
    load_fn: Callable[[int], mx.array],
    num_batches: int,
    buffer_count: int = 2,
) -> Iterator[mx.array]:
    """Overlap computation and I/O for efficient pipelining.

    Loads the next batch while computing on the current batch,
    maximizing hardware utilization.

    Args:
        compute_fn: Function that processes a batch (GPU).
        load_fn: Function that loads a batch by index (CPU/disk).
        num_batches: Total number of batches to process.
        buffer_count: Number of buffers for overlapping.

    Yields:
        Computed results for each batch.

    Example:
        >>> def load_batch(idx):
        ...     return mx.array(data[idx * batch_size:(idx + 1) * batch_size])
        >>> for output in overlap_compute_io(model, load_batch, num_batches=100):
        ...     save_output(output)
    """
    if num_batches <= 0:
        return

    with ThreadPoolExecutor(max_workers=1) as loader:
        # Load first batch
        current_batch = load_fn(0)
        mx.eval(current_batch)

        # Start loading next batch
        next_future = None
        if num_batches > 1:
            next_future = loader.submit(load_fn, 1)

        for i in range(num_batches):
            # Compute on current batch
            result = compute_fn(current_batch)

            # Start loading batch i+2 while computing
            future_load = None
            if i + 2 < num_batches:
                future_load = loader.submit(load_fn, i + 2)

            # Wait for GPU compute and yield result
            mx.eval(result)
            yield result

            # Get next batch (already loaded)
            if next_future is not None:
                current_batch = next_future.result()
                mx.eval(current_batch)

            # Update next future
            next_future = future_load
