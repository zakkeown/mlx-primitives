"""Streaming operations for large datasets on Apple Silicon.

This module provides memory-mapped tensor operations that leverage
Apple Silicon's unified memory architecture for processing datasets
larger than available memory.
"""

import mmap
import os
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import mlx.core as mx
import numpy as np


class StreamingTensor:
    """Memory-mapped tensor for streaming large datasets.

    Maps a file directly into unified memory, allowing both CPU and GPU
    to access data without loading the entire dataset. This is particularly
    efficient on Apple Silicon due to the unified memory architecture.

    Example:
        >>> # Create a streaming tensor from a file
        >>> tensor = StreamingTensor("large_weights.bin", shape=(10000, 4096))
        >>> # Iterate over chunks
        >>> for chunk in tensor.iter_chunks(chunk_size=1000):
        ...     process(chunk)
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape: tuple,
        dtype: mx.Dtype = mx.float32,
        mode: str = "r",
    ):
        """Initialize a memory-mapped tensor.

        Args:
            path: Path to the data file.
            shape: Tensor shape (must match file size).
            dtype: Data type of elements.
            mode: Access mode - "r" (read), "r+" (read-write), "w+" (create).
        """
        self._path = Path(path)
        self._shape = shape
        self._dtype = dtype
        self._mode = mode
        self._mmap: Optional[mmap.mmap] = None
        self._np_array: Optional[np.ndarray] = None
        self._dtype_size = _get_dtype_size(dtype)

        # Calculate expected file size
        self._total_elements = 1
        for dim in shape:
            self._total_elements *= dim
        self._file_size = self._total_elements * self._dtype_size

        # Open the file
        self._open()

    def _open(self) -> None:
        """Open or create the memory-mapped file."""
        np_dtype = _mlx_to_numpy_dtype(self._dtype)

        if self._mode == "w+":
            # Create new file
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                f.write(b"\x00" * self._file_size)
            file_mode = "r+b"
        elif self._mode == "r+":
            file_mode = "r+b"
        else:
            file_mode = "rb"

        # Check file exists and size
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

        actual_size = self._path.stat().st_size
        if actual_size != self._file_size:
            raise ValueError(
                f"File size mismatch: expected {self._file_size} bytes, "
                f"got {actual_size} bytes"
            )

        # Open file and create mmap
        self._file = open(self._path, file_mode)
        mmap_mode = mmap.ACCESS_WRITE if "+" in self._mode else mmap.ACCESS_READ
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap_mode)

        # Create numpy view of mmap
        self._np_array = np.frombuffer(self._mmap, dtype=np_dtype).reshape(self._shape)

    def close(self) -> None:
        """Close the memory-mapped file.

        Note: Must delete numpy array reference before closing mmap,
        otherwise BufferError is raised due to exported pointers.
        """
        # Delete numpy array first to release buffer reference
        if hasattr(self, "_np_array"):
            del self._np_array
            self._np_array = None

        # Now safe to close mmap
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # Ignore if there are still external references
                pass
            self._mmap = None

        if hasattr(self, "_file") and self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        """Clean up resources."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __enter__(self) -> "StreamingTensor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def shape(self) -> tuple:
        """Get tensor shape."""
        return self._shape

    @property
    def dtype(self) -> mx.Dtype:
        """Get tensor dtype."""
        return self._dtype

    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self._total_elements

    def __getitem__(self, idx) -> mx.array:
        """Get slice as MLX array.

        For small slices, this is zero-copy on unified memory.
        For large slices, consider using iter_chunks for efficiency.

        Args:
            idx: Index or slice.

        Returns:
            MLX array containing the requested data.
        """
        np_slice = self._np_array[idx]
        return mx.array(np_slice)

    def __setitem__(self, idx, value: Union[mx.array, np.ndarray]) -> None:
        """Set slice from MLX or NumPy array.

        Args:
            idx: Index or slice.
            value: Data to write.
        """
        if self._mode == "r":
            raise ValueError("Cannot write to read-only tensor")

        if isinstance(value, mx.array):
            mx.eval(value)
            value = np.array(value)

        self._np_array[idx] = value
        self._mmap.flush()

    def iter_chunks(
        self,
        chunk_size: int,
        dim: int = 0,
        prefetch_ahead: int = 2,
    ) -> Iterator[mx.array]:
        """Iterate over chunks with prefetching.

        Yields chunks of the tensor along the specified dimension,
        with optional prefetching for better performance.

        Args:
            chunk_size: Number of elements per chunk along dim.
            dim: Dimension to chunk along.
            prefetch_ahead: Number of chunks to prefetch (hint).

        Yields:
            MLX arrays containing each chunk.
        """
        total_size = self._shape[dim]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)

            # Build slice tuple
            slices = [slice(None)] * len(self._shape)
            slices[dim] = slice(start, end)

            # Get chunk as MLX array
            np_chunk = self._np_array[tuple(slices)]
            yield mx.array(np_chunk)

    def prefetch(
        self,
        start: int,
        end: int,
        dim: int = 0,
    ) -> None:
        """Prefetch a range into cache.

        Hints to the system to bring the specified range into cache.
        This is advisory and may not have effect on all systems.

        Args:
            start: Start index along dimension.
            end: End index along dimension.
            dim: Dimension to prefetch along.
        """
        # Build slice for the range
        slices = [slice(None)] * len(self._shape)
        slices[dim] = slice(start, end)

        # Touch the memory to bring it into cache
        # This is a hint - the system may or may not honor it
        _ = self._np_array[tuple(slices)].sum()

    def to_mlx(self) -> mx.array:
        """Load entire tensor into MLX array.

        Warning: This loads the entire tensor into memory.
        Use only if the tensor fits in available memory.

        Returns:
            MLX array containing all data.
        """
        return mx.array(self._np_array)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        path: Union[str, Path],
    ) -> "StreamingTensor":
        """Create a streaming tensor from a NumPy array.

        Writes the array to disk and returns a memory-mapped view.

        Args:
            arr: Source NumPy array.
            path: Path to write the file.

        Returns:
            StreamingTensor backed by the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine MLX dtype
        mlx_dtype = _numpy_to_mlx_dtype(arr.dtype)

        # Write array to file
        arr.tofile(path)

        return cls(path, arr.shape, mlx_dtype, mode="r+")

    @classmethod
    def from_mlx(
        cls,
        arr: mx.array,
        path: Union[str, Path],
    ) -> "StreamingTensor":
        """Create a streaming tensor from an MLX array.

        Writes the array to disk and returns a memory-mapped view.

        Args:
            arr: Source MLX array.
            path: Path to write the file.

        Returns:
            StreamingTensor backed by the file.
        """
        mx.eval(arr)
        np_arr = np.array(arr)
        return cls.from_numpy(np_arr, path)


class StreamingDataLoader:
    """Efficient data loading using unified memory streaming.

    Pipelines data from disk through memory-mapping to GPU compute,
    with optional prefetching for improved throughput.

    Example:
        >>> tensor = StreamingTensor("data.bin", shape=(100000, 512))
        >>> loader = StreamingDataLoader(tensor, batch_size=64, shuffle=True)
        >>> for batch in loader:
        ...     output = model(batch)
    """

    def __init__(
        self,
        source: StreamingTensor,
        batch_size: int,
        shuffle: bool = False,
        prefetch_batches: int = 2,
        transform: Optional[Callable[[mx.array], mx.array]] = None,
        drop_last: bool = False,
    ):
        """Initialize the data loader.

        Args:
            source: StreamingTensor to load from.
            batch_size: Samples per batch.
            shuffle: Whether to shuffle (uses index shuffling, not data copy).
            prefetch_batches: Number of batches to prefetch.
            transform: Optional transform to apply to each batch.
            drop_last: Drop last incomplete batch if True.
        """
        self._source = source
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._prefetch_batches = prefetch_batches
        self._transform = transform
        self._drop_last = drop_last

        # Calculate number of samples (first dimension)
        self._num_samples = source.shape[0]
        self._num_batches = self._num_samples // batch_size
        if not drop_last and self._num_samples % batch_size != 0:
            self._num_batches += 1

    def __len__(self) -> int:
        """Get number of batches."""
        return self._num_batches

    def __iter__(self) -> Iterator[mx.array]:
        """Iterate over batches.

        Yields:
            MLX arrays containing each batch.
        """
        # Generate indices
        indices = np.arange(self._num_samples)
        if self._shuffle:
            np.random.shuffle(indices)

        for batch_idx in range(self._num_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, self._num_samples)

            if self._drop_last and end - start < self._batch_size:
                break

            # Get batch indices
            batch_indices = indices[start:end]

            # Load batch
            batch = self._source[batch_indices]

            # Apply transform if provided
            if self._transform is not None:
                batch = self._transform(batch)

            yield batch


def streaming_reduce(
    tensor: StreamingTensor,
    op: str,
    axis: int = 0,
    chunk_size: int = 1024,
) -> mx.array:
    """Streaming reduction for large tensors.

    Performs a reduction operation on a streaming tensor by
    processing it in chunks, enabling reductions on datasets
    larger than available memory.

    Args:
        tensor: StreamingTensor to reduce.
        op: Reduction operation - "sum", "mean", "max", "min".
        axis: Axis to reduce along.
        chunk_size: Chunk size for streaming.

    Returns:
        Reduced MLX array.
    """
    result = None
    count = 0

    for chunk in tensor.iter_chunks(chunk_size, dim=axis):
        if op == "sum":
            chunk_result = mx.sum(chunk, axis=axis)
        elif op == "mean":
            chunk_result = mx.sum(chunk, axis=axis)
            count += chunk.shape[axis]
        elif op == "max":
            chunk_result = mx.max(chunk, axis=axis)
        elif op == "min":
            chunk_result = mx.min(chunk, axis=axis)
        else:
            raise ValueError(f"Unknown operation: {op}")

        if result is None:
            result = chunk_result
        else:
            if op == "sum" or op == "mean":
                result = result + chunk_result
            elif op == "max":
                result = mx.maximum(result, chunk_result)
            elif op == "min":
                result = mx.minimum(result, chunk_result)

    if op == "mean" and result is not None:
        result = result / count

    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _get_dtype_size(dtype: mx.Dtype) -> int:
    """Get size in bytes for an MLX dtype."""
    sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int16: 2,
        mx.int8: 1,
        mx.uint32: 4,
        mx.uint16: 2,
        mx.uint8: 1,
    }
    return sizes.get(dtype, 4)


def _mlx_to_numpy_dtype(dtype: mx.Dtype) -> np.dtype:
    """Convert MLX dtype to NumPy dtype."""
    mapping = {
        mx.float32: np.float32,
        mx.float16: np.float16,
        mx.int32: np.int32,
        mx.int16: np.int16,
        mx.int8: np.int8,
        mx.uint32: np.uint32,
        mx.uint16: np.uint16,
        mx.uint8: np.uint8,
    }
    return mapping.get(dtype, np.float32)


def _numpy_to_mlx_dtype(dtype: np.dtype) -> mx.Dtype:
    """Convert NumPy dtype to MLX dtype."""
    mapping = {
        np.float32: mx.float32,
        np.float16: mx.float16,
        np.int32: mx.int32,
        np.int16: mx.int16,
        np.int8: mx.int8,
        np.uint32: mx.uint32,
        np.uint16: mx.uint16,
        np.uint8: mx.uint8,
    }
    # Handle numpy dtype objects
    dtype = np.dtype(dtype)
    return mapping.get(dtype.type, mx.float32)
