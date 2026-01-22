"""DataLoader for MLX.

This module provides the DataLoader class for efficient batched data loading.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import mlx.core as mx

from mlx_primitives.data.dataset import Dataset, IterableDataset
from mlx_primitives.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


def default_collate(batch: List[Any]) -> Any:
    """Default collate function that stacks tensors.

    Handles various data structures:
    - MLX arrays: Stack along new first dimension
    - Tuples/Lists: Recursively collate each element
    - Dicts: Recursively collate each value
    - Other: Return as list

    Args:
        batch: List of samples to collate.

    Returns:
        Collated batch.
    """
    if not batch:
        return batch

    elem = batch[0]

    if isinstance(elem, mx.array):
        return mx.stack(batch)

    elif isinstance(elem, (int, float)):
        return mx.array(batch)

    elif isinstance(elem, tuple):
        # Handle named tuples specially
        if hasattr(elem, "_fields"):
            return type(elem)(*(default_collate(samples) for samples in zip(*batch)))
        return tuple(default_collate(samples) for samples in zip(*batch))

    elif isinstance(elem, list):
        return [default_collate(samples) for samples in zip(*batch)]

    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    else:
        # For other types, just return as list
        return batch


class DataLoader:
    """Data loader for MLX datasets.

    Combines a dataset with a sampler to provide an iterable over batches.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch (default: 1).
        shuffle: If True, shuffle data at each epoch (default: False).
        sampler: Custom sampler for drawing samples. Mutually exclusive with shuffle.
        batch_sampler: Custom batch sampler. Mutually exclusive with batch_size,
            shuffle, sampler, and drop_last.
        num_workers: Number of worker threads for loading (default: 0).
        collate_fn: Function to collate samples into batches.
        drop_last: Drop the last incomplete batch (default: False).
        prefetch_factor: Number of batches to prefetch per worker (default: 2).
        seed: Random seed for shuffling.

    Example:
        >>> dataset = TensorDataset(mx.random.normal((100, 10)), mx.arange(100))
        >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for x, y in loader:
        ...     # x.shape = (16, 10), y.shape = (16,)
        ...     pass
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.prefetch_factor = prefetch_factor

        # Handle batch_sampler
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, "
                    "shuffle, sampler, and drop_last"
                )
            self.batch_size = None
            self.drop_last = False
            self.sampler = None
            self.batch_sampler = batch_sampler
        else:
            self.batch_size = batch_size
            self.drop_last = drop_last

            # Set up sampler
            if sampler is not None:
                if shuffle:
                    raise ValueError("sampler and shuffle are mutually exclusive")
                self.sampler = sampler
            elif isinstance(dataset, IterableDataset):
                self.sampler = None
            elif shuffle:
                self.sampler = RandomSampler(dataset, seed=seed)
            else:
                self.sampler = SequentialSampler(dataset)

            # Set up batch sampler
            if self.sampler is not None:
                self.batch_sampler = BatchSampler(
                    self.sampler, batch_size, drop_last
                )
            else:
                self.batch_sampler = None

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible shuffling.

        Args:
            epoch: Epoch number.
        """
        self._epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[Any]:
        if self.num_workers == 0:
            return self._single_process_iter()
        else:
            return self._multi_process_iter()

    def _single_process_iter(self) -> Iterator[Any]:
        """Single-process data loading."""
        if isinstance(self.dataset, IterableDataset):
            # For iterable datasets, batch manually
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self.collate_fn(batch)
        else:
            # Use batch sampler for map-style datasets
            for indices in self.batch_sampler:
                batch = [self.dataset[i] for i in indices]
                yield self.collate_fn(batch)

    def _multi_process_iter(self) -> Iterator[Any]:
        """Multi-threaded data loading with prefetching."""
        # Use a queue for prefetching
        prefetch_queue: queue.Queue = queue.Queue(
            maxsize=self.num_workers * self.prefetch_factor
        )
        stop_event = threading.Event()
        # Store worker exceptions to propagate to main thread
        worker_exception: List[Optional[BaseException]] = [None]

        def worker():
            """Worker thread that loads and queues batches."""
            try:
                if isinstance(self.dataset, IterableDataset):
                    batch = []
                    for sample in self.dataset:
                        if stop_event.is_set():
                            return
                        batch.append(sample)
                        if len(batch) == self.batch_size:
                            prefetch_queue.put(self.collate_fn(batch))
                            batch = []
                    if batch and not self.drop_last:
                        prefetch_queue.put(self.collate_fn(batch))
                else:
                    for indices in self.batch_sampler:
                        if stop_event.is_set():
                            return
                        batch = [self.dataset[i] for i in indices]
                        prefetch_queue.put(self.collate_fn(batch))
            except BaseException as e:
                # Capture exception to propagate to main thread
                worker_exception[0] = e
            finally:
                prefetch_queue.put(None)  # Signal completion

        # Start worker thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    # Check if worker failed with exception
                    if worker_exception[0] is not None:
                        raise worker_exception[0]
                    break
                yield batch
        finally:
            stop_event.set()
            # Drain the queue to unblock worker
            while not prefetch_queue.empty():
                try:
                    prefetch_queue.get_nowait()
                except queue.Empty:
                    break

    def __len__(self) -> int:
        """Return the number of batches."""
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        elif isinstance(self.dataset, IterableDataset):
            raise TypeError("Cannot determine length of IterableDataset")
        else:
            if self.drop_last:
                return len(self.dataset) // self.batch_size
            else:
                return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class InfiniteDataLoader:
    """DataLoader that cycles through data infinitely.

    Useful for training loops that iterate by steps rather than epochs.

    Args:
        dataloader: DataLoader to cycle through.

    Example:
        >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> infinite_loader = InfiniteDataLoader(loader)
        >>> for step, batch in enumerate(infinite_loader):
        ...     if step >= 10000:
        ...         break
        ...     # Training step
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._epoch = 0

    def __iter__(self) -> Iterator[Any]:
        while True:
            self.dataloader.set_epoch(self._epoch)
            for batch in self.dataloader:
                yield batch
            self._epoch += 1
