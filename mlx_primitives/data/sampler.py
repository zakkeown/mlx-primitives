"""Samplers for data loading in MLX.

This module provides samplers that determine the order of data access:
- SequentialSampler: Sequential access
- RandomSampler: Random shuffled access
- WeightedRandomSampler: Weighted random sampling
- BatchSampler: Wraps another sampler to yield batches
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence

import mlx.core as mx

from mlx_primitives.data.dataset import Dataset


class Sampler(ABC):
    """Base class for all samplers.

    Every sampler should provide an __iter__ method that yields
    indices and optionally a __len__ method for the number of samples.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Yield indices for sampling."""
        pass

    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.

    Example:
        >>> dataset = TensorDataset(mx.arange(10))
        >>> sampler = SequentialSampler(dataset)
        >>> list(sampler)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly.

    Args:
        data_source: Dataset to sample from.
        replacement: If True, samples with replacement (default: False).
        num_samples: Number of samples to draw. Defaults to len(data_source).
        seed: Random seed for reproducibility.

    Example:
        >>> dataset = TensorDataset(mx.arange(10))
        >>> sampler = RandomSampler(dataset, seed=42)
        >>> indices = list(sampler)  # Random permutation of 0-9
    """

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self._epoch = 0

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible shuffling across epochs.

        Args:
            epoch: Epoch number.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        # Set seed for reproducibility
        if self.seed is not None:
            mx.random.seed(self.seed + self._epoch)

        if self.replacement:
            # Sample with replacement
            indices = mx.random.randint(0, n, (self.num_samples,))
        else:
            # Random permutation
            indices = mx.random.permutation(n)
            if self.num_samples != n:
                indices = indices[: self.num_samples]

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler):
    """Samples elements with given probabilities (weights).

    Args:
        weights: Sequence of weights (probabilities) for each sample.
        num_samples: Number of samples to draw.
        replacement: If True, samples with replacement (default: True).
        seed: Random seed for reproducibility.

    Example:
        >>> # Sample more from first half of dataset
        >>> weights = [2.0] * 50 + [1.0] * 50  # First half twice as likely
        >>> sampler = WeightedRandomSampler(weights, num_samples=100)
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        seed: Optional[int] = None,
    ):
        if not replacement and num_samples > len(weights):
            raise ValueError(
                "num_samples cannot exceed len(weights) without replacement"
            )

        self.weights = mx.array(weights, dtype=mx.float32)
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.seed is not None:
            mx.random.seed(self.seed)

        # Normalize weights to probabilities
        probs = self.weights / mx.sum(self.weights)

        # Sample indices according to weights
        indices = mx.random.categorical(
            mx.log(probs), num_samples=self.num_samples
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples randomly from a specified subset of indices.

    Args:
        indices: Sequence of indices to sample from.
        seed: Random seed for reproducibility.

    Example:
        >>> # Only sample from indices 0, 2, 4, 6, 8
        >>> sampler = SubsetRandomSampler([0, 2, 4, 6, 8])
    """

    def __init__(self, indices: Sequence[int], seed: Optional[int] = None):
        self.indices = list(indices)
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.seed is not None:
            mx.random.seed(self.seed)

        perm = mx.random.permutation(len(self.indices))
        return iter([self.indices[i] for i in perm.tolist()])

    def __len__(self) -> int:
        return len(self.indices)


class BatchSampler(Sampler):
    """Wraps another sampler to yield batches of indices.

    Args:
        sampler: Base sampler to draw indices from.
        batch_size: Size of each batch.
        drop_last: If True, drop the last incomplete batch (default: False).

    Example:
        >>> dataset = TensorDataset(mx.arange(10))
        >>> sampler = SequentialSampler(dataset)
        >>> batch_sampler = BatchSampler(sampler, batch_size=3)
        >>> list(batch_sampler)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DistributedSampler(Sampler):
    """Sampler for distributed training.

    Restricts data loading to a subset of the dataset, ensuring each
    process gets a unique subset.

    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of processes participating in training.
        rank: Rank of the current process (0 to num_replicas-1).
        shuffle: If True, shuffle indices (default: True).
        seed: Random seed for reproducibility.
        drop_last: If True, drop samples to make evenly divisible (default: False).

    Example:
        >>> # Process 0 of 4
        >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, must be in [0, {num_replicas})"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        # Calculate number of samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = (
                len(self.dataset) + self.num_replicas - 1
            ) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling.

        Args:
            epoch: Epoch number.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Generate indices
        if self.shuffle:
            mx.random.seed(self.seed + self._epoch)
            indices = mx.random.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Pad to make evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.total_size]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
