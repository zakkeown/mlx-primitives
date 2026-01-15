"""Dataset base classes for MLX.

This module provides base classes for datasets:
- Dataset: Map-style dataset with random access
- IterableDataset: For streaming/large datasets
- TensorDataset: Simple dataset from tensors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx


class Dataset(ABC):
    """Abstract base class for map-style datasets.

    A map-style dataset implements __getitem__ and __len__ for
    random access to samples.

    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        ...     def __getitem__(self, idx):
        ...         return self.data[idx]
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get a sample by index."""
        pass

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]


class IterableDataset(ABC):
    """Abstract base class for iterable datasets.

    An iterable dataset implements __iter__ for sequential access.
    Useful for streaming data or very large datasets that don't fit in memory.

    Example:
        >>> class StreamingDataset(IterableDataset):
        ...     def __init__(self, file_path):
        ...         self.file_path = file_path
        ...
        ...     def __iter__(self):
        ...         with open(self.file_path) as f:
        ...             for line in f:
        ...                 yield process(line)
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the dataset."""
        pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample is retrieved by indexing tensors along the first dimension.

    Args:
        *tensors: MLX arrays with the same size in the first dimension.

    Example:
        >>> x = mx.random.normal((100, 10))
        >>> y = mx.random.normal((100, 1))
        >>> dataset = TensorDataset(x, y)
        >>> len(dataset)
        100
        >>> sample = dataset[0]  # Returns (x[0], y[0])
    """

    def __init__(self, *tensors: mx.array):
        if not tensors:
            raise ValueError("At least one tensor required")

        # Verify all tensors have same first dimension
        size = tensors[0].shape[0]
        for i, tensor in enumerate(tensors):
            if tensor.shape[0] != size:
                raise ValueError(
                    f"Tensor {i} has size {tensor.shape[0]} in first dimension, "
                    f"expected {size}"
                )

        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int) -> Tuple[mx.array, ...]:
        return tuple(tensor[index] for tensor in self.tensors)


class ListDataset(Dataset):
    """Dataset wrapping a list of samples.

    Args:
        data: List of samples.

    Example:
        >>> data = [{"x": mx.array([1, 2]), "y": 0} for _ in range(100)]
        >>> dataset = ListDataset(data)
    """

    def __init__(self, data: List[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        return self.data[index]


class ConcatDataset(Dataset):
    """Dataset that concatenates multiple datasets.

    Args:
        datasets: List of datasets to concatenate.

    Example:
        >>> dataset1 = TensorDataset(mx.ones((50, 10)))
        >>> dataset2 = TensorDataset(mx.zeros((50, 10)))
        >>> combined = ConcatDataset([dataset1, dataset2])
        >>> len(combined)
        100
    """

    def __init__(self, datasets: List[Dataset]):
        if not datasets:
            raise ValueError("At least one dataset required")

        self.datasets = datasets
        self.cumulative_sizes = []

        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            if -index > len(self):
                raise IndexError("Index out of range")
            index = len(self) + index

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if index < cum_size:
                dataset_idx = i
                break

        # Compute local index
        if dataset_idx == 0:
            local_idx = index
        else:
            local_idx = index - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]


class SubsetDataset(Dataset):
    """Dataset that is a subset of another dataset.

    Args:
        dataset: The original dataset.
        indices: Indices to include in the subset.

    Example:
        >>> full_dataset = TensorDataset(mx.arange(100))
        >>> subset = SubsetDataset(full_dataset, indices=[0, 10, 20, 30])
        >>> len(subset)
        4
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


class TransformDataset(Dataset):
    """Dataset that applies a transform to another dataset.

    Args:
        dataset: The original dataset.
        transform: Transform function to apply to each sample.

    Example:
        >>> dataset = TensorDataset(mx.arange(100))
        >>> transformed = TransformDataset(dataset, lambda x: x * 2)
    """

    def __init__(self, dataset: Dataset, transform: callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        return self.transform(sample)


class ChainDataset(IterableDataset):
    """Chain multiple iterable datasets together.

    Args:
        datasets: List of iterable datasets to chain.

    Example:
        >>> class NumberStream(IterableDataset):
        ...     def __init__(self, start, end):
        ...         self.start, self.end = start, end
        ...     def __iter__(self):
        ...         for i in range(self.start, self.end):
        ...             yield i
        >>> chained = ChainDataset([NumberStream(0, 5), NumberStream(5, 10)])
        >>> list(chained)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, datasets: List[IterableDataset]):
        self.datasets = datasets

    def __iter__(self) -> Iterator[Any]:
        for dataset in self.datasets:
            yield from dataset


def random_split(
    dataset: Dataset,
    lengths: List[int],
    seed: Optional[int] = None,
) -> List[SubsetDataset]:
    """Randomly split a dataset into non-overlapping subsets.

    Args:
        dataset: Dataset to split.
        lengths: Lengths of splits to create.
        seed: Random seed for reproducibility.

    Returns:
        List of SubsetDataset instances.

    Example:
        >>> dataset = TensorDataset(mx.arange(100))
        >>> train, val, test = random_split(dataset, [80, 10, 10])
        >>> len(train), len(val), len(test)
        (80, 10, 10)
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Sum of lengths ({sum(lengths)}) must equal dataset length ({len(dataset)})"
        )

    # Generate random permutation
    if seed is not None:
        mx.random.seed(seed)

    indices = mx.random.permutation(len(dataset)).tolist()

    # Split indices according to lengths
    splits = []
    offset = 0
    for length in lengths:
        splits.append(SubsetDataset(dataset, indices[offset : offset + length]))
        offset += length

    return splits
