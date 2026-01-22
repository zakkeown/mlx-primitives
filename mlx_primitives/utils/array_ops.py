"""Array operation utilities for MLX.

These utilities provide functionality that MLX doesn't natively support,
such as nonzero/argwhere operations with dynamic output sizes.
"""

from typing import Tuple

import mlx.core as mx


def nonzero(mask: mx.array) -> Tuple[mx.array, mx.array]:
    """Return indices where mask is True.

    Since MLX doesn't support dynamic output sizes, this returns a tuple
    of (sorted_indices, count) where the first `count` elements of
    sorted_indices are the True indices.

    Uses argsort trick: sorts indices by (-mask, index) to place True
    indices first while preserving their relative order.

    Args:
        mask: Boolean array of any shape (will be flattened).

    Returns:
        Tuple of:
        - sorted_indices: Int32 array of length mask.size. The first `count`
          elements are indices where mask was True (in ascending order).
          Remaining elements are indices where mask was False.
        - count: Scalar int32, number of True elements.

    Example:
        >>> mask = mx.array([True, False, True, False, True])
        >>> indices, count = nonzero(mask)
        >>> mx.eval(indices, count)
        >>> true_indices = indices[:count.item()]  # [0, 2, 4]
    """
    mask_int = mask.astype(mx.int32).reshape(-1)
    n = mask_int.shape[0]

    # Count True values
    count = mx.sum(mask_int)

    # Create index array
    indices = mx.arange(n, dtype=mx.int32)

    # Sort by (-mask * (n+1) + indices) to put True values first
    # True: -1 * (n+1) + idx = -(n+1) + idx (negative, sorted first)
    # False: 0 * (n+1) + idx = idx (positive, sorted after)
    # The +idx ensures stable sort (preserves order among True indices)
    sort_key = -mask_int * (n + 1) + indices
    sorted_indices = mx.argsort(sort_key)

    return sorted_indices, count


def argwhere(condition: mx.array) -> Tuple[mx.array, mx.array]:
    """Return indices where condition is True (alias for nonzero).

    Args:
        condition: Boolean array of any shape.

    Returns:
        Same as nonzero(): (sorted_indices, count).
    """
    return nonzero(condition)


def gather_where(
    x: mx.array,
    mask: mx.array,
    capacity: int,
) -> Tuple[mx.array, mx.array]:
    """Gather elements from x where mask is True, up to capacity.

    This is useful for MoE-style sparse computation where we want to
    gather only the tokens routed to a specific expert.

    Args:
        x: Source array of shape (n, ...).
        mask: Boolean mask of shape (n,) indicating which elements to gather.
        capacity: Maximum number of elements to gather.

    Returns:
        Tuple of:
        - gathered: Array of shape (capacity, ...) containing gathered elements.
          If fewer than capacity elements match, remaining elements are zeros.
        - actual_count: Scalar int32, actual number of gathered elements.

    Example:
        >>> x = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> mask = mx.array([True, False, True, False])
        >>> gathered, count = gather_where(x, mask, capacity=3)
        >>> # gathered[:count.item()] contains [[1, 2], [5, 6]]
    """
    indices, count = nonzero(mask)

    # Clamp indices to capacity (for safety)
    gather_indices = indices[:capacity]

    # Handle case where we have fewer than capacity elements
    # Pad gather_indices with 0s (which will gather x[0] but we'll mask it)
    pad_count = capacity - min(capacity, mask.shape[0])
    if pad_count > 0:
        gather_indices = mx.concatenate([gather_indices, mx.zeros(pad_count, dtype=mx.int32)])

    # Gather elements
    gathered = x[gather_indices]

    # The actual count is min(count, capacity)
    actual_count = mx.minimum(count, mx.array(capacity, dtype=mx.int32))

    return gathered, actual_count


def scatter_where(
    gathered: mx.array,
    mask: mx.array,
    count: mx.array,
    output_shape: Tuple[int, ...],
) -> mx.array:
    """Scatter gathered elements back to positions where mask was True.

    Inverse of gather_where: takes gathered values and places them back
    at their original positions.

    Args:
        gathered: Array of shape (capacity, ...) from gather_where.
        mask: Original boolean mask of shape (n,).
        count: Actual count from gather_where.
        output_shape: Shape of output array (n, ...).

    Returns:
        Array of output_shape with gathered values at True positions, zeros elsewhere.
    """
    indices, _ = nonzero(mask)
    n = output_shape[0]
    capacity = gathered.shape[0]

    # Create output
    output = mx.zeros(output_shape, dtype=gathered.dtype)

    # For each gathered element, scatter to its position
    # We need count.item() here, which is a sync point, but it's unavoidable
    # for dynamic scattering. In practice, this is called once per expert per forward.

    # Use a loop-based approach since MLX doesn't have scatter_add with indices
    # This is O(capacity) GPU operations, not ideal but workable
    count_val = int(count.item())
    if count_val == 0:
        return output

    # Batch the indices extraction to avoid O(n) GPU syncs in the loop.
    # Previously, indices[i].item() was called per iteration causing a sync each time.
    # Now we sync once with eval+tolist, reducing O(n) syncs to O(1).
    valid_indices = indices[:count_val]
    mx.eval(valid_indices)
    idx_list = valid_indices.tolist()

    for i in range(count_val):
        output = output.at[idx_list[i]].add(gathered[i])

    return output
