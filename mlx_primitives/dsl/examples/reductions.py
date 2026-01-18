"""Reduction operation examples for Metal-Triton DSL.

Demonstrates parallel reduction patterns using SIMD operations.
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def sum_reduce(
    x_ptr,
    out_ptr,
    N: constexpr,
):
    """Parallel sum reduction using SIMD shuffles.

    Each SIMD group computes partial sum, then atomically adds to output.
    """
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()

    # Load element (with bounds check)
    pid = mt.program_id(0)
    block_size = mt.threads_per_threadgroup()
    idx = pid * block_size + tid

    val = 0.0
    if idx < N:
        val = mt.load(x_ptr + idx)

    # SIMD tree reduction (32-wide SIMD groups)
    val = val + mt.simd_shuffle_down(val, 16)
    val = val + mt.simd_shuffle_down(val, 8)
    val = val + mt.simd_shuffle_down(val, 4)
    val = val + mt.simd_shuffle_down(val, 2)
    val = val + mt.simd_shuffle_down(val, 1)

    # First lane of each SIMD group has partial sum
    if simd_lane == 0:
        mt.atomic_add(out_ptr, val)


@metal_kernel
def max_reduce(
    x_ptr,
    out_ptr,
    N: constexpr,
):
    """Parallel max reduction using SIMD operations."""
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()

    pid = mt.program_id(0)
    block_size = mt.threads_per_threadgroup()
    idx = pid * block_size + tid

    # Initialize to -inf for max reduction
    val = float('-inf')
    if idx < N:
        val = mt.load(x_ptr + idx)

    # SIMD max reduction
    val = mt.maximum(val, mt.simd_shuffle_down(val, 16))
    val = mt.maximum(val, mt.simd_shuffle_down(val, 8))
    val = mt.maximum(val, mt.simd_shuffle_down(val, 4))
    val = mt.maximum(val, mt.simd_shuffle_down(val, 2))
    val = mt.maximum(val, mt.simd_shuffle_down(val, 1))

    # Use atomic max (would need atomic_max support)
    # For now, first lane writes
    if simd_lane == 0:
        # Note: This is not fully correct for multi-block reductions
        # A proper implementation needs a second pass or atomic max
        mt.store(out_ptr + mt.simd_group_id(), val)


@metal_kernel
def dot_product(
    a_ptr,
    b_ptr,
    out_ptr,
    N: constexpr,
):
    """Dot product: sum(a * b).

    Combines element-wise multiply with reduction.
    """
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()

    pid = mt.program_id(0)
    block_size = mt.threads_per_threadgroup()
    idx = pid * block_size + tid

    # Load and multiply
    val = 0.0
    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        val = a * b

    # SIMD sum reduction
    val = val + mt.simd_shuffle_down(val, 16)
    val = val + mt.simd_shuffle_down(val, 8)
    val = val + mt.simd_shuffle_down(val, 4)
    val = val + mt.simd_shuffle_down(val, 2)
    val = val + mt.simd_shuffle_down(val, 1)

    # Accumulate partial sums
    if simd_lane == 0:
        mt.atomic_add(out_ptr, val)


@metal_kernel
def softmax_stable(
    x_ptr,
    out_ptr,
    N: constexpr,
):
    """Numerically stable softmax (single-row version).

    Demonstrates online max-subtract pattern for numerical stability.

    For full softmax over batches, see attention examples.
    """
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()

    # Phase 1: Find max (for numerical stability)
    pid = mt.program_id(0)
    block_size = mt.threads_per_threadgroup()
    idx = pid * block_size + tid

    val = float('-inf')
    if idx < N:
        val = mt.load(x_ptr + idx)

    # SIMD max reduction
    max_val = val
    max_val = mt.maximum(max_val, mt.simd_shuffle_down(max_val, 16))
    max_val = mt.maximum(max_val, mt.simd_shuffle_down(max_val, 8))
    max_val = mt.maximum(max_val, mt.simd_shuffle_down(max_val, 4))
    max_val = mt.maximum(max_val, mt.simd_shuffle_down(max_val, 2))
    max_val = mt.maximum(max_val, mt.simd_shuffle_down(max_val, 1))

    # Broadcast max to all lanes
    max_val = mt.simd_broadcast(max_val, 0)

    # Phase 2: Compute exp(x - max) and sum
    exp_val = 0.0
    if idx < N:
        exp_val = mt.exp(val - max_val)

    # SIMD sum for denominator
    sum_exp = exp_val
    sum_exp = sum_exp + mt.simd_shuffle_down(sum_exp, 16)
    sum_exp = sum_exp + mt.simd_shuffle_down(sum_exp, 8)
    sum_exp = sum_exp + mt.simd_shuffle_down(sum_exp, 4)
    sum_exp = sum_exp + mt.simd_shuffle_down(sum_exp, 2)
    sum_exp = sum_exp + mt.simd_shuffle_down(sum_exp, 1)

    # Broadcast sum
    sum_exp = mt.simd_broadcast(sum_exp, 0)

    # Phase 3: Write normalized values
    if idx < N:
        mt.store(out_ptr + idx, exp_val / sum_exp)
