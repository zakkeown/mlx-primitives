"""Vector operation examples for Metal-Triton DSL.

Simple element-wise kernels demonstrating basic DSL usage.
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def vector_add(
    a_ptr,
    b_ptr,
    c_ptr,
    N: constexpr,
):
    """Element-wise vector addition: c = a + b

    Each thread handles one element.

    Args:
        a_ptr: Pointer to first input array
        b_ptr: Pointer to second input array
        c_ptr: Pointer to output array
        N: Number of elements (compile-time constant)
    """
    # Global thread index
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    # Bounds check
    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        mt.store(c_ptr + idx, a + b)


@metal_kernel
def vector_mul(
    a_ptr,
    b_ptr,
    c_ptr,
    N: constexpr,
):
    """Element-wise vector multiplication: c = a * b"""
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        mt.store(c_ptr + idx, a * b)


@metal_kernel
def relu(
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """ReLU activation: y = max(0, x)"""
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        y = mt.maximum(x, 0.0)
        mt.store(y_ptr + idx, y)


@metal_kernel
def fused_add_relu(
    a_ptr,
    b_ptr,
    c_ptr,
    N: constexpr,
):
    """Fused add + ReLU: c = max(0, a + b)

    Demonstrates kernel fusion - single kernel for two operations.
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        a = mt.load(a_ptr + idx)
        b = mt.load(b_ptr + idx)
        result = mt.maximum(a + b, 0.0)
        mt.store(c_ptr + idx, result)


@metal_kernel
def saxpy(
    a: mt.float32,
    x_ptr,
    y_ptr,
    N: constexpr,
):
    """SAXPY: y = a * x + y

    Classic BLAS operation demonstrating scalar parameters.
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    idx = pid * block_size + tid

    if idx < N:
        x = mt.load(x_ptr + idx)
        y = mt.load(y_ptr + idx)
        result = a * x + y
        mt.store(y_ptr + idx, result)


@metal_kernel
def sum_reduce(
    x_ptr,
    out_ptr,
    N: constexpr,
):
    """Simple sum reduction using SIMD and atomics.

    Each threadgroup reduces a portion using SIMD operations,
    then uses atomic_add to accumulate the final result.
    """
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    simd_lane = mt.simd_lane_id()
    block_size = mt.threads_per_threadgroup()

    # Global index
    idx = pid * block_size + tid

    # Load value (0 if out of bounds)
    val = mt.load(x_ptr + idx, mask=idx < N, other=0.0)

    # SIMD reduction within warp
    val = val + mt.simd_shuffle_down(val, 16)
    val = val + mt.simd_shuffle_down(val, 8)
    val = val + mt.simd_shuffle_down(val, 4)
    val = val + mt.simd_shuffle_down(val, 2)
    val = val + mt.simd_shuffle_down(val, 1)

    # First lane of each SIMD group adds to output
    if simd_lane == 0:
        mt.atomic_add(out_ptr, val)


def test_vector_add():
    """Test vector_add kernel.

    Example usage:
        python -c "from mlx_primitives.dsl.examples.vector_ops import test_vector_add; test_vector_add()"
    """
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        print("MLX or NumPy not available, skipping test")
        return

    # Create test data
    N = 1024
    a = mx.random.normal((N,))
    b = mx.random.normal((N,))
    c = mx.zeros((N,))  # Template for output shape/dtype

    # Compute grid (ignored in MLX mode - computed from output shape)
    block_size = 128
    num_blocks = (N + block_size - 1) // block_size

    # Run kernel - returns new output tensor
    result = vector_add(a, b, c, N=N, grid=(num_blocks,))

    # Unpack single output from list
    if isinstance(result, list):
        result = result[0]

    # Verify
    mx.eval(result)
    expected = a + b
    mx.eval(expected)

    if mx.allclose(result, expected):
        print("vector_add: PASSED")
    else:
        print("vector_add: FAILED")
        print(f"  Max error: {mx.max(mx.abs(result - expected))}")


if __name__ == "__main__":
    test_vector_add()
