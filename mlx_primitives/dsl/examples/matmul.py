"""Matrix multiplication example using block pointers.

Demonstrates Triton-style tiled matrix multiplication with
make_block_ptr, advance, load_block, and store_block.
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def matmul_tiled(
    A_ptr,
    B_ptr,
    C_ptr,
    M: constexpr,
    N: constexpr,
    K: constexpr,
    BLOCK_M: constexpr = 32,
    BLOCK_N: constexpr = 32,
    BLOCK_K: constexpr = 16,
):
    """Tiled matrix multiplication: C = A @ B.

    Uses block pointers for efficient 2D tile loading with automatic
    bounds checking.

    Grid: (M // BLOCK_M, N // BLOCK_N)
    Each threadgroup computes one BLOCK_M x BLOCK_N tile of C.
    """
    # Block indices
    pid_m = mt.program_id(0)
    pid_n = mt.program_id(1)

    # Create block pointers for A, B tiles
    # A is M x K, we load BLOCK_M x BLOCK_K tiles
    a_block_ptr = mt.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
    )

    # B is K x N, we load BLOCK_K x BLOCK_N tiles
    b_block_ptr = mt.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
    )

    # Accumulator in shared memory
    acc = mt.shared_memory(BLOCK_M, BLOCK_N, dtype=mt.float32)
    tid = mt.thread_id_in_threadgroup()

    # Initialize accumulator to zero
    for i in range(0, BLOCK_M * BLOCK_N, 256):
        idx = i + tid
        if idx < BLOCK_M * BLOCK_N:
            acc[idx] = 0.0

    mt.threadgroup_barrier()

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A tile into shared memory
        a_tile = mt.load_block(a_block_ptr, boundary_check=(0, 1))
        mt.threadgroup_barrier()

        # Load B tile into shared memory
        b_tile = mt.load_block(b_block_ptr, boundary_check=(0, 1))
        mt.threadgroup_barrier()

        # Compute tile product and accumulate
        # Each thread computes part of the accumulator
        for row in range(0, BLOCK_M, 8):
            for col in range(0, BLOCK_N, 8):
                local_row = row + tid // 32
                local_col = col + tid % 32
                if local_row < BLOCK_M and local_col < BLOCK_N:
                    dot_sum = 0.0
                    for kk in range(BLOCK_K):
                        a_val = a_tile[local_row * BLOCK_K + kk]
                        b_val = b_tile[kk * BLOCK_N + local_col]
                        dot_sum = dot_sum + a_val * b_val
                    acc[local_row * BLOCK_N + local_col] = acc[local_row * BLOCK_N + local_col] + dot_sum

        mt.threadgroup_barrier()

        # Advance block pointers
        a_block_ptr = mt.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = mt.advance(b_block_ptr, (BLOCK_K, 0))

    mt.threadgroup_barrier()

    # Create output block pointer
    c_block_ptr = mt.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
    )

    # Store result
    mt.store_block(c_block_ptr, acc, boundary_check=(0, 1))
