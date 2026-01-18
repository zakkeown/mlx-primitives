"""Metal kernel wrappers for associative scan operations."""

from pathlib import Path
from typing import Literal, Optional

import mlx.core as mx

# Kernel cache to avoid recompilation
_scan_add_kernel: Optional[mx.fast.metal_kernel] = None
_scan_add_batched_kernel: Optional[mx.fast.metal_kernel] = None
_ssm_scan_kernel: Optional[mx.fast.metal_kernel] = None
_scan_mul_kernel: Optional[mx.fast.metal_kernel] = None

# Maximum sequence length for single-block scan
MAX_SINGLE_BLOCK_SEQ = 1024


def _load_metal_source(filename: str) -> str:
    """Load Metal source from file."""
    metal_dir = Path(__file__).parent.parent.parent.parent / "metal" / "primitives"
    metal_file = metal_dir / filename
    if metal_file.exists():
        return metal_file.read_text()
    raise FileNotFoundError(f"Metal source file not found: {metal_file}")


def _get_scan_add_kernel() -> mx.fast.metal_kernel:
    """Get or create the additive scan kernel."""
    global _scan_add_kernel
    if _scan_add_kernel is None:
        source = """
        uint tid = thread_position_in_threadgroup.x;
        uint block_size = threads_per_threadgroup.x;

        // Load input into shared memory
        if (tid < seq_len) {
            shared_mem[tid] = input[tid];
        } else {
            shared_mem[tid] = 0.0f;
        }
        float original = shared_mem[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep (reduce) phase
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    shared_mem[bi] += shared_mem[ai];
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Clear the last element for exclusive scan
        if (tid == 0) {
            shared_mem[block_size - 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep phase
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float t = shared_mem[ai];
                    shared_mem[ai] = shared_mem[bi];
                    shared_mem[bi] += t;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output (inclusive scan: add original value back)
        if (tid < seq_len) {
            output[tid] = shared_mem[tid] + original;
        }
        """
        _scan_add_kernel = mx.fast.metal_kernel(
            name="associative_scan_add",
            input_names=["input", "seq_len"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_kernel


def _get_scan_add_batched_kernel() -> mx.fast.metal_kernel:
    """Get or create the batched additive scan kernel."""
    global _scan_add_batched_kernel
    if _scan_add_batched_kernel is None:
        source = """
        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.x;
        uint block_size = threads_per_threadgroup.x;

        uint base_offset = batch_idx * seq_len;

        // Load input into shared memory
        if (tid < seq_len) {
            shared_mem[tid] = input[base_offset + tid];
        } else {
            shared_mem[tid] = 0.0f;
        }
        float original = shared_mem[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep (reduce) phase
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    shared_mem[bi] += shared_mem[ai];
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Clear the last element for exclusive scan
        if (tid == 0) {
            shared_mem[block_size - 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep phase
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float t = shared_mem[ai];
                    shared_mem[ai] = shared_mem[bi];
                    shared_mem[bi] += t;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output (inclusive scan)
        if (tid < seq_len) {
            output[base_offset + tid] = shared_mem[tid] + original;
        }
        """
        _scan_add_batched_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_batched",
            input_names=["input", "seq_len"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_batched_kernel


def _get_ssm_scan_kernel() -> mx.fast.metal_kernel:
    """Get or create the SSM scan kernel for diagonal A."""
    global _ssm_scan_kernel
    if _ssm_scan_kernel is None:
        source = """
        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.x;
        uint dim_idx = threadgroup_position_in_grid.y;
        uint block_size = threads_per_threadgroup.x;

        // Base offset for this (batch, dim) slice
        uint base = batch_idx * seq_len * d_inner + dim_idx;
        uint stride = d_inner;

        // shared_mem layout: first half for A_shared, second half for h_shared
        threadgroup float* A_shared = shared_mem;
        threadgroup float* h_shared = shared_mem + block_size;

        // Load A and x into shared memory for this sequence dimension
        if (tid < seq_len) {
            A_shared[tid] = A[base + tid * stride];
            h_shared[tid] = x[base + tid * stride];
        } else {
            A_shared[tid] = 1.0f;  // Identity for A product
            h_shared[tid] = 0.0f;  // Zero for h accumulation
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep: compute products of A and partial weighted sums
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    // Combine: (A[ai], h[ai]) ⊕ (A[bi], h[bi])
                    float A_right = A_shared[bi];
                    h_shared[bi] = A_right * h_shared[ai] + h_shared[bi];
                    A_shared[bi] = A_shared[ai] * A_right;
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Set identity at root for down-sweep
        if (tid == 0) {
            A_shared[block_size - 1] = 1.0f;
            h_shared[block_size - 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep: propagate prefix values
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float A_ai = A_shared[ai];
                    float h_ai = h_shared[ai];
                    float A_bi = A_shared[bi];
                    float h_bi = h_shared[bi];

                    A_shared[ai] = A_bi;
                    h_shared[ai] = h_bi;

                    A_shared[bi] = A_ai * A_bi;
                    h_shared[bi] = A_bi * h_ai + h_bi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output: inclusive result
        if (tid < seq_len) {
            float A_t = A[base + tid * stride];
            float x_t = x[base + tid * stride];
            h[base + tid * stride] = A_t * h_shared[tid] + x_t;
        }
        """
        _ssm_scan_kernel = mx.fast.metal_kernel(
            name="ssm_scan_diagonal",
            input_names=["A", "x", "seq_len", "d_inner"],
            output_names=["h"],
            source=source,
        )
    return _ssm_scan_kernel


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def metal_associative_scan(
    x: mx.array,
    operator: Literal["add", "mul"] = "add",
    axis: int = -1,
    inclusive: bool = True,
) -> mx.array:
    """Parallel associative scan using Metal kernel.

    Args:
        x: Input tensor.
        operator: Scan operator - "add" (cumsum) or "mul" (cumprod).
        axis: Axis along which to scan. Must be the last axis for now.
        inclusive: If True, include current element in scan result.

    Returns:
        Scanned tensor of same shape as input.
    """
    # Normalize axis
    if axis < 0:
        axis = x.ndim + axis

    # For now, only support scan along last axis
    if axis != x.ndim - 1:
        # Transpose to move scan axis to last position
        perm = list(range(x.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x = mx.transpose(x, perm)
        result = metal_associative_scan(x, operator, -1, inclusive)
        return mx.transpose(result, perm)

    original_shape = x.shape
    seq_len = x.shape[-1]

    # Flatten to (batch, seq_len)
    if x.ndim == 1:
        batch_size = 1
        x_flat = x.reshape(1, -1)
    else:
        batch_size = int(mx.prod(mx.array(x.shape[:-1])).item())
        x_flat = x.reshape(batch_size, seq_len)

    # Determine block size (must be power of 2 and >= seq_len)
    block_size = _next_power_of_2(seq_len)
    block_size = min(block_size, MAX_SINGLE_BLOCK_SEQ)

    if seq_len > MAX_SINGLE_BLOCK_SEQ:
        # Fall back to sequential for long sequences
        # TODO: Implement multi-block scan
        return _sequential_scan(x, operator, inclusive)

    # Get kernel
    if operator == "add":
        kernel = _get_scan_add_batched_kernel()
    else:
        raise NotImplementedError(f"Metal kernel for operator '{operator}' not implemented yet")

    # Prepare inputs
    x_flat = mx.ascontiguousarray(x_flat.astype(mx.float32))
    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)

    # Shared memory: block_size floats
    shared_mem_size = block_size * 4  # 4 bytes per float

    # Execute kernel
    outputs = kernel(
        inputs=[x_flat, seq_len_arr],
        grid=(batch_size, 1, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(batch_size, seq_len)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    result = outputs[0]

    # Reshape back to original shape
    return result.reshape(original_shape)


def metal_ssm_scan(
    A: mx.array,
    x: mx.array,
) -> mx.array:
    """Parallel SSM scan for diagonal A matrices.

    Computes h[t] = A[t] * h[t-1] + x[t] in O(log n) parallel steps.

    Args:
        A: Discretized state transition (batch, seq_len, d_inner).
           For Mamba, this is exp(delta * A_diag).
        x: Scaled input (batch, seq_len, d_inner).
           For Mamba, this is delta * B * u.

    Returns:
        Hidden states h of shape (batch, seq_len, d_inner).
    """
    if A.shape != x.shape:
        raise ValueError(f"A and x must have same shape, got {A.shape} and {x.shape}")

    if A.ndim != 3:
        raise ValueError(f"Expected 3D tensors (batch, seq, d_inner), got {A.ndim}D")

    batch_size, seq_len, d_inner = A.shape

    if seq_len > MAX_SINGLE_BLOCK_SEQ:
        # Fall back to sequential for long sequences
        return _sequential_ssm_scan(A, x)

    # Get kernel
    kernel = _get_ssm_scan_kernel()

    # Prepare inputs
    A = mx.ascontiguousarray(A.astype(mx.float32))
    x = mx.ascontiguousarray(x.astype(mx.float32))
    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)
    d_inner_arr = mx.array([d_inner], dtype=mx.uint32)

    # Block size for sequence dimension
    block_size = _next_power_of_2(seq_len)
    block_size = min(block_size, MAX_SINGLE_BLOCK_SEQ)

    # Shared memory: 2 * block_size floats (for A_shared and h_shared)
    shared_mem_size = 2 * block_size * 4  # 4 bytes per float

    # Execute kernel: grid is (batch, d_inner), each threadgroup processes one sequence
    outputs = kernel(
        inputs=[A, x, seq_len_arr, d_inner_arr],
        grid=(batch_size, d_inner, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(batch_size, seq_len, d_inner)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _sequential_scan(
    x: mx.array,
    operator: Literal["add", "mul"],
    inclusive: bool = True,
) -> mx.array:
    """Sequential fallback for scan operations."""
    if operator == "add":
        return mx.cumsum(x, axis=-1)
    elif operator == "mul":
        return mx.cumprod(x, axis=-1)
    else:
        raise ValueError(f"Unknown operator: {operator}")


def _sequential_ssm_scan(A: mx.array, x: mx.array) -> mx.array:
    """Sequential fallback for SSM scan."""
    batch_size, seq_len, d_inner = A.shape
    h = mx.zeros((batch_size, seq_len, d_inner), dtype=A.dtype)

    # Sequential recurrence: h[t] = A[t] * h[t-1] + x[t]
    h_prev = mx.zeros((batch_size, d_inner), dtype=A.dtype)
    outputs = []
    for t in range(seq_len):
        h_t = A[:, t, :] * h_prev + x[:, t, :]
        outputs.append(h_t)
        h_prev = h_t

    return mx.stack(outputs, axis=1)
