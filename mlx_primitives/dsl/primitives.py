"""DSL primitives for Metal-Triton.

These functions serve as markers during AST parsing. They are not
executed at runtime - instead, the compiler recognizes them and
generates corresponding Metal code.

Usage:
    import mlx_primitives.dsl as mt

    @mt.metal_kernel
    def my_kernel(x_ptr, y_ptr, N: mt.constexpr):
        pid = mt.program_id(0)  # Recognized and compiled to Metal
        ...
"""

from __future__ import annotations
from typing import Union, Optional, Any, overload, Literal
from mlx_primitives.dsl.types import DType, float32, int32, uint32


# =============================================================================
# Thread/Block Indexing
# =============================================================================

def program_id(axis: Literal[0, 1, 2]) -> int:
    """Get threadgroup position in grid.

    Args:
        axis: 0 for x, 1 for y, 2 for z

    Metal equivalent: threadgroup_position_in_grid.{x,y,z}

    Example:
        pid_x = mt.program_id(0)  # Block index in x dimension
        pid_y = mt.program_id(1)  # Block index in y dimension
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def num_programs(axis: Literal[0, 1, 2]) -> int:
    """Get number of threadgroups in grid.

    Args:
        axis: 0 for x, 1 for y, 2 for z

    Metal equivalent: threadgroups_per_grid.{x,y,z}
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def thread_id_in_threadgroup() -> int:
    """Get thread position within threadgroup (flattened).

    Metal equivalent: thread_position_in_threadgroup.x
        (assuming 1D threadgroup)

    For multi-dimensional threadgroups, use thread_id_in_threadgroup_3d().
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def thread_id_in_threadgroup_3d() -> tuple[int, int, int]:
    """Get 3D thread position within threadgroup.

    Metal equivalent: thread_position_in_threadgroup.{x,y,z}
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def threads_per_threadgroup() -> int:
    """Get number of threads per threadgroup (flattened).

    Metal equivalent: threads_per_threadgroup.x
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_lane_id() -> int:
    """Get thread index within SIMD group (0-31).

    Metal equivalent: thread_index_in_simdgroup

    Apple Silicon uses 32-wide SIMD groups.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_group_id() -> int:
    """Get SIMD group index within threadgroup.

    Metal equivalent: simdgroup_index_in_threadgroup
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Memory Operations
# =============================================================================

def load(
    ptr: Any,
    *,
    mask: Optional[Any] = None,
    other: Any = 0.0,
    use_shared: bool = False,
) -> Any:
    """Load value(s) from device memory.

    Args:
        ptr: Pointer expression (base + offset)
        mask: Optional boolean mask for conditional loads
        other: Default value where mask is False
        use_shared: If True, load cooperatively to threadgroup memory

    Metal equivalent (scalar): ptr[offset]
    Metal equivalent (shared): threadgroup memory with cooperative load

    Example:
        val = mt.load(x_ptr + idx)
        vals = mt.load(x_ptr + idx, mask=idx < N, other=0.0)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def store(ptr: Any, value: Any, *, mask: Optional[Any] = None) -> None:
    """Store value(s) to device memory.

    Args:
        ptr: Pointer expression (base + offset)
        value: Value to store
        mask: Optional boolean mask for conditional stores

    Metal equivalent: ptr[offset] = value

    Example:
        mt.store(y_ptr + idx, result)
        mt.store(y_ptr + idx, result, mask=idx < N)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Shared Memory (Threadgroup Memory)
# =============================================================================

def shared_memory(
    *shape: int,
    dtype: DType = float32,
    padding: int = 4,
) -> Any:
    """Allocate threadgroup (shared) memory.

    Allocates memory shared across all threads in the threadgroup.
    Apple Silicon has 32KB threadgroup memory limit.

    Args:
        *shape: Dimensions of the shared memory block
        dtype: Data type (default: float32)
        padding: Padding for bank conflict avoidance (default: 4)

    Metal equivalent: threadgroup float shared[size];

    Example:
        # Allocate 64x64 tile with padding
        shared_tile = mt.shared_memory(64, 64, dtype=mt.float32)

        # Load cooperatively
        tid = mt.thread_id_in_threadgroup()
        for i in range(0, 64*64, 256):
            idx = i + tid
            if idx < 64*64:
                shared_tile[idx] = mt.load(ptr + idx)

        mt.threadgroup_barrier()

        # Read from shared memory
        val = shared_tile[local_idx]
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def load_shared(
    shared: Any,
    src_ptr: Any,
    count: int,
    *,
    stride: int = 1,
) -> None:
    """Cooperatively load data into shared memory.

    All threads in threadgroup participate in loading data.
    Automatically handles thread-to-element mapping.

    Args:
        shared: Shared memory allocation
        src_ptr: Source pointer in device memory
        count: Number of elements to load
        stride: Stride between elements (default: 1)

    Example:
        mt.load_shared(shared_k, K_ptr + block_start, BLOCK_N * head_dim)
        mt.threadgroup_barrier()
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def store_shared(
    dst_ptr: Any,
    shared: Any,
    count: int,
    *,
    stride: int = 1,
) -> None:
    """Cooperatively store data from shared memory to device memory.

    Args:
        dst_ptr: Destination pointer in device memory
        shared: Shared memory allocation
        count: Number of elements to store
        stride: Stride between elements (default: 1)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Block Pointers (Triton-style tiled access)
# =============================================================================

def make_block_ptr(
    base: Any,
    shape: tuple,
    strides: tuple,
    offsets: tuple,
    block_shape: tuple,
    order: tuple = None,
) -> Any:
    """Create a block pointer for tiled memory access.

    Block pointers enable efficient loading/storing of 2D tiles
    with automatic bounds checking and strided access.

    Args:
        base: Base pointer to tensor data
        shape: Shape of the underlying tensor (M, N, ...)
        strides: Strides for each dimension (stride_m, stride_n, ...)
        offsets: Starting offset for this block (offset_m, offset_n, ...)
        block_shape: Shape of the block to load (BLOCK_M, BLOCK_N)
        order: Memory layout order (default: row-major)

    Example:
        # Create block pointer for Q matrix tile
        q_block_ptr = mt.make_block_ptr(
            base=Q_ptr,
            shape=(seq_len, head_dim),
            strides=(head_dim, 1),
            offsets=(block_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, head_dim),
        )

        # Load the block into shared memory
        q = mt.load(q_block_ptr, use_shared=True)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def advance(block_ptr: Any, offsets: tuple) -> Any:
    """Advance a block pointer by given offsets.

    Args:
        block_ptr: Block pointer to advance
        offsets: Offset to add to each dimension

    Example:
        # Move to next K block
        k_block_ptr = mt.advance(k_block_ptr, (BLOCK_K, 0))
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def load_block(
    block_ptr: Any,
    *,
    boundary_check: tuple = None,
    padding_option: str = "zero",
) -> Any:
    """Load a block of data using a block pointer.

    Efficiently loads a 2D tile with optional boundary checking.

    Args:
        block_ptr: Block pointer created with make_block_ptr
        boundary_check: Which dimensions to check bounds (default: all)
        padding_option: How to handle out-of-bounds ("zero" or "nan")

    Example:
        k = mt.load_block(k_block_ptr, boundary_check=(0,))
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def store_block(
    block_ptr: Any,
    value: Any,
    *,
    boundary_check: tuple = None,
) -> None:
    """Store a block of data using a block pointer.

    Args:
        block_ptr: Block pointer for destination
        value: Block of data to store
        boundary_check: Which dimensions to check bounds
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Array Creation
# =============================================================================

def zeros(shape: Union[int, tuple], dtype: DType = float32) -> Any:
    """Create zero-initialized array.

    Args:
        shape: Scalar or tuple of dimensions
        dtype: Data type

    Example:
        acc = mt.zeros((BLOCK_M, BLOCK_N), dtype=mt.float32)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def full(shape: Union[int, tuple], fill_value: Any, dtype: DType = float32) -> Any:
    """Create array filled with a value.

    Args:
        shape: Scalar or tuple of dimensions
        fill_value: Value to fill
        dtype: Data type

    Example:
        m_i = mt.full((BLOCK_M,), float('-inf'), dtype=mt.float32)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def arange(start: int, end: int, dtype: DType = int32) -> Any:
    """Create array with sequential values.

    Args:
        start: Start value (inclusive)
        end: End value (exclusive)
        dtype: Data type

    Example:
        offsets = mt.arange(0, BLOCK_SIZE)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Math Operations
# =============================================================================

def dot(a: Any, b: Any) -> Any:
    """Matrix multiplication or dot product.

    Args:
        a: Left operand
        b: Right operand (transposed if needed via mt.trans())

    Example:
        scores = mt.dot(q, mt.trans(k)) * scale
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def trans(x: Any) -> Any:
    """Transpose a 2D block.

    Args:
        x: 2D array to transpose

    Example:
        k_t = mt.trans(k)  # Transpose K for QK^T
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def maximum(a: Any, b: Any) -> Any:
    """Element-wise maximum.

    Metal equivalent: max(a, b)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def minimum(a: Any, b: Any) -> Any:
    """Element-wise minimum.

    Metal equivalent: min(a, b)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def exp(x: Any) -> Any:
    """Element-wise exponential.

    Metal equivalent: exp(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def log(x: Any) -> Any:
    """Element-wise natural logarithm.

    Metal equivalent: log(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def sqrt(x: Any) -> Any:
    """Element-wise square root.

    Metal equivalent: sqrt(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def abs(x: Any) -> Any:
    """Element-wise absolute value.

    Metal equivalent: abs(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def where(condition: Any, x: Any, y: Any) -> Any:
    """Conditional selection.

    Args:
        condition: Boolean condition
        x: Value where True
        y: Value where False

    Metal equivalent: condition ? x : y (or select())
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def fma(a: Any, b: Any, c: Any) -> Any:
    """Fused multiply-add: a * b + c.

    Single instruction that is both faster and more accurate than
    separate multiply and add operations.

    Metal equivalent: fma(a, b, c)

    Example:
        result = mt.fma(weight, input, bias)  # weight * input + bias
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def tanh(x: Any) -> Any:
    """Element-wise hyperbolic tangent.

    Metal equivalent: tanh(x)

    Commonly used in activation functions like GELU.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def erf(x: Any) -> Any:
    """Element-wise error function.

    Metal equivalent: erf(x)

    Used in exact GELU computation.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def cos(x: Any) -> Any:
    """Element-wise cosine.

    Metal equivalent: cos(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def sin(x: Any) -> Any:
    """Element-wise sine.

    Metal equivalent: sin(x)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def rsqrt(x: Any) -> Any:
    """Element-wise reciprocal square root (1 / sqrt(x)).

    Metal equivalent: rsqrt(x)

    More efficient than 1.0 / sqrt(x) for normalization.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Type Casting
# =============================================================================

def cast(x: Any, dtype: DType) -> Any:
    """Cast value to specified data type.

    Used for explicit precision control, especially in mixed-precision
    operations.

    Args:
        x: Value to cast
        dtype: Target data type (mt.float16, mt.float32, mt.int32, etc.)

    Metal equivalent: T(x) or static_cast<T>(x)

    Example:
        # Compute in fp32, store as fp16
        result = compute_in_fp32(...)
        mt.store(ptr, mt.cast(result, mt.float16))

        # Load fp16, compute in fp32
        val_fp16 = mt.load(ptr)  # half
        val_fp32 = mt.cast(val_fp16, mt.float32)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def reinterpret_cast(x: Any, dtype: DType) -> Any:
    """Reinterpret bit pattern as different type.

    Does not convert values - interprets the same bits differently.
    Use for bit manipulation or accessing raw representations.

    Metal equivalent: as_type<T>(x)

    Example:
        # Get bit representation of float
        bits = mt.reinterpret_cast(float_val, mt.uint32)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Reduction Operations
# =============================================================================

def sum(x: Any, axis: Optional[int] = None) -> Any:
    """Sum reduction.

    Args:
        x: Input array
        axis: Axis to reduce (None for all)

    Example:
        total = mt.sum(x)  # Sum all elements
        row_sums = mt.sum(x, axis=1)  # Sum along axis 1
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def max(x: Any, axis: Optional[int] = None) -> Any:
    """Maximum reduction.

    Args:
        x: Input array
        axis: Axis to reduce (None for all)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def min(x: Any, axis: Optional[int] = None) -> Any:
    """Minimum reduction.

    Args:
        x: Input array
        axis: Axis to reduce (None for all)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Synchronization
# =============================================================================

def threadgroup_barrier() -> None:
    """Synchronize all threads in threadgroup.

    Metal equivalent: threadgroup_barrier(mem_flags::mem_threadgroup)

    Use after writing to shared memory before other threads read it.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_barrier() -> None:
    """Synchronize threads within SIMD group.

    Metal equivalent: simdgroup_barrier(mem_flags::mem_none)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# SIMD Group Operations
# =============================================================================

def simd_shuffle_down(value: Any, delta: int) -> Any:
    """Shuffle value down within SIMD group.

    Gets value from lane (current_lane + delta).

    Metal equivalent: simd_shuffle_down(value, delta)

    Example (tree reduction):
        val = mt.simd_shuffle_down(val, 16)  # Get from lane+16
        val = mt.simd_shuffle_down(val, 8)   # Get from lane+8
        ...
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_shuffle_up(value: Any, delta: int) -> Any:
    """Shuffle value up within SIMD group.

    Gets value from lane (current_lane - delta).

    Metal equivalent: simd_shuffle_up(value, delta)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_shuffle_xor(value: Any, mask: int) -> Any:
    """Shuffle value using XOR mask within SIMD group.

    Gets value from lane (current_lane ^ mask).

    Metal equivalent: simd_shuffle_xor(value, mask)

    Example (butterfly reduction):
        val += mt.simd_shuffle_xor(val, 16)
        val += mt.simd_shuffle_xor(val, 8)
        val += mt.simd_shuffle_xor(val, 4)
        val += mt.simd_shuffle_xor(val, 2)
        val += mt.simd_shuffle_xor(val, 1)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_broadcast(value: Any, lane: int) -> Any:
    """Broadcast value from specific lane to all lanes.

    Metal equivalent: simd_broadcast(value, lane)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_sum(value: Any) -> Any:
    """Sum across SIMD group.

    Metal equivalent: simd_sum(value)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_max(value: Any) -> Any:
    """Maximum across SIMD group.

    Metal equivalent: simd_max(value)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def simd_min(value: Any) -> Any:
    """Minimum across SIMD group.

    Metal equivalent: simd_min(value)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Atomic Operations
# =============================================================================

def atomic_add(ptr: Any, value: Any) -> Any:
    """Atomic addition. Returns old value.

    Metal equivalent: atomic_fetch_add_explicit(ptr, value, memory_order_relaxed)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def atomic_max(ptr: Any, value: Any) -> Any:
    """Atomic maximum. Returns old value.

    Metal equivalent: atomic_fetch_max_explicit(ptr, value, memory_order_relaxed)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def atomic_min(ptr: Any, value: Any) -> Any:
    """Atomic minimum. Returns old value.

    Metal equivalent: atomic_fetch_min_explicit(ptr, value, memory_order_relaxed)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def atomic_cas(ptr: Any, expected: Any, desired: Any) -> Any:
    """Atomic compare-and-swap. Returns old value.

    Metal equivalent: atomic_compare_exchange_weak_explicit(...)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Debug Utilities
# =============================================================================

def debug_print(fmt: str, *args: Any) -> None:
    """Print debug output (Metal printf, only works in debug mode).

    This is a no-op in release builds.
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def debug_barrier() -> None:
    """Barrier with optional debug check."""
    raise NotImplementedError("DSL primitive - compiled to Metal")


def static_assert(condition: bool, message: str = "") -> None:
    """Compile-time assertion.

    Fails compilation if condition is not met. Useful for validating
    constexpr values.

    Args:
        condition: Must be compile-time evaluable to true
        message: Error message if assertion fails

    Metal equivalent: static_assert(condition, "message");

    Example:
        mt.static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of 32")
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


# =============================================================================
# Loop Control
# =============================================================================

def static_for(start: int, end: int, step: int = 1):
    """Unrolled iteration over compile-time known range.

    The loop is fully unrolled at compile time. All bounds must be
    compile-time constants (constexpr values or literals).

    Args:
        start: Starting value (inclusive)
        end: Ending value (exclusive)
        step: Step size (default: 1)

    Returns:
        Iterator for use in for loop

    Example:
        # Fully unrolled loop - generates 4 separate statements
        for i in mt.static_for(0, 4):
            val = mt.load(ptr + i)
            acc += val

        # With step
        for i in mt.static_for(0, 16, 4):
            vec = mt.load_vec4(ptr + i)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


class unroll:
    """Decorator hint for loop unrolling.

    Can specify unroll factor or use 0 for full unroll.
    Apply as a decorator on for loops using range().

    Note: In Python DSL, this is typically used as a comment hint
    or by wrapping the range with this marker.

    Example:
        # In practice, use mt.static_for for full unrolling:
        for i in mt.static_for(0, 16):
            ...

        # Or use a pragma-style comment (recognized by parser):
        # @mt.unroll(4)
        for i in range(N):
            ...
    """
    def __init__(self, factor: int = 0):
        """
        Args:
            factor: Unroll factor. 0 means full unroll.
        """
        self.factor = factor

    def __call__(self, loop):
        return loop


# =============================================================================
# Vector (SIMD) Types
# =============================================================================

def load_vec2(ptr: Any, *, mask: Optional[Any] = None, other: Any = 0.0) -> Any:
    """Load 2 consecutive elements as a vector.

    Metal equivalent: *(device float2*)(ptr)

    More efficient than 2 separate loads for aligned data.

    Args:
        ptr: Pointer to first element (must be 8-byte aligned for float2)
        mask: Optional boolean mask
        other: Default value where mask is False

    Example:
        vec = mt.load_vec2(ptr + idx * 2)
        x, y = vec.x, vec.y
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def load_vec4(ptr: Any, *, mask: Optional[Any] = None, other: Any = 0.0) -> Any:
    """Load 4 consecutive elements as a vector.

    Metal equivalent: *(device float4*)(ptr)

    4x more memory efficient than scalar loads for aligned data.

    Args:
        ptr: Pointer to first element (must be 16-byte aligned for float4)
        mask: Optional boolean mask
        other: Default value where mask is False

    Example:
        vec = mt.load_vec4(ptr + idx * 4)  # Load 4 floats
        # vec.x, vec.y, vec.z, vec.w are the 4 elements
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def store_vec2(ptr: Any, value: Any, *, mask: Optional[Any] = None) -> None:
    """Store 2-element vector to consecutive memory locations.

    Metal equivalent: *(device float2*)(ptr) = value

    Args:
        ptr: Pointer to first element (must be 8-byte aligned)
        value: 2-element vector to store
        mask: Optional boolean mask
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def store_vec4(ptr: Any, value: Any, *, mask: Optional[Any] = None) -> None:
    """Store 4-element vector to consecutive memory locations.

    Metal equivalent: *(device float4*)(ptr) = value

    Args:
        ptr: Pointer to first element (must be 16-byte aligned)
        value: 4-element vector to store
        mask: Optional boolean mask
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def vec2(x: Any, y: Any) -> Any:
    """Construct a 2-element vector.

    Metal equivalent: float2(x, y)

    Example:
        v = mt.vec2(1.0, 2.0)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def vec4(x: Any, y: Any, z: Any, w: Any) -> Any:
    """Construct a 4-element vector.

    Metal equivalent: float4(x, y, z, w)

    Example:
        v = mt.vec4(1.0, 2.0, 3.0, 4.0)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")


def swizzle(vec: Any, components: str) -> Any:
    """Swizzle vector components.

    Select and reorder vector components using component names.

    Args:
        vec: Input vector
        components: String of component names (x, y, z, w or r, g, b, a)

    Metal equivalent: vec.{components}

    Example:
        v = mt.vec4(1, 2, 3, 4)
        xy = mt.swizzle(v, "xy")  # float2(1, 2)
        wzyx = mt.swizzle(v, "wzyx")  # float4(4, 3, 2, 1)
        xx = mt.swizzle(v, "xx")  # float2(1, 1)
    """
    raise NotImplementedError("DSL primitive - compiled to Metal")
