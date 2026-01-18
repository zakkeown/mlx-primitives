"""Metal-Triton: A Triton-like DSL for Apple Silicon.

Write Metal kernels using Python syntax:

    from mlx_primitives.dsl import metal_kernel
    import mlx_primitives.dsl as mt

    @metal_kernel
    def vector_add(a_ptr, b_ptr, c_ptr, N: mt.constexpr):
        pid = mt.program_id(0)
        idx = pid * 256 + mt.thread_id_in_threadgroup()
        if idx < N:
            a = mt.load(a_ptr + idx)
            b = mt.load(b_ptr + idx)
            mt.store(c_ptr + idx, a + b)
"""

from mlx_primitives.dsl.types import (
    constexpr,
    float16,
    float32,
    int32,
    uint32,
    bool_,
    Pointer,
    dtype,
)

from mlx_primitives.dsl.primitives import (
    # Thread/block indexing
    program_id,
    num_programs,
    thread_id_in_threadgroup,
    thread_id_in_threadgroup_3d,
    threads_per_threadgroup,
    simd_lane_id,
    simd_group_id,
    # Memory operations
    load,
    store,
    # Shared memory
    shared_memory,
    load_shared,
    store_shared,
    # Block pointers
    make_block_ptr,
    advance,
    load_block,
    store_block,
    # Array creation
    zeros,
    full,
    arange,
    # Math operations
    dot,
    trans,
    maximum,
    minimum,
    exp,
    log,
    sqrt,
    rsqrt,
    abs,
    where,
    fma,
    tanh,
    erf,
    cos,
    sin,
    # Type casting
    cast,
    reinterpret_cast,
    # Reductions
    sum,
    max,
    min,
    # Synchronization
    threadgroup_barrier,
    simd_barrier,
    # SIMD operations
    simd_shuffle_down,
    simd_shuffle_up,
    simd_shuffle_xor,
    simd_broadcast,
    simd_sum,
    simd_max,
    simd_min,
    # Atomics
    atomic_add,
    atomic_max,
    atomic_min,
    atomic_cas,
    # Vector operations
    load_vec2,
    load_vec4,
    store_vec2,
    store_vec4,
    vec2,
    vec4,
    swizzle,
    # Loop control
    static_for,
    unroll,
    # Debug utilities
    debug_print,
    debug_barrier,
    static_assert,
)

from mlx_primitives.dsl.decorators import metal_kernel, autotune, Config

__all__ = [
    # Decorators
    "metal_kernel",
    "autotune",
    "Config",
    # Types
    "constexpr",
    "float16",
    "float32",
    "int32",
    "uint32",
    "bool_",
    "Pointer",
    "dtype",
    # Thread/block indexing
    "program_id",
    "num_programs",
    "thread_id_in_threadgroup",
    "thread_id_in_threadgroup_3d",
    "threads_per_threadgroup",
    "simd_lane_id",
    "simd_group_id",
    # Memory operations
    "load",
    "store",
    # Shared memory
    "shared_memory",
    "load_shared",
    "store_shared",
    # Block pointers
    "make_block_ptr",
    "advance",
    "load_block",
    "store_block",
    # Array creation
    "zeros",
    "full",
    "arange",
    # Math operations
    "dot",
    "trans",
    "maximum",
    "minimum",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "abs",
    "where",
    "fma",
    "tanh",
    "erf",
    "cos",
    "sin",
    # Type casting
    "cast",
    "reinterpret_cast",
    # Reductions
    "sum",
    "max",
    "min",
    # Synchronization
    "threadgroup_barrier",
    "simd_barrier",
    # SIMD operations
    "simd_shuffle_down",
    "simd_shuffle_up",
    "simd_shuffle_xor",
    "simd_broadcast",
    "simd_sum",
    "simd_max",
    "simd_min",
    # Atomics
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_cas",
    # Vector operations
    "load_vec2",
    "load_vec4",
    "store_vec2",
    "store_vec4",
    "vec2",
    "vec4",
    "swizzle",
    # Loop control
    "static_for",
    "unroll",
    # Debug utilities
    "debug_print",
    "debug_barrier",
    "static_assert",
]
