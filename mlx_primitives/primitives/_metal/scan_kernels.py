"""Metal kernel wrappers for associative scan operations.

Optimizations:
1. Bank conflict-free shared memory access
2. SIMD warp-level intrinsics for intra-warp scan
3. Multi-block scan for sequences > 1024
4. SSM-specific SIMD optimizations
5. Vectorized (float4) memory access for improved bandwidth
"""

from pathlib import Path
from typing import Literal, Optional

import mlx.core as mx

# Kernel cache to avoid recompilation
_scan_add_kernel: Optional[mx.fast.metal_kernel] = None
_scan_add_batched_kernel: Optional[mx.fast.metal_kernel] = None
_scan_add_simd_kernel: Optional[mx.fast.metal_kernel] = None
_scan_add_batched_simd_kernel: Optional[mx.fast.metal_kernel] = None
_scan_add_vectorized_kernel: Optional[mx.fast.metal_kernel] = None
_ssm_scan_kernel: Optional[mx.fast.metal_kernel] = None
_ssm_scan_simd_kernel: Optional[mx.fast.metal_kernel] = None
_scan_mul_kernel: Optional[mx.fast.metal_kernel] = None
_multiblock_phase1_kernel: Optional[mx.fast.metal_kernel] = None
_multiblock_phase2_kernel: Optional[mx.fast.metal_kernel] = None

# Maximum sequence length for single-block scan
MAX_SINGLE_BLOCK_SEQ = 1024

# Bank conflict avoidance offset
CONFLICT_FREE_OFFSET = lambda n: n >> 5  # n / 32


def _load_metal_source(filename: str) -> str:
    """Load Metal source from file."""
    metal_dir = Path(__file__).parent.parent.parent.parent / "metal" / "primitives"
    metal_file = metal_dir / filename
    if metal_file.exists():
        return metal_file.read_text()
    raise FileNotFoundError(f"Metal source file not found: {metal_file}")


def _get_scan_add_simd_kernel() -> mx.fast.metal_kernel:
    """Get or create the SIMD-optimized additive scan kernel."""
    global _scan_add_simd_kernel
    if _scan_add_simd_kernel is None:
        source = """
        // Threadgroup memory for warp totals (max 32 warps)
        threadgroup float shared[32];

        uint tid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];

        // Load input value
        float val = (tid < slen) ? input[tid] : 0.0f;

        // Phase 1: Intra-warp inclusive scan using SIMD intrinsics
        float warp_scan = simd_prefix_inclusive_sum(val);

        // Phase 2: Store warp totals
        uint num_warps = (block_size + 31) / 32;
        if (simd_lane == 31) {
            shared[simd_group] = warp_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Scan warp totals (first warp only)
        if (simd_group == 0 && simd_lane < num_warps) {
            float warp_total = shared[simd_lane];
            float scanned_total = simd_prefix_exclusive_sum(warp_total);
            shared[simd_lane] = scanned_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Add warp prefix
        float warp_prefix = (simd_group > 0) ? shared[simd_group] : 0.0f;
        float result = warp_scan + warp_prefix;

        // Write output
        if (tid < slen) {
            output[tid] = result;
        }
        """
        _scan_add_simd_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_simd",
            input_names=["input", "seq_len"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_simd_kernel


def _get_scan_add_batched_simd_kernel() -> mx.fast.metal_kernel:
    """Get or create the batched SIMD-optimized additive scan kernel."""
    global _scan_add_batched_simd_kernel
    if _scan_add_batched_simd_kernel is None:
        source = """
        // Threadgroup memory for warp totals (max 32 warps)
        threadgroup float shared[32];

        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.y;  // 2D grid: y = batch
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];
        uint bsize = batch_size[0];

        if (batch_idx >= bsize) return;

        uint base_offset = batch_idx * slen;

        // Load input value
        float val = (tid < slen) ? input[base_offset + tid] : 0.0f;

        // Phase 1: Intra-warp inclusive scan
        float warp_scan = simd_prefix_inclusive_sum(val);

        // Phase 2: Store warp totals
        uint num_warps = (block_size + 31) / 32;
        if (simd_lane == 31) {
            shared[simd_group] = warp_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Scan warp totals
        if (simd_group == 0 && simd_lane < num_warps) {
            float warp_total = shared[simd_lane];
            float scanned_total = simd_prefix_exclusive_sum(warp_total);
            shared[simd_lane] = scanned_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Add warp prefix
        float warp_prefix = (simd_group > 0) ? shared[simd_group] : 0.0f;
        float result = warp_scan + warp_prefix;

        // Write output
        if (tid < slen) {
            output[base_offset + tid] = result;
        }
        """
        _scan_add_batched_simd_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_batched_simd",
            input_names=["input", "seq_len", "batch_size"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_batched_simd_kernel


def _get_scan_add_vectorized_kernel() -> mx.fast.metal_kernel:
    """Get or create the vectorized (float4) additive scan kernel.

    Uses float4 vector loads/stores for better memory bandwidth utilization.
    Each thread processes 4 consecutive elements.
    """
    global _scan_add_vectorized_kernel
    if _scan_add_vectorized_kernel is None:
        source = """
        // Threadgroup memory for chunk totals (max 256 chunks = 1024 elements / 4)
        threadgroup float shared[256];

        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.y;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint block_size = threads_per_threadgroup.x;  // Number of chunks
        uint slen = seq_len[0];
        uint bsize = batch_size[0];
        uint num_chunks = (slen + 3) / 4;

        if (batch_idx >= bsize) return;

        uint base_offset = batch_idx * slen;

        // Each thread handles a chunk of 4 elements
        uint chunk_start = tid * 4;

        // Load 4 elements (handle boundary)
        float4 vals;
        if (chunk_start + 3 < slen) {
            // Full chunk - use vector load
            vals = float4(
                input[base_offset + chunk_start],
                input[base_offset + chunk_start + 1],
                input[base_offset + chunk_start + 2],
                input[base_offset + chunk_start + 3]
            );
        } else {
            // Partial chunk at boundary
            vals.x = (chunk_start < slen) ? input[base_offset + chunk_start] : 0.0f;
            vals.y = (chunk_start + 1 < slen) ? input[base_offset + chunk_start + 1] : 0.0f;
            vals.z = (chunk_start + 2 < slen) ? input[base_offset + chunk_start + 2] : 0.0f;
            vals.w = (chunk_start + 3 < slen) ? input[base_offset + chunk_start + 3] : 0.0f;
        }

        // Local inclusive scan within chunk
        float v0 = vals.x;
        float v1 = v0 + vals.y;
        float v2 = v1 + vals.z;
        float v3 = v2 + vals.w;

        // Store chunk total for inter-chunk scan
        float chunk_total = v3;

        // Phase 1: Intra-warp scan of chunk totals
        float warp_scan = simd_prefix_inclusive_sum(chunk_total);

        // Phase 2: Store warp totals
        uint num_warps = (block_size + 31) / 32;
        if (simd_lane == 31) {
            shared[simd_group] = warp_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Scan warp totals (first warp only)
        if (simd_group == 0 && simd_lane < num_warps) {
            float warp_total = shared[simd_lane];
            float scanned_total = simd_prefix_exclusive_sum(warp_total);
            shared[simd_lane] = scanned_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Compute chunk prefix
        float warp_prefix = (simd_group > 0) ? shared[simd_group] : 0.0f;
        float chunk_prefix = warp_scan - chunk_total + warp_prefix;  // Exclusive prefix for this chunk

        // Apply chunk prefix to local results
        v0 += chunk_prefix;
        v1 += chunk_prefix;
        v2 += chunk_prefix;
        v3 += chunk_prefix;

        // Write output
        if (chunk_start < slen) {
            output[base_offset + chunk_start] = v0;
        }
        if (chunk_start + 1 < slen) {
            output[base_offset + chunk_start + 1] = v1;
        }
        if (chunk_start + 2 < slen) {
            output[base_offset + chunk_start + 2] = v2;
        }
        if (chunk_start + 3 < slen) {
            output[base_offset + chunk_start + 3] = v3;
        }
        """
        _scan_add_vectorized_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_vectorized",
            input_names=["input", "seq_len", "batch_size"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_vectorized_kernel


def _get_scan_add_kernel() -> mx.fast.metal_kernel:
    """Get or create the additive scan kernel (bank conflict-free fallback)."""
    global _scan_add_kernel
    if _scan_add_kernel is None:
        source = """
        // Threadgroup memory with padding for bank conflict avoidance
        // 1024 elements + 32 padding = 1056
        threadgroup float shared[1056];

        uint tid = thread_position_in_threadgroup.x;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];

        // Conflict-free index helper
        #define CF_IDX(n) ((n) + ((n) >> 5))

        // Load input into shared memory with conflict-free indexing
        if (tid < slen) {
            shared[CF_IDX(tid)] = input[tid];
        } else {
            shared[CF_IDX(tid)] = 0.0f;
        }
        float original = shared[CF_IDX(tid)];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep (reduce) phase
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    shared[CF_IDX(bi)] += shared[CF_IDX(ai)];
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Clear the last element for exclusive scan
        if (tid == 0) {
            shared[CF_IDX(block_size - 1)] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep phase
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float t = shared[CF_IDX(ai)];
                    shared[CF_IDX(ai)] = shared[CF_IDX(bi)];
                    shared[CF_IDX(bi)] += t;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output (inclusive scan: add original value back)
        if (tid < slen) {
            output[tid] = shared[CF_IDX(tid)] + original;
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
    """Get or create the batched additive scan kernel (bank conflict-free fallback)."""
    global _scan_add_batched_kernel
    if _scan_add_batched_kernel is None:
        source = """
        // Threadgroup memory with padding for bank conflict avoidance
        // 1024 elements + 32 padding = 1056
        threadgroup float shared[1056];

        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.x;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];

        // Conflict-free index helper
        #define CF_IDX(n) ((n) + ((n) >> 5))

        uint base_offset = batch_idx * slen;

        // Load input into shared memory
        if (tid < slen) {
            shared[CF_IDX(tid)] = input[base_offset + tid];
        } else {
            shared[CF_IDX(tid)] = 0.0f;
        }
        float original = shared[CF_IDX(tid)];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep (reduce) phase
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    shared[CF_IDX(bi)] += shared[CF_IDX(ai)];
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Clear the last element for exclusive scan
        if (tid == 0) {
            shared[CF_IDX(block_size - 1)] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep phase
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float t = shared[CF_IDX(ai)];
                    shared[CF_IDX(ai)] = shared[CF_IDX(bi)];
                    shared[CF_IDX(bi)] += t;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output (inclusive scan)
        if (tid < slen) {
            output[base_offset + tid] = shared[CF_IDX(tid)] + original;
        }
        """
        _scan_add_batched_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_batched",
            input_names=["input", "seq_len"],
            output_names=["output"],
            source=source,
        )
    return _scan_add_batched_kernel


def _get_ssm_scan_simd_kernel() -> mx.fast.metal_kernel:
    """Get or create the SIMD-optimized SSM scan kernel for diagonal A."""
    global _ssm_scan_simd_kernel
    if _ssm_scan_simd_kernel is None:
        source = """
        // Threadgroup memory for warp totals (max 32 warps * 2 values each)
        threadgroup float shared[64];

        uint tid = thread_position_in_threadgroup.x;
        uint dim_idx = threadgroup_position_in_grid.y;   // 3D grid: y = d_inner
        uint batch_idx = threadgroup_position_in_grid.z; // 3D grid: z = batch
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];
        uint dinn = d_inner[0];
        uint bsize = batch_size[0];

        if (batch_idx >= bsize || dim_idx >= dinn) return;

        uint base = batch_idx * slen * dinn + dim_idx;
        uint stride = dinn;

        // Load values
        float A_val = (tid < slen) ? A[base + tid * stride] : 1.0f;
        float h_val = (tid < slen) ? x[base + tid * stride] : 0.0f;

        // Phase 1: Intra-warp SSM scan using shuffle
        float A_scan = A_val;
        float h_scan = h_val;

        for (uint delta = 1; delta < 32; delta *= 2) {
            float A_other = simd_shuffle_up(A_scan, delta);
            float h_other = simd_shuffle_up(h_scan, delta);

            if (simd_lane >= delta) {
                h_scan = A_scan * h_other + h_scan;
                A_scan = A_other * A_scan;
            }
        }

        // Phase 2: Store warp totals
        uint num_warps = (block_size + 31) / 32;
        if (simd_lane == 31) {
            shared[simd_group * 2] = A_scan;
            shared[simd_group * 2 + 1] = h_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Scan warp totals (first warp only)
        float A_prefix = 1.0f;
        float h_prefix = 0.0f;

        if (simd_group == 0 && simd_lane < num_warps) {
            float A_wt = shared[simd_lane * 2];
            float h_wt = shared[simd_lane * 2 + 1];

            float A_scan_wt = A_wt;
            float h_scan_wt = h_wt;

            for (uint delta = 1; delta < 32; delta *= 2) {
                float A_other = simd_shuffle_up(A_scan_wt, delta);
                float h_other = simd_shuffle_up(h_scan_wt, delta);

                if (simd_lane >= delta) {
                    h_scan_wt = A_scan_wt * h_other + h_scan_wt;
                    A_scan_wt = A_other * A_scan_wt;
                }
            }

            // Store exclusive prefix
            float A_exc = simd_shuffle_up(A_scan_wt, 1);
            float h_exc = simd_shuffle_up(h_scan_wt, 1);
            if (simd_lane == 0) {
                A_exc = 1.0f;
                h_exc = 0.0f;
            }

            shared[simd_lane * 2] = A_exc;
            shared[simd_lane * 2 + 1] = h_exc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Apply warp prefix
        if (simd_group > 0) {
            A_prefix = shared[simd_group * 2];
            h_prefix = shared[simd_group * 2 + 1];
        }

        // Combine: (A_prefix, h_prefix) op (A_scan, h_scan)
        float h_final = A_scan * h_prefix + h_scan;

        // Write output
        if (tid < slen) {
            h[base + tid * stride] = h_final;
        }
        """
        _ssm_scan_simd_kernel = mx.fast.metal_kernel(
            name="ssm_scan_diagonal_simd",
            input_names=["A", "x", "seq_len", "d_inner", "batch_size"],
            output_names=["h"],
            source=source,
        )
    return _ssm_scan_simd_kernel


def _get_ssm_scan_kernel() -> mx.fast.metal_kernel:
    """Get or create the SSM scan kernel for diagonal A (bank conflict-free fallback)."""
    global _ssm_scan_kernel
    if _ssm_scan_kernel is None:
        source = """
        // Threadgroup memory: two arrays with padding for bank conflict avoidance
        // Each array: 1024 elements + 32 padding = 1056
        // Total: 2112 floats for A_shared and h_shared
        threadgroup float shared[2112];

        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = threadgroup_position_in_grid.x;
        uint dim_idx = threadgroup_position_in_grid.y;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];
        uint dinn = d_inner[0];

        // Conflict-free index helper
        #define CF_IDX(n) ((n) + ((n) >> 5))

        // Base offset for this (batch, dim) slice
        uint base = batch_idx * slen * dinn + dim_idx;
        uint stride = dinn;

        // shared layout: first half for A_shared, second half for h_shared
        uint shared_size = 1056;
        threadgroup float* A_shared = shared;
        threadgroup float* h_shared = shared + shared_size;

        // Load A and x into shared memory for this sequence dimension
        if (tid < slen) {
            A_shared[CF_IDX(tid)] = A[base + tid * stride];
            h_shared[CF_IDX(tid)] = x[base + tid * stride];
        } else {
            A_shared[CF_IDX(tid)] = 1.0f;  // Identity for A product
            h_shared[CF_IDX(tid)] = 0.0f;  // Zero for h accumulation
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep: compute products of A and partial weighted sums
        uint offset = 1;
        for (uint d = block_size >> 1; d > 0; d >>= 1) {
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float A_right = A_shared[CF_IDX(bi)];
                    h_shared[CF_IDX(bi)] = A_right * h_shared[CF_IDX(ai)] + h_shared[CF_IDX(bi)];
                    A_shared[CF_IDX(bi)] = A_shared[CF_IDX(ai)] * A_right;
                }
            }
            offset *= 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Set identity at root for down-sweep
        if (tid == 0) {
            A_shared[CF_IDX(block_size - 1)] = 1.0f;
            h_shared[CF_IDX(block_size - 1)] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep: propagate prefix values
        for (uint d = 1; d < block_size; d *= 2) {
            offset >>= 1;
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                if (bi < block_size) {
                    float A_ai = A_shared[CF_IDX(ai)];
                    float h_ai = h_shared[CF_IDX(ai)];
                    float A_bi = A_shared[CF_IDX(bi)];
                    float h_bi = h_shared[CF_IDX(bi)];

                    A_shared[CF_IDX(ai)] = A_bi;
                    h_shared[CF_IDX(ai)] = h_bi;

                    A_shared[CF_IDX(bi)] = A_ai * A_bi;
                    h_shared[CF_IDX(bi)] = A_bi * h_ai + h_bi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write output: inclusive result
        if (tid < slen) {
            float A_t = A[base + tid * stride];
            float x_t = x[base + tid * stride];
            h[base + tid * stride] = A_t * h_shared[CF_IDX(tid)] + x_t;
        }
        """
        _ssm_scan_kernel = mx.fast.metal_kernel(
            name="ssm_scan_diagonal",
            input_names=["A", "x", "seq_len", "d_inner"],
            output_names=["h"],
            source=source,
        )
    return _ssm_scan_kernel


def _get_multiblock_phase1_kernel() -> mx.fast.metal_kernel:
    """Get or create the multi-block Phase 1 kernel (SIMD-optimized)."""
    global _multiblock_phase1_kernel
    if _multiblock_phase1_kernel is None:
        source = """
        // Threadgroup memory for warp totals (max 32 warps)
        threadgroup float shared[32];

        uint tid = thread_position_in_threadgroup.x;
        uint block_idx = threadgroup_position_in_grid.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];

        uint global_idx = block_idx * block_size + tid;

        // Load input
        float val = (global_idx < slen) ? input[global_idx] : 0.0f;

        // SIMD-optimized scan within block
        float warp_scan = simd_prefix_inclusive_sum(val);

        uint num_warps = (block_size + 31) / 32;
        if (simd_lane == 31) {
            shared[simd_group] = warp_scan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scan warp totals
        if (simd_group == 0 && simd_lane < num_warps) {
            float warp_total = shared[simd_lane];
            float scanned_total = simd_prefix_exclusive_sum(warp_total);
            shared[simd_lane] = scanned_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float warp_prefix = (simd_group > 0) ? shared[simd_group] : 0.0f;
        float result = warp_scan + warp_prefix;

        // Save block sum (last thread in block)
        if (tid == block_size - 1) {
            block_sums[block_idx] = result;
        }

        // Write output (inclusive)
        if (global_idx < slen) {
            output[global_idx] = result;
        }
        """
        _multiblock_phase1_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_multiblock_phase1",
            input_names=["input", "seq_len"],
            output_names=["output", "block_sums"],
            source=source,
        )
    return _multiblock_phase1_kernel


def _get_multiblock_phase2_kernel() -> mx.fast.metal_kernel:
    """Get or create the multi-block Phase 2 kernel."""
    global _multiblock_phase2_kernel
    if _multiblock_phase2_kernel is None:
        source = """
        uint tid = thread_position_in_threadgroup.x;
        uint block_idx = threadgroup_position_in_grid.x;
        uint block_size = threads_per_threadgroup.x;
        uint slen = seq_len[0];

        if (block_idx == 0) return;  // First block has no prefix to add

        uint global_idx = block_idx * block_size + tid;
        if (global_idx < slen) {
            output[global_idx] += block_prefixes[block_idx - 1];
        }
        """
        _multiblock_phase2_kernel = mx.fast.metal_kernel(
            name="associative_scan_add_multiblock_phase2",
            input_names=["block_prefixes", "seq_len"],
            output_names=["output"],
            source=source,
        )
    return _multiblock_phase2_kernel


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


def _shared_mem_size_with_padding(block_size: int) -> int:
    """Calculate shared memory size with conflict-free padding."""
    # Add 1 extra float per 32 elements for conflict-free access
    return (block_size + CONFLICT_FREE_OFFSET(block_size)) * 4


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

    # Handle multi-block case for long sequences
    if seq_len > MAX_SINGLE_BLOCK_SEQ:
        result = _metal_associative_scan_multiblock(x_flat, operator, inclusive)
        return result.reshape(original_shape)

    # Prepare inputs
    x_flat = mx.contiguous(x_flat.astype(mx.float32))
    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)
    batch_size_arr = mx.array([batch_size], dtype=mx.uint32)

    # Choose kernel based on operator and sequence length
    if operator == "add":
        # Use vectorized kernel for larger sequences (>= 64 elements)
        # Vectorization reduces thread count by 4x and improves memory bandwidth
        use_vectorized = seq_len >= 64

        if use_vectorized:
            kernel = _get_scan_add_vectorized_kernel()
            # Each thread handles 4 elements
            num_chunks = (seq_len + 3) // 4
            block_size = _next_power_of_2(num_chunks)
            block_size = min(block_size, 256)  # Max 256 chunks = 1024 elements
        else:
            kernel = _get_scan_add_batched_simd_kernel()
            block_size = _next_power_of_2(seq_len)
            block_size = min(block_size, MAX_SINGLE_BLOCK_SEQ)
    else:
        raise NotImplementedError(f"Metal kernel for operator '{operator}' not implemented yet")

    # Execute kernel
    # Grid is total threads: (block_size, batch_size) gives block_size threads per batch
    # threadgroup = (block_size, 1, 1) means 1 threadgroup per batch
    outputs = kernel(
        inputs=[x_flat, seq_len_arr, batch_size_arr],
        grid=(block_size, batch_size, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(batch_size, seq_len)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    result = outputs[0]

    # Reshape back to original shape
    return result.reshape(original_shape)


def _metal_associative_scan_multiblock(
    x: mx.array,
    operator: Literal["add", "mul"] = "add",
    inclusive: bool = True,
) -> mx.array:
    """Multi-block associative scan for sequences > 1024.

    Uses a three-phase algorithm:
    1. Each block scans its local segment, outputs partial results + block total
    2. Scan the block totals (recursively if needed)
    3. Add block prefix to each element

    Args:
        x: Input tensor of shape (batch, seq_len).
        operator: Scan operator.
        inclusive: If True, include current element in scan result.

    Returns:
        Scanned tensor of same shape as input.
    """
    if operator != "add":
        raise NotImplementedError(f"Multi-block scan for operator '{operator}' not implemented yet")

    batch_size, seq_len = x.shape
    block_size = MAX_SINGLE_BLOCK_SEQ
    num_blocks = (seq_len + block_size - 1) // block_size

    # For batched inputs, process each batch separately
    # (Could be optimized with a batched multi-block kernel)
    results = []
    for b in range(batch_size):
        x_batch = x[b : b + 1].reshape(-1)  # Shape: (seq_len,)
        result_batch = _metal_associative_scan_multiblock_single(
            x_batch, block_size, num_blocks, inclusive
        )
        results.append(result_batch)

    return mx.stack(results, axis=0)


def _metal_associative_scan_multiblock_single(
    x: mx.array,
    block_size: int,
    num_blocks: int,
    inclusive: bool = True,
) -> mx.array:
    """Multi-block scan for a single sequence."""
    seq_len = x.shape[0]

    # Phase 1: Local scans + block sums
    phase1_kernel = _get_multiblock_phase1_kernel()

    x = mx.contiguous(x.astype(mx.float32))
    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)

    # Shared memory for warp totals
    shared_mem_size = 32 * 4

    outputs = phase1_kernel(
        inputs=[x, seq_len_arr],
        grid=(num_blocks, 1, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(seq_len,), (num_blocks,)],
        output_dtypes=[mx.float32, mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    partial_results, block_sums = outputs

    # Phase 2: Scan block sums
    if num_blocks <= MAX_SINGLE_BLOCK_SEQ:
        # Single-block scan of block sums
        block_prefixes = metal_associative_scan(block_sums, "add", -1, True)
    else:
        # Recursive multi-block scan
        block_prefixes = _metal_associative_scan_multiblock(
            block_sums.reshape(1, -1), "add", True
        ).reshape(-1)

    # Phase 3: Add block prefixes to partial results
    phase2_kernel = _get_multiblock_phase2_kernel()

    # Need to pass output as input for in-place update
    # Create a copy since Metal kernels can't do true in-place
    final_result = mx.array(partial_results)

    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)

    outputs = phase2_kernel(
        inputs=[block_prefixes, seq_len_arr],
        grid=(num_blocks, 1, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
        init_value=partial_results,  # Initialize output with partial results
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


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

    # Prefer SIMD-optimized kernel
    kernel = _get_ssm_scan_simd_kernel()

    # Prepare inputs
    A = mx.contiguous(A.astype(mx.float32))
    x = mx.contiguous(x.astype(mx.float32))
    seq_len_arr = mx.array([seq_len], dtype=mx.uint32)
    d_inner_arr = mx.array([d_inner], dtype=mx.uint32)
    batch_size_arr = mx.array([batch_size], dtype=mx.uint32)

    # Block size for sequence dimension
    block_size = _next_power_of_2(seq_len)
    block_size = min(block_size, MAX_SINGLE_BLOCK_SEQ)

    # Shared memory: 2 * num_warps floats for A and h warp totals
    num_warps = (block_size + 31) // 32
    shared_mem_size = 2 * num_warps * 4

    # Execute kernel: 3D grid (block_size, d_inner, batch_size)
    # Each threadgroup of block_size threads handles one (batch, dim) slice
    outputs = kernel(
        inputs=[A, x, seq_len_arr, d_inner_arr, batch_size_arr],
        grid=(block_size, d_inner, batch_size),
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
