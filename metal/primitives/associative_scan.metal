// Associative Scan (Parallel Prefix Sum) - Optimized Implementation
// Enables efficient O(log n) parallel scan for SSMs like Mamba
//
// Optimizations:
// 1. Bank conflict-free shared memory access
// 2. SIMD warp-level intrinsics for intra-warp scan
// 3. Blelloch algorithm for inter-warp coordination
//
// The Blelloch algorithm has two phases:
// 1. Up-sweep (reduce): Build a balanced binary tree of partial sums
// 2. Down-sweep: Traverse the tree to compute all prefix sums

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Bank Conflict Avoidance
// Apple Silicon has 32 memory banks. Adding an offset based on index
// ensures threads don't access the same bank simultaneously.
// ============================================================================

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// Padded index for conflict-free access
inline uint cf_idx(uint idx) {
    return idx + CONFLICT_FREE_OFFSET(idx);
}

// ============================================================================
// SIMD-Optimized Associative Scan (Addition)
// Uses warp-level intrinsics for O(1) intra-warp scan
// ============================================================================

kernel void associative_scan_add_simd(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant bool& inclusive [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Load input value
    float val = (tid < seq_len) ? input[tid] : 0.0f;
    float original = val;

    // Phase 1: Intra-warp inclusive scan using SIMD intrinsics (O(1) for 32 elements)
    float warp_scan = simd_prefix_inclusive_sum(val);

    // Phase 2: Store warp totals to shared memory for inter-warp scan
    uint num_warps = (block_size + 31) / 32;
    if (simd_lane == 31) {
        shared[cf_idx(simd_group)] = warp_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Scan warp totals (first warp only)
    float warp_prefix = 0.0f;
    if (simd_group == 0 && simd_lane < num_warps) {
        float warp_total = shared[cf_idx(simd_lane)];
        float scanned_total = simd_prefix_exclusive_sum(warp_total);
        shared[cf_idx(simd_lane)] = scanned_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Add warp prefix to get final result
    if (simd_group > 0) {
        warp_prefix = shared[cf_idx(simd_group - 1)] + shared[cf_idx(simd_group - 1) + 1 - cf_idx(simd_group - 1)];
        // Simpler: just read the exclusive scan result
        warp_prefix = shared[cf_idx(simd_group)];
    }

    float result;
    if (inclusive) {
        result = warp_scan + warp_prefix;
    } else {
        // Exclusive: shift by one (use shuffle)
        float prev = simd_shuffle_up(warp_scan, 1);
        if (simd_lane == 0) {
            prev = warp_prefix;
        } else {
            prev = prev + warp_prefix;
        }
        result = prev;
    }

    // Write output
    if (tid < seq_len) {
        output[tid] = result;
    }
}

// ============================================================================
// Batched SIMD-Optimized Associative Scan (Addition)
// Process multiple sequences in parallel with SIMD optimization
// ============================================================================

kernel void associative_scan_add_batched_simd(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant bool& inclusive [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;

    uint base_offset = batch_idx * seq_len;

    // Load input value
    float val = (tid < seq_len) ? input[base_offset + tid] : 0.0f;

    // Phase 1: Intra-warp inclusive scan
    float warp_scan = simd_prefix_inclusive_sum(val);

    // Phase 2: Store warp totals
    uint num_warps = (block_size + 31) / 32;
    if (simd_lane == 31) {
        shared[cf_idx(simd_group)] = warp_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Scan warp totals
    if (simd_group == 0 && simd_lane < num_warps) {
        float warp_total = shared[cf_idx(simd_lane)];
        float scanned_total = simd_prefix_exclusive_sum(warp_total);
        shared[cf_idx(simd_lane)] = scanned_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Add warp prefix
    float warp_prefix = (simd_group > 0) ? shared[cf_idx(simd_group)] : 0.0f;

    float result;
    if (inclusive) {
        result = warp_scan + warp_prefix;
    } else {
        float prev = simd_shuffle_up(warp_scan, 1);
        if (simd_lane == 0) {
            prev = warp_prefix;
        } else {
            prev = prev + warp_prefix;
        }
        result = prev;
    }

    // Write output
    if (tid < seq_len) {
        output[base_offset + tid] = result;
    }
}

// ============================================================================
// Simple Associative Scan (Addition) - Bank Conflict Free
// For sequences that fit in a single threadgroup (up to 1024 elements)
// Fallback for devices without SIMD support
// ============================================================================

kernel void associative_scan_add_single_block(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant bool& inclusive [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Load input into shared memory with conflict-free indexing
    if (tid < seq_len) {
        shared[cf_idx(tid)] = input[tid];
    } else {
        shared[cf_idx(tid)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase with conflict-free access
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[cf_idx(bi)] += shared[cf_idx(ai)];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        shared[cf_idx(block_size - 1)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase with conflict-free access
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[cf_idx(ai)];
                shared[cf_idx(ai)] = shared[cf_idx(bi)];
                shared[cf_idx(bi)] += t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            output[tid] = shared[cf_idx(tid)] + input[tid];
        } else {
            output[tid] = shared[cf_idx(tid)];
        }
    }
}

// ============================================================================
// Batched Associative Scan (Addition) - Bank Conflict Free
// Process multiple sequences in parallel
// ============================================================================

kernel void associative_scan_add_batched(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant bool& inclusive [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;

    uint base_offset = batch_idx * seq_len;

    // Load input into shared memory with conflict-free indexing
    if (tid < seq_len) {
        shared[cf_idx(tid)] = input[base_offset + tid];
    } else {
        shared[cf_idx(tid)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[cf_idx(bi)] += shared[cf_idx(ai)];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        shared[cf_idx(block_size - 1)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[cf_idx(ai)];
                shared[cf_idx(ai)] = shared[cf_idx(bi)];
                shared[cf_idx(bi)] += t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            output[base_offset + tid] = shared[cf_idx(tid)] + input[base_offset + tid];
        } else {
            output[base_offset + tid] = shared[cf_idx(tid)];
        }
    }
}

// ============================================================================
// SSM Scan - State Space Model Recurrence (SIMD Optimized)
// Computes h[t] = A[t] * h[t-1] + x[t] in parallel
//
// The scan operator is: (A1, h1) ⊕ (A2, h2) = (A1 * A2, A2 * h1 + h2)
// This is associative because matrix multiplication distributes over addition.
//
// For diagonal A (common in Mamba), this simplifies to element-wise operations.
// ============================================================================

kernel void ssm_scan_diagonal_simd(
    device const float* A [[buffer(0)]],        // (batch, seq, d_inner) - discretized diagonal A
    device const float* x [[buffer(1)]],        // (batch, seq, d_inner) - delta * B * input
    device float* h [[buffer(2)]],              // (batch, seq, d_inner) - output hidden states
    constant uint& seq_len [[buffer(3)]],
    constant uint& d_inner [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],  // Warp totals: 2 * num_warps floats
    uint tid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.x;
    uint dim_idx = group_id.y;

    if (batch_idx >= batch_size || dim_idx >= d_inner) return;

    uint base = batch_idx * seq_len * d_inner + dim_idx;
    uint stride = d_inner;

    // Load values
    float A_val = (tid < seq_len) ? A[base + tid * stride] : 1.0f;
    float h_val = (tid < seq_len) ? x[base + tid * stride] : 0.0f;

    // Phase 1: Intra-warp SSM scan using shuffle
    // Operator: (A1, h1) op (A2, h2) = (A1 * A2, A2 * h1 + h2)
    float A_scan = A_val;
    float h_scan = h_val;

    for (uint delta = 1; delta < 32; delta *= 2) {
        float A_other = simd_shuffle_up(A_scan, delta);
        float h_other = simd_shuffle_up(h_scan, delta);

        if (simd_lane >= delta) {
            // Apply SSM operator: (A_other, h_other) op (A_scan, h_scan)
            // Result: (A_other * A_scan, A_scan * h_other + h_scan)
            h_scan = A_scan * h_other + h_scan;
            A_scan = A_other * A_scan;
        }
    }

    // Phase 2: Store warp totals (both A and h)
    uint num_warps = (block_size + 31) / 32;
    threadgroup float* A_warp_totals = shared;
    threadgroup float* h_warp_totals = shared + num_warps + CONFLICT_FREE_OFFSET(num_warps);

    if (simd_lane == 31) {
        A_warp_totals[cf_idx(simd_group)] = A_scan;
        h_warp_totals[cf_idx(simd_group)] = h_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Scan warp totals (first warp only)
    float A_prefix = 1.0f;
    float h_prefix = 0.0f;

    if (simd_group == 0 && simd_lane < num_warps) {
        float A_wt = A_warp_totals[cf_idx(simd_lane)];
        float h_wt = h_warp_totals[cf_idx(simd_lane)];

        // Exclusive scan of warp totals
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

        // Store exclusive prefix (shift by one)
        float A_exc = simd_shuffle_up(A_scan_wt, 1);
        float h_exc = simd_shuffle_up(h_scan_wt, 1);
        if (simd_lane == 0) {
            A_exc = 1.0f;
            h_exc = 0.0f;
        }

        A_warp_totals[cf_idx(simd_lane)] = A_exc;
        h_warp_totals[cf_idx(simd_lane)] = h_exc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: Apply warp prefix
    if (simd_group > 0) {
        A_prefix = A_warp_totals[cf_idx(simd_group)];
        h_prefix = h_warp_totals[cf_idx(simd_group)];
    }

    // Combine: (A_prefix, h_prefix) op (A_scan, h_scan)
    float h_final = A_scan * h_prefix + h_scan;

    // Write output
    if (tid < seq_len) {
        h[base + tid * stride] = h_final;
    }
}

// ============================================================================
// SSM Scan - State Space Model Recurrence (Bank Conflict Free Fallback)
// ============================================================================

kernel void ssm_scan_diagonal(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* h [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& d_inner [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.x;
    uint dim_idx = group_id.y;

    if (batch_idx >= batch_size || dim_idx >= d_inner) return;

    uint base = batch_idx * seq_len * d_inner + dim_idx;
    uint stride = d_inner;

    // Shared memory layout with conflict-free offset
    // First half: A_shared, Second half: h_shared
    uint shared_size = block_size + CONFLICT_FREE_OFFSET(block_size);
    threadgroup float* A_shared = shared;
    threadgroup float* h_shared = shared + shared_size;

    // Load A and x into shared memory
    if (tid < seq_len) {
        A_shared[cf_idx(tid)] = A[base + tid * stride];
        h_shared[cf_idx(tid)] = x[base + tid * stride];
    } else {
        A_shared[cf_idx(tid)] = 1.0f;
        h_shared[cf_idx(tid)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep with conflict-free access
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float A_right = A_shared[cf_idx(bi)];
                h_shared[cf_idx(bi)] = A_right * h_shared[cf_idx(ai)] + h_shared[cf_idx(bi)];
                A_shared[cf_idx(bi)] = A_shared[cf_idx(ai)] * A_right;
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Set identity at root
    if (tid == 0) {
        A_shared[cf_idx(block_size - 1)] = 1.0f;
        h_shared[cf_idx(block_size - 1)] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep with conflict-free access
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float A_ai = A_shared[cf_idx(ai)];
                float h_ai = h_shared[cf_idx(ai)];
                float A_bi = A_shared[cf_idx(bi)];
                float h_bi = h_shared[cf_idx(bi)];

                A_shared[cf_idx(ai)] = A_bi;
                h_shared[cf_idx(ai)] = h_bi;

                A_shared[cf_idx(bi)] = A_ai * A_bi;
                h_shared[cf_idx(bi)] = A_bi * h_ai + h_bi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        float A_t = A[base + tid * stride];
        float x_t = x[base + tid * stride];
        h[base + tid * stride] = A_t * h_shared[cf_idx(tid)] + x_t;
    }
}

// ============================================================================
// Multi-Block Scan Components (Bank Conflict Free)
// For sequences longer than a single threadgroup
// ============================================================================

// Phase 1: Scan each block independently, store block sums
kernel void associative_scan_add_multiblock_phase1(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* block_sums [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant bool& inclusive [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    uint global_idx = block_idx * block_size + tid;

    // Load input
    float val = (global_idx < seq_len) ? input[global_idx] : 0.0f;
    float original = val;

    // SIMD-optimized scan within block
    float warp_scan = simd_prefix_inclusive_sum(val);

    uint num_warps = (block_size + 31) / 32;
    if (simd_lane == 31) {
        shared[cf_idx(simd_group)] = warp_scan;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scan warp totals
    if (simd_group == 0 && simd_lane < num_warps) {
        float warp_total = shared[cf_idx(simd_lane)];
        float scanned_total = simd_prefix_exclusive_sum(warp_total);
        shared[cf_idx(simd_lane)] = scanned_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float warp_prefix = (simd_group > 0) ? shared[cf_idx(simd_group)] : 0.0f;
    float result = warp_scan + warp_prefix;

    // Save block sum (last thread in block)
    if (tid == block_size - 1) {
        block_sums[block_idx] = result;
    }

    // Write output
    if (global_idx < seq_len) {
        if (inclusive) {
            output[global_idx] = result;
        } else {
            // Exclusive: need to shift
            float prev = simd_shuffle_up(result, 1);
            if (simd_lane == 0) {
                prev = warp_prefix;
            } else {
                prev = simd_shuffle_up(warp_scan, 1) + warp_prefix;
            }
            output[global_idx] = prev;
        }
    }
}

// Phase 2: Add block prefix to all elements in each block
kernel void associative_scan_add_multiblock_phase2(
    device float* output [[buffer(0)]],
    device const float* block_prefixes [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_idx [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    if (block_idx == 0) return;  // First block has no prefix to add

    uint global_idx = block_idx * block_size + tid;
    if (global_idx < seq_len) {
        output[global_idx] += block_prefixes[block_idx - 1];
    }
}

// ============================================================================
// Multiplication Scan (Cumulative Product) - Bank Conflict Free
// ============================================================================

kernel void associative_scan_mul_single_block(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant bool& inclusive [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Load input into shared memory with conflict-free indexing
    if (tid < seq_len) {
        shared[cf_idx(tid)] = input[tid];
    } else {
        shared[cf_idx(tid)] = 1.0f;  // Identity for multiplication
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep with multiplication
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[cf_idx(bi)] *= shared[cf_idx(ai)];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear last element with identity
    if (tid == 0) {
        shared[cf_idx(block_size - 1)] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep with multiplication
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[cf_idx(ai)];
                shared[cf_idx(ai)] = shared[cf_idx(bi)];
                shared[cf_idx(bi)] *= t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            output[tid] = shared[cf_idx(tid)] * input[tid];
        } else {
            output[tid] = shared[cf_idx(tid)];
        }
    }
}
