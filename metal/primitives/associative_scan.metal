// Associative Scan (Parallel Prefix Sum) - Blelloch Algorithm
// Enables efficient O(log n) parallel scan for SSMs like Mamba
//
// The Blelloch algorithm has two phases:
// 1. Up-sweep (reduce): Build a balanced binary tree of partial sums
// 2. Down-sweep: Traverse the tree to compute all prefix sums

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Simple Associative Scan (Addition)
// For sequences that fit in a single threadgroup (up to 1024 elements)
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
    // Load input into shared memory
    if (tid < seq_len) {
        shared[tid] = input[tid];
    } else {
        shared[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[bi] += shared[ai];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        shared[block_size - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[ai];
                shared[ai] = shared[bi];
                shared[bi] += t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            // Inclusive scan: add original value back
            output[tid] = shared[tid] + input[tid];
        } else {
            // Exclusive scan: result is already correct
            output[tid] = shared[tid];
        }
    }
}

// ============================================================================
// Batched Associative Scan (Addition)
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

    // Load input into shared memory
    if (tid < seq_len) {
        shared[tid] = input[base_offset + tid];
    } else {
        shared[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[bi] += shared[ai];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        shared[block_size - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[ai];
                shared[ai] = shared[bi];
                shared[bi] += t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            output[base_offset + tid] = shared[tid] + input[base_offset + tid];
        } else {
            output[base_offset + tid] = shared[tid];
        }
    }
}

// ============================================================================
// SSM Scan - State Space Model Recurrence
// Computes h[t] = A[t] * h[t-1] + x[t] in parallel
//
// The scan operator is: (A1, h1) ⊕ (A2, h2) = (A1 * A2, A2 * h1 + h2)
// This is associative because matrix multiplication distributes over addition.
//
// For diagonal A (common in Mamba), this simplifies to element-wise operations.
// ============================================================================

kernel void ssm_scan_diagonal(
    device const float* A [[buffer(0)]],        // (batch, seq, d_inner) - discretized diagonal A
    device const float* x [[buffer(1)]],        // (batch, seq, d_inner) - delta * B * input
    device float* h [[buffer(2)]],              // (batch, seq, d_inner) - output hidden states
    constant uint& seq_len [[buffer(3)]],
    constant uint& d_inner [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    threadgroup float* A_shared [[threadgroup(0)]],    // (seq,) per dimension
    threadgroup float* h_shared [[threadgroup(1)]],    // (seq,) per dimension
    uint tid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],   // (batch, d_inner)
    uint block_size [[threads_per_threadgroup]]
) {
    uint batch_idx = group_id.x;
    uint dim_idx = group_id.y;

    if (batch_idx >= batch_size || dim_idx >= d_inner) return;

    // Base offset for this (batch, dim) slice
    uint base = batch_idx * seq_len * d_inner + dim_idx;
    uint stride = d_inner;

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
    // At each level, we combine adjacent pairs:
    // (A_left, h_left) ⊕ (A_right, h_right) = (A_left * A_right, A_right * h_left + h_right)
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
                // Swap and accumulate
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

    // Write output: h[t] = A_prefix[t] * 0 + h_prefix[t] + A[t] * h[t-1] + x[t]
    // Since we computed exclusive prefix, need to add current element contribution
    if (tid < seq_len) {
        float A_t = A[base + tid * stride];
        float x_t = x[base + tid * stride];
        // Inclusive result: apply operator one more time with current element
        h[base + tid * stride] = A_t * h_shared[tid] + x_t;
    }
}

// ============================================================================
// Multi-Block Scan Components
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
    uint block_size [[threads_per_threadgroup]]
) {
    uint global_idx = block_idx * block_size + tid;

    // Load input into shared memory
    if (global_idx < seq_len) {
        shared[tid] = input[global_idx];
    } else {
        shared[tid] = 0.0f;
    }
    float original = shared[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block sum before clearing
    if (tid == block_size - 1) {
        block_sums[block_idx] = shared[tid];
    }

    // Clear last element
    if (tid == 0) {
        shared[block_size - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            float t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (global_idx < seq_len) {
        if (inclusive) {
            output[global_idx] = shared[tid] + original;
        } else {
            output[global_idx] = shared[tid];
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
// Multiplication Scan (Cumulative Product)
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
    // Load input into shared memory
    if (tid < seq_len) {
        shared[tid] = input[tid];
    } else {
        shared[tid] = 1.0f;  // Identity for multiplication
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep with multiplication
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                shared[bi] *= shared[ai];
            }
        }
        offset *= 2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear last element with identity
    if (tid == 0) {
        shared[block_size - 1] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep with multiplication
    for (uint d = 1; d < block_size; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            if (bi < block_size) {
                float t = shared[ai];
                shared[ai] = shared[bi];
                shared[bi] *= t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tid < seq_len) {
        if (inclusive) {
            output[tid] = shared[tid] * input[tid];
        } else {
            output[tid] = shared[tid];
        }
    }
}
