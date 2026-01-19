"""Flash Attention implementation using Metal-Triton DSL.

This demonstrates how complex kernels like Flash Attention can be
expressed in a Triton-like Python syntax instead of raw Metal.

Compare to the 378-line hand-written Metal kernel in:
    metal/attention/flash_attention.metal
"""

from mlx_primitives.dsl import metal_kernel, autotune, Config, constexpr
import mlx_primitives.dsl as mt


@autotune(
    configs=[
        Config(BLOCK_M=32, BLOCK_N=32, num_warps=4),
        Config(BLOCK_M=64, BLOCK_N=32, num_warps=8),
        Config(BLOCK_M=64, BLOCK_N=64, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
@metal_kernel
def flash_attention_fwd(
    Q_ptr,           # Query: (batch, seq_len, num_heads, head_dim)
    K_ptr,           # Key: (batch, seq_len, num_heads, head_dim)
    V_ptr,           # Value: (batch, seq_len, num_heads, head_dim)
    O_ptr,           # Output: (batch, seq_len, num_heads, head_dim)
    seq_len: constexpr,
    head_dim: constexpr,
    scale: mt.float32,
    causal: mt.uint32,
    BLOCK_M: constexpr = 32,  # Query block size
    BLOCK_N: constexpr = 32,  # KV block size
):
    """Flash Attention forward pass with tiled online softmax and shared memory.

    Memory-efficient exact attention using:
    1. Block-wise computation to fit in shared memory
    2. Online softmax for numerical stability without materializing O(n²) matrix
    3. Cooperative loading of K/V blocks into threadgroup shared memory
    4. Full head dimension handling with per-thread accumulation

    Each threadgroup processes one (batch, head, query_block) tile.

    Algorithm:
    - Q block: BLOCK_M queries loaded into registers
    - For each KV block of size BLOCK_N:
      - Cooperatively load K block and V block into shared memory
      - Compute Q @ K^T for the block (BLOCK_M x BLOCK_N scores)
      - Update online softmax state
      - Accumulate weighted V contributions
    - Write normalized output
    """
    # Grid indices
    pid_m = mt.program_id(0)   # Query block index (0 to seq_len/BLOCK_M)
    pid_h = mt.program_id(1)   # Head index
    pid_b = mt.program_id(2)   # Batch index

    # Thread position within threadgroup
    tid = mt.thread_id_in_threadgroup()
    num_threads = mt.threads_per_threadgroup()

    # Compute base offsets for this block
    head_stride = seq_len * head_dim
    batch_stride = head_stride * mt.num_programs(1)

    # Base offset for this (batch, head)
    base_offset = pid_b * batch_stride + pid_h * head_stride

    # Allocate shared memory for K and V blocks
    # Each block: BLOCK_N positions × head_dim elements
    k_shared = mt.shared_memory(BLOCK_N * head_dim, mt.float32)
    v_shared = mt.shared_memory(BLOCK_N * head_dim, mt.float32)

    # Each thread handles one query position within the block
    # and iterates over all head_dim elements
    q_local_idx = tid
    q_global_idx = pid_m * BLOCK_M + q_local_idx

    # Initialize online softmax state (per head dim element)
    # Using registers for accumulators (max head_dim = 128)
    m_i = float('-inf')   # Running maximum score
    l_i = 0.0            # Running sum of exponentials

    # Accumulator for weighted V sum (one per head_dim element)
    # Each thread accumulates full head_dim vector for its query position
    acc = mt.zeros(head_dim, mt.float32)

    # Load Q vector for this thread's query position into registers
    q_vec = mt.zeros(head_dim, mt.float32)
    if q_global_idx < seq_len:
        for d in range(head_dim):
            q_vec[d] = mt.load(Q_ptr + base_offset + q_global_idx * head_dim + d)

    # Determine KV iteration limit for causal masking
    kv_limit = seq_len
    if causal:
        kv_limit = mt.minimum((pid_m + 1) * BLOCK_M, seq_len)

    # Iterate over KV blocks
    for kv_start in range(0, kv_limit, BLOCK_N):
        # === Cooperative loading phase ===
        # All threads help load K and V blocks into shared memory
        kv_block_size = mt.minimum(BLOCK_N, seq_len - kv_start)

        # Load K block cooperatively
        mt.load_shared(
            k_shared,
            K_ptr + base_offset + kv_start * head_dim,
            kv_block_size * head_dim
        )

        # Load V block cooperatively
        mt.load_shared(
            v_shared,
            V_ptr + base_offset + kv_start * head_dim,
            kv_block_size * head_dim
        )

        # Synchronize after loading
        mt.barrier()

        # === Computation phase ===
        if q_global_idx < seq_len:
            # Process each KV position in the block
            for kv_offset in range(BLOCK_N):
                kv_idx = kv_start + kv_offset

                # Skip if beyond sequence
                if kv_idx >= seq_len:
                    continue

                # Skip if causal and KV position > query position
                if causal and kv_idx > q_global_idx:
                    continue

                # Compute attention score: Q·K / sqrt(d)
                score = 0.0
                for d in range(head_dim):
                    score = score + q_vec[d] * k_shared[kv_offset * head_dim + d]
                score = score * scale

                # Online softmax update
                m_new = mt.maximum(m_i, score)

                # Rescale factors
                alpha = mt.exp(m_i - m_new)
                beta = mt.exp(score - m_new)

                # Update running sum
                l_new = alpha * l_i + beta

                # Update accumulator: rescale old + add new V contribution
                for d in range(head_dim):
                    acc[d] = alpha * acc[d] + beta * v_shared[kv_offset * head_dim + d]

                # Update state
                m_i = m_new
                l_i = l_new

        # Synchronize before next block load
        mt.barrier()

    # === Output phase ===
    # Normalize by sum of exponentials and store
    if q_global_idx < seq_len and l_i > 0.0:
        inv_l = 1.0 / l_i
        for d in range(head_dim):
            out_val = acc[d] * inv_l
            mt.store(O_ptr + base_offset + q_global_idx * head_dim + d, out_val)


@metal_kernel
def flash_attention_simple(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    seq_len: constexpr,
    head_dim: constexpr,
    scale: mt.float32,
):
    """Simplified Flash Attention for single query.

    This version processes one query position per thread.
    Good for understanding the algorithm before the full tiled version.
    """
    # Each thread handles one query position
    pid = mt.program_id(0)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    q_idx = pid * block_size + tid

    if q_idx >= seq_len:
        return

    # Load query
    q_offset = q_idx * head_dim
    q_val = mt.load(Q_ptr + q_offset)

    # Initialize online softmax
    m_i = float('-inf')
    l_i = 0.0
    acc = 0.0

    # Iterate over all KV positions
    for kv_idx in range(seq_len):
        # Load K, V
        kv_offset = kv_idx * head_dim
        k_val = mt.load(K_ptr + kv_offset)
        v_val = mt.load(V_ptr + kv_offset)

        # Attention score
        score = q_val * k_val * scale

        # Online softmax
        m_new = mt.maximum(m_i, score)
        alpha = mt.exp(m_i - m_new)
        beta = mt.exp(score - m_new)
        l_new = alpha * l_i + beta

        acc = alpha * acc + beta * v_val

        m_i = m_new
        l_i = l_new

    # Normalize and store
    out_val = acc / l_i if l_i > 0.0 else 0.0
    mt.store(O_ptr + q_offset, out_val)


def demo_generated_metal():
    """Show the generated Metal code for flash_attention_fwd."""
    print("=" * 70)
    print("Generated Metal for flash_attention_fwd:")
    print("=" * 70)

    # Get generated Metal source
    metal_source = flash_attention_fwd.inspect_metal()
    print(metal_source)

    print("=" * 70)
    print(f"Generated code: {len(metal_source.splitlines())} lines")
    print("Compare to hand-written: ~378 lines in flash_attention.metal")
    print("=" * 70)


def demo_simple_attention():
    """Show the generated Metal for simplified attention."""
    print("\n" + "=" * 70)
    print("Generated Metal for flash_attention_simple:")
    print("=" * 70)

    metal_source = flash_attention_simple.inspect_metal()
    print(metal_source)
    print("=" * 70)


if __name__ == "__main__":
    demo_generated_metal()
    demo_simple_attention()
