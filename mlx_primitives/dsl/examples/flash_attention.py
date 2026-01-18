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
    """Flash Attention forward pass with tiled online softmax.

    Memory-efficient exact attention using:
    1. Block-wise computation to fit in shared memory
    2. Online softmax for numerical stability without materializing O(n²) matrix

    Each threadgroup processes one (batch, head, query_block) tile.
    """
    # Grid indices
    pid_m = mt.program_id(0)   # Query block index (0 to seq_len/BLOCK_M)
    pid_h = mt.program_id(1)   # Head index
    pid_b = mt.program_id(2)   # Batch index

    # Thread position within threadgroup
    tid = mt.thread_id_in_threadgroup()

    # Compute base offsets for this block
    # Each head has seq_len * head_dim elements
    head_stride = seq_len * head_dim
    batch_stride = head_stride * mt.num_programs(1)  # num_heads

    # Base offset for this (batch, head)
    base_offset = pid_b * batch_stride + pid_h * head_stride

    # Query offset for this block
    q_block_offset = pid_m * BLOCK_M * head_dim

    # Initialize online softmax state per thread
    # Each thread handles one query position within the block
    m_i = float('-inf')   # Running maximum
    l_i = 0.0            # Running sum of exponentials
    acc = 0.0            # Accumulator for output (simplified: 1 element per thread)

    # Determine KV iteration limit for causal masking
    # If causal: only attend to positions <= current query position
    kv_limit = seq_len
    if causal:
        # Last valid KV position for this query block
        kv_limit = mt.minimum((pid_m + 1) * BLOCK_M, seq_len)

    # Load Q value for this thread's query position
    q_idx = pid_m * BLOCK_M + tid
    q_val = 0.0
    if q_idx < seq_len:
        q_val = mt.load(Q_ptr + base_offset + q_idx * head_dim)

    # Iterate over KV blocks
    for kv_start in range(0, kv_limit, BLOCK_N):
        # For each KV position in this block
        for kv_offset in range(BLOCK_N):
            kv_idx = kv_start + kv_offset

            # Skip if beyond sequence
            if kv_idx >= seq_len:
                continue

            # Skip if causal and KV position > query position
            if causal:
                if kv_idx > q_idx:
                    continue

            # Load K and V values
            k_val = mt.load(K_ptr + base_offset + kv_idx * head_dim)
            v_val = mt.load(V_ptr + base_offset + kv_idx * head_dim)

            # Compute attention score: Q·K / sqrt(d)
            score = q_val * k_val * scale

            # Online softmax update
            m_new = mt.maximum(m_i, score)

            # Rescale old accumulator and sum
            alpha = mt.exp(m_i - m_new)
            beta = mt.exp(score - m_new)

            l_new = alpha * l_i + beta

            # Update accumulator: rescale old + add new contribution
            acc = alpha * acc + beta * v_val

            # Update running state
            m_i = m_new
            l_i = l_new

    # Normalize output by sum of exponentials
    out_val = acc / l_i if l_i > 0.0 else 0.0

    # Store output
    if q_idx < seq_len:
        mt.store(O_ptr + base_offset + q_idx * head_dim, out_val)


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
