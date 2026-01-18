"""Rotary Position Embedding (RoPE) kernels using Metal-Triton DSL.

RoPE applies position-dependent rotations to query/key vectors:
    For each pair (x[2i], x[2i+1]):
        cos_theta = cos(pos * freq[i])
        sin_theta = sin(pos * freq[i])
        y[2i]   = x[2i] * cos_theta - x[2i+1] * sin_theta
        y[2i+1] = x[2i] * sin_theta + x[2i+1] * cos_theta

Where freq[i] = 1 / (base ^ (2i / dim)), typically base=10000

Used in LLaMA, GPT-NeoX, PaLM, and many modern LLMs.

This module includes:
- Basic RoPE with precomputed cos/sin tables
- RoPE with inline frequency computation
- Fused Q/K RoPE application
"""

from mlx_primitives.dsl import metal_kernel, constexpr
import mlx_primitives.dsl as mt


@metal_kernel
def rope_forward(
    x_ptr,           # (batch, seq, heads, dim)
    cos_cache_ptr,   # (max_seq, dim/2) - precomputed cos values
    sin_cache_ptr,   # (max_seq, dim/2) - precomputed sin values
    out_ptr,
    batch_size: constexpr,
    seq_len: constexpr,
    num_heads: constexpr,
    head_dim: constexpr,
    start_pos: mt.uint32 = 0,  # For incremental decoding
):
    """Apply Rotary Position Embedding (forward pass).

    Uses precomputed cos/sin tables for efficiency.
    Tables should be precomputed as:
        freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))
        cos_cache[pos, i] = cos(pos * freqs[i])
        sin_cache[pos, i] = sin(pos * freqs[i])

    Args:
        x_ptr: Input tensor (batch, seq, heads, dim)
        cos_cache_ptr: Precomputed cos values (max_seq, dim/2)
        sin_cache_ptr: Precomputed sin values (max_seq, dim/2)
        out_ptr: Output tensor (batch, seq, heads, dim)
        batch_size: Batch dimension
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension of each head (must be even)
        start_pos: Starting position for incremental decoding

    Grid: (batch_size, seq_len, num_heads)
    Threadgroup: (head_dim // 2,) or 32
    """
    pid_b = mt.program_id(0)
    pid_s = mt.program_id(1)
    pid_h = mt.program_id(2)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    if pid_b >= batch_size:
        return
    if pid_s >= seq_len:
        return
    if pid_h >= num_heads:
        return

    half_dim = head_dim // 2
    pos = start_pos + pid_s

    # Compute base offset for this (batch, seq, head)
    # x layout: (batch, seq, heads, dim)
    head_stride = head_dim
    heads_stride = num_heads * head_dim
    seq_stride = seq_len * num_heads * head_dim
    base_offset = pid_b * seq_len * num_heads * head_dim + pid_s * num_heads * head_dim + pid_h * head_dim

    # Each thread handles one pair (or multiple pairs if dim > block_size)
    for i in range(tid, half_dim, block_size):
        # Load the pair
        x0 = mt.load(x_ptr + base_offset + 2 * i)
        x1 = mt.load(x_ptr + base_offset + 2 * i + 1)

        # Load precomputed cos/sin
        cache_offset = pos * half_dim + i
        cos_val = mt.load(cos_cache_ptr + cache_offset)
        sin_val = mt.load(sin_cache_ptr + cache_offset)

        # Apply rotation
        y0 = x0 * cos_val - x1 * sin_val
        y1 = x0 * sin_val + x1 * cos_val

        # Store results
        mt.store(out_ptr + base_offset + 2 * i, y0)
        mt.store(out_ptr + base_offset + 2 * i + 1, y1)


@metal_kernel
def rope_inline(
    x_ptr,
    out_ptr,
    batch_size: constexpr,
    seq_len: constexpr,
    num_heads: constexpr,
    head_dim: constexpr,
    base: mt.float32 = 10000.0,
    start_pos: mt.uint32 = 0,
):
    """RoPE with inline frequency computation.

    Computes cos/sin values on-the-fly instead of using lookup tables.
    Slightly slower but doesn't require precomputed tables.

    Args:
        x_ptr: Input tensor (batch, seq, heads, dim)
        out_ptr: Output tensor (batch, seq, heads, dim)
        batch_size: Batch dimension
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        base: RoPE base (default 10000)
        start_pos: Starting position

    Grid: (batch_size, seq_len, num_heads)
    Threadgroup: 32
    """
    pid_b = mt.program_id(0)
    pid_s = mt.program_id(1)
    pid_h = mt.program_id(2)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    if pid_b >= batch_size:
        return
    if pid_s >= seq_len:
        return
    if pid_h >= num_heads:
        return

    half_dim = head_dim // 2
    pos = start_pos + pid_s

    base_offset = pid_b * seq_len * num_heads * head_dim + pid_s * num_heads * head_dim + pid_h * head_dim

    for i in range(tid, half_dim, block_size):
        # Compute frequency: 1 / (base ^ (2*i / dim))
        # freq = base^(-2*i/dim) = exp(-2*i/dim * log(base))
        exponent = -2.0 * i / head_dim
        freq = mt.exp(exponent * mt.log(base))

        # Compute angle
        theta = pos * freq

        # Compute cos/sin (could use sincos if available)
        cos_val = mt.cos(theta)
        sin_val = mt.sin(theta)

        # Load pair
        x0 = mt.load(x_ptr + base_offset + 2 * i)
        x1 = mt.load(x_ptr + base_offset + 2 * i + 1)

        # Apply rotation
        y0 = x0 * cos_val - x1 * sin_val
        y1 = x0 * sin_val + x1 * cos_val

        mt.store(out_ptr + base_offset + 2 * i, y0)
        mt.store(out_ptr + base_offset + 2 * i + 1, y1)


@metal_kernel
def rope_qk_fused(
    q_ptr,           # (batch, seq, heads, dim)
    k_ptr,           # (batch, seq, kv_heads, dim)
    cos_cache_ptr,   # (max_seq, dim/2)
    sin_cache_ptr,   # (max_seq, dim/2)
    q_out_ptr,
    k_out_ptr,
    batch_size: constexpr,
    seq_len: constexpr,
    num_q_heads: constexpr,
    num_kv_heads: constexpr,
    head_dim: constexpr,
    start_pos: mt.uint32 = 0,
):
    """Apply RoPE to both Q and K tensors in one kernel.

    For GQA, num_q_heads > num_kv_heads.
    Each KV head is shared by (num_q_heads / num_kv_heads) Q heads.

    Args:
        q_ptr: Query tensor (batch, seq, num_q_heads, dim)
        k_ptr: Key tensor (batch, seq, num_kv_heads, dim)
        cos_cache_ptr: Precomputed cos values (max_seq, dim/2)
        sin_cache_ptr: Precomputed sin values (max_seq, dim/2)
        q_out_ptr: Output query tensor
        k_out_ptr: Output key tensor
        batch_size: Batch dimension
        seq_len: Sequence length
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Head dimension
        start_pos: Starting position

    Grid: (batch_size, seq_len, max(num_q_heads, num_kv_heads))
    Threadgroup: 32
    """
    pid_b = mt.program_id(0)
    pid_s = mt.program_id(1)
    pid_h = mt.program_id(2)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    if pid_b >= batch_size:
        return
    if pid_s >= seq_len:
        return

    half_dim = head_dim // 2
    pos = start_pos + pid_s

    # Process Q head if valid
    if pid_h < num_q_heads:
        q_base = pid_b * seq_len * num_q_heads * head_dim + pid_s * num_q_heads * head_dim + pid_h * head_dim

        for i in range(tid, half_dim, block_size):
            cache_offset = pos * half_dim + i
            cos_val = mt.load(cos_cache_ptr + cache_offset)
            sin_val = mt.load(sin_cache_ptr + cache_offset)

            q0 = mt.load(q_ptr + q_base + 2 * i)
            q1 = mt.load(q_ptr + q_base + 2 * i + 1)

            qy0 = q0 * cos_val - q1 * sin_val
            qy1 = q0 * sin_val + q1 * cos_val

            mt.store(q_out_ptr + q_base + 2 * i, qy0)
            mt.store(q_out_ptr + q_base + 2 * i + 1, qy1)

    # Process K head if valid
    if pid_h < num_kv_heads:
        k_base = pid_b * seq_len * num_kv_heads * head_dim + pid_s * num_kv_heads * head_dim + pid_h * head_dim

        for i in range(tid, half_dim, block_size):
            cache_offset = pos * half_dim + i
            cos_val = mt.load(cos_cache_ptr + cache_offset)
            sin_val = mt.load(sin_cache_ptr + cache_offset)

            k0 = mt.load(k_ptr + k_base + 2 * i)
            k1 = mt.load(k_ptr + k_base + 2 * i + 1)

            ky0 = k0 * cos_val - k1 * sin_val
            ky1 = k0 * sin_val + k1 * cos_val

            mt.store(k_out_ptr + k_base + 2 * i, ky0)
            mt.store(k_out_ptr + k_base + 2 * i + 1, ky1)


@metal_kernel
def rope_neox(
    x_ptr,
    cos_cache_ptr,
    sin_cache_ptr,
    out_ptr,
    batch_size: constexpr,
    seq_len: constexpr,
    num_heads: constexpr,
    head_dim: constexpr,
    start_pos: mt.uint32 = 0,
):
    """RoPE with GPT-NeoX style rotation.

    Unlike standard RoPE which pairs adjacent elements [x0, x1, x2, x3...],
    NeoX pairs first half with second half: [x0, x2, x4...] with [x1, x3, x5...]

    This is the "neox" or "gptj" style rotation.

    Args:
        x_ptr: Input tensor (batch, seq, heads, dim)
        cos_cache_ptr: Precomputed cos values (max_seq, dim/2)
        sin_cache_ptr: Precomputed sin values (max_seq, dim/2)
        out_ptr: Output tensor
        batch_size: Batch dimension
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        start_pos: Starting position

    Grid: (batch_size, seq_len, num_heads)
    Threadgroup: 32
    """
    pid_b = mt.program_id(0)
    pid_s = mt.program_id(1)
    pid_h = mt.program_id(2)
    tid = mt.thread_id_in_threadgroup()
    block_size = mt.threads_per_threadgroup()

    if pid_b >= batch_size:
        return
    if pid_s >= seq_len:
        return
    if pid_h >= num_heads:
        return

    half_dim = head_dim // 2
    pos = start_pos + pid_s

    base_offset = pid_b * seq_len * num_heads * head_dim + pid_s * num_heads * head_dim + pid_h * head_dim

    # NeoX style: pair x[i] with x[i + half_dim]
    for i in range(tid, half_dim, block_size):
        cache_offset = pos * half_dim + i
        cos_val = mt.load(cos_cache_ptr + cache_offset)
        sin_val = mt.load(sin_cache_ptr + cache_offset)

        # Load from first half and second half
        x0 = mt.load(x_ptr + base_offset + i)
        x1 = mt.load(x_ptr + base_offset + half_dim + i)

        # Apply rotation
        y0 = x0 * cos_val - x1 * sin_val
        y1 = x0 * sin_val + x1 * cos_val

        # Store to first half and second half
        mt.store(out_ptr + base_offset + i, y0)
        mt.store(out_ptr + base_offset + half_dim + i, y1)


def precompute_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
):
    """Precompute cos/sin cache for RoPE.

    Call this once to create the lookup tables.

    Args:
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head
        base: RoPE base frequency

    Returns:
        (cos_cache, sin_cache) as numpy arrays of shape (max_seq_len, head_dim//2)
    """
    import numpy as np

    half_dim = head_dim // 2

    # Compute frequencies: 1 / (base ** (2i / dim))
    freqs = 1.0 / (base ** (np.arange(0, half_dim) * 2.0 / head_dim))

    # Positions
    positions = np.arange(max_seq_len)

    # Compute angles: positions x freqs
    angles = np.outer(positions, freqs)

    # Compute cos/sin
    cos_cache = np.cos(angles).astype(np.float32)
    sin_cache = np.sin(angles).astype(np.float32)

    return cos_cache, sin_cache
