"""Flash Attention - Memory-efficient exact attention via tiled online softmax.

Flash Attention computes exact attention with O(n) memory instead of O(n²) by:
1. Processing Q and KV in tiles that fit in fast memory (registers/shared)
2. Using online softmax to accumulate results without materializing the full matrix
3. Never storing the O(n²) attention weights

This is the algorithm from "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" (Dao et al., 2022).

Memory: O(n * d) instead of O(n²)
Compute: Same O(n² * d) as standard attention
IO: Significantly reduced memory bandwidth

Example:
    >>> q = mx.random.normal((2, 8192, 32, 128))
    >>> k = mx.random.normal((2, 8192, 32, 128))
    >>> v = mx.random.normal((2, 8192, 32, 128))
    >>> out = flash_attention(q, k, v, causal=True)
    >>> # Memory: ~O(8192 * 128) instead of O(8192² * 32)
"""

import math
from typing import Optional, Tuple

import mlx.core as mx

from mlx_primitives.attention._online_softmax import (
    compute_chunk_attention,
    online_softmax_merge,
)

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Kernel cache
_flash_attention_kernel: Optional[mx.fast.metal_kernel] = None


def flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
    block_q: int = 32,
    block_kv: int = 32,
    use_metal: bool = True,
) -> mx.array:
    """Flash Attention - O(n) memory attention via tiled online softmax.

    Computes exact attention identical to standard scaled dot-product attention
    but without materializing the O(n²) attention matrix. Instead, processes
    queries and keys/values in tiles, using online softmax to accumulate results.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        v: Value tensor of shape (batch, seq_len, num_heads, head_dim).
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        causal: If True, apply causal masking (queries can only attend to
            earlier positions).
        block_q: Query block size for tiling. Larger = faster but more memory.
        block_kv: KV block size for tiling. Larger = faster but more memory.
        use_metal: Use Metal kernel if available.

    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim).

    Example:
        >>> # Standard usage
        >>> q = mx.random.normal((2, 1024, 8, 64))
        >>> k = mx.random.normal((2, 1024, 8, 64))
        >>> v = mx.random.normal((2, 1024, 8, 64))
        >>> out = flash_attention(q, k, v, causal=True)
        >>>
        >>> # Equivalent to (but more memory efficient than):
        >>> scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(64)
        >>> # ... apply causal mask ...
        >>> weights = mx.softmax(scores, axis=-1)
        >>> out_standard = weights @ v

    Note:
        For best performance, block sizes should be tuned based on:
        - Hardware (Apple Silicon generation, available memory)
        - Sequence length (longer sequences benefit from larger blocks)
        - Head dimension (larger dims need smaller blocks to fit in shared memory)

        Default block_q=32, block_kv=32 is reasonable for most configurations.
        For head_dim=128, consider reducing to block_q=16, block_kv=16.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors (batch, seq, heads, dim), got {q.ndim}D")

    batch_size, seq_len, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Try Metal kernel for longer sequences
    if use_metal and _HAS_METAL and seq_len >= 64:
        try:
            return _metal_flash_attention(
                q, k, v, scale, causal, block_q, block_kv
            )
        except Exception as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("flash_attention", e)

    # Python implementation with tiled computation
    return _tiled_flash_attention(q, k, v, scale, causal, block_q, block_kv)


def _tiled_flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    block_q: int,
    block_kv: int,
) -> mx.array:
    """Python implementation of Flash Attention using tiled computation."""
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Output buffer
    output = mx.zeros_like(q)

    # Process query blocks
    for q_start in range(0, seq_len, block_q):
        q_end = min(q_start + block_q, seq_len)
        q_block = q[:, q_start:q_end]  # (batch, block_q, heads, dim)
        block_q_actual = q_end - q_start

        # Initialize running statistics for this query block
        # Shape: (batch, block_q, heads)
        m_i = mx.full((batch_size, block_q_actual, num_heads), -1e9)
        l_i = mx.zeros((batch_size, block_q_actual, num_heads))
        o_i = mx.zeros((batch_size, block_q_actual, num_heads, head_dim))

        # KV range for causal: only up to q_end
        # For non-causal: full sequence
        kv_limit = q_end if causal else seq_len

        # Process KV blocks
        for kv_start in range(0, kv_limit, block_kv):
            kv_end = min(kv_start + block_kv, kv_limit)
            k_block = k[:, kv_start:kv_end]
            v_block = v[:, kv_start:kv_end]

            # Compute attention for this block pair
            chunk_output, chunk_max, chunk_sum = compute_chunk_attention(
                q_block,
                k_block,
                v_block,
                scale,
                causal=causal,
                q_offset=q_start,
                kv_offset=kv_start,
            )

            # Online softmax merge
            o_i, m_i, l_i = online_softmax_merge(
                o_i, m_i, l_i,
                chunk_output, chunk_max, chunk_sum,
            )

        # Write output block (already normalized by online_softmax_merge)
        output = _update_output_block(output, o_i, q_start, q_end)

    return output


def _update_output_block(
    output: mx.array,
    block_output: mx.array,
    start: int,
    end: int,
) -> mx.array:
    """Update a slice of the output tensor.

    MLX doesn't support in-place assignment, so we use scatter or rebuild.
    """
    # Use index update
    indices = mx.arange(start, end)
    # Expand indices for proper broadcasting
    batch_size, _, num_heads, head_dim = output.shape

    # Create full output by concatenating slices
    if start == 0:
        before = mx.zeros((batch_size, 0, num_heads, head_dim))
    else:
        before = output[:, :start]

    if end >= output.shape[1]:
        after = mx.zeros((batch_size, 0, num_heads, head_dim))
    else:
        after = output[:, end:]

    return mx.concatenate([before, block_output, after], axis=1)


def _get_flash_attention_kernel() -> mx.fast.metal_kernel:
    """Get or create the Flash Attention Metal kernel."""
    global _flash_attention_kernel
    if _flash_attention_kernel is None:
        source = """
        // Flash Attention Metal Kernel
        // Processes one (batch, head, q_block) per threadgroup

        uint batch_idx = threadgroup_position_in_grid.z;
        uint head_idx = threadgroup_position_in_grid.y;
        uint q_block_idx = threadgroup_position_in_grid.x;
        uint local_tid = thread_position_in_threadgroup.x;

        if (batch_idx >= batch_size || head_idx >= num_heads) return;

        uint q_block_start = q_block_idx * block_q;
        uint q_pos = q_block_start + local_tid;
        bool valid_q = q_pos < seq_len;

        // Padded dimension for bank-conflict-free access
        uint head_dim_pad = head_dim + 4;

        // Initialize per-thread state (online softmax)
        float max_val = -1e38f;
        float sum_exp = 0.0f;
        float acc[132];  // Support up to head_dim=128 + padding
        for (uint d = 0; d < head_dim; d++) {
            acc[d] = 0.0f;
        }

        // Load Q into registers
        float q_reg[132];
        uint qkv_stride = num_heads * head_dim;
        uint q_offset = batch_idx * seq_len * qkv_stride +
                        q_pos * qkv_stride +
                        head_idx * head_dim;

        if (valid_q) {
            for (uint d = 0; d < head_dim; d++) {
                q_reg[d] = Q[q_offset + d];
            }
        }

        // KV limit for causal masking
        uint kv_limit = causal ? (q_block_start + block_q) : seq_len;
        if (kv_limit > seq_len) kv_limit = seq_len;

        // Base offset for K/V
        uint kv_base = batch_idx * seq_len * qkv_stride + head_idx * head_dim;

        // Process KV in blocks (streaming through memory)
        for (uint kv_start = 0; kv_start < kv_limit; kv_start += block_kv) {
            uint kv_end = kv_start + block_kv;
            if (kv_end > kv_limit) kv_end = kv_limit;
            uint kv_count = kv_end - kv_start;

            // Cooperative load K tile into shared memory
            for (uint i = local_tid; i < kv_count * head_dim; i += threads_per_threadgroup.x) {
                uint kv_local = i / head_dim;
                uint d = i % head_dim;
                uint kv_global = kv_start + kv_local;
                K_shared[kv_local * head_dim_pad + d] = K[kv_base + kv_global * qkv_stride + d];
            }

            // Cooperative load V tile
            for (uint i = local_tid; i < kv_count * head_dim; i += threads_per_threadgroup.x) {
                uint kv_local = i / head_dim;
                uint d = i % head_dim;
                uint kv_global = kv_start + kv_local;
                V_shared[kv_local * head_dim_pad + d] = V[kv_base + kv_global * qkv_stride + d];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Process this KV tile
            if (valid_q) {
                for (uint kv_local = 0; kv_local < kv_count; kv_local++) {
                    uint kv_global = kv_start + kv_local;

                    // Causal mask check
                    if (causal && kv_global > q_pos) continue;

                    // Compute score: Q[q_pos] @ K[kv_pos]
                    float score = 0.0f;
                    for (uint d = 0; d < head_dim; d++) {
                        score += q_reg[d] * K_shared[kv_local * head_dim_pad + d];
                    }
                    score *= scale;

                    // Online softmax update
                    if (score > max_val) {
                        float ratio = exp(max_val - score);
                        sum_exp = sum_exp * ratio + 1.0f;
                        for (uint d = 0; d < head_dim; d++) {
                            acc[d] *= ratio;
                        }
                        max_val = score;

                        // Add V contribution with weight 1
                        for (uint d = 0; d < head_dim; d++) {
                            acc[d] += V_shared[kv_local * head_dim_pad + d];
                        }
                    } else {
                        float weight = exp(score - max_val);
                        sum_exp += weight;
                        for (uint d = 0; d < head_dim; d++) {
                            acc[d] += weight * V_shared[kv_local * head_dim_pad + d];
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Normalize and write output
        if (valid_q) {
            uint o_offset = batch_idx * seq_len * qkv_stride +
                            q_pos * qkv_stride +
                            head_idx * head_dim;

            float inv_sum = 1.0f / sum_exp;
            for (uint d = 0; d < head_dim; d++) {
                O[o_offset + d] = acc[d] * inv_sum;
            }
        }
        """

        _flash_attention_kernel = mx.fast.metal_kernel(
            name="flash_attention",
            input_names=[
                "Q", "K", "V",
                "batch_size", "seq_len", "num_heads", "head_dim",
                "block_q", "block_kv", "causal", "scale"
            ],
            output_names=["O"],
            source=source,
        )
    return _flash_attention_kernel


def _metal_flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    block_q: int,
    block_kv: int,
) -> mx.array:
    """Metal kernel implementation of Flash Attention."""
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Adjust block sizes for large head dimensions
    if head_dim > 64:
        block_q = min(block_q, 16)
        block_kv = min(block_kv, 16)

    kernel = _get_flash_attention_kernel()

    # Ensure contiguous float32 tensors
    q = mx.ascontiguousarray(q.astype(mx.float32))
    k = mx.ascontiguousarray(k.astype(mx.float32))
    v = mx.ascontiguousarray(v.astype(mx.float32))

    # Prepare scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    heads_arr = mx.array([num_heads], dtype=mx.uint32)
    dim_arr = mx.array([head_dim], dtype=mx.uint32)
    block_q_arr = mx.array([block_q], dtype=mx.uint32)
    block_kv_arr = mx.array([block_kv], dtype=mx.uint32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Calculate grid dimensions
    num_q_blocks = (seq_len + block_q - 1) // block_q

    # Grid: (num_q_blocks, num_heads, batch_size)
    outputs = kernel(
        inputs=[
            q, k, v,
            batch_arr, seq_arr, heads_arr, dim_arr,
            block_q_arr, block_kv_arr, causal_arr, scale_arr
        ],
        grid=(num_q_blocks, num_heads, batch_size),
        threadgroup=(block_q, 1, 1),
        output_shapes=[(batch_size, seq_len, num_heads, head_dim)],
        output_dtypes=[mx.float32],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def _reference_flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float],
    causal: bool,
) -> mx.array:
    """Reference implementation using standard O(n²) attention.

    Used for testing correctness of Flash Attention.
    """
    batch, seq, heads, dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Full attention matrix
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    if causal:
        mask = mx.tril(mx.ones((seq, seq)))
        scores = mx.where(mask[None, None, :, :] == 1, scores, -1e9)

    weights = mx.softmax(scores, axis=-1)
    output = weights @ v_t

    return output.transpose(0, 2, 1, 3)


class FlashAttention:
    """Flash Attention module with configurable block sizes.

    A reusable attention module that uses Flash Attention for memory
    efficiency. Produces identical results to standard attention but
    with O(n) memory instead of O(n²).

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        causal: Whether to use causal masking.
        block_q: Query block size (default: auto-tune based on head_dim).
        block_kv: KV block size (default: auto-tune based on head_dim).

    Example:
        >>> attn = FlashAttention(num_heads=8, head_dim=64, causal=True)
        >>> q = mx.random.normal((2, 4096, 8, 64))
        >>> k = mx.random.normal((2, 4096, 8, 64))
        >>> v = mx.random.normal((2, 4096, 8, 64))
        >>> out = attn(q, k, v)  # O(n) memory
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        causal: bool = True,
        block_q: Optional[int] = None,
        block_kv: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)

        # Auto-tune block sizes based on head_dim
        if block_q is None:
            block_q = 32 if head_dim <= 64 else 16
        if block_kv is None:
            block_kv = 32 if head_dim <= 64 else 16

        self.block_q = block_q
        self.block_kv = block_kv

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Apply Flash Attention.

        Args:
            q: Query tensor (batch, seq, num_heads, head_dim).
            k: Key tensor (batch, seq, num_heads, head_dim).
            v: Value tensor (batch, seq, num_heads, head_dim).

        Returns:
            Output tensor of same shape as q.
        """
        return flash_attention(
            q, k, v,
            scale=self.scale,
            causal=self.causal,
            block_q=self.block_q,
            block_kv=self.block_kv,
        )


def get_optimal_flash_config(
    seq_len: int,
    head_dim: int,
    num_heads: int,
) -> Tuple[int, int]:
    """Get optimal Flash Attention block sizes for given dimensions.

    Heuristics based on Apple Silicon shared memory constraints (32KB)
    and typical performance characteristics.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension.
        num_heads: Number of attention heads.

    Returns:
        (block_q, block_kv) tuple of optimal block sizes.
    """
    # Shared memory budget: ~26KB for Q, K, V tiles
    # Each tile: block_size * (head_dim + 4) * 4 bytes
    head_dim_pad = head_dim + 4
    bytes_per_block = head_dim_pad * 4  # Per element in block

    # Target: 3 tiles (Q, K, V) fit in 26KB
    max_tile_bytes = 26000 // 3
    max_block_size = max_tile_bytes // bytes_per_block

    # Clamp to reasonable range
    if head_dim <= 64:
        block_q = min(32, max_block_size, seq_len)
        block_kv = min(32, max_block_size, seq_len)
    elif head_dim <= 128:
        block_q = min(16, max_block_size, seq_len)
        block_kv = min(16, max_block_size, seq_len)
    else:
        block_q = min(8, max_block_size, seq_len)
        block_kv = min(8, max_block_size, seq_len)

    return block_q, block_kv
