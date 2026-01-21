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
import threading
from typing import Optional, Tuple

import mlx.core as mx

from mlx_primitives.attention._online_softmax import (
    compute_chunk_attention,
    online_softmax_merge,
)
from mlx_primitives.constants import (
    ATTENTION_MASK_VALUE,
    METAL_ATTENTION_MAX_HEAD_DIM,
    METAL_SOFTMAX_EPSILON,
)

# Check if Metal kernels are available
_HAS_METAL = hasattr(mx.fast, "metal_kernel")

# Check if MLX's optimized SDPA is available (preferred over custom kernels)
_HAS_SDPA = hasattr(mx.fast, "scaled_dot_product_attention")

# Kernel cache with thread safety
_flash_attention_kernel: Optional[mx.fast.metal_kernel] = None
_flash_kernel_lock = threading.Lock()


def flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
    block_q: int = 32,
    block_kv: int = 32,
    use_metal: bool = True,
    dropout_p: float = 0.0,
    training: bool = True,
    layout: str = "BSHD",
) -> mx.array:
    """Flash Attention - O(n) memory attention via tiled online softmax.

    Computes exact attention identical to standard scaled dot-product attention
    but without materializing the O(n²) attention matrix. Instead, processes
    queries and keys/values in tiles, using online softmax to accumulate results.

    Args:
        q: Query tensor of shape depending on layout parameter.
        k: Key tensor of shape depending on layout parameter.
        v: Value tensor of shape depending on layout parameter.
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        causal: If True, apply causal masking (queries can only attend to
            earlier positions).
        block_q: Query block size for tiling. Larger = faster but more memory.
        block_kv: KV block size for tiling. Larger = faster but more memory.
        use_metal: Use Metal kernel if available.
        dropout_p: Dropout probability on attention weights. Default 0.0 (no dropout).
            When dropout_p > 0, falls back to reference implementation.
        training: If False, dropout is not applied. Default True.
        layout: Tensor layout format. Options:
            - "BSHD": (batch, seq_len, num_heads, head_dim) - default
            - "BHSD": (batch, num_heads, seq_len, head_dim) - no transpose overhead

    Returns:
        Output tensor of same shape as input (matching layout parameter).

    Example:
        >>> # Standard usage (BSHD layout)
        >>> q = mx.random.normal((2, 1024, 8, 64))
        >>> k = mx.random.normal((2, 1024, 8, 64))
        >>> v = mx.random.normal((2, 1024, 8, 64))
        >>> out = flash_attention(q, k, v, causal=True)
        >>>
        >>> # BHSD layout (no transpose overhead - faster for short sequences)
        >>> q = mx.random.normal((2, 8, 1024, 64))  # (batch, heads, seq, dim)
        >>> k = mx.random.normal((2, 8, 1024, 64))
        >>> v = mx.random.normal((2, 8, 1024, 64))
        >>> out = flash_attention(q, k, v, causal=True, layout="BHSD")
        >>>
        >>> # With dropout (training)
        >>> out = flash_attention(q, k, v, causal=True, dropout_p=0.1)

    Note:
        For best performance, block sizes should be tuned based on:
        - Hardware (Apple Silicon generation, available memory)
        - Sequence length (longer sequences benefit from larger blocks)
        - Head dimension (larger dims need smaller blocks to fit in shared memory)

        Default block_q=32, block_kv=32 is reasonable for most configurations.
        For head_dim=128, consider reducing to block_q=16, block_kv=16.

        When dropout_p > 0 and training=True, falls back to the reference
        implementation since dropout requires materializing attention weights.

        For short sequences (seq_len <= 512) with BSHD layout, transpose overhead
        may dominate. Use layout="BHSD" or mx.fast.scaled_dot_product_attention
        directly for maximum performance.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got {q.ndim}D")

    if layout not in ("BSHD", "BHSD"):
        raise ValueError(f"layout must be 'BSHD' or 'BHSD', got '{layout}'")

    # Handle BHSD layout - call SDPA directly without transpose overhead
    if layout == "BHSD":
        batch_size, num_heads, seq_len, head_dim = q.shape
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # BHSD layout matches SDPA's expected format - no transpose needed
        if _HAS_SDPA and use_metal:
            return mx.fast.scaled_dot_product_attention(
                q, k, v,
                scale=scale,
                mask="causal" if causal else None,
            )
        else:
            # Fallback to standard attention for BHSD
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            if causal:
                mask = mx.tril(mx.ones((seq_len, seq_len)))
                scores = mx.where(mask[None, None, :, :] == 1, scores, ATTENTION_MASK_VALUE)
            weights = mx.softmax(scores, axis=-1)
            return weights @ v

    # BSHD layout (default) - extract dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Fall back to reference implementation when dropout is enabled
    # (dropout requires materializing the full attention matrix)
    if dropout_p > 0.0 and training:
        return _reference_flash_attention_with_dropout(
            q, k, v, scale, causal, dropout_p
        )

    # For very short sequences or small batch+sequence combos, use reference attention
    # which now uses SDPA when available. This avoids the overhead of the tiled Python
    # implementation or custom Metal kernel for sizes where O(n²) memory is acceptable.
    # The reference path is kept for correctness testing and SDPA-unavailable fallback.
    if seq_len <= 256 or (batch_size <= 4 and seq_len <= 512):
        return _reference_flash_attention(q, k, v, scale, causal)

    # Priority 1: Use MLX's built-in SDPA when available
    # This is the most optimized path, using MLX's native implementation
    # which has proper multi-threading and shared memory support
    if _HAS_SDPA and use_metal:
        try:
            # MLX SDPA expects (batch, heads, seq, dim) but our input is (batch, seq, heads, dim)
            # Transpose to match SDPA's expected format
            q_sdpa = mx.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, dim)
            k_sdpa = mx.transpose(k, (0, 2, 1, 3))
            v_sdpa = mx.transpose(v, (0, 2, 1, 3))

            out_sdpa = mx.fast.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                scale=scale,
                mask="causal" if causal else None,
            )

            # Transpose back to (batch, seq, heads, dim)
            return mx.transpose(out_sdpa, (0, 2, 1, 3))
        except (RuntimeError, TypeError) as e:
            # Fall through to custom implementation if SDPA fails
            # (e.g., unsupported configuration)
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("flash_attention (SDPA)", e)

    # Priority 2: Custom Metal kernel (1 thread per threadgroup limitation)
    # This is a workaround implementation - SDPA is preferred when available
    if use_metal and _HAS_METAL and seq_len >= 64:
        try:
            return _metal_flash_attention(
                q, k, v, scale, causal, block_q, block_kv
            )
        except RuntimeError as e:
            # Catch Metal kernel errors, but let programming bugs propagate
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("flash_attention", e)

    # Priority 3: Python implementation with tiled computation
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

    # Collect output blocks for single concatenation at the end (O(n) instead of O(n^2))
    output_blocks = []

    # Process query blocks
    for q_start in range(0, seq_len, block_q):
        q_end = min(q_start + block_q, seq_len)
        q_block = q[:, q_start:q_end]  # (batch, block_q, heads, dim)
        block_q_actual = q_end - q_start

        # Initialize running statistics for this query block
        # Shape: (batch, block_q, heads)
        m_i = mx.full((batch_size, block_q_actual, num_heads), ATTENTION_MASK_VALUE)
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

        # Normalize at the end of each query block
        o_i_normalized = o_i / (l_i[..., None] + METAL_SOFTMAX_EPSILON)
        output_blocks.append(o_i_normalized)

    # Single concatenation at the end (O(n) instead of O(n^2))
    return mx.concatenate(output_blocks, axis=1)


def _get_flash_attention_kernel() -> mx.fast.metal_kernel:
    """Get or create the Flash Attention Metal kernel (thread-safe)."""
    global _flash_attention_kernel
    if _flash_attention_kernel is None:
        with _flash_kernel_lock:
            # Double-check after acquiring lock
            if _flash_attention_kernel is not None:
                return _flash_attention_kernel
            # Flash Attention kernel - simple per-element online softmax
            # Each threadgroup handles one query position (1 thread per threadgroup)
            # This is a workaround for MLX metal_kernel not supporting multiple threads
            # Single-pass online softmax - computes scores once and accumulates V in same loop
            # This is ~2x faster than the two-pass approach
            source = """
        // Flash Attention Metal Kernel - Single Pass Online Softmax
        // One threadgroup (1 thread) per query position
        // Optimized: computes scores once, accumulates V in same loop

        // Dereference scalar parameters (passed as single-element arrays)
        uint _batch_size = batch_size[0];
        uint _seq_len = seq_len[0];
        uint _num_heads = num_heads[0];
        uint _head_dim = head_dim[0];
        uint _causal = causal[0];
        float _scale = scale[0];

        // Grid layout: (seq_len, num_heads, batch_size)
        uint batch_idx = threadgroup_position_in_grid.z;
        uint head_idx = threadgroup_position_in_grid.y;
        uint q_idx = threadgroup_position_in_grid.x;

        if (batch_idx >= _batch_size || head_idx >= _num_heads || q_idx >= _seq_len) return;

        uint qkv_stride = _num_heads * _head_dim;
        uint q_offset = batch_idx * _seq_len * qkv_stride + q_idx * qkv_stride + head_idx * _head_dim;
        uint kv_base = batch_idx * _seq_len * qkv_stride + head_idx * _head_dim;

        // Initialize online softmax state
        float max_val = -1e38f;
        float sum_exp = 0.0f;

        // Accumulator for weighted V (max head_dim = 128)
        float acc[128];
        for (uint d = 0; d < _head_dim; d++) {
            acc[d] = 0.0f;
        }

        // Load Q into registers for reuse
        float q_reg[128];
        for (uint d = 0; d < _head_dim; d++) {
            q_reg[d] = Q[q_offset + d];
        }

        // KV limit for causal
        uint kv_limit = _causal ? (q_idx + 1) : _seq_len;

        // Single-pass: compute score, update running max/sum, and accumulate weighted V
        for (uint kv_idx = 0; kv_idx < kv_limit; kv_idx++) {
            uint kv_offset = kv_base + kv_idx * qkv_stride;

            // Compute Q @ K score
            float score = 0.0f;
            for (uint d = 0; d < _head_dim; d++) {
                score += q_reg[d] * K[kv_offset + d];
            }
            score *= _scale;

            // Online softmax update with V accumulation
            if (score > max_val) {
                // New max found - rescale previous accumulator
                float ratio = exp(max_val - score);
                sum_exp = sum_exp * ratio + 1.0f;
                for (uint d = 0; d < _head_dim; d++) {
                    acc[d] = acc[d] * ratio + V[kv_offset + d];
                }
                max_val = score;
            } else {
                // Accumulate with current weight
                float weight = exp(score - max_val);
                sum_exp += weight;
                for (uint d = 0; d < _head_dim; d++) {
                    acc[d] += weight * V[kv_offset + d];
                }
            }
        }

        // Normalize and write output
        uint o_offset = batch_idx * _seq_len * qkv_stride + q_idx * qkv_stride + head_idx * _head_dim;
        float inv_sum = 1.0f / (sum_exp + SOFTMAX_EPSf);
        for (uint d = 0; d < _head_dim; d++) {
            O[o_offset + d] = acc[d] * inv_sum;
        }
        """.replace("SOFTMAX_EPS", str(METAL_SOFTMAX_EPSILON))

        _flash_attention_kernel = mx.fast.metal_kernel(
            name="flash_attention",
            input_names=[
                "Q", "K", "V",
                "batch_size", "seq_len", "num_heads", "head_dim",
                "causal", "scale"
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
    """Metal kernel implementation of Flash Attention.

    Note: block_q and block_kv parameters are kept for API compatibility but
    are not used by the current kernel implementation. The kernel processes
    one query position per threadgroup due to MLX metal_kernel limitations.
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Validate head_dim against Metal kernel limits
    if head_dim > METAL_ATTENTION_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={head_dim} exceeds Metal kernel limit of "
            f"{METAL_ATTENTION_MAX_HEAD_DIM}. Use use_metal=False for Python fallback."
        )

    kernel = _get_flash_attention_kernel()

    # Ensure contiguous float32 tensors
    q = mx.contiguous(q.astype(mx.float32))
    k = mx.contiguous(k.astype(mx.float32))
    v = mx.contiguous(v.astype(mx.float32))

    # Prepare scalar inputs
    batch_arr = mx.array([batch_size], dtype=mx.uint32)
    seq_arr = mx.array([seq_len], dtype=mx.uint32)
    heads_arr = mx.array([num_heads], dtype=mx.uint32)
    dim_arr = mx.array([head_dim], dtype=mx.uint32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Grid: (seq_len, num_heads, batch_size) - one threadgroup per query position
    # Threadgroup: (1, 1, 1) - MLX metal_kernel only supports 1 thread per threadgroup
    outputs = kernel(
        inputs=[
            q, k, v,
            batch_arr, seq_arr, heads_arr, dim_arr,
            causal_arr, scale_arr
        ],
        grid=(seq_len, num_heads, batch_size),
        threadgroup=(1, 1, 1),
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
    """Reference implementation using MLX's optimized SDPA when available.

    Falls back to standard O(n²) attention if SDPA is unavailable.
    Used for small sequence lengths where O(n²) memory is acceptable.
    """
    batch, seq, heads, dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Use MLX SDPA when available - same transposes but optimized kernel
    if _HAS_SDPA:
        q_sdpa = mx.transpose(q, (0, 2, 1, 3))
        k_sdpa = mx.transpose(k, (0, 2, 1, 3))
        v_sdpa = mx.transpose(v, (0, 2, 1, 3))

        out = mx.fast.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            scale=scale,
            mask="causal" if causal else None,
        )
        return mx.transpose(out, (0, 2, 1, 3))

    # Manual fallback for systems without SDPA
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Full attention matrix
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    if causal:
        mask = mx.tril(mx.ones((seq, seq)))
        scores = mx.where(mask[None, None, :, :] == 1, scores, ATTENTION_MASK_VALUE)

    weights = mx.softmax(scores, axis=-1)
    output = weights @ v_t

    return output.transpose(0, 2, 1, 3)


def _reference_flash_attention_with_dropout(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> mx.array:
    """Reference implementation with dropout support.

    Uses standard O(n²) attention since dropout requires materializing
    the full attention weight matrix to apply the dropout mask.

    Args:
        q: Query tensor (batch, seq, heads, dim).
        k: Key tensor (batch, seq, heads, dim).
        v: Value tensor (batch, seq, heads, dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability (0.0 to 1.0).

    Returns:
        Output tensor (batch, seq, heads, dim).
    """
    batch, seq, heads, dim = q.shape

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    # Full attention matrix
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    if causal:
        mask = mx.tril(mx.ones((seq, seq)))
        scores = mx.where(mask[None, None, :, :] == 1, scores, ATTENTION_MASK_VALUE)

    weights = mx.softmax(scores, axis=-1)

    # Apply dropout to attention weights
    if dropout_p > 0.0:
        # Generate dropout mask: keep with probability (1 - dropout_p)
        dropout_mask = mx.random.bernoulli(1.0 - dropout_p, weights.shape)
        # Scale by 1/(1-p) to maintain expected value (inverted dropout)
        weights = weights * dropout_mask / (1.0 - dropout_p)

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
