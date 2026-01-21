"""Optimized Grouped Query Attention for MLX.

This module provides GQA implementations that avoid K/V expansion:

1. **Primary path (recommended)**: Uses mx.fast.scaled_dot_product_attention
   which provides ~6x speedup and handles GQA natively without expansion.

2. **Custom Metal kernel path**: For research/experimentation. Infrastructure
   is correct but slower than MLX's optimized SDPA without compiled Metal.

3. **Reference path**: Expands K/V for compatibility testing.
"""

from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.constants import METAL_ATTENTION_MAX_HEAD_DIM

# Kernel caches with thread safety
_gqa_kernel = None
_gqa_kernel_tiled = None
_gqa_kernel_lock = threading.Lock()

# Check if metal kernels are available
_USE_METAL_KERNELS = hasattr(mx.fast, 'metal_kernel')

# Check if MLX fast SDPA is available (handles GQA natively)
_HAS_MLX_FAST_SDPA = hasattr(mx.fast, 'scaled_dot_product_attention')


def _mlx_fast_gqa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = False,
) -> mx.array:
    """Use MLX's optimized SDPA for GQA (~6x faster).

    MLX's SDPA handles GQA natively without K/V expansion.

    Args:
        q: Query tensor (batch, seq_q, num_q_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.

    Returns:
        Output tensor (batch, seq_q, num_q_heads, head_dim).
    """
    # MLX SDPA expects (batch, heads, seq, dim) layout
    q_t = q.transpose(0, 2, 1, 3)  # (batch, q_heads, seq_q, dim)
    k_t = k.transpose(0, 2, 1, 3)  # (batch, kv_heads, seq_kv, dim)
    v_t = v.transpose(0, 2, 1, 3)  # (batch, kv_heads, seq_kv, dim)

    # MLX SDPA handles causal mask with string "causal"
    mask = "causal" if causal else None

    # MLX SDPA handles GQA natively - no expansion needed!
    output = mx.fast.scaled_dot_product_attention(
        q_t, k_t, v_t,
        scale=scale,
        mask=mask,
    )

    # Transpose back to (batch, seq, heads, dim)
    return output.transpose(0, 2, 1, 3)


def _get_gqa_kernel():
    """Simple GQA kernel without K/V expansion.

    Each thread processes one (batch, q_head, q_pos) output.
    K/V are indexed using kv_head = q_head / num_groups.
    """
    global _gqa_kernel

    if _gqa_kernel is None:
        with _gqa_kernel_lock:
            # Double-check after acquiring lock
            if _gqa_kernel is not None:
                return _gqa_kernel

            # Use constant for array sizes to match METAL_ATTENTION_MAX_HEAD_DIM
            max_dim = METAL_ATTENTION_MAX_HEAD_DIM
            source = f"""
            uint tid = thread_position_in_grid.x;

            // Get dimensions from shapes
            uint batch_size = q_shape[0];
            uint seq_len_q = q_shape[1];
            uint num_q_heads = q_shape[2];
            uint head_dim = q_shape[3];
            uint seq_len_kv = k_shape[1];
            uint num_kv_heads = k_shape[2];

            // Read parameters: [scale, num_groups, is_causal]
            float scale_val = params[0];
            uint num_groups = uint(params[1]);
            bool is_causal = params[2] > 0.5f;

            // Total work items
            uint total_items = batch_size * num_q_heads * seq_len_q;
            if (tid >= total_items) return;

            // Decode thread index to (batch, q_head, q_pos)
            uint q_pos = tid % seq_len_q;
            uint tmp = tid / seq_len_q;
            uint q_head_idx = tmp % num_q_heads;
            uint batch_idx = tmp / num_q_heads;

            // Map Q head to KV head
            uint kv_head_idx = q_head_idx / num_groups;

            // Memory layout: (batch, seq, heads, dim)
            uint q_base = batch_idx * (seq_len_q * num_q_heads * head_dim) +
                          q_pos * (num_q_heads * head_dim) +
                          q_head_idx * head_dim;

            // KV stride info
            uint kv_stride_batch = seq_len_kv * num_kv_heads * head_dim;
            uint kv_stride_seq = num_kv_heads * head_dim;
            uint kv_stride_head = head_dim;

            // Load Q vector for this thread (max head_dim = {max_dim})
            float q_vec[{max_dim}];
            for (uint d = 0; d < head_dim; d++) {{
                q_vec[d] = q[q_base + d];
            }}

            // Online softmax state
            float max_val = -INFINITY;
            float sum_val = 0.0f;
            float acc[{max_dim}];
            for (uint d = 0; d < head_dim; d++) acc[d] = 0.0f;

            // Iterate over KV sequence
            for (uint kv_pos = 0; kv_pos < seq_len_kv; kv_pos++) {{
                // Causal masking
                if (is_causal && kv_pos > q_pos) break;

                // K/V base offset (using kv_head_idx, not q_head_idx!)
                uint kv_base = batch_idx * kv_stride_batch +
                               kv_pos * kv_stride_seq +
                               kv_head_idx * kv_stride_head;

                // Compute Q @ K^T for this position
                float dot = 0.0f;
                for (uint d = 0; d < head_dim; d++) {{
                    dot += q_vec[d] * k[kv_base + d];
                }}
                dot *= scale_val;

                // Online softmax
                float old_max = max_val;
                max_val = max(max_val, dot);

                if (old_max == -INFINITY) {{
                    sum_val = 1.0f;
                }} else {{
                    float rescale = exp(old_max - max_val);
                    sum_val = sum_val * rescale + exp(dot - max_val);
                    for (uint d = 0; d < head_dim; d++) acc[d] *= rescale;
                }}

                // Accumulate V
                float w = exp(dot - max_val);
                for (uint d = 0; d < head_dim; d++) {{
                    acc[d] += w * v[kv_base + d];
                }}
            }}

            // Write normalized output
            uint out_base = batch_idx * (seq_len_q * num_q_heads * head_dim) +
                            q_pos * (num_q_heads * head_dim) +
                            q_head_idx * head_dim;

            // Get output type for explicit casting (required for bf16)
            typedef decltype(out[0] + out[0]) OutT;

            if (sum_val > 0.0f) {{
                float inv_sum = 1.0f / sum_val;
                for (uint d = 0; d < head_dim; d++) {{
                    out[out_base + d] = OutT(acc[d] * inv_sum);
                }}
            }} else {{
                for (uint d = 0; d < head_dim; d++) {{
                    out[out_base + d] = OutT(0.0f);
                }}
            }}
            """

            _gqa_kernel = mx.fast.metal_kernel(
                name="gqa_optimized",
                input_names=["q", "k", "v", "params"],
                output_names=["out"],
                source=source,
            )

    return _gqa_kernel


def _get_gqa_kernel_tiled():
    """Tiled GQA kernel with shared memory for K/V reuse.

    Architecture:
    - Grid: (num_Q_blocks, num_Q_heads, batch)
    - Threadgroup: (BLOCK_SIZE, 1, 1) - each thread owns one Q position
    - All threads in a threadgroup share the same KV head
    - K/V tiles loaded to shared memory, reused across all Q positions

    This provides significant speedup when num_groups > 1 because K/V
    are loaded once and used by multiple Q heads.
    """
    global _gqa_kernel_tiled

    if _gqa_kernel_tiled is None:
        with _gqa_kernel_lock:
            # Double-check after acquiring lock
            if _gqa_kernel_tiled is not None:
                return _gqa_kernel_tiled

            # Use constants for array sizes
            max_dim = METAL_ATTENTION_MAX_HEAD_DIM
            block_kv = 32  # Must match BLOCK_KV in kernel
            source = f"""
            // Block size for Q tiling
            const uint BLOCK_Q = 32;
            const uint BLOCK_KV = 32;

            // Thread indices
            uint local_tid = thread_index_in_threadgroup;
            uint q_block_id = threadgroup_position_in_grid.x;
            uint q_head_idx = threadgroup_position_in_grid.y;
            uint batch_idx = threadgroup_position_in_grid.z;

            // Get dimensions
            uint batch_size = q_shape[0];
            uint seq_len_q = q_shape[1];
            uint num_q_heads = q_shape[2];
            uint head_dim = q_shape[3];
            uint seq_len_kv = k_shape[1];
            uint num_kv_heads = k_shape[2];

            // Read parameters
            float scale_val = params[0];
            uint num_groups = uint(params[1]);
            bool is_causal = params[2] > 0.5f;

            // Map Q head to KV head
            uint kv_head_idx = q_head_idx / num_groups;

            // This thread's global Q index
            uint global_q_idx = q_block_id * BLOCK_Q + local_tid;
            bool valid_q = (global_q_idx < seq_len_q);

            // Shared memory for K and V tiles (indexed by KV head, shared across Q group)
            // Layout: [BLOCK_KV * head_dim], sizes derived from METAL_ATTENTION_MAX_HEAD_DIM
            threadgroup float K_shared[{block_kv} * {max_dim}];
            threadgroup float V_shared[{block_kv} * {max_dim}];

            // Memory strides
            uint q_stride_batch = seq_len_q * num_q_heads * head_dim;
            uint q_stride_seq = num_q_heads * head_dim;
            uint q_stride_head = head_dim;

            uint kv_stride_batch = seq_len_kv * num_kv_heads * head_dim;
            uint kv_stride_seq = num_kv_heads * head_dim;
            uint kv_stride_head = head_dim;

            // Load this thread's Q vector (max head_dim = {max_dim})
            float q_vec[{max_dim}];
            if (valid_q) {{
                uint q_base = batch_idx * q_stride_batch +
                              global_q_idx * q_stride_seq +
                              q_head_idx * q_stride_head;

                for (uint d = 0; d < head_dim; d++) {{
                    q_vec[d] = q[q_base + d];
                }}
            }}

            // Online softmax state
            float max_val = -INFINITY;
            float sum_val = 0.0f;
            float acc[{max_dim}];
            for (uint d = 0; d < head_dim; d++) acc[d] = 0.0f;

            // Number of KV blocks
            uint num_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;

            // Iterate over KV blocks
            for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {{
                uint kv_start = kv_block * BLOCK_KV;

                // === Cooperative load: K/V tiles to shared memory ===
                // Thread i loads K[kv_start + i] and V[kv_start + i]
                uint kv_load_idx = kv_start + local_tid;
                if (kv_load_idx < seq_len_kv && local_tid < BLOCK_KV) {{
                    uint kv_base = batch_idx * kv_stride_batch +
                                   kv_load_idx * kv_stride_seq +
                                   kv_head_idx * kv_stride_head;  // Note: using kv_head_idx!

                    for (uint d = 0; d < head_dim; d++) {{
                        K_shared[local_tid * head_dim + d] = k[kv_base + d];
                        V_shared[local_tid * head_dim + d] = v[kv_base + d];
                    }}
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // === Compute attention scores against K tile ===
                if (valid_q) {{
                    uint kv_end = min(kv_start + BLOCK_KV, seq_len_kv);
                    uint num_valid_kv = kv_end - kv_start;

                    for (uint kv_local = 0; kv_local < num_valid_kv; kv_local++) {{
                        uint kv_tensor_idx = kv_start + kv_local;

                        // Causal masking
                        if (is_causal && kv_tensor_idx > global_q_idx) continue;

                        // Dot product: Q @ K_shared[kv_local]
                        float dot = 0.0f;
                        for (uint d = 0; d < head_dim; d++) {{
                            dot += q_vec[d] * K_shared[kv_local * head_dim + d];
                        }}
                        dot *= scale_val;

                        // Online softmax update
                        float old_max = max_val;
                        max_val = max(max_val, dot);

                        if (old_max == -INFINITY) {{
                            sum_val = 1.0f;
                        }} else {{
                            float rescale = exp(old_max - max_val);
                            sum_val = sum_val * rescale + exp(dot - max_val);
                            for (uint d = 0; d < head_dim; d++) {{
                                acc[d] *= rescale;
                            }}
                        }}

                        // Accumulate weighted V
                        float w = exp(dot - max_val);
                        for (uint d = 0; d < head_dim; d++) {{
                            acc[d] += w * V_shared[kv_local * head_dim + d];
                        }}
                    }}
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            // === Write output ===
            if (valid_q) {{
                uint out_base = batch_idx * q_stride_batch +
                                global_q_idx * q_stride_seq +
                                q_head_idx * q_stride_head;

                // Get output type for explicit casting (required for bf16)
                typedef decltype(out[0] + out[0]) OutT;

                if (sum_val > 0.0f) {{
                    float inv_sum = 1.0f / sum_val;
                    for (uint d = 0; d < head_dim; d++) {{
                        out[out_base + d] = OutT(acc[d] * inv_sum);
                    }}
                }} else {{
                    for (uint d = 0; d < head_dim; d++) {{
                        out[out_base + d] = OutT(0.0f);
                    }}
                }}
            }}
            """

            _gqa_kernel_tiled = mx.fast.metal_kernel(
                name="gqa_optimized_tiled",
                input_names=["q", "k", "v", "params"],
                output_names=["out"],
                source=source,
            )

    return _gqa_kernel_tiled


def fast_gqa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    num_kv_groups: int,
    scale: Optional[float] = None,
    causal: bool = False,
    use_tiled: Optional[bool] = None,
) -> mx.array:
    """Optimized GQA attention without K/V expansion.

    This kernel computes grouped query attention by indexing into the
    appropriate KV head directly, rather than expanding K/V to match
    the number of Q heads.

    Args:
        q: Query tensor (batch, seq_q, num_q_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        num_kv_groups: Number of Q heads per KV head.
        scale: Attention scale factor (default: 1/sqrt(head_dim)).
        causal: Whether to apply causal masking.
        use_tiled: Force tiled kernel (True), non-tiled (False), or auto (None).

    Returns:
        Output tensor (batch, seq_q, num_q_heads, head_dim).

    Example:
        >>> # 64 Q heads, 8 KV heads (8 groups)
        >>> q = mx.random.normal((2, 512, 64, 64))
        >>> k = mx.random.normal((2, 512, 8, 64))
        >>> v = mx.random.normal((2, 512, 8, 64))
        >>> out = fast_gqa_attention(q, k, v, num_kv_groups=8, causal=True)
        >>> out.shape
        (2, 512, 64, 64)
    """
    if q.ndim != 4:
        raise ValueError(f"Q must be 4D (batch, seq_q, num_q_heads, head_dim), got {q.ndim}D")
    if k.ndim != 4:
        raise ValueError(f"K must be 4D (batch, seq_kv, num_kv_heads, head_dim), got {k.ndim}D")
    if v.ndim != 4:
        raise ValueError(f"V must be 4D (batch, seq_kv, num_kv_heads, head_dim), got {v.ndim}D")

    batch_size, seq_q, num_q_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    if num_q_heads != num_kv_heads * num_kv_groups:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must equal "
            f"num_kv_heads ({num_kv_heads}) * num_kv_groups ({num_kv_groups})"
        )
    if head_dim > METAL_ATTENTION_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim ({head_dim}) exceeds METAL_ATTENTION_MAX_HEAD_DIM ({METAL_ATTENTION_MAX_HEAD_DIM})"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Auto-select tiled kernel for longer sequences
    if use_tiled is None:
        use_tiled = seq_q >= 32 and seq_kv >= 32

    # Pack parameters
    params = mx.array([scale, float(num_kv_groups), 1.0 if causal else 0.0])

    if use_tiled:
        # Tiled kernel: grid is (num_Q_blocks, num_Q_heads, batch)
        block_size = 32
        num_q_blocks = (seq_q + block_size - 1) // block_size
        grid = (num_q_blocks * block_size, num_q_heads, batch_size)
        threadgroup = (block_size, 1, 1)

        kernel = _get_gqa_kernel_tiled()
    else:
        # Simple kernel: one thread per output
        total_threads = batch_size * num_q_heads * seq_q
        threadgroup_size = 256
        num_groups = (total_threads + threadgroup_size - 1) // threadgroup_size
        grid = (num_groups * threadgroup_size, 1, 1)
        threadgroup = (threadgroup_size, 1, 1)

        kernel = _get_gqa_kernel()

    # Output shape matches Q shape
    output_shape = (batch_size, seq_q, num_q_heads, head_dim)

    outputs = kernel(
        inputs=[q, k, v, params],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[output_shape],
        output_dtypes=[q.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def gqa_attention_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    num_kv_groups: int,
    scale: Optional[float] = None,
    causal: bool = False,
) -> mx.array:
    """Reference GQA implementation using K/V expansion.

    This is the naive implementation that expands K/V to match Q heads.
    Used as fallback and for correctness testing.
    """
    batch_size, seq_q, num_q_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Expand K and V: (batch, seq, kv_heads, groups, dim) -> (batch, seq, q_heads, dim)
    k_expanded = k[:, :, :, None, :]
    k_expanded = mx.broadcast_to(
        k_expanded, (batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim)
    )
    k_expanded = k_expanded.reshape(batch_size, seq_kv, num_q_heads, head_dim)

    v_expanded = v[:, :, :, None, :]
    v_expanded = mx.broadcast_to(
        v_expanded, (batch_size, seq_kv, num_kv_heads, num_kv_groups, head_dim)
    )
    v_expanded = v_expanded.reshape(batch_size, seq_kv, num_q_heads, head_dim)

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k_expanded.transpose(0, 2, 1, 3)
    v_t = v_expanded.transpose(0, 2, 1, 3)

    # Compute attention scores
    scores = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    if causal:
        mask = mx.triu(
            mx.full((seq_q, seq_kv), float("-inf")),
            k=seq_kv - seq_q + 1,
        )
        scores = scores + mask

    # Softmax and output
    weights = mx.softmax(scores, axis=-1)
    output = weights @ v_t

    # Transpose back
    return output.transpose(0, 2, 1, 3)


def gqa_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    num_kv_groups: int,
    scale: Optional[float] = None,
    causal: bool = False,
    use_mlx_sdpa: bool = True,
    use_metal: bool = True,
) -> mx.array:
    """Grouped Query Attention with automatic optimization.

    Uses MLX's fast SDPA when available (~6x faster), falls back to
    custom Metal kernel or reference implementation.

    Args:
        q: Query tensor (batch, seq_q, num_q_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        num_kv_groups: Number of Q heads per KV head.
        scale: Attention scale factor (default: 1/sqrt(head_dim)).
        causal: Whether to apply causal masking.
        use_mlx_sdpa: Use mx.fast.scaled_dot_product_attention (default: True).
            This is ~6x faster and should be preferred.
        use_metal: Whether to use custom Metal kernel as fallback (default: True).

    Returns:
        Output tensor (batch, seq_q, num_q_heads, head_dim).
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Primary path: MLX's optimized SDPA (~6x faster)
    # Handles GQA natively without K/V expansion
    if use_mlx_sdpa and _HAS_MLX_FAST_SDPA:
        try:
            return _mlx_fast_gqa(q, k, v, scale, causal)
        except (RuntimeError, TypeError) as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("gqa_attention (SDPA)", e)

    # Secondary path: Custom Metal kernel (correct but slower)
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_gqa_attention(q, k, v, num_kv_groups, scale, causal)
        except RuntimeError as e:
            from mlx_primitives.utils.logging import log_fallback
            log_fallback("gqa_attention (Metal)", e)

    # Fallback: Reference implementation with K/V expansion
    return gqa_attention_reference(q, k, v, num_kv_groups, scale, causal)


class OptimizedGQA(nn.Module):
    """Optimized Grouped Query Attention module.

    Uses MLX's fast SDPA (~6x faster) which handles GQA natively without
    K/V expansion, providing significant memory bandwidth savings for
    models with many Q heads per KV head (e.g., Llama 2 70B with 64 Q
    heads / 8 KV heads).

    Args:
        dims: Model dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head (default: dims // num_heads).
        causal: Whether to use causal masking (default: False).
        bias: Whether to use bias in projections (default: False).
        use_mlx_sdpa: Use mx.fast.scaled_dot_product_attention (default: True).

    Example:
        >>> # Llama 2 70B style configuration
        >>> attn = OptimizedGQA(
        ...     dims=8192,
        ...     num_heads=64,
        ...     num_kv_heads=8,
        ...     causal=True,
        ... )
        >>> x = mx.random.normal((2, 1024, 8192))
        >>> output, cache = attn(x)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        causal: bool = False,
        bias: bool = False,
        use_mlx_sdpa: bool = True,
    ):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        self.dims = dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim or dims // num_heads
        self.causal = causal
        self.use_mlx_sdpa = use_mlx_sdpa

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        q_dim = num_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(dims, q_dim, bias=bias)
        self.k_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.v_proj = nn.Linear(dims, kv_dim, bias=bias)
        self.out_proj = nn.Linear(q_dim, dims, bias=bias)

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass with optimized GQA.

        Args:
            queries: Query input (batch, seq_q, dims).
            keys: Key input (batch, seq_kv, dims). If None, uses queries.
            values: Value input (batch, seq_kv, dims). If None, uses keys.
            mask: Optional attention mask (not yet supported in kernel).
            cache: Optional KV cache (k_cache, v_cache).

        Returns:
            Tuple of (output, new_cache).
        """
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        batch_size, seq_q, _ = queries.shape
        _, seq_kv, _ = keys.shape

        # Project
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        # Reshape: (batch, seq, heads, dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_kv_heads, self.head_dim)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Optimized GQA attention (uses MLX SDPA by default, ~6x faster)
        output = gqa_attention(
            q, k, v,
            num_kv_groups=self.num_kv_groups,
            scale=self.scale,
            causal=self.causal,
            use_mlx_sdpa=self.use_mlx_sdpa,
        )

        # Reshape and project output
        output = output.reshape(batch_size, seq_q, -1)
        return self.out_proj(output), new_cache
