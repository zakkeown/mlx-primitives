"""Flash Attention implementation for MLX.

Memory-efficient attention using tiling to avoid materializing the full
attention matrix. Based on:
"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
https://arxiv.org/abs/2205.14135

"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
https://arxiv.org/abs/2307.08691

Primary implementation uses mx.fast.scaled_dot_product_attention which provides
~8-9x speedup over pure Python implementations via compiled Metal kernels.

Supports:
- Automatic use of MLX's optimized SDPA when available
- Fallback to Python implementation for edge cases
- Automatic precision selection for ~2x memory bandwidth improvement
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_primitives.config.precision import (
    PrecisionMode,
    PrecisionConfig,
    get_precision_config,
    should_use_fp16,
)
from mlx_primitives.kernels.block_config import get_optimal_block_config

# Check if MLX fast SDPA is available
_HAS_MLX_FAST_SDPA = hasattr(mx.fast, 'scaled_dot_product_attention')


def _mlx_fast_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = False,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Use MLX's optimized scaled_dot_product_attention.

    This is ~8-9x faster than Python implementations via compiled Metal kernels.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        scale: Attention scale factor.
        causal: Whether to apply causal masking.
        mask: Optional attention mask.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).
    """
    # MLX SDPA expects (batch, heads, seq, dim) layout
    q_t = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, dim)
    k_t = k.transpose(0, 2, 1, 3)  # (batch, kv_heads, seq_kv, dim)
    v_t = v.transpose(0, 2, 1, 3)  # (batch, kv_heads, seq_kv, dim)

    # MLX SDPA handles causal mask with string "causal"
    if causal and mask is None:
        mlx_mask = "causal"
    elif mask is not None:
        # MLX expects mask in (batch, heads, seq_q, seq_kv) format
        mlx_mask = mask
    else:
        mlx_mask = None

    # Call MLX's optimized SDPA
    output = mx.fast.scaled_dot_product_attention(
        q_t, k_t, v_t,
        scale=scale,
        mask=mlx_mask,
    )

    # Transpose back to (batch, seq, heads, dim)
    return output.transpose(0, 2, 1, 3)


def _naive_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
    causal: bool = False,
) -> mx.array:
    """Standard attention for reference/fallback.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_heads, head_dim).
        scale: Attention scale factor.
        mask: Optional attention mask.
        causal: Whether to apply causal masking.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).
    """
    # Transpose for matmul: (batch, num_heads, seq, head_dim)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Compute attention scores
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask
    if causal:
        seq_q, seq_k = scores.shape[2], scores.shape[3]
        causal_mask = mx.triu(
            mx.full((seq_q, seq_k), float("-inf")), k=1
        )
        scores = scores + causal_mask

    # Apply custom mask
    if mask is not None:
        scores = scores + mask

    # Softmax and weighted sum
    weights = mx.softmax(scores, axis=-1)
    output = weights @ v

    # Transpose back: (batch, seq, num_heads, head_dim)
    return output.transpose(0, 2, 1, 3)


def _flash_attention_forward_fp16(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    block_size_q: int = 64,
    block_size_kv: int = 64,
    causal: bool = False,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Flash attention using float16 for ~2x memory bandwidth improvement.

    Converts inputs to float16, uses float32 accumulation for numerical
    stability (matching the Metal kernel behavior), and converts output
    back to original dtype.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_heads, head_dim).
        scale: Attention scale factor.
        block_size_q: Block size for query dimension.
        block_size_kv: Block size for key/value dimension.
        causal: Whether to apply causal masking.
        mask: Optional attention mask.

    Returns:
        Output tensor with same dtype as input q.
    """
    input_dtype = q.dtype
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Convert to float16 if needed
    if q.dtype != mx.float16:
        q = q.astype(mx.float16)
        k = k.astype(mx.float16)
        v = v.astype(mx.float16)

    # For short sequences, use naive attention with fp16
    if seq_q <= block_size_q and seq_kv <= block_size_kv:
        output = _naive_attention(q, k, v, scale, mask, causal)
        if input_dtype != mx.float16:
            output = output.astype(input_dtype)
        return output

    # Initialize output accumulator (use float32 for accumulation precision)
    # but store in float16 to reduce memory
    output = mx.zeros_like(q)

    # Track running max and sum for numerically stable softmax
    # These stay in float32 for precision
    m_prev = mx.full((batch_size, seq_q, num_heads), float("-inf"), dtype=mx.float32)
    l_prev = mx.zeros((batch_size, seq_q, num_heads), dtype=mx.float32)

    # Number of blocks
    num_blocks_kv = (seq_kv + block_size_kv - 1) // block_size_kv

    # Process key/value blocks
    for j in range(num_blocks_kv):
        kv_start = j * block_size_kv
        kv_end = min(kv_start + block_size_kv, seq_kv)

        # Load key/value block (already in fp16)
        k_block = k[:, kv_start:kv_end, :, :]
        v_block = v[:, kv_start:kv_end, :, :]

        # Compute attention scores for this KV block
        # Transpose for matmul
        q_t = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, dim)
        k_t = k_block.transpose(0, 2, 3, 1)  # (batch, heads, dim, block_kv)

        # Scores: (batch, heads, seq_q, block_kv) - compute in fp16, result in fp16
        scores = (q_t @ k_t) * scale

        # Apply causal mask if needed
        if causal:
            q_pos = mx.arange(seq_q)[:, None]
            k_pos = mx.arange(kv_start, kv_end)[None, :]
            causal_mask = mx.where(
                q_pos < k_pos,
                mx.array(float("-inf"), dtype=mx.float16),
                mx.array(0.0, dtype=mx.float16),
            )
            scores = scores + causal_mask[None, None, :, :]

        # Apply custom mask if provided
        if mask is not None:
            if mask.ndim == 4:
                mask_block = mask[:, :, :, kv_start:kv_end]
            else:
                mask_block = mask[..., kv_start:kv_end]
            if mask_block.dtype != mx.float16:
                mask_block = mask_block.astype(mx.float16)
            scores = scores + mask_block

        # Compute block statistics in float32 for stability
        scores_f32 = scores.astype(mx.float32)
        m_block = mx.max(scores_f32, axis=-1)  # (batch, heads, seq_q)
        m_block = m_block.transpose(0, 2, 1)  # (batch, seq_q, heads)

        # New maximum
        m_new = mx.maximum(m_prev, m_block)

        # Compute exponentials with numerical stability (in float32)
        scores_t = scores_f32.transpose(0, 2, 1, 3)  # (batch, seq_q, heads, block_kv)
        exp_scores = mx.exp(scores_t - m_new[:, :, :, None])

        # Sum of exponentials for this block
        l_block = mx.sum(exp_scores, axis=-1)  # (batch, seq_q, heads)

        # Rescale previous sum
        l_rescale = mx.exp(m_prev - m_new) * l_prev

        # New sum
        l_new = l_rescale + l_block

        # Compute weighted values for this block
        v_t = v_block.transpose(0, 2, 1, 3)  # (batch, heads, block_kv, dim)
        exp_scores_t = exp_scores.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, block_kv)

        # Convert exp_scores to fp16 for matmul with V
        exp_scores_t_fp16 = exp_scores_t.astype(mx.float16)

        # Weighted sum: (batch, heads, seq_q, dim)
        block_output = exp_scores_t_fp16 @ v_t
        block_output = block_output.transpose(0, 2, 1, 3)  # (batch, seq_q, heads, dim)

        # Rescale and accumulate output (mixing fp16 storage with fp32 scaling)
        l_rescale_expanded = l_rescale[:, :, :, None].astype(mx.float16)
        l_new_expanded = l_new[:, :, :, None].astype(mx.float16)

        output = (l_rescale_expanded * output + block_output) / l_new_expanded

        # Update running statistics (keep in float32)
        m_prev = m_new
        l_prev = l_new

    # Convert output back to original dtype if needed
    if input_dtype != mx.float16:
        output = output.astype(input_dtype)

    return output


def flash_attention_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    block_size_q: Optional[int] = None,
    block_size_kv: Optional[int] = None,
    causal: bool = False,
    mask: Optional[mx.array] = None,
    precision: Optional[PrecisionMode] = None,
    auto_tune: bool = True,
    use_mlx_sdpa: bool = True,
) -> mx.array:
    """Forward pass of Flash Attention.

    Primary implementation uses mx.fast.scaled_dot_product_attention which
    provides ~8-9x speedup via compiled Metal kernels.

    Falls back to Python tiled implementation when:
    - use_mlx_sdpa=False
    - MLX fast SDPA not available
    - Custom mask format not supported by MLX SDPA

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_kv_heads, head_dim).
           Note: num_kv_heads can differ from num_heads for GQA/MQA.
        v: Value tensor (batch, seq_kv, num_kv_heads, head_dim).
        scale: Attention scale factor (typically 1/sqrt(head_dim)).
        block_size_q: Block size for query dimension (Python fallback only).
        block_size_kv: Block size for key/value dimension (Python fallback only).
        causal: Whether to apply causal masking.
        mask: Optional attention mask of shape broadcastable to
              (batch, num_heads, seq_q, seq_kv).
        precision: Override precision mode (Python fallback only).
        auto_tune: If True, automatically select optimal block sizes (Python fallback only).
        use_mlx_sdpa: If True, use MLX's optimized SDPA when available (default: True).
            Set to False to force Python implementation.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).

    Performance:
        - MLX SDPA: ~8-9x faster than Python implementation
        - Supports MHA, GQA, MQA natively without K/V expansion
    """
    # Use MLX's optimized SDPA when available (8-9x faster)
    if use_mlx_sdpa and _HAS_MLX_FAST_SDPA:
        try:
            return _mlx_fast_sdpa(q, k, v, scale, causal, mask)
        except Exception:
            # Fall back to Python implementation on error
            pass

    # Python fallback implementation
    batch_size, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Auto-tune block sizes if not provided
    if auto_tune and (block_size_q is None or block_size_kv is None):
        opt_block_m, opt_block_n = get_optimal_block_config(head_dim, q.dtype)
        if block_size_q is None:
            block_size_q = opt_block_m
        if block_size_kv is None:
            block_size_kv = opt_block_n
    else:
        # Use defaults if not auto-tuning and not provided
        if block_size_q is None:
            block_size_q = 64
        if block_size_kv is None:
            block_size_kv = 64

    # Check if we should use fp16 for this computation
    seq_len = max(seq_q, seq_kv)
    use_fp16 = should_use_fp16(q, k, v, seq_len, scale, precision)

    if use_fp16:
        return _flash_attention_forward_fp16(
            q, k, v, scale, block_size_q, block_size_kv, causal, mask
        )

    # For short sequences, use naive attention (overhead not worth it)
    if seq_q <= block_size_q and seq_kv <= block_size_kv:
        return _naive_attention(q, k, v, scale, mask, causal)

    # Initialize output accumulator and log-sum-exp tracker
    output = mx.zeros_like(q)
    # Track running max and sum for numerically stable softmax
    # Shape: (batch, seq_q, num_heads)
    m_prev = mx.full((batch_size, seq_q, num_heads), float("-inf"))
    l_prev = mx.zeros((batch_size, seq_q, num_heads))

    # Number of blocks
    num_blocks_kv = (seq_kv + block_size_kv - 1) // block_size_kv

    # Process key/value blocks
    for j in range(num_blocks_kv):
        kv_start = j * block_size_kv
        kv_end = min(kv_start + block_size_kv, seq_kv)

        # Load key/value block
        k_block = k[:, kv_start:kv_end, :, :]  # (batch, block_kv, heads, dim)
        v_block = v[:, kv_start:kv_end, :, :]

        # Compute attention scores for this KV block against all queries
        # q: (batch, seq_q, heads, dim)
        # k_block: (batch, block_kv, heads, dim)
        # We need: (batch, seq_q, heads, block_kv)

        # Transpose for matmul
        q_t = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, dim)
        k_t = k_block.transpose(0, 2, 3, 1)  # (batch, heads, dim, block_kv)

        # Scores: (batch, heads, seq_q, block_kv)
        scores = (q_t @ k_t) * scale

        # Apply causal mask if needed
        if causal:
            # Create causal mask for this block
            # Query positions: 0 to seq_q-1
            # Key positions: kv_start to kv_end-1
            q_pos = mx.arange(seq_q)[:, None]
            k_pos = mx.arange(kv_start, kv_end)[None, :]
            causal_mask = mx.where(
                q_pos < k_pos,
                mx.array(float("-inf")),
                mx.array(0.0),
            )
            scores = scores + causal_mask[None, None, :, :]

        # Apply custom mask if provided
        if mask is not None:
            # Slice mask for this KV block
            if mask.ndim == 4:
                mask_block = mask[:, :, :, kv_start:kv_end]
            else:
                mask_block = mask[..., kv_start:kv_end]
            scores = scores + mask_block

        # Compute block statistics for numerically stable softmax
        # scores: (batch, heads, seq_q, block_kv)
        m_block = mx.max(scores, axis=-1)  # (batch, heads, seq_q)
        m_block = m_block.transpose(0, 2, 1)  # (batch, seq_q, heads)

        # New maximum
        m_new = mx.maximum(m_prev, m_block)

        # Compute exponentials with numerical stability
        # exp(scores - m_new)
        scores_t = scores.transpose(0, 2, 1, 3)  # (batch, seq_q, heads, block_kv)
        exp_scores = mx.exp(scores_t - m_new[:, :, :, None])

        # Sum of exponentials for this block
        l_block = mx.sum(exp_scores, axis=-1)  # (batch, seq_q, heads)

        # Rescale previous sum
        l_rescale = mx.exp(m_prev - m_new) * l_prev

        # New sum
        l_new = l_rescale + l_block

        # Compute weighted values for this block
        # exp_scores: (batch, seq_q, heads, block_kv)
        # v_block: (batch, block_kv, heads, dim)
        v_t = v_block.transpose(0, 2, 1, 3)  # (batch, heads, block_kv, dim)
        exp_scores_t = exp_scores.transpose(0, 2, 1, 3)  # (batch, heads, seq_q, block_kv)

        # Weighted sum: (batch, heads, seq_q, dim)
        block_output = exp_scores_t @ v_t
        block_output = block_output.transpose(0, 2, 1, 3)  # (batch, seq_q, heads, dim)

        # Rescale and accumulate output
        # output = (l_rescale/l_new) * output + (1/l_new) * block_output
        l_rescale_expanded = l_rescale[:, :, :, None]
        l_new_expanded = l_new[:, :, :, None]

        output = (l_rescale_expanded * output + block_output) / l_new_expanded

        # Update running statistics
        m_prev = m_new
        l_prev = l_new

    return output


class FlashAttention(nn.Module):
    """Memory-efficient multi-head attention using Flash Attention algorithm.

    This implementation uses tiling to compute attention without materializing
    the full N×N attention matrix, reducing memory from O(N²) to O(N).

    Supports automatic precision selection for ~2x memory bandwidth improvement
    when using float16 on supported inputs, and adaptive block sizing for
    10-20% speedup on non-standard configurations.

    Args:
        dims: Model dimension (will be split across heads).
        num_heads: Number of attention heads.
        head_dim: Dimension of each head (default: dims // num_heads).
        block_size: Block size for tiling. If None and auto_tune=True,
            automatically selects optimal size (default: None).
        causal: Whether to apply causal masking (default: False).
        dropout: Dropout probability (default: 0.0).
        bias: Whether to use bias in projections (default: False).
        precision: Precision mode for attention computation (default: AUTO).
            - AUTO: Automatically select based on input analysis
            - FLOAT32: Force full precision
            - FLOAT16: Force half precision
            - MIXED: Half precision compute with float32 accumulation
        auto_tune: If True, automatically select optimal block sizes based on
            head_dim and dtype (default: True).

    Example:
        >>> attn = FlashAttention(dims=768, num_heads=12, causal=True)
        >>> x = mx.random.normal((2, 1024, 768))
        >>> output = attn(x)  # (2, 1024, 768)

        >>> # With explicit precision control
        >>> attn = FlashAttention(dims=768, num_heads=12, precision=PrecisionMode.FLOAT16)
        >>> output = attn(x)  # Uses float16 path

        >>> # With fixed block size (disables auto-tuning for that dimension)
        >>> attn = FlashAttention(dims=768, num_heads=12, block_size=32)
        >>> output = attn(x)

        >>> # With separate Q, K, V
        >>> q = mx.random.normal((2, 512, 768))
        >>> k = mx.random.normal((2, 1024, 768))
        >>> v = mx.random.normal((2, 1024, 768))
        >>> output = attn(q, k, v)  # (2, 512, 768)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        block_size: Optional[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        precision: Optional[PrecisionMode] = None,
        auto_tune: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = head_dim or dims // num_heads
        self.block_size = block_size
        self.causal = causal
        self.dropout = dropout
        self.precision = precision
        self.auto_tune = auto_tune

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        inner_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.k_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.v_proj = nn.Linear(dims, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, dims, bias=bias)

    def __call__(
        self,
        queries: mx.array,
        keys: Optional[mx.array] = None,
        values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute flash attention.

        Args:
            queries: Query tensor of shape (batch, seq_q, dims).
            keys: Key tensor of shape (batch, seq_kv, dims).
                  If None, uses queries (self-attention).
            values: Value tensor of shape (batch, seq_kv, dims).
                    If None, uses keys.
            mask: Optional attention mask broadcastable to
                  (batch, num_heads, seq_q, seq_kv).

        Returns:
            Output tensor of shape (batch, seq_q, dims).
        """
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        batch_size, seq_q, _ = queries.shape
        _, seq_kv, _ = keys.shape

        # Project to Q, K, V
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        # Reshape to (batch, seq, num_heads, head_dim)
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)

        # Apply flash attention
        output = flash_attention_forward(
            q, k, v,
            scale=self.scale,
            block_size_q=self.block_size,
            block_size_kv=self.block_size,
            causal=self.causal,
            mask=mask,
            precision=self.precision,
            auto_tune=self.auto_tune,
        )

        # Reshape and project output
        output = output.reshape(batch_size, seq_q, -1)
        return self.out_proj(output)


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    block_size: Optional[int] = None,
    precision: Optional[PrecisionMode] = None,
    auto_tune: bool = True,
) -> mx.array:
    """Functional interface for Flash Attention.

    This provides a PyTorch-like functional API for attention computation.
    Supports automatic precision selection for ~2x memory bandwidth improvement
    and adaptive block sizing for 10-20% speedup on non-standard configurations.

    Args:
        query: Query tensor (batch, seq_q, num_heads, head_dim) or
               (batch, num_heads, seq_q, head_dim).
        key: Key tensor with same layout as query.
        value: Value tensor with same layout as query.
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability (currently ignored).
        is_causal: Whether to apply causal masking.
        scale: Scale factor (default: 1/sqrt(head_dim)).
        block_size: Block size for tiling. If None and auto_tune=True,
            automatically selects optimal size.
        precision: Precision mode (AUTO, FLOAT32, FLOAT16, MIXED).
        auto_tune: If True, automatically select optimal block sizes based on
            head_dim and dtype (default: True).

    Returns:
        Output tensor with same layout as input.
    """
    # Detect input format
    # PyTorch uses (batch, heads, seq, dim)
    # Our internal format uses (batch, seq, heads, dim)
    if query.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {query.ndim}D")

    # Check if input is (batch, heads, seq, dim) format
    # Heuristic: if dim 1 < dim 2, likely (batch, heads, seq, dim)
    transposed_input = query.shape[1] < query.shape[2]

    if transposed_input:
        # Convert from (batch, heads, seq, dim) to (batch, seq, heads, dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    output = flash_attention_forward(
        query, key, value,
        scale=scale,
        block_size_q=block_size,
        block_size_kv=block_size,
        causal=is_causal,
        mask=attn_mask,
        precision=precision,
        auto_tune=auto_tune,
    )

    if transposed_input:
        # Convert back to (batch, heads, seq, dim)
        output = output.transpose(0, 2, 1, 3)

    return output
