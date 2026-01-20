"""Fast RoPE Metal kernel implementation.

Provides fused rotary position embedding that avoids intermediate allocations
from rotate_half and concatenation operations.
"""

from typing import Tuple, Optional
import mlx.core as mx

# Cache compiled kernels
_rope_kernel = None
_rope_qk_kernel = None


def _get_rope_kernel():
    """Get or create the RoPE kernel for single tensor."""
    global _rope_kernel

    if _rope_kernel is None:
        # Each thread handles one (batch, seq, head, dim_pair) element
        # x_shape: (batch, seq_len, num_heads, head_dim)
        source = """
            uint idx = thread_position_in_grid.x;

            // Get dimensions from x_shape (auto-provided by MLX)
            uint batch_size = x_shape[0];
            uint seq_len = x_shape[1];
            uint num_heads = x_shape[2];
            uint head_dim = x_shape[3];
            uint half_dim = head_dim / 2;

            // Total dimension pairs to process
            uint total_pairs = batch_size * seq_len * num_heads * half_dim;
            if (idx >= total_pairs) return;

            // Decompose linear index to (batch, seq, head, dim_pair)
            uint dim_pair = idx % half_dim;
            uint rem = idx / half_dim;
            uint head_idx = rem % num_heads;
            rem = rem / num_heads;
            uint seq_idx = rem % seq_len;
            uint batch_idx = rem / seq_len;

            // Compute index into x tensor
            // Layout: batch * (seq * heads * dim) + seq * (heads * dim) + head * dim + d
            uint base_idx = batch_idx * (seq_len * num_heads * head_dim) +
                           seq_idx * (num_heads * head_dim) +
                           head_idx * head_dim;

            // Get the two elements that form a rotation pair
            // Standard RoPE pairs: (0, half_dim), (1, half_dim+1), etc.
            uint d1 = dim_pair;
            uint d2 = dim_pair + half_dim;

            float x1 = x[base_idx + d1];
            float x2 = x[base_idx + d2];

            // Get cos/sin from precomputed cache
            // Cache layout: (seq_len, half_dim)
            uint cache_idx = seq_idx * half_dim + dim_pair;
            float cos_val = cos_cache[cache_idx];
            float sin_val = sin_cache[cache_idx];

            // Apply rotation: out1 = x1*cos - x2*sin, out2 = x1*sin + x2*cos
            out[base_idx + d1] = x1 * cos_val - x2 * sin_val;
            out[base_idx + d2] = x1 * sin_val + x2 * cos_val;
        """

        _rope_kernel = mx.fast.metal_kernel(
            name="rope_forward",
            input_names=["x", "cos_cache", "sin_cache"],
            output_names=["out"],
            source=source,
        )

    return _rope_kernel


def _get_rope_qk_kernel():
    """Get or create the fused RoPE kernel for Q and K together."""
    global _rope_qk_kernel

    if _rope_qk_kernel is None:
        # Process Q and K simultaneously - more efficient memory access
        source = """
            uint idx = thread_position_in_grid.x;

            // Get dimensions from q_shape
            uint batch_size = q_shape[0];
            uint seq_len = q_shape[1];
            uint num_heads = q_shape[2];
            uint head_dim = q_shape[3];
            uint half_dim = head_dim / 2;

            uint total_pairs = batch_size * seq_len * num_heads * half_dim;
            if (idx >= total_pairs) return;

            // Decompose linear index
            uint dim_pair = idx % half_dim;
            uint rem = idx / half_dim;
            uint head_idx = rem % num_heads;
            rem = rem / num_heads;
            uint seq_idx = rem % seq_len;
            uint batch_idx = rem / seq_len;

            uint base_idx = batch_idx * (seq_len * num_heads * head_dim) +
                           seq_idx * (num_heads * head_dim) +
                           head_idx * head_dim;

            uint d1 = dim_pair;
            uint d2 = dim_pair + half_dim;

            // Load Q values
            float q1 = q[base_idx + d1];
            float q2 = q[base_idx + d2];

            // Load K values (may have different num_heads for GQA)
            // For simplicity, assume same layout - user should handle GQA separately
            float k1 = k[base_idx + d1];
            float k2 = k[base_idx + d2];

            // Get cos/sin from cache
            uint cache_idx = seq_idx * half_dim + dim_pair;
            float cos_val = cos_cache[cache_idx];
            float sin_val = sin_cache[cache_idx];

            // Apply rotation to Q
            q_out[base_idx + d1] = q1 * cos_val - q2 * sin_val;
            q_out[base_idx + d2] = q1 * sin_val + q2 * cos_val;

            // Apply rotation to K
            k_out[base_idx + d1] = k1 * cos_val - k2 * sin_val;
            k_out[base_idx + d2] = k1 * sin_val + k2 * cos_val;
        """

        _rope_qk_kernel = mx.fast.metal_kernel(
            name="rope_forward_qk",
            input_names=["q", "k", "cos_cache", "sin_cache"],
            output_names=["q_out", "k_out"],
            source=source,
        )

    return _rope_qk_kernel


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Precompute cos/sin cache for RoPE.

    Args:
        seq_len: Sequence length to precompute.
        head_dim: Head dimension (must be even).
        base: Base for frequency computation.
        dtype: Output dtype.

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape (seq_len, head_dim // 2).
    """
    half_dim = head_dim // 2

    # Compute inverse frequencies: theta_i = base^(-2i/dim)
    inv_freq = 1.0 / (base ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim))

    # Position indices
    t = mx.arange(seq_len, dtype=mx.float32)

    # Outer product: (seq_len, half_dim)
    freqs = mx.outer(t, inv_freq)

    cos_cache = mx.cos(freqs).astype(dtype)
    sin_cache = mx.sin(freqs).astype(dtype)

    return cos_cache, sin_cache


def fast_rope(
    x: mx.array,
    cos_cache: mx.array,
    sin_cache: mx.array,
    offset: int = 0,
) -> mx.array:
    """Fast Metal-accelerated RoPE for single tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos_cache: Precomputed cosines of shape (max_seq, head_dim // 2).
        sin_cache: Precomputed sines of shape (max_seq, head_dim // 2).
        offset: Position offset for KV cache decoding.

    Returns:
        Rotated tensor of same shape as input.
    """
    assert x.ndim == 4, "Input must be (batch, seq_len, num_heads, head_dim)"
    batch_size, seq_len, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even"

    half_dim = head_dim // 2

    # Slice cache to match sequence (with offset)
    cos_slice = cos_cache[offset:offset + seq_len]
    sin_slice = sin_cache[offset:offset + seq_len]

    kernel = _get_rope_kernel()

    # Total dimension pairs to process
    total_pairs = batch_size * seq_len * num_heads * half_dim

    threadgroup_size = 256
    num_groups = (total_pairs + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[x, cos_slice, sin_slice],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0]


def fast_rope_qk(
    q: mx.array,
    k: mx.array,
    cos_cache: mx.array,
    sin_cache: mx.array,
    offset: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Fast Metal-accelerated RoPE for Q and K together.

    More efficient than calling fast_rope twice because it reads
    the cos/sin cache only once.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim).
           Note: For GQA where num_kv_heads != num_heads, Q and K must
           have the same shape here (expand K before calling).
        cos_cache: Precomputed cosines of shape (max_seq, head_dim // 2).
        sin_cache: Precomputed sines of shape (max_seq, head_dim // 2).
        offset: Position offset for KV cache decoding.

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    assert q.ndim == 4 and k.ndim == 4, "Inputs must be 4D"
    assert q.shape == k.shape, "Q and K must have same shape for fused kernel"

    batch_size, seq_len, num_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim must be even"

    half_dim = head_dim // 2

    # Slice cache
    cos_slice = cos_cache[offset:offset + seq_len]
    sin_slice = sin_cache[offset:offset + seq_len]

    kernel = _get_rope_qk_kernel()

    total_pairs = batch_size * seq_len * num_heads * half_dim

    threadgroup_size = 256
    num_groups = (total_pairs + threadgroup_size - 1) // threadgroup_size
    grid = (num_groups * threadgroup_size, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[q, k, cos_slice, sin_slice],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
        stream=mx.default_stream(mx.default_device()),
    )

    return outputs[0], outputs[1]


# Check if metal kernels are available
_USE_METAL_KERNELS = hasattr(mx.fast, 'metal_kernel')


def rope(
    x: mx.array,
    cos_cache: mx.array,
    sin_cache: mx.array,
    offset: int = 0,
    use_metal: bool = True,
) -> mx.array:
    """RoPE with automatic Metal acceleration.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos_cache: Precomputed cosines.
        sin_cache: Precomputed sines.
        offset: Position offset.
        use_metal: Whether to use Metal kernels.

    Returns:
        Rotated tensor.
    """
    if use_metal and _USE_METAL_KERNELS:
        try:
            return fast_rope(x, cos_cache, sin_cache, offset)
        except Exception:
            pass

    # Fallback: Python implementation
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    cos = cos_cache[offset:offset + seq_len]
    sin = sin_cache[offset:offset + seq_len]

    # Reshape for broadcasting: (seq_len, half_dim) -> (1, seq_len, 1, half_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Split into first and second halves
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return mx.concatenate([out1, out2], axis=-1)


def rope_qk(
    q: mx.array,
    k: mx.array,
    cos_cache: mx.array,
    sin_cache: mx.array,
    offset: int = 0,
    use_metal: bool = True,
) -> Tuple[mx.array, mx.array]:
    """RoPE for Q and K with automatic Metal acceleration.

    Args:
        q: Query tensor.
        k: Key tensor (same shape as q for fused path).
        cos_cache: Precomputed cosines.
        sin_cache: Precomputed sines.
        offset: Position offset.
        use_metal: Whether to use Metal kernels.

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    # Try fused path if shapes match
    if use_metal and _USE_METAL_KERNELS and q.shape == k.shape:
        try:
            return fast_rope_qk(q, k, cos_cache, sin_cache, offset)
        except Exception:
            pass

    # Fallback: apply separately
    q_rot = rope(q, cos_cache, sin_cache, offset, use_metal)
    k_rot = rope(k, cos_cache, sin_cache, offset, use_metal)

    return q_rot, k_rot
