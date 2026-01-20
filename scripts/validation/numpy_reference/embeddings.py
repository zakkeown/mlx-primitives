"""NumPy reference implementations for positional embeddings."""

import numpy as np


def sinusoidal_embedding(
    positions: np.ndarray,
    dim: int,
    max_period: float = 10000.0,
) -> np.ndarray:
    """Sinusoidal positional embedding.

    PE(pos, 2i) = sin(pos / max_period^(2i/dim))
    PE(pos, 2i+1) = cos(pos / max_period^(2i/dim))

    Args:
        positions: Position indices, shape (seq_len,) or (batch, seq_len)
        dim: Embedding dimension
        max_period: Maximum period for sinusoids

    Returns:
        Positional embeddings, shape (*positions.shape, dim)
    """
    positions = np.asarray(positions)

    half_dim = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half_dim) / half_dim)

    # Outer product of positions and frequencies
    if positions.ndim == 1:
        args = positions[:, None] * freqs[None, :]
    else:
        args = positions[..., None] * freqs

    # Interleave sin and cos
    emb = np.zeros((*positions.shape, dim), dtype=np.float32)
    emb[..., 0::2] = np.sin(args)
    emb[..., 1::2] = np.cos(args)

    return emb


def rotary_embedding(
    x: np.ndarray,
    positions: np.ndarray = None,
    dim: int = None,
    base: float = 10000.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply Rotary Position Embedding (RoPE).

    Rotates pairs of dimensions based on position.

    Args:
        x: Input tensor, shape (..., seq_len, head_dim)
        positions: Position indices, shape (seq_len,). Default: 0, 1, 2, ...
        dim: Number of dimensions to apply RoPE to (default: head_dim)
        base: Base for frequency computation
        scale: Scale factor for positions

    Returns:
        Rotated tensor, same shape as x
    """
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]

    if dim is None:
        dim = head_dim

    if positions is None:
        positions = np.arange(seq_len)

    positions = positions * scale

    # Compute frequencies
    half_dim = dim // 2
    freqs = 1.0 / (base ** (np.arange(half_dim) / half_dim))

    # Compute angles: (seq_len, half_dim)
    angles = positions[:, None] * freqs[None, :]

    # Compute sin and cos
    sin = np.sin(angles)
    cos = np.cos(angles)

    # Split x into pairs
    x1 = x[..., :dim:2]
    x2 = x[..., 1:dim:2]

    # Apply rotation
    out = x.copy()
    out[..., :dim:2] = x1 * cos - x2 * sin
    out[..., 1:dim:2] = x1 * sin + x2 * cos

    return out


def alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute ALiBi slopes for attention.

    Slopes are powers of 2: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    where n is the number of heads.

    Args:
        num_heads: Number of attention heads

    Returns:
        Slopes array, shape (num_heads,)
    """
    # Get closest power of 2 >= num_heads
    closest_power_of_2 = 2 ** np.ceil(np.log2(num_heads))
    base = 2 ** (-(2 ** -(np.log2(closest_power_of_2) - 3)))

    powers = np.arange(1, num_heads + 1)
    slopes = base ** powers

    return slopes.astype(np.float32)


def alibi_bias(
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    slopes: np.ndarray = None,
) -> np.ndarray:
    """Compute ALiBi attention bias.

    bias[h, i, j] = -slope[h] * |i - j|

    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        num_heads: Number of attention heads
        slopes: Pre-computed slopes. If None, will be computed.

    Returns:
        Bias tensor, shape (num_heads, seq_len_q, seq_len_k)
    """
    if slopes is None:
        slopes = alibi_slopes(num_heads)

    # Compute position differences: (seq_len_q, seq_len_k)
    positions_q = np.arange(seq_len_q)[:, None]
    positions_k = np.arange(seq_len_k)[None, :]

    # Relative positions (for causal: j - i, clamped to <= 0)
    rel_pos = positions_k - positions_q

    # Compute bias: (num_heads, seq_len_q, seq_len_k)
    # bias = -slope * |relative_position|
    # For causal attention, typically use -slope * max(0, j - i) or similar
    bias = -slopes[:, None, None] * np.abs(rel_pos)[None, :, :]

    return bias.astype(np.float32)
