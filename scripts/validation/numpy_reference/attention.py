"""NumPy reference implementations for attention mechanisms."""

import numpy as np
from scipy import special


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float = None,
    causal: bool = False,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Scaled Dot-Product Attention.

    SDPA(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

    Args:
        q: Query tensor, shape (batch, seq_q, heads, head_dim)
        k: Key tensor, shape (batch, seq_kv, heads, head_dim)
        v: Value tensor, shape (batch, seq_kv, heads, head_dim)
        scale: Attention scale (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        mask: Optional attention mask

    Returns:
        Output tensor, shape (batch, seq_q, heads, head_dim)
    """
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    # Transpose to (batch, heads, seq, head_dim) for matmul
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Compute attention scores: (batch, heads, seq_q, seq_kv)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale

    # Apply causal mask
    if causal:
        seq_q, seq_kv = scores.shape[-2], scores.shape[-1]
        causal_mask = np.triu(np.full((seq_q, seq_kv), -np.inf), k=1)
        scores = scores + causal_mask

    # Apply optional mask
    if mask is not None:
        scores = scores + mask

    # Softmax and matmul with values
    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, v)

    # Transpose back to (batch, seq, heads, head_dim)
    return np.transpose(out, (0, 2, 1, 3))


def linear_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    feature_map: str = "elu",
    eps: float = 1e-6,
) -> np.ndarray:
    """Linear Attention with kernel feature maps.

    LinearAttention(Q, K, V) = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))

    Args:
        q: Query tensor, shape (batch, seq_q, heads, head_dim)
        k: Key tensor, shape (batch, seq_kv, heads, head_dim)
        v: Value tensor, shape (batch, seq_kv, heads, head_dim)
        feature_map: Feature map to use ("elu", "relu", "identity")
        eps: Small constant for numerical stability

    Returns:
        Output tensor, shape (batch, seq_q, heads, head_dim)
    """
    # Apply feature map
    if feature_map == "elu":
        q_prime = np.where(q > 0, q + 1, np.exp(q))
        k_prime = np.where(k > 0, k + 1, np.exp(k))
    elif feature_map == "relu":
        q_prime = np.maximum(0, q)
        k_prime = np.maximum(0, k)
    else:  # identity
        q_prime = q
        k_prime = k

    # Transpose to (batch, heads, seq, head_dim)
    q_prime = np.transpose(q_prime, (0, 2, 1, 3))
    k_prime = np.transpose(k_prime, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Compute KV = K^T @ V: (batch, heads, head_dim, head_dim)
    kv = np.matmul(np.swapaxes(k_prime, -2, -1), v)

    # Compute Q @ KV: (batch, heads, seq_q, head_dim)
    qkv = np.matmul(q_prime, kv)

    # Compute normalization: Q @ sum(K, axis=seq)
    k_sum = np.sum(k_prime, axis=-2, keepdims=True)  # (batch, heads, 1, head_dim)
    normalizer = np.matmul(q_prime, np.swapaxes(k_sum, -2, -1)) + eps  # (batch, heads, seq_q, 1)

    out = qkv / normalizer

    # Transpose back
    return np.transpose(out, (0, 2, 1, 3))


def performer_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_random_features: int = None,
    eps: float = 1e-6,
    seed: int = 42,
) -> np.ndarray:
    """Performer attention with FAVOR+ random features.

    Uses random orthogonal features for efficient attention approximation.

    Args:
        q: Query tensor, shape (batch, seq_q, heads, head_dim)
        k: Key tensor, shape (batch, seq_kv, heads, head_dim)
        v: Value tensor, shape (batch, seq_kv, heads, head_dim)
        num_random_features: Number of random features (default: head_dim)
        eps: Small constant for numerical stability
        seed: Random seed for reproducibility

    Returns:
        Output tensor, shape (batch, seq_q, heads, head_dim)
    """
    head_dim = q.shape[-1]
    if num_random_features is None:
        num_random_features = head_dim

    np.random.seed(seed)

    # Generate random projection matrix
    random_matrix = np.random.randn(head_dim, num_random_features) / np.sqrt(num_random_features)

    def softmax_kernel(x):
        """Positive random feature map for softmax kernel."""
        proj = np.matmul(x, random_matrix)
        return np.exp(proj - np.max(proj, axis=-1, keepdims=True)) / np.sqrt(num_random_features)

    # Apply kernel to Q and K
    q_prime = softmax_kernel(q)
    k_prime = softmax_kernel(k)

    # Use linear attention with transformed features
    # Transpose to (batch, heads, seq, features)
    q_prime = np.transpose(q_prime, (0, 2, 1, 3))
    k_prime = np.transpose(k_prime, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # KV: (batch, heads, features, head_dim)
    kv = np.matmul(np.swapaxes(k_prime, -2, -1), v)

    # Q @ KV
    qkv = np.matmul(q_prime, kv)

    # Normalization
    k_sum = np.sum(k_prime, axis=-2, keepdims=True)
    normalizer = np.matmul(q_prime, np.swapaxes(k_sum, -2, -1)) + eps

    out = qkv / normalizer

    return np.transpose(out, (0, 2, 1, 3))


def cosformer_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """CosFormer attention with cosine reweighting.

    Uses cos(pi * i / (2 * M)) positional reweighting for linear attention.

    Args:
        q: Query tensor, shape (batch, seq_q, heads, head_dim)
        k: Key tensor, shape (batch, seq_kv, heads, head_dim)
        v: Value tensor, shape (batch, seq_kv, heads, head_dim)
        eps: Small constant for numerical stability

    Returns:
        Output tensor, shape (batch, seq_q, heads, head_dim)
    """
    batch, seq_q, heads, head_dim = q.shape
    seq_kv = k.shape[1]

    # Compute position weights
    pos_q = np.arange(seq_q).reshape(1, -1, 1, 1)
    pos_k = np.arange(seq_kv).reshape(1, -1, 1, 1)

    cos_q = np.cos(np.pi * pos_q / (2 * seq_q))
    sin_q = np.sin(np.pi * pos_q / (2 * seq_q))
    cos_k = np.cos(np.pi * pos_k / (2 * seq_kv))
    sin_k = np.sin(np.pi * pos_k / (2 * seq_kv))

    # Apply ReLU feature map
    q_relu = np.maximum(0, q)
    k_relu = np.maximum(0, k)

    # Reweight with cos/sin
    q_cos = q_relu * cos_q
    q_sin = q_relu * sin_q
    k_cos = k_relu * cos_k
    k_sin = k_relu * sin_k

    # Transpose for matmul
    q_cos = np.transpose(q_cos, (0, 2, 1, 3))
    q_sin = np.transpose(q_sin, (0, 2, 1, 3))
    k_cos = np.transpose(k_cos, (0, 2, 1, 3))
    k_sin = np.transpose(k_sin, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Compute cos-cos and sin-sin terms
    kv_cos = np.matmul(np.swapaxes(k_cos, -2, -1), v)
    kv_sin = np.matmul(np.swapaxes(k_sin, -2, -1), v)

    out = np.matmul(q_cos, kv_cos) + np.matmul(q_sin, kv_sin)

    # Normalization
    k_sum_cos = np.sum(k_cos, axis=-2, keepdims=True)
    k_sum_sin = np.sum(k_sin, axis=-2, keepdims=True)
    normalizer = (
        np.matmul(q_cos, np.swapaxes(k_sum_cos, -2, -1)) +
        np.matmul(q_sin, np.swapaxes(k_sum_sin, -2, -1)) +
        eps
    )

    out = out / normalizer

    return np.transpose(out, (0, 2, 1, 3))
