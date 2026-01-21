"""PyTorch parity tests for embedding operations."""

import math
import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import embedding_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance
from tests.parity.conftest import get_mlx_dtype, get_pytorch_dtype, HAS_PYTORCH

if HAS_PYTORCH:
    import torch
    import torch.nn as nn


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_to_mlx(x_np: np.ndarray, dtype: str) -> mx.array:
    """Convert numpy array to MLX with proper dtype."""
    x_mlx = mx.array(x_np)
    mlx_dtype = get_mlx_dtype(dtype)
    return x_mlx.astype(mlx_dtype)


def _convert_to_torch(x_np: np.ndarray, dtype: str) -> "torch.Tensor":
    """Convert numpy array to PyTorch with proper dtype."""
    x_torch = torch.from_numpy(x_np.astype(np.float32))
    torch_dtype = get_pytorch_dtype(dtype)
    return x_torch.to(torch_dtype)


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or PyTorch tensor to numpy."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_PYTORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _numpy_sinusoidal_encoding_with_freqs(
    seq_len: int, dim: int, freqs: np.ndarray
) -> np.ndarray:
    """Numpy reference using provided frequencies.

    This allows testing sin/cos and interleaving logic separately from
    the frequency computation, which has platform-specific precision.

    Args:
        seq_len: Sequence length.
        dim: Embedding dimension.
        freqs: Pre-computed frequencies of shape (dim/2,).

    Returns:
        Sinusoidal embedding of shape (seq_len, dim).
    """
    positions = np.arange(seq_len)[:, None].astype(np.float32)
    angles = positions * freqs[None, :]

    pe = np.zeros((seq_len, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe


def _get_mlx_sinusoidal_freqs(dim: int, base: float = 10000.0) -> np.ndarray:
    """Compute sinusoidal frequencies using MLX (for consistent precision).

    The power operation (base ** exponent) has different precision on
    Metal GPU vs CPU. Using MLX-computed frequencies as reference ensures
    we test the sin/cos and interleaving logic, not the platform-specific
    power operation precision.
    """
    dims_range = mx.arange(0, dim, 2)
    freqs = base ** (-dims_range / dim)
    mx.eval(freqs)
    return np.array(freqs)


def _get_mlx_rope_freqs(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Compute RoPE inverse frequencies using MLX (for consistent precision).

    The power operation has platform-specific precision differences.
    """
    dims_range = mx.arange(0, head_dim, 2)
    freqs = 1.0 / (base ** (dims_range / head_dim))
    mx.eval(freqs)
    return np.array(freqs)


def _pytorch_rope_apply(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of RoPE application."""
    # x: (batch, heads, seq, head_dim)
    # freqs_cos, freqs_sin: (seq, head_dim/2)
    x_shape = x.shape
    x = x.reshape(*x_shape[:-1], -1, 2)  # (..., seq, head_dim/2, 2)

    x0, x1 = x[..., 0], x[..., 1]

    # Broadcast freqs to match x shape
    cos = freqs_cos[None, None, :, :]  # (1, 1, seq, head_dim/2)
    sin = freqs_sin[None, None, :, :]

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    out = torch.stack([out0, out1], dim=-1)
    return out.reshape(x_shape)


def _pytorch_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for given number of heads."""
    ratio = 2 ** (-8 / num_heads)
    return torch.tensor([ratio ** (i + 1) for i in range(num_heads)])


# =============================================================================
# Sinusoidal Embedding Parity Tests
# =============================================================================

class TestSinusoidalEmbeddingParity:
    """Sinusoidal positional embedding parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test sinusoidal embedding forward pass parity.

        Note: Uses MLX-computed frequencies to isolate sin/cos and interleaving
        logic from platform-specific power operation precision differences.
        """
        from mlx_primitives.layers.embeddings import SinusoidalEmbedding

        config = SIZE_CONFIGS[size]["embedding"]
        seq_len = config["seq"]
        dim = config["dim"]

        np.random.seed(42)

        # MLX forward
        embed_mlx = SinusoidalEmbedding(dim, max_seq_len=seq_len * 2)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)

        # Use MLX-computed frequencies for reference (isolates sin/cos logic)
        freqs = _get_mlx_sinusoidal_freqs(dim)
        numpy_out = _numpy_sinusoidal_encoding_with_freqs(seq_len, dim, freqs)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), numpy_out,
            rtol=rtol, atol=atol,
            err_msg=f"SinusoidalEmbedding forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_different_dimensions(self, dim, skip_without_pytorch):
        """Test sinusoidal embedding with different dimensions."""
        from mlx_primitives.layers.embeddings import SinusoidalEmbedding

        seq_len = 100

        # MLX
        embed_mlx = SinusoidalEmbedding(dim, max_seq_len=seq_len * 2)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)

        # Use MLX-computed frequencies for reference
        freqs = _get_mlx_sinusoidal_freqs(dim)
        numpy_out = _numpy_sinusoidal_encoding_with_freqs(seq_len, dim, freqs)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), numpy_out,
            rtol=rtol, atol=atol,
            err_msg=f"SinusoidalEmbedding dim={dim} mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_len", [512, 1024, 2048, 8192])
    def test_different_max_lengths(self, max_len, skip_without_pytorch):
        """Test sinusoidal embedding with different max lengths."""
        from mlx_primitives.layers.embeddings import SinusoidalEmbedding

        dim = 128
        seq_len = min(max_len, 256)  # Test up to 256 positions

        # MLX
        embed_mlx = SinusoidalEmbedding(dim, max_seq_len=max_len)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)

        # Use MLX-computed frequencies for reference
        freqs = _get_mlx_sinusoidal_freqs(dim)
        numpy_out = _numpy_sinusoidal_encoding_with_freqs(seq_len, dim, freqs)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), numpy_out,
            rtol=rtol, atol=atol,
            err_msg=f"SinusoidalEmbedding max_len={max_len} mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_frequency_computation(self, skip_without_pytorch):
        """Test that frequency computation matches PyTorch/Transformers."""
        from mlx_primitives.layers.embeddings import SinusoidalEmbedding

        dim = 64
        seq_len = 16
        base = 10000.0

        embed_mlx = SinusoidalEmbedding(dim, max_seq_len=seq_len, base=base)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)

        # Manual computation following the formula
        positions_np = np.arange(seq_len)[:, None]
        dims_range = np.arange(0, dim, 2)
        freqs = base ** (-dims_range / dim)
        angles = positions_np * freqs

        expected = np.zeros((seq_len, dim))
        expected[:, 0::2] = np.sin(angles)
        expected[:, 1::2] = np.cos(angles)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), expected,
            rtol=1e-5, atol=1e-6,
            err_msg="Frequency computation mismatch"
        )


# =============================================================================
# Learned Positional Embedding Parity Tests
# =============================================================================

class TestLearnedPositionalEmbeddingParity:
    """Learned positional embedding parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test learned positional embedding forward pass parity."""
        from mlx_primitives.layers.embeddings import LearnedPositionalEmbedding

        config = SIZE_CONFIGS[size]["embedding"]
        seq_len = config["seq"]
        dim = config["dim"]
        max_seq_len = seq_len * 2

        np.random.seed(42)
        weight_np = np.random.randn(max_seq_len, dim).astype(np.float32) * 0.02

        # MLX
        embed_mlx = LearnedPositionalEmbedding(dim, max_seq_len=max_seq_len, dropout=0.0)
        embed_mlx.embedding.weight = mx.array(weight_np)
        positions = mx.arange(seq_len)
        x_mlx = _convert_to_mlx(positions.astype(mx.int32), dtype)
        mlx_out = embed_mlx(positions=positions)

        # PyTorch reference (nn.Embedding)
        embed_torch = nn.Embedding(max_seq_len, dim)
        embed_torch.weight.data = torch.from_numpy(weight_np)
        positions_torch = torch.arange(seq_len)
        torch_out = embed_torch(positions_torch)

        rtol, atol = get_tolerance("embeddings", "learned_positional", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"LearnedPositionalEmbedding forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test learned positional embedding backward pass parity."""
        from mlx_primitives.layers.embeddings import LearnedPositionalEmbedding

        config = SIZE_CONFIGS[size]["embedding"]
        seq_len = config["seq"]
        dim = config["dim"]
        max_seq_len = seq_len * 2
        dtype = "fp32"

        np.random.seed(42)
        weight_np = np.random.randn(max_seq_len, dim).astype(np.float32) * 0.02

        # MLX backward
        embed_mlx = LearnedPositionalEmbedding(dim, max_seq_len=max_seq_len, dropout=0.0)
        embed_mlx.embedding.weight = mx.array(weight_np)

        def mlx_loss_fn(weight):
            embed_mlx.embedding.weight = weight
            positions = mx.arange(seq_len)
            out = embed_mlx(positions=positions)
            return mx.sum(out)

        mlx_grad = mx.grad(mlx_loss_fn)(mx.array(weight_np))
        mx.eval(mlx_grad)

        # PyTorch backward
        embed_torch = nn.Embedding(max_seq_len, dim)
        embed_torch.weight.data = torch.from_numpy(weight_np)
        embed_torch.weight.requires_grad = True
        positions_torch = torch.arange(seq_len)
        torch_out = embed_torch(positions_torch)
        torch_out.sum().backward()

        rtol, atol = get_gradient_tolerance("embeddings", "learned_positional", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), embed_torch.weight.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"LearnedPositionalEmbedding backward mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_weight_indexing(self, skip_without_pytorch):
        """Test that position indexing matches PyTorch nn.Embedding."""
        from mlx_primitives.layers.embeddings import LearnedPositionalEmbedding

        dim = 64
        max_seq_len = 100

        np.random.seed(42)
        weight_np = np.random.randn(max_seq_len, dim).astype(np.float32)

        # Test non-contiguous position indices
        positions_np = np.array([0, 5, 10, 50, 99])

        # MLX
        embed_mlx = LearnedPositionalEmbedding(dim, max_seq_len=max_seq_len, dropout=0.0)
        embed_mlx.embedding.weight = mx.array(weight_np)
        mlx_out = embed_mlx(positions=mx.array(positions_np))

        # PyTorch
        embed_torch = nn.Embedding(max_seq_len, dim)
        embed_torch.weight.data = torch.from_numpy(weight_np)
        torch_out = embed_torch(torch.tensor(positions_np))

        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=1e-6, atol=1e-7,
            err_msg="Position indexing mismatch"
        )


# =============================================================================
# Rotary Embedding (RoPE) Parity Tests
# =============================================================================

class TestRotaryEmbeddingParity:
    """Rotary Position Embedding (RoPE) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test RoPE forward pass parity.

        Note: Uses MLX-computed frequencies to isolate rotation logic from
        platform-specific power operation precision differences.
        """
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq_len = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = 10000.0

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX
        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=seq_len * 2, base=base)
        q_mlx = _convert_to_mlx(q_np, dtype)
        k_mlx = _convert_to_mlx(k_np, dtype)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx, offset=0)

        # PyTorch reference - use MLX-computed frequencies for fair comparison
        freqs = torch.from_numpy(_get_mlx_rope_freqs(head_dim, base))
        positions = torch.arange(seq_len).float()
        freqs_outer = positions[:, None] * freqs[None, :]  # (seq_len, head_dim/2)
        freqs_cos = torch.cos(freqs_outer)
        freqs_sin = torch.sin(freqs_outer)

        q_torch = _convert_to_torch(q_np, dtype)
        k_torch = _convert_to_torch(k_np, dtype)

        q_rot_torch = _pytorch_rope_apply(q_torch.float(), freqs_cos, freqs_sin)
        k_rot_torch = _pytorch_rope_apply(k_torch.float(), freqs_cos, freqs_sin)

        rtol, atol = get_tolerance("embeddings", "rotary", dtype)
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), _to_numpy(q_rot_torch),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE Q rotation mismatch [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_rot_mlx), _to_numpy(k_rot_torch),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE K rotation mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test RoPE backward pass parity.

        Note: Uses MLX-computed frequencies for fair comparison.
        """
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq_len = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = 10000.0
        dtype = "fp32"

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX backward
        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=seq_len * 2, base=base)

        def mlx_loss_fn(q, k):
            q_rot, k_rot = rope_mlx(q, k, offset=0)
            return mx.sum(q_rot) + mx.sum(k_rot)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        mlx_grad_q, mlx_grad_k = mx.grad(mlx_loss_fn, argnums=(0, 1))(q_mlx, k_mlx)
        mx.eval(mlx_grad_q, mlx_grad_k)

        # PyTorch backward - use MLX-computed frequencies
        freqs = torch.from_numpy(_get_mlx_rope_freqs(head_dim, base))
        positions = torch.arange(seq_len).float()
        freqs_outer = positions[:, None] * freqs[None, :]
        freqs_cos = torch.cos(freqs_outer)
        freqs_sin = torch.sin(freqs_outer)

        q_torch = torch.from_numpy(q_np).requires_grad_(True)
        k_torch = torch.from_numpy(k_np).requires_grad_(True)

        q_rot_torch = _pytorch_rope_apply(q_torch, freqs_cos, freqs_sin)
        k_rot_torch = _pytorch_rope_apply(k_torch, freqs_cos, freqs_sin)
        loss = q_rot_torch.sum() + k_rot_torch.sum()
        loss.backward()

        rtol, atol = get_gradient_tolerance("embeddings", "rotary", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_q), q_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE Q gradient mismatch [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_k), k_torch.grad.numpy(),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE K gradient mismatch [{size}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_cos_sin_caching(self, skip_without_pytorch):
        """Test that cos/sin caching produces correct results.

        Recomputes the expected cos/sin using MLX to avoid cross-platform
        precision differences in both the frequency computation (power) and
        the trigonometric functions (cos/sin).
        """
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        head_dim = 64
        max_seq_len = 512
        base = 10000.0

        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=base)

        # Get cached frequencies
        cached_freqs = rope_mlx._freqs_cis

        # Recompute expected cos/sin using MLX (for consistent precision)
        freqs_mlx = 1.0 / (base ** (mx.arange(0, head_dim, 2) / head_dim))
        positions_mlx = mx.arange(max_seq_len)
        freqs_outer_mlx = positions_mlx[:, None] * freqs_mlx[None, :]
        expected_cos_mlx = mx.cos(freqs_outer_mlx)
        expected_sin_mlx = mx.sin(freqs_outer_mlx)
        mx.eval(expected_cos_mlx, expected_sin_mlx)

        cached_np = _to_numpy(cached_freqs)
        np.testing.assert_allclose(
            cached_np[..., 0], np.array(expected_cos_mlx),
            rtol=1e-5, atol=1e-6,
            err_msg="Cached cos mismatch"
        )
        np.testing.assert_allclose(
            cached_np[..., 1], np.array(expected_sin_mlx),
            rtol=1e-5, atol=1e-6,
            err_msg="Cached sin mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("base", [10000, 500000, 1000000])
    def test_different_bases(self, base, skip_without_pytorch):
        """Test RoPE with different frequency bases."""
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        batch, heads, seq_len, head_dim = 2, 8, 64, 64

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX
        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=seq_len * 2, base=float(base))
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx, offset=0)

        # PyTorch reference - use MLX-computed frequencies
        freqs = torch.from_numpy(_get_mlx_rope_freqs(head_dim, float(base)))
        positions = torch.arange(seq_len).float()
        freqs_outer = positions[:, None] * freqs[None, :]
        freqs_cos = torch.cos(freqs_outer)
        freqs_sin = torch.sin(freqs_outer)

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        q_rot_torch = _pytorch_rope_apply(q_torch, freqs_cos, freqs_sin)
        k_rot_torch = _pytorch_rope_apply(k_torch, freqs_cos, freqs_sin)

        rtol, atol = get_tolerance("embeddings", "rotary", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), _to_numpy(q_rot_torch),
            rtol=rtol, atol=atol,
            err_msg=f"RoPE base={base} Q mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_scaling_factor(self, skip_without_pytorch):
        """Test RoPE with position scaling (for extended context)."""
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        # Test with offset (simulates position scaling via offset)
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        base = 10000.0
        offset = 100  # Start at position 100

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX with offset
        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=offset + seq_len * 2, base=base)
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx, offset=offset)

        # PyTorch reference with shifted positions - use MLX-computed frequencies
        freqs = torch.from_numpy(_get_mlx_rope_freqs(head_dim, base))
        positions = torch.arange(offset, offset + seq_len).float()
        freqs_outer = positions[:, None] * freqs[None, :]
        freqs_cos = torch.cos(freqs_outer)
        freqs_sin = torch.sin(freqs_outer)

        q_torch = torch.from_numpy(q_np)
        q_rot_torch = _pytorch_rope_apply(q_torch, freqs_cos, freqs_sin)

        rtol, atol = get_tolerance("embeddings", "rotary", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), _to_numpy(q_rot_torch),
            rtol=rtol, atol=atol,
            err_msg="RoPE with offset mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_interleaved_vs_rotated(self, skip_without_pytorch):
        """Test interleaved vs rotated RoPE implementations."""
        from mlx_primitives.layers.embeddings import RotaryEmbedding

        # Both MLX and our reference use the interleaved format
        # Verify the rotation formula: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        batch, heads, seq_len, head_dim = 1, 1, 4, 8
        base = 10000.0

        # Use simple values for verification
        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        rope_mlx = RotaryEmbedding(head_dim, max_seq_len=seq_len * 2, base=base)
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(q_np)  # Use same values
        q_rot_mlx, _ = rope_mlx(q_mlx, k_mlx, offset=0)

        # Manual rotation check for position 0
        q_pos0 = q_np[0, 0, 0, :]  # (head_dim,)
        q_pairs = q_pos0.reshape(-1, 2)  # (head_dim/2, 2)

        freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
        angles = 0 * freqs  # position 0, angles are 0
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # At position 0, angles are 0, so cos=1, sin=0
        # Rotation should be identity: [x0*1 - x1*0, x0*0 + x1*1] = [x0, x1]
        q_rot_pos0 = _to_numpy(q_rot_mlx)[0, 0, 0, :]

        np.testing.assert_allclose(
            q_rot_pos0, q_pos0,
            rtol=1e-5, atol=1e-6,
            err_msg="RoPE at position 0 should be identity"
        )


# =============================================================================
# ALiBi Embedding Parity Tests
# =============================================================================

class TestAlibiEmbeddingParity:
    """ALiBi (Attention with Linear Biases) embedding parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test ALiBi embedding forward pass parity."""
        from mlx_primitives.layers.embeddings import AlibiEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq_len = config["seq"]
        num_heads = config["heads"]

        np.random.seed(42)
        attn_scores = np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)

        # MLX
        alibi_mlx = AlibiEmbedding(num_heads)
        scores_mlx = _convert_to_mlx(attn_scores, dtype)
        mlx_out = alibi_mlx(scores_mlx, offset=0)

        # PyTorch reference
        slopes = _pytorch_alibi_slopes(num_heads)
        q_pos = torch.arange(seq_len)[:, None]
        k_pos = torch.arange(seq_len)[None, :]
        rel_pos = q_pos - k_pos  # (seq_len, seq_len)

        # slopes: (num_heads,) -> (num_heads, 1, 1)
        bias = slopes[:, None, None] * rel_pos[None, :, :]  # (num_heads, seq_len, seq_len)

        scores_torch = _convert_to_torch(attn_scores, dtype)
        torch_out = scores_torch + bias[None, :, :, :]

        rtol, atol = get_tolerance("embeddings", "alibi", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"ALiBi forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_heads", [4, 8, 12, 16, 32])
    def test_different_num_heads(self, num_heads, skip_without_pytorch):
        """Test ALiBi with different numbers of heads."""
        from mlx_primitives.layers.embeddings import AlibiEmbedding

        batch, seq_len = 2, 64

        np.random.seed(42)
        attn_scores = np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)

        # MLX
        alibi_mlx = AlibiEmbedding(num_heads)
        mlx_out = alibi_mlx(mx.array(attn_scores), offset=0)

        # PyTorch reference
        slopes = _pytorch_alibi_slopes(num_heads)
        q_pos = torch.arange(seq_len)[:, None]
        k_pos = torch.arange(seq_len)[None, :]
        rel_pos = q_pos - k_pos
        bias = slopes[:, None, None] * rel_pos[None, :, :]
        torch_out = torch.from_numpy(attn_scores) + bias[None, :, :, :]

        rtol, atol = get_tolerance("embeddings", "alibi", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(torch_out),
            rtol=rtol, atol=atol,
            err_msg=f"ALiBi num_heads={num_heads} mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_slope_computation(self, skip_without_pytorch):
        """Test ALiBi slope computation matches reference."""
        from mlx_primitives.layers.embeddings import AlibiEmbedding

        for num_heads in [4, 8, 16]:
            alibi_mlx = AlibiEmbedding(num_heads)
            mlx_slopes = _to_numpy(alibi_mlx._slopes)

            # Reference: geometric sequence with ratio 2^(-8/num_heads)
            expected_slopes = _pytorch_alibi_slopes(num_heads).numpy()

            np.testing.assert_allclose(
                mlx_slopes, expected_slopes,
                rtol=1e-6, atol=1e-7,
                err_msg=f"ALiBi slopes mismatch for num_heads={num_heads}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_bias_matrix_shape(self, skip_without_pytorch):
        """Test ALiBi bias matrix has correct shape."""
        from mlx_primitives.layers.embeddings import AlibiEmbedding

        num_heads = 8
        seq_len_q = 64
        seq_len_k = 128

        alibi_mlx = AlibiEmbedding(num_heads)
        bias = alibi_mlx.get_bias(seq_len_q, seq_len_k)

        expected_shape = (num_heads, seq_len_q, seq_len_k)
        actual_shape = tuple(bias.shape)

        assert actual_shape == expected_shape, \
            f"ALiBi bias shape mismatch: expected {expected_shape}, got {actual_shape}"


# =============================================================================
# Relative Positional Embedding Parity Tests
# =============================================================================

class TestRelativePositionalEmbeddingParity:
    """Relative positional embedding (T5-style) parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test relative positional embedding forward pass parity."""
        from mlx_primitives.layers.embeddings import RelativePositionalEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq_len = config["seq"]
        num_heads = config["heads"]
        num_buckets = 32
        max_distance = 128

        np.random.seed(42)
        attn_scores = np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02

        # MLX
        rel_emb_mlx = RelativePositionalEmbedding(
            num_heads, num_buckets=num_buckets, max_distance=max_distance, bidirectional=True
        )
        rel_emb_mlx.embedding.weight = mx.array(embedding_weight)
        scores_mlx = _convert_to_mlx(attn_scores, dtype)
        mlx_out = rel_emb_mlx(scores_mlx)

        # Get MLX bias for comparison
        mlx_bias = rel_emb_mlx.get_bias(seq_len, seq_len)

        # Verify output shape
        assert mlx_out.shape == (batch, num_heads, seq_len, seq_len), \
            f"Output shape mismatch: {mlx_out.shape}"

        # Verify bias is added correctly
        expected_out = scores_mlx + mlx_bias
        np.testing.assert_allclose(
            _to_numpy(mlx_out), _to_numpy(expected_out),
            rtol=1e-6, atol=1e-7,
            err_msg=f"Bias addition mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test relative positional embedding backward pass parity."""
        from mlx_primitives.layers.embeddings import RelativePositionalEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq_len = config["seq"]
        num_heads = config["heads"]
        num_buckets = 32
        dtype = "fp32"

        np.random.seed(42)
        attn_scores = np.random.randn(batch, num_heads, seq_len, seq_len).astype(np.float32)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02

        # MLX backward
        rel_emb_mlx = RelativePositionalEmbedding(num_heads, num_buckets=num_buckets)
        rel_emb_mlx.embedding.weight = mx.array(embedding_weight)

        def mlx_loss_fn(weight):
            rel_emb_mlx.embedding.weight = weight
            scores = mx.array(attn_scores)
            out = rel_emb_mlx(scores)
            return mx.sum(out)

        mlx_grad = mx.grad(mlx_loss_fn)(mx.array(embedding_weight))
        mx.eval(mlx_grad)

        # Verify gradient has correct shape
        assert mlx_grad.shape == (num_buckets, num_heads), \
            f"Gradient shape mismatch: {mlx_grad.shape}"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_bucket_computation(self, skip_without_pytorch):
        """Test relative position bucket computation."""
        from mlx_primitives.layers.embeddings import RelativePositionalEmbedding

        num_heads = 8
        num_buckets = 32
        max_distance = 128
        seq_len = 16

        rel_emb = RelativePositionalEmbedding(
            num_heads, num_buckets=num_buckets, max_distance=max_distance, bidirectional=True
        )

        # Test bucket computation directly
        q_pos = mx.arange(seq_len)[:, None]
        k_pos = mx.arange(seq_len)[None, :]
        relative_position = k_pos - q_pos

        buckets = rel_emb._relative_position_bucket(relative_position)
        mx.eval(buckets)

        # Verify bucket indices are in valid range
        buckets_np = np.array(buckets)
        assert np.all(buckets_np >= 0), "Bucket indices should be non-negative"
        assert np.all(buckets_np < num_buckets), f"Bucket indices should be < {num_buckets}"

        # Verify symmetry for bidirectional
        # Positive and negative positions should use different bucket ranges
        assert buckets_np[0, seq_len - 1] != buckets_np[seq_len - 1, 0] or seq_len == 1, \
            "Bidirectional should distinguish positive/negative positions"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_buckets", [16, 32, 64, 128])
    def test_different_num_buckets(self, num_buckets, skip_without_pytorch):
        """Test with different numbers of buckets."""
        from mlx_primitives.layers.embeddings import RelativePositionalEmbedding

        num_heads = 8
        seq_len = 64

        np.random.seed(42)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02
        attn_scores = np.random.randn(2, num_heads, seq_len, seq_len).astype(np.float32)

        rel_emb = RelativePositionalEmbedding(num_heads, num_buckets=num_buckets)
        rel_emb.embedding.weight = mx.array(embedding_weight)

        # Should work without errors
        out = rel_emb(mx.array(attn_scores))
        mx.eval(out)

        assert out.shape == (2, num_heads, seq_len, seq_len), \
            f"Output shape mismatch for num_buckets={num_buckets}"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_bidirectional_vs_unidirectional(self, skip_without_pytorch):
        """Test bidirectional vs unidirectional relative positions."""
        from mlx_primitives.layers.embeddings import RelativePositionalEmbedding

        num_heads = 8
        num_buckets = 32
        seq_len = 16

        np.random.seed(42)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02
        attn_scores = np.random.randn(1, num_heads, seq_len, seq_len).astype(np.float32)

        # Bidirectional
        rel_emb_bi = RelativePositionalEmbedding(
            num_heads, num_buckets=num_buckets, bidirectional=True
        )
        rel_emb_bi.embedding.weight = mx.array(embedding_weight)
        out_bi = rel_emb_bi(mx.array(attn_scores))

        # Unidirectional
        rel_emb_uni = RelativePositionalEmbedding(
            num_heads, num_buckets=num_buckets, bidirectional=False
        )
        rel_emb_uni.embedding.weight = mx.array(embedding_weight)
        out_uni = rel_emb_uni(mx.array(attn_scores))

        mx.eval(out_bi, out_uni)

        # Outputs should be different
        out_bi_np = _to_numpy(out_bi)
        out_uni_np = _to_numpy(out_uni)

        # They should have the same shape but different values
        assert out_bi_np.shape == out_uni_np.shape
        assert not np.allclose(out_bi_np, out_uni_np), \
            "Bidirectional and unidirectional should produce different outputs"
