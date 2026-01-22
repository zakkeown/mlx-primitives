"""JAX Metal parity tests for embedding operations."""

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from tests.reference_jax_extended import (
        jax_sinusoidal_embedding,
        jax_learned_positional_embedding,
        jax_rotary_embedding,
        jax_alibi_embedding,
        jax_relative_positional_embedding,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


class TestSinusoidalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test sinusoidal embedding forward pass parity with JAX."""
        from mlx_primitives.layers import SinusoidalEmbedding

        config = SIZE_CONFIGS[size]["embedding"]
        seq = config["seq"]
        dim = config["dim"]

        # Create positions
        positions_np = np.arange(seq).astype(np.int32)

        # MLX sinusoidal embedding
        embed_mlx = SinusoidalEmbedding(dims=dim, max_seq_len=seq * 2)
        mx.eval(embed_mlx.parameters())

        positions_mlx = mx.array(positions_np)
        mlx_out = embed_mlx(positions_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_sinusoidal_embedding(positions_np, dim, base=embed_mlx.base)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Sinusoidal embedding forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_different_dimensions(self, dim, skip_without_jax):
        """Test sinusoidal embedding with different dimensions."""
        from mlx_primitives.layers import SinusoidalEmbedding

        seq_len = 100

        # MLX
        embed_mlx = SinusoidalEmbedding(dims=dim, max_seq_len=seq_len * 2)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)
        mx.eval(mlx_out)

        # JAX reference
        positions_np = np.arange(seq_len).astype(np.int32)
        jax_out = jax_sinusoidal_embedding(positions_np, dim, base=embed_mlx.base)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Sinusoidal embedding dim={dim} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("max_len", [512, 1024, 2048, 8192])
    def test_different_max_lengths(self, max_len, skip_without_jax):
        """Test sinusoidal embedding with different max lengths."""
        from mlx_primitives.layers import SinusoidalEmbedding

        dim = 128
        seq_len = min(max_len, 256)  # Test up to 256 positions

        # MLX
        embed_mlx = SinusoidalEmbedding(dims=dim, max_seq_len=max_len)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)
        mx.eval(mlx_out)

        # JAX reference
        positions_np = np.arange(seq_len).astype(np.int32)
        jax_out = jax_sinusoidal_embedding(positions_np, dim, base=embed_mlx.base)

        rtol, atol = get_tolerance("embeddings", "sinusoidal", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Sinusoidal embedding max_len={max_len} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_frequency_computation(self, skip_without_jax):
        """Test that frequency computation matches JAX reference."""
        from mlx_primitives.layers import SinusoidalEmbedding

        dim = 64
        seq_len = 16
        base = 10000.0

        embed_mlx = SinusoidalEmbedding(dims=dim, max_seq_len=seq_len, base=base)
        positions = mx.arange(seq_len)
        mlx_out = embed_mlx(positions)
        mx.eval(mlx_out)

        # JAX reference
        positions_np = np.arange(seq_len).astype(np.int32)
        jax_out = jax_sinusoidal_embedding(positions_np, dim, base=base)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=1e-5, atol=1e-6,
            err_msg="Frequency computation mismatch (JAX)"
        )


class TestLearnedPositionalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test learned positional embedding forward pass parity with JAX."""
        from mlx_primitives.layers import LearnedPositionalEmbedding

        config = SIZE_CONFIGS[size]["embedding"]
        seq = config["seq"]
        dim = config["dim"]

        # Create positions
        positions_np = np.arange(seq).astype(np.int32)

        # MLX learned positional embedding
        embed_mlx = LearnedPositionalEmbedding(dims=dim, max_seq_len=seq * 2)
        mx.eval(embed_mlx.parameters())
        weight_np = np.array(embed_mlx.embedding.weight)

        mlx_dtype = get_mlx_dtype(dtype)
        positions_mlx = mx.array(positions_np)
        mlx_out = embed_mlx(positions=positions_mlx)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_learned_positional_embedding(positions_np, weight_np)

        rtol, atol = get_tolerance("embeddings", "learned_positional", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Learned positional embedding forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    def test_backward_parity(self, skip_without_jax):
        """Test learned positional embedding backward pass parity with JAX."""
        from mlx_primitives.layers import LearnedPositionalEmbedding

        config = SIZE_CONFIGS["small"]["embedding"]
        seq = config["seq"]
        dim = config["dim"]

        np.random.seed(42)
        positions_np = np.arange(seq).astype(np.int32)

        # MLX backward
        embed_mlx = LearnedPositionalEmbedding(dims=dim, max_seq_len=seq * 2)
        mx.eval(embed_mlx.parameters())
        weight_np = np.array(embed_mlx.embedding.weight)

        def mlx_fn(weight):
            embed_mlx.embedding.weight = weight
            return embed_mlx(positions=mx.array(positions_np)).sum()

        mlx_grad = mx.grad(mlx_fn)(embed_mlx.embedding.weight)
        mx.eval(mlx_grad)

        # JAX backward
        def jax_fn(weight):
            return weight[positions_np].sum()

        weight_jax = jnp.array(weight_np, dtype=jnp.float32)
        jax_grad = jax.grad(jax_fn)(weight_jax)

        rtol, atol = get_gradient_tolerance("embeddings", "learned_positional", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad), np.array(jax_grad),
            rtol=rtol, atol=atol,
            err_msg="Learned positional embedding backward mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_weight_indexing(self, skip_without_jax):
        """Test that position indexing matches JAX embedding lookup."""
        from mlx_primitives.layers import LearnedPositionalEmbedding

        dim = 64
        max_seq_len = 100

        np.random.seed(42)
        weight_np = np.random.randn(max_seq_len, dim).astype(np.float32)

        # Test non-contiguous position indices
        positions_np = np.array([0, 5, 10, 50, 99])

        # MLX
        embed_mlx = LearnedPositionalEmbedding(dims=dim, max_seq_len=max_seq_len, dropout=0.0)
        embed_mlx.embedding.weight = mx.array(weight_np)
        mlx_out = embed_mlx(positions=mx.array(positions_np))
        mx.eval(mlx_out)

        # JAX reference (simple indexing)
        jax_out = jax_learned_positional_embedding(positions_np, weight_np)

        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=1e-6, atol=1e-7,
            err_msg="Position indexing mismatch (JAX)"
        )


class TestRotaryEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test rotary embedding forward pass parity with JAX."""
        from mlx_primitives.layers import RotaryEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        np.random.seed(42)
        # Query/Key tensors: MLX expects (batch, heads, seq_len, head_dim)
        q_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)

        # Convert to target dtype for fair comparison
        jax_dtype = get_jax_dtype(dtype)
        q_typed = np.array(jnp.array(q_np).astype(jax_dtype).astype(jnp.float32))
        k_typed = np.array(jnp.array(k_np).astype(jax_dtype).astype(jnp.float32))

        # MLX rotary embedding
        rope_mlx = RotaryEmbedding(dims=head_dim)
        mx.eval(rope_mlx.parameters())

        mlx_dtype = get_mlx_dtype(dtype)
        q_mlx = mx.array(q_np).astype(mlx_dtype)
        k_mlx = mx.array(k_np).astype(mlx_dtype)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx)
        mx.eval(q_rot_mlx, k_rot_mlx)

        # Get cos/sin from MLX for JAX reference
        # _freqs_cis has shape (seq_len, dims/2, 2) where [..., 0]=cos, [..., 1]=sin
        freqs_cis = rope_mlx._freqs_cis[:seq]
        mx.eval(freqs_cis)
        freqs_cis_np = np.array(freqs_cis)
        cos_np = freqs_cis_np[..., 0]  # (seq, head_dim/2)
        sin_np = freqs_cis_np[..., 1]  # (seq, head_dim/2)

        # Reshape for broadcasting: (seq, head_dim/2) -> (1, 1, seq, head_dim/2)
        # to match (batch, heads, seq, head_dim/2) for JAX reference
        cos_np = cos_np[None, None, :, :]
        sin_np = sin_np[None, None, :, :]

        # JAX reference - note: JAX ref expects (batch, seq, heads, head_dim)
        # so we transpose q/k for the JAX call, then transpose back
        # Use dtype-converted inputs for fair comparison
        q_for_jax = np.transpose(q_typed, (0, 2, 1, 3))  # (batch, seq, heads, head_dim)
        k_for_jax = np.transpose(k_typed, (0, 2, 1, 3))
        # Reshape cos/sin for JAX ref: (1, seq, 1, head_dim/2)
        cos_for_jax = np.transpose(cos_np, (0, 2, 1, 3))
        sin_for_jax = np.transpose(sin_np, (0, 2, 1, 3))

        q_rot_jax = jax_rotary_embedding(q_for_jax, cos_for_jax, sin_for_jax)
        k_rot_jax = jax_rotary_embedding(k_for_jax, cos_for_jax, sin_for_jax)

        # Transpose JAX outputs back to MLX order
        q_rot_jax = np.transpose(q_rot_jax, (0, 2, 1, 3))
        k_rot_jax = np.transpose(k_rot_jax, (0, 2, 1, 3))

        rtol, atol = get_tolerance("embeddings", "rotary", dtype)
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), q_rot_jax,
            rtol=rtol, atol=atol,
            err_msg=f"Rotary embedding (Q) forward mismatch (JAX) [{size}, {dtype}]"
        )
        np.testing.assert_allclose(
            _to_numpy(k_rot_mlx), k_rot_jax,
            rtol=rtol, atol=atol,
            err_msg=f"Rotary embedding (K) forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test RoPE backward pass parity with JAX."""
        from mlx_primitives.layers import RotaryEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        base = 10000.0

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)

        # MLX backward
        rope_mlx = RotaryEmbedding(dims=head_dim, max_seq_len=seq * 2, base=base)

        def mlx_loss_fn(q, k):
            q_rot, k_rot = rope_mlx(q, k, offset=0)
            return mx.sum(q_rot) + mx.sum(k_rot)

        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        mlx_grad_q, mlx_grad_k = mx.grad(mlx_loss_fn, argnums=(0, 1))(q_mlx, k_mlx)
        mx.eval(mlx_grad_q, mlx_grad_k)

        # Get cos/sin from MLX for JAX reference
        freqs_cis = rope_mlx._freqs_cis[:seq]
        mx.eval(freqs_cis)
        freqs_cis_np = np.array(freqs_cis)
        cos_np = freqs_cis_np[..., 0]
        sin_np = freqs_cis_np[..., 1]

        # JAX backward - transpose to JAX format (batch, seq, heads, head_dim)
        q_for_jax = np.transpose(q_np, (0, 2, 1, 3))
        k_for_jax = np.transpose(k_np, (0, 2, 1, 3))
        cos_for_jax = cos_np[None, :, None, :]  # (1, seq, 1, head_dim/2)
        sin_for_jax = sin_np[None, :, None, :]

        def jax_loss_fn(q_j, k_j):
            q_rot = jax_rotary_embedding(q_j, cos_for_jax, sin_for_jax)
            k_rot = jax_rotary_embedding(k_j, cos_for_jax, sin_for_jax)
            return jnp.sum(q_rot) + jnp.sum(k_rot)

        q_jax = jnp.array(q_for_jax)
        k_jax = jnp.array(k_for_jax)
        jax_grad_q, jax_grad_k = jax.grad(jax_loss_fn, argnums=(0, 1))(q_jax, k_jax)

        # Transpose JAX gradients back to MLX order
        jax_grad_q = np.transpose(np.array(jax_grad_q), (0, 2, 1, 3))
        jax_grad_k = np.transpose(np.array(jax_grad_k), (0, 2, 1, 3))

        rtol, atol = get_gradient_tolerance("embeddings", "rotary", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_q), jax_grad_q,
            rtol=rtol, atol=atol,
            err_msg=f"RoPE Q gradient mismatch (JAX) [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(mlx_grad_k), jax_grad_k,
            rtol=rtol, atol=atol,
            err_msg=f"RoPE K gradient mismatch (JAX) [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_cos_sin_caching(self, skip_without_jax):
        """Test that cos/sin caching produces correct results."""
        from mlx_primitives.layers import RotaryEmbedding

        head_dim = 64
        max_seq_len = 512
        base = 10000.0

        rope_mlx = RotaryEmbedding(dims=head_dim, max_seq_len=max_seq_len, base=base)

        # Get cached frequencies
        cached_freqs = rope_mlx._freqs_cis
        mx.eval(cached_freqs)

        # Recompute expected cos/sin using JAX
        half_dim = head_dim // 2
        freqs = 1.0 / (base ** (jnp.arange(0, half_dim) * 2.0 / head_dim))
        positions = jnp.arange(max_seq_len)
        angles = jnp.outer(positions, freqs)
        expected_cos = np.array(jnp.cos(angles))
        expected_sin = np.array(jnp.sin(angles))

        cached_np = _to_numpy(cached_freqs)
        np.testing.assert_allclose(
            cached_np[..., 0], expected_cos,
            rtol=1e-5, atol=1e-6,
            err_msg="Cached cos mismatch (JAX)"
        )
        np.testing.assert_allclose(
            cached_np[..., 1], expected_sin,
            rtol=1e-5, atol=1e-6,
            err_msg="Cached sin mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("base", [10000, 500000, 1000000])
    def test_different_bases(self, base, skip_without_jax):
        """Test RoPE with different frequency bases."""
        from mlx_primitives.layers import RotaryEmbedding

        batch, heads, seq_len, head_dim = 2, 8, 64, 64

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX
        rope_mlx = RotaryEmbedding(dims=head_dim, max_seq_len=seq_len * 2, base=float(base))
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx, offset=0)
        mx.eval(q_rot_mlx, k_rot_mlx)

        # Get cos/sin from MLX
        freqs_cis = rope_mlx._freqs_cis[:seq_len]
        mx.eval(freqs_cis)
        freqs_cis_np = np.array(freqs_cis)
        cos_np = freqs_cis_np[..., 0]
        sin_np = freqs_cis_np[..., 1]

        # JAX reference
        q_for_jax = np.transpose(q_np, (0, 2, 1, 3))
        k_for_jax = np.transpose(k_np, (0, 2, 1, 3))
        cos_for_jax = cos_np[None, :, None, :]
        sin_for_jax = sin_np[None, :, None, :]

        q_rot_jax = jax_rotary_embedding(q_for_jax, cos_for_jax, sin_for_jax)
        k_rot_jax = jax_rotary_embedding(k_for_jax, cos_for_jax, sin_for_jax)

        # Transpose back
        q_rot_jax = np.transpose(q_rot_jax, (0, 2, 1, 3))
        k_rot_jax = np.transpose(k_rot_jax, (0, 2, 1, 3))

        rtol, atol = get_tolerance("embeddings", "rotary", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), q_rot_jax,
            rtol=rtol, atol=atol,
            err_msg=f"RoPE base={base} Q mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_scaling_factor(self, skip_without_jax):
        """Test RoPE with position scaling (for extended context)."""
        from mlx_primitives.layers import RotaryEmbedding

        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        base = 10000.0
        offset = 100  # Start at position 100

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        # MLX with offset
        rope_mlx = RotaryEmbedding(dims=head_dim, max_seq_len=offset + seq_len * 2, base=base)
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(k_np)
        q_rot_mlx, k_rot_mlx = rope_mlx(q_mlx, k_mlx, offset=offset)
        mx.eval(q_rot_mlx, k_rot_mlx)

        # Get cos/sin from MLX at offset positions
        freqs_cis = rope_mlx._freqs_cis[offset:offset + seq_len]
        mx.eval(freqs_cis)
        freqs_cis_np = np.array(freqs_cis)
        cos_np = freqs_cis_np[..., 0]
        sin_np = freqs_cis_np[..., 1]

        # JAX reference with shifted positions
        q_for_jax = np.transpose(q_np, (0, 2, 1, 3))
        cos_for_jax = cos_np[None, :, None, :]
        sin_for_jax = sin_np[None, :, None, :]

        q_rot_jax = jax_rotary_embedding(q_for_jax, cos_for_jax, sin_for_jax)
        q_rot_jax = np.transpose(q_rot_jax, (0, 2, 1, 3))

        rtol, atol = get_tolerance("embeddings", "rotary", "fp32")
        np.testing.assert_allclose(
            _to_numpy(q_rot_mlx), q_rot_jax,
            rtol=rtol, atol=atol,
            err_msg="RoPE with offset mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_interleaved_vs_rotated(self, skip_without_jax):
        """Test interleaved vs rotated RoPE implementations."""
        from mlx_primitives.layers import RotaryEmbedding

        batch, heads, seq_len, head_dim = 1, 1, 4, 8
        base = 10000.0

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        rope_mlx = RotaryEmbedding(dims=head_dim, max_seq_len=seq_len * 2, base=base)
        q_mlx = mx.array(q_np)
        k_mlx = mx.array(q_np)
        q_rot_mlx, _ = rope_mlx(q_mlx, k_mlx, offset=0)
        mx.eval(q_rot_mlx)

        # At position 0, angles are 0, so cos=1, sin=0
        # Rotation should be identity: [x0*1 - x1*0, x0*0 + x1*1] = [x0, x1]
        q_pos0 = q_np[0, 0, 0, :]
        q_rot_pos0 = _to_numpy(q_rot_mlx)[0, 0, 0, :]

        np.testing.assert_allclose(
            q_rot_pos0, q_pos0,
            rtol=1e-5, atol=1e-6,
            err_msg="RoPE at position 0 should be identity (JAX)"
        )


class TestAlibiEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test ALiBi embedding forward pass parity with JAX."""
        from mlx_primitives.layers import AlibiEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        seq = config["seq"]
        heads = config["heads"]

        # MLX ALiBi embedding
        alibi_mlx = AlibiEmbedding(num_heads=heads)
        mx.eval(alibi_mlx.parameters())

        mlx_out = alibi_mlx.get_bias(seq, seq)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_alibi_embedding(seq, heads)

        rtol, atol = get_tolerance("embeddings", "alibi", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"ALiBi embedding forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_heads", [4, 8, 12, 16, 32])
    def test_different_num_heads(self, num_heads, skip_without_jax):
        """Test ALiBi with different numbers of heads."""
        from mlx_primitives.layers import AlibiEmbedding

        seq_len = 64

        # MLX
        alibi_mlx = AlibiEmbedding(num_heads=num_heads)
        mlx_out = alibi_mlx.get_bias(seq_len, seq_len)
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_alibi_embedding(seq_len, num_heads)

        rtol, atol = get_tolerance("embeddings", "alibi", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"ALiBi num_heads={num_heads} mismatch (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_slope_computation(self, skip_without_jax):
        """Test ALiBi slope computation matches JAX reference."""
        from mlx_primitives.layers import AlibiEmbedding

        for num_heads in [4, 8, 16]:
            alibi_mlx = AlibiEmbedding(num_heads=num_heads)
            mlx_slopes = _to_numpy(alibi_mlx._slopes)

            # Reference: geometric sequence with ratio 2^(-8/num_heads)
            ratio = 2 ** (-8 / num_heads)
            expected_slopes = np.array([ratio ** (i + 1) for i in range(num_heads)])

            np.testing.assert_allclose(
                mlx_slopes, expected_slopes,
                rtol=1e-6, atol=1e-7,
                err_msg=f"ALiBi slopes mismatch for num_heads={num_heads} (JAX)"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_bias_matrix_shape(self, skip_without_jax):
        """Test ALiBi bias matrix has correct shape."""
        from mlx_primitives.layers import AlibiEmbedding

        num_heads = 8
        seq_len_q = 64
        seq_len_k = 128

        alibi_mlx = AlibiEmbedding(num_heads=num_heads)
        bias = alibi_mlx.get_bias(seq_len_q, seq_len_k)
        mx.eval(bias)

        expected_shape = (num_heads, seq_len_q, seq_len_k)
        actual_shape = tuple(bias.shape)

        assert actual_shape == expected_shape, \
            f"ALiBi bias shape mismatch: expected {expected_shape}, got {actual_shape}"


class TestRelativePositionalEmbeddingParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test relative positional embedding forward pass parity with JAX."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]

        # MLX relative positional embedding
        rel_embed_mlx = RelativePositionalEmbedding(
            num_heads=heads,
            num_buckets=32,
            max_distance=128,
        )
        mx.eval(rel_embed_mlx.parameters())

        # Get bucket indices - need to compute relative positions first
        q_pos = mx.arange(seq)[:, None]
        k_pos = mx.arange(seq)[None, :]
        relative_position = k_pos - q_pos
        mlx_buckets = rel_embed_mlx._relative_position_bucket(relative_position)
        mx.eval(mlx_buckets)

        # JAX reference
        jax_buckets = jax_relative_positional_embedding(seq, seq, num_buckets=32, max_distance=128)

        rtol, atol = get_tolerance("embeddings", "relative_positional", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_buckets), jax_buckets,
            rtol=rtol, atol=atol,
            err_msg=f"Relative positional embedding forward mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test relative positional embedding backward pass parity with JAX."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        config = SIZE_CONFIGS[size]["attention"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        num_buckets = 32

        np.random.seed(42)
        attn_scores = np.random.randn(batch, heads, seq, seq).astype(np.float32)
        embedding_weight = np.random.randn(num_buckets, heads).astype(np.float32) * 0.02

        # MLX backward
        rel_emb_mlx = RelativePositionalEmbedding(num_heads=heads, num_buckets=num_buckets)
        rel_emb_mlx.embedding.weight = mx.array(embedding_weight)

        def mlx_loss_fn(weight):
            rel_emb_mlx.embedding.weight = weight
            scores = mx.array(attn_scores)
            out = rel_emb_mlx(scores)
            return mx.sum(out)

        mlx_grad = mx.grad(mlx_loss_fn)(mx.array(embedding_weight))
        mx.eval(mlx_grad)

        # Verify gradient has correct shape
        assert mlx_grad.shape == (num_buckets, heads), \
            f"Gradient shape mismatch: {mlx_grad.shape}"

        # Gradient should be non-zero (basic sanity check)
        assert np.any(np.abs(_to_numpy(mlx_grad)) > 0), "Gradient should be non-zero"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_bucket_computation(self, skip_without_jax):
        """Test relative position bucket computation."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        num_heads = 8
        num_buckets = 32
        max_distance = 128
        seq_len = 16

        rel_emb = RelativePositionalEmbedding(
            num_heads=num_heads, num_buckets=num_buckets, max_distance=max_distance, bidirectional=True
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
        if seq_len > 1:
            assert buckets_np[0, seq_len - 1] != buckets_np[seq_len - 1, 0], \
                "Bidirectional should distinguish positive/negative positions"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_buckets", [16, 32, 64, 128])
    def test_different_num_buckets(self, num_buckets, skip_without_jax):
        """Test with different numbers of buckets."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        num_heads = 8
        seq_len = 64

        np.random.seed(42)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02
        attn_scores = np.random.randn(2, num_heads, seq_len, seq_len).astype(np.float32)

        rel_emb = RelativePositionalEmbedding(num_heads=num_heads, num_buckets=num_buckets)
        rel_emb.embedding.weight = mx.array(embedding_weight)

        # Should work without errors
        out = rel_emb(mx.array(attn_scores))
        mx.eval(out)

        assert out.shape == (2, num_heads, seq_len, seq_len), \
            f"Output shape mismatch for num_buckets={num_buckets}"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_bidirectional_vs_unidirectional(self, skip_without_jax):
        """Test bidirectional vs unidirectional relative positions."""
        from mlx_primitives.layers import RelativePositionalEmbedding

        num_heads = 8
        num_buckets = 32
        seq_len = 16

        np.random.seed(42)
        embedding_weight = np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02
        attn_scores = np.random.randn(1, num_heads, seq_len, seq_len).astype(np.float32)

        # Bidirectional
        rel_emb_bi = RelativePositionalEmbedding(
            num_heads=num_heads, num_buckets=num_buckets, bidirectional=True
        )
        rel_emb_bi.embedding.weight = mx.array(embedding_weight)
        out_bi = rel_emb_bi(mx.array(attn_scores))

        # Unidirectional
        rel_emb_uni = RelativePositionalEmbedding(
            num_heads=num_heads, num_buckets=num_buckets, bidirectional=False
        )
        rel_emb_uni.embedding.weight = mx.array(embedding_weight)
        out_uni = rel_emb_uni(mx.array(attn_scores))

        mx.eval(out_bi, out_uni)

        out_bi_np = _to_numpy(out_bi)
        out_uni_np = _to_numpy(out_uni)

        # They should have the same shape but different values
        assert out_bi_np.shape == out_uni_np.shape
        assert not np.allclose(out_bi_np, out_uni_np), \
            "Bidirectional and unidirectional should produce different outputs"
