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
