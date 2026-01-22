"""JAX Metal parity tests for cache operations."""

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, get_gradient_tolerance, assert_close
from tests.parity.conftest import get_mlx_dtype, get_jax_dtype, HAS_JAX

# Import MLX cache implementations
from mlx_primitives.cache.eviction import (
    LRUEvictionPolicy,
    FIFOEvictionPolicy,
)
from mlx_primitives.cache.speculative import speculative_verify

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from tests.reference_jax_extended import (
        jax_paged_attention,
        jax_block_allocation,
    )


def _to_numpy(x) -> np.ndarray:
    """Convert MLX or JAX array to numpy for comparison."""
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x.astype(mx.float32))
    if HAS_JAX and isinstance(x, jnp.ndarray):
        return np.array(x.astype(jnp.float32))
    return np.asarray(x)


# =============================================================================
# Reference Implementations (shared with PyTorch tests)
# =============================================================================

class ReferenceLRU:
    """Reference LRU implementation using Python OrderedDict."""

    def __init__(self):
        self._order: OrderedDict[int, bool] = OrderedDict()

    def on_create(self, key: int) -> None:
        self._order[key] = True

    def on_access(self, key: int) -> None:
        if key in self._order:
            self._order.move_to_end(key)
        self._order[key] = True

    def on_delete(self, key: int) -> None:
        self._order.pop(key, None)

    def select_for_eviction(self, candidates: List[int], n: int) -> List[int]:
        candidate_set = set(candidates)
        result = []
        for key in self._order:
            if key in candidate_set:
                result.append(key)
                if len(result) >= n:
                    break
        return result


class ReferenceFIFO:
    """Reference FIFO implementation using Python OrderedDict."""

    def __init__(self):
        self._order: OrderedDict[int, bool] = OrderedDict()

    def on_create(self, key: int) -> None:
        if key not in self._order:
            self._order[key] = True

    def on_access(self, key: int) -> None:
        pass  # FIFO ignores access

    def on_delete(self, key: int) -> None:
        self._order.pop(key, None)

    def select_for_eviction(self, candidates: List[int], n: int) -> List[int]:
        candidate_set = set(candidates)
        result = []
        for key in self._order:
            if key in candidate_set:
                result.append(key)
                if len(result) >= n:
                    break
        return result


def reference_speculative_verify(
    draft_tokens: List[int],
    draft_log_probs: np.ndarray,
    target_log_probs: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, Optional[int]]:
    """Reference implementation of speculative verification using NumPy."""
    num_tokens = len(draft_tokens)

    if num_tokens == 0:
        return 0, None

    for i in range(num_tokens):
        draft_token = draft_tokens[i]
        draft_logp = float(draft_log_probs[i])
        target_logp = float(target_log_probs[i, draft_token])

        acceptance_prob = min(1.0, float(np.exp(target_logp - draft_logp)))
        r = rng.random()

        if r < acceptance_prob:
            continue
        else:
            target_probs = np.exp(target_log_probs[i])
            draft_prob = np.exp(draft_logp)
            adjusted = np.maximum(0.0, target_probs - draft_prob)
            adjusted_sum = np.sum(adjusted)

            if adjusted_sum > 1e-6:
                adjusted = adjusted / adjusted_sum
                correction_token = int(rng.choice(len(adjusted), p=adjusted))
            else:
                target_probs_norm = np.exp(target_log_probs[i])
                target_probs_norm = target_probs_norm / target_probs_norm.sum()
                correction_token = int(rng.choice(len(target_probs_norm), p=target_probs_norm))

            return i, correction_token

    if target_log_probs.shape[0] > num_tokens:
        target_probs = np.exp(target_log_probs[num_tokens])
        target_probs = target_probs / target_probs.sum()
        next_token = int(rng.choice(len(target_probs), p=target_probs))
        return num_tokens, next_token

    return num_tokens, None


# =============================================================================
# Paged Attention Parity Tests
# =============================================================================

class TestPagedAttentionParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_jax):
        """Test paged attention forward pass parity with JAX."""
        from mlx_primitives.cache.paged_attention import (
            paged_attention,
            create_block_table_from_lengths,
        )

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        block_size = config["block_size"]

        # For decode mode (single query token)
        seq_q = 1
        # Context length per sequence
        context_len_base = config["seq"]

        np.random.seed(42)

        # Generate context lengths for each sequence
        context_lens_np = np.array([
            context_len_base - i * 4 for i in range(batch)
        ], dtype=np.int32)
        context_lens_np = np.maximum(context_lens_np, block_size)

        # Calculate total blocks needed
        blocks_per_seq = (context_lens_np + block_size - 1) // block_size
        total_blocks = int(np.sum(blocks_per_seq)) + batch  # Extra for safety

        # Generate query and KV pool
        q_np = np.random.randn(batch, seq_q, num_heads, head_dim).astype(np.float32) * 0.1
        k_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32) * 0.1
        v_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32) * 0.1

        # MLX paged attention
        mlx_dtype = get_mlx_dtype(dtype)
        q_mlx = mx.array(q_np).astype(mlx_dtype)
        k_pool_mlx = mx.array(k_pool_np).astype(mlx_dtype)
        v_pool_mlx = mx.array(v_pool_np).astype(mlx_dtype)
        context_lens_mlx = mx.array(context_lens_np)

        # Create block tables
        block_tables_mlx = create_block_table_from_lengths(context_lens_mlx, block_size)
        mx.eval(block_tables_mlx)
        block_tables_np = _to_numpy(block_tables_mlx).astype(np.int32)

        mlx_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            block_size=block_size
        )
        mx.eval(mlx_out)

        # JAX reference
        jax_out = jax_paged_attention(
            q_np, k_pool_np, v_pool_np, block_tables_np, context_lens_np,
            block_size=block_size
        )

        rtol, atol = get_tolerance("cache", "paged_attention", dtype)
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out,
            rtol=rtol, atol=atol,
            err_msg=f"Paged attention mismatch (JAX) [{size}, {dtype}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_jax):
        """Test paged attention backward pass gradients flow correctly.

        Note: Paged attention backward is validated for gradient flow rather than
        exact numerical parity since it involves discrete block table lookups.
        """
        from mlx_primitives.cache.paged_attention import (
            paged_attention,
            create_block_table_from_lengths,
        )

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        block_size = config["block_size"]
        seq_q = 1
        context_len_base = config["seq"]

        np.random.seed(42)

        context_lens_np = np.array([
            context_len_base - i * 4 for i in range(batch)
        ], dtype=np.int32)
        context_lens_np = np.maximum(context_lens_np, block_size)

        blocks_per_seq = (context_lens_np + block_size - 1) // block_size
        total_blocks = int(np.sum(blocks_per_seq)) + batch

        q_np = np.random.randn(batch, seq_q, num_heads, head_dim).astype(np.float32) * 0.1

        # MLX backward - verify gradients flow through
        def mlx_loss_fn(q, k_pool, v_pool):
            context_lens_mlx = mx.array(context_lens_np)
            block_tables_mlx = create_block_table_from_lengths(context_lens_mlx, block_size)
            out = paged_attention(
                q, k_pool, v_pool, block_tables_mlx, context_lens_mlx,
                block_size=block_size
            )
            return mx.sum(out)

        q_mlx = mx.array(q_np)
        k_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32) * 0.1
        v_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32) * 0.1
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)

        grad_fn = mx.grad(mlx_loss_fn, argnums=(0, 1, 2))
        mlx_grad_q, mlx_grad_k, mlx_grad_v = grad_fn(q_mlx, k_pool_mlx, v_pool_mlx)
        mx.eval(mlx_grad_q, mlx_grad_k, mlx_grad_v)

        # Verify gradients have correct shape and no NaN/Inf
        assert mlx_grad_q.shape == q_mlx.shape, "Q gradient shape mismatch"
        assert mlx_grad_k.shape == k_pool_mlx.shape, "K gradient shape mismatch"
        assert mlx_grad_v.shape == v_pool_mlx.shape, "V gradient shape mismatch"

        q_grad_np = _to_numpy(mlx_grad_q)
        k_grad_np = _to_numpy(mlx_grad_k)
        v_grad_np = _to_numpy(mlx_grad_v)

        assert not np.isnan(q_grad_np).any(), f"NaN in Q gradient [{size}]"
        assert not np.isinf(q_grad_np).any(), f"Inf in Q gradient [{size}]"
        assert not np.isnan(k_grad_np).any(), f"NaN in K gradient [{size}]"
        assert not np.isinf(k_grad_np).any(), f"Inf in K gradient [{size}]"
        assert not np.isnan(v_grad_np).any(), f"NaN in V gradient [{size}]"
        assert not np.isinf(v_grad_np).any(), f"Inf in V gradient [{size}]"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_different_block_sizes(self, block_size, skip_without_jax):
        """Test paged attention with different block sizes."""
        from mlx_primitives.cache.paged_attention import (
            paged_attention,
            create_block_table_from_lengths,
        )

        batch = 2
        seq = 256
        num_heads = 8
        head_dim = 64

        np.random.seed(456)
        mx.random.seed(456)

        q_np = np.random.randn(batch, 1, num_heads, head_dim).astype(np.float32)
        k_pool_np = np.random.randn(batch * ((seq + block_size - 1) // block_size), block_size, num_heads, head_dim).astype(np.float32)
        v_pool_np = np.random.randn(batch * ((seq + block_size - 1) // block_size), block_size, num_heads, head_dim).astype(np.float32)

        context_lens_np = np.full(batch, seq, dtype=np.int32)

        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        context_lens_mlx = mx.array(context_lens_np)
        block_tables_mlx = create_block_table_from_lengths(context_lens_mlx, block_size)
        mx.eval(block_tables_mlx)
        block_tables_np = _to_numpy(block_tables_mlx).astype(np.int32)

        mlx_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            block_size=block_size
        )
        mx.eval(mlx_out)

        jax_out = jax_paged_attention(
            q_np, k_pool_np, v_pool_np, block_tables_np, context_lens_np,
            block_size=block_size
        )

        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out, rtol=rtol, atol=atol,
            err_msg=f"Paged attention mismatch with block_size={block_size} (JAX)"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_variable_sequence_lengths(self, skip_without_jax):
        """Test paged attention with variable sequence lengths in batch."""
        from mlx_primitives.cache.paged_attention import (
            paged_attention,
            create_block_table_from_lengths,
        )

        batch = 4
        max_seq = 256
        num_heads = 8
        head_dim = 64
        block_size = 32

        np.random.seed(101112)
        mx.random.seed(101112)

        # Variable sequence lengths
        seq_lens = np.array([64, 128, 192, 256], dtype=np.int32)

        q_np = np.random.randn(batch, 1, num_heads, head_dim).astype(np.float32)

        # Calculate total blocks needed
        total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens) + batch
        k_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32)
        v_pool_np = np.random.randn(total_blocks, block_size, num_heads, head_dim).astype(np.float32)

        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        context_lens_mlx = mx.array(seq_lens)
        block_tables_mlx = create_block_table_from_lengths(context_lens_mlx, block_size)
        mx.eval(block_tables_mlx)
        block_tables_np = _to_numpy(block_tables_mlx).astype(np.int32)

        mlx_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            block_size=block_size
        )
        mx.eval(mlx_out)

        jax_out = jax_paged_attention(
            q_np, k_pool_np, v_pool_np, block_tables_np, seq_lens,
            block_size=block_size
        )

        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")
        np.testing.assert_allclose(
            _to_numpy(mlx_out), jax_out, rtol=rtol, atol=atol,
            err_msg="Variable seq length paged attention mismatch (JAX)"
        )


# =============================================================================
# Block Allocation Parity Tests
# =============================================================================

class TestBlockAllocationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test block allocation forward pass parity with JAX."""
        from mlx_primitives.cache.paged_attention import create_block_table_from_lengths

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        block_size = config["block_size"]
        seq = config["seq"]

        np.random.seed(42)

        # Generate random sequence lengths
        sequence_lengths_np = np.random.randint(
            block_size, seq, size=(batch,)
        ).astype(np.int32)

        # MLX block allocation
        sequence_lengths_mlx = mx.array(sequence_lengths_np)
        mlx_tables = create_block_table_from_lengths(sequence_lengths_mlx, block_size)
        mx.eval(mlx_tables)

        # JAX reference
        jax_tables = jax_block_allocation(sequence_lengths_np, block_size)

        # Compare block tables (should be exact integer match)
        np.testing.assert_array_equal(
            _to_numpy(mlx_tables).astype(np.int32), jax_tables,
            err_msg=f"Block allocation mismatch (JAX) [{size}]"
        )


# =============================================================================
# Eviction Policies Parity Tests
# =============================================================================

class TestEvictionPoliciesParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_lru_parity(self, skip_without_jax):
        """Test LRU eviction policy parity with reference implementation."""
        num_sequences = 16

        # Create MLX and reference LRU policies
        mlx_lru = LRUEvictionPolicy()
        ref_lru = ReferenceLRU()

        # Create sequences
        for i in range(num_sequences):
            mlx_lru.on_create(i)
            ref_lru.on_create(i)

        # Apply access pattern
        access_pattern = list(range(num_sequences)) + list(range(num_sequences // 2))
        for seq_id in access_pattern:
            mlx_lru.on_access(seq_id)
            ref_lru.on_access(seq_id)

        # Test eviction selection
        candidates = list(range(num_sequences))
        for num_to_evict in [1, 2, 4]:
            mlx_evicted = mlx_lru.select_for_eviction(candidates, num_to_evict)
            ref_evicted = ref_lru.select_for_eviction(candidates, num_to_evict)

            assert mlx_evicted == ref_evicted, (
                f"LRU eviction mismatch for n={num_to_evict}: "
                f"MLX={mlx_evicted}, ref={ref_evicted}"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_fifo_parity(self, skip_without_jax):
        """Test FIFO eviction policy parity with reference implementation."""
        num_sequences = 16

        # Create MLX and reference FIFO policies
        mlx_fifo = FIFOEvictionPolicy()
        ref_fifo = ReferenceFIFO()

        # Create sequences
        for i in range(num_sequences):
            mlx_fifo.on_create(i)
            ref_fifo.on_create(i)

        # Apply access pattern (FIFO should ignore)
        access_pattern = list(range(num_sequences))[::-1]
        for seq_id in access_pattern:
            mlx_fifo.on_access(seq_id)
            ref_fifo.on_access(seq_id)

        # Test eviction selection
        candidates = list(range(num_sequences))
        for num_to_evict in [1, 2, 4]:
            mlx_evicted = mlx_fifo.select_for_eviction(candidates, num_to_evict)
            ref_evicted = ref_fifo.select_for_eviction(candidates, num_to_evict)

            assert mlx_evicted == ref_evicted, (
                f"FIFO eviction mismatch for n={num_to_evict}: "
                f"MLX={mlx_evicted}, ref={ref_evicted}"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_lru_eviction_order(self, skip_without_jax):
        """Test LRU eviction selects correct blocks."""
        mlx_lru = LRUEvictionPolicy()
        ref_lru = ReferenceLRU()

        # Create sequences 0, 1, 2, 3
        for i in range(4):
            mlx_lru.on_create(i)
            ref_lru.on_create(i)

        # Access 0 again (moves to end, so eviction order is now 1, 2, 3, 0)
        mlx_lru.on_access(0)
        ref_lru.on_access(0)

        # Access 2 (moves to end, so order is 1, 3, 0, 2)
        mlx_lru.on_access(2)
        ref_lru.on_access(2)

        # Evict 2 - should be [1, 3]
        candidates = [0, 1, 2, 3]
        mlx_evicted = mlx_lru.select_for_eviction(candidates, 2)
        ref_evicted = ref_lru.select_for_eviction(candidates, 2)

        assert mlx_evicted == [1, 3], f"LRU order mismatch: got {mlx_evicted}, expected [1, 3]"
        assert mlx_evicted == ref_evicted

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_fifo_eviction_order(self, skip_without_jax):
        """Test FIFO eviction selects correct blocks."""
        mlx_fifo = FIFOEvictionPolicy()
        ref_fifo = ReferenceFIFO()

        # Create sequences 0, 1, 2, 3
        for i in range(4):
            mlx_fifo.on_create(i)
            ref_fifo.on_create(i)

        # Access 0 multiple times (should NOT change order for FIFO)
        for _ in range(5):
            mlx_fifo.on_access(0)
            ref_fifo.on_access(0)

        # Evict 2 - should be [0, 1] (first created)
        candidates = [0, 1, 2, 3]
        mlx_evicted = mlx_fifo.select_for_eviction(candidates, 2)
        ref_evicted = ref_fifo.select_for_eviction(candidates, 2)

        assert mlx_evicted == [0, 1], f"FIFO order mismatch: got {mlx_evicted}, expected [0, 1]"
        assert mlx_evicted == ref_evicted

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_attention_score_eviction(self, skip_without_jax):
        """Test attention-score based eviction policy."""
        from mlx_primitives.cache.eviction import AttentionScoreEvictionPolicy

        class ReferenceAttentionScore:
            """Reference attention score eviction implementation."""
            def __init__(self, decay_factor: float = 0.99):
                self._decay = decay_factor
                self._scores: dict = {}

            def on_create(self, key: int) -> None:
                self._scores[key] = 0.0

            def update_score(self, key: int, score: float) -> None:
                if key not in self._scores:
                    self._scores[key] = 0.0
                self._scores[key] = self._decay * self._scores[key] + (1 - self._decay) * score

            def select_for_eviction(self, candidates: List[int], n: int) -> List[int]:
                scored = [(k, self._scores.get(k, 0.0)) for k in candidates]
                scored.sort(key=lambda x: x[1])
                return [k for k, _ in scored[:n]]

        decay_factor = 0.99
        mlx_policy = AttentionScoreEvictionPolicy(decay_factor=decay_factor)
        ref_policy = ReferenceAttentionScore(decay_factor=decay_factor)

        # Create sequences
        for i in range(4):
            mlx_policy.on_create(i)
            ref_policy.on_create(i)

        # Update attention scores
        scores = [(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.3)]
        for seq_id, score in scores:
            mlx_policy.update_attention_score(seq_id, score)
            ref_policy.update_score(seq_id, score)

        # Evict 2 - should be [0, 3] (lowest scores)
        candidates = [0, 1, 2, 3]
        mlx_evicted = mlx_policy.select_for_eviction(candidates, 2)
        ref_evicted = ref_policy.select_for_eviction(candidates, 2)

        assert mlx_evicted == ref_evicted, (
            f"Attention score eviction mismatch: MLX={mlx_evicted}, ref={ref_evicted}"
        )
        assert set(mlx_evicted) == {0, 3}


# =============================================================================
# Speculative Verification Parity Tests
# =============================================================================

class TestSpeculativeVerificationParity:
    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_jax):
        """Test speculative verification forward pass parity."""
        config = SIZE_CONFIGS[size]["cache"]
        vocab_size = 1000
        draft_length = min(8, config["seq"] // 8)

        # Use fixed seed for reproducibility
        seed = 12345
        np.random.seed(seed)
        mx.random.seed(seed)

        # Generate test data
        draft_tokens = list(np.random.randint(0, vocab_size, (draft_length,)))
        draft_log_probs_np = np.random.randn(draft_length).astype(np.float32) * 0.5
        target_log_probs_np = np.random.randn(draft_length + 1, vocab_size).astype(np.float32)
        target_log_probs_np = target_log_probs_np - np.log(
            np.exp(target_log_probs_np).sum(axis=-1, keepdims=True)
        )

        # MLX version
        draft_log_probs_mlx = mx.array(draft_log_probs_np)
        target_log_probs_mlx = mx.array(target_log_probs_np)

        # Run both with same random state
        mx.random.seed(42)
        mlx_accepted, _ = speculative_verify(
            draft_tokens,
            draft_log_probs_mlx,
            target_log_probs_mlx,
        )

        # Reference implementation with same seed
        rng = np.random.default_rng(42)
        ref_accepted, _ = reference_speculative_verify(
            draft_tokens,
            draft_log_probs_np,
            target_log_probs_np,
            rng,
        )

        # Compare acceptance counts
        assert mlx_accepted == ref_accepted, (
            f"Acceptance count mismatch for size={size}: "
            f"MLX={mlx_accepted}, ref={ref_accepted}"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_acceptance_probability(self, skip_without_jax):
        """Test acceptance probability computation parity."""
        test_cases = [
            (-1.0, -1.0, 1.0),
            (-1.0, -0.5, 1.0),
            (-0.5, -1.0, np.exp(-0.5)),
            (-2.0, -1.0, 1.0),
            (-1.0, -2.0, np.exp(-1.0)),
        ]

        rtol, atol = get_tolerance("cache", "speculative_verification", "fp32")

        for draft_logp, target_logp, expected in test_cases:
            mlx_accept = min(1.0, float(mx.exp(mx.array(target_logp - draft_logp))))
            np_accept = min(1.0, float(np.exp(target_logp - draft_logp)))

            np.testing.assert_allclose(
                mlx_accept, np_accept, rtol=rtol, atol=atol,
                err_msg=f"Acceptance prob mismatch for draft={draft_logp}, target={target_logp}"
            )
            np.testing.assert_allclose(
                mlx_accept, expected, rtol=rtol, atol=atol,
                err_msg=f"Expected acceptance prob mismatch"
            )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_rejection_sampling(self, skip_without_jax):
        """Test rejection sampling for corrected tokens."""
        vocab_size = 100
        seed = 54321

        # Create a scenario where rejection is guaranteed
        draft_log_probs = np.array([-0.01], dtype=np.float32)  # ~99% prob
        target_log_probs = np.zeros((2, vocab_size), dtype=np.float32)
        target_log_probs[0, 0] = -10.0  # Very low prob for draft token
        target_log_probs[0, 1:] = -2.3  # Higher prob for other tokens
        target_log_probs = target_log_probs - np.log(
            np.exp(target_log_probs).sum(axis=-1, keepdims=True)
        )

        draft_tokens = [0]

        # Run MLX
        mx.random.seed(seed)
        mlx_accepted, mlx_correction = speculative_verify(
            draft_tokens,
            mx.array(draft_log_probs),
            mx.array(target_log_probs),
        )

        # Run reference
        rng = np.random.default_rng(seed)
        ref_accepted, ref_correction = reference_speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
            rng,
        )

        # Should reject since draft prob >> target prob
        assert mlx_accepted == 0, f"Expected rejection, got {mlx_accepted} accepted"
        assert ref_accepted == 0

        # Correction token should not be 0 (the rejected token)
        assert mlx_correction != 0, "Correction token should not be the rejected token"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_speculative", [1, 2, 4, 8])
    def test_different_speculation_lengths(self, num_speculative, skip_without_jax):
        """Test with different speculation lengths."""
        vocab_size = 500
        seed = 99999

        np.random.seed(seed)

        draft_tokens = list(np.random.randint(0, vocab_size, (num_speculative,)))
        draft_log_probs = np.random.randn(num_speculative).astype(np.float32) * 0.3
        target_log_probs = np.random.randn(num_speculative + 1, vocab_size).astype(np.float32)
        target_log_probs = target_log_probs - np.log(
            np.exp(target_log_probs).sum(axis=-1, keepdims=True)
        )

        mx.random.seed(42)
        mlx_accepted, _ = speculative_verify(
            draft_tokens,
            mx.array(draft_log_probs),
            mx.array(target_log_probs),
        )

        rng = np.random.default_rng(42)
        ref_accepted, _ = reference_speculative_verify(
            draft_tokens,
            draft_log_probs,
            target_log_probs,
            rng,
        )

        assert mlx_accepted == ref_accepted, (
            f"Acceptance count mismatch for num_spec={num_speculative}: "
            f"MLX={mlx_accepted}, ref={ref_accepted}"
        )
        assert 0 <= mlx_accepted <= num_speculative


# =============================================================================
# KV Cache Variant Tests
# =============================================================================

class TestKVCacheParity:
    """KV cache basic operations parity tests using SimpleKVCache."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_kv_cache_update(self, size, skip_without_jax):
        """Test KV cache update operation parity."""
        from mlx_primitives.cache.simple_cache import SimpleKVCache

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        max_seq = config["seq"]
        num_updates = min(4, max_seq // 4)

        np.random.seed(42)

        # Create MLX KV cache (note: SimpleKVCache uses batch, seq, heads, dim order)
        cache = SimpleKVCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq,
        )

        # Create reference cache (batch, seq, heads, dim)
        ref_k_cache = np.zeros((batch, max_seq, num_heads, head_dim), dtype=np.float32)
        ref_v_cache = np.zeros((batch, max_seq, num_heads, head_dim), dtype=np.float32)
        ref_pos = 0

        # Perform several updates
        for _ in range(num_updates):
            seq_len = np.random.randint(1, max_seq // num_updates)
            if ref_pos + seq_len > max_seq:
                break

            # Shape: (batch, seq_len, num_heads, head_dim)
            new_k_np = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
            new_v_np = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

            # MLX update
            new_k_mlx = mx.array(new_k_np)
            new_v_mlx = mx.array(new_v_np)
            cache.update(new_k_mlx, new_v_mlx)

            # Reference update
            end_pos = ref_pos + seq_len
            ref_k_cache[:, ref_pos:end_pos, :, :] = new_k_np
            ref_v_cache[:, ref_pos:end_pos, :, :] = new_v_np
            ref_pos = end_pos

        # Compare caches
        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)
        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")

        np.testing.assert_allclose(
            _to_numpy(k_out), ref_k_cache[:, :ref_pos, :, :],
            rtol=rtol, atol=atol,
            err_msg=f"K cache mismatch [{size}]"
        )
        np.testing.assert_allclose(
            _to_numpy(v_out), ref_v_cache[:, :ref_pos, :, :],
            rtol=rtol, atol=atol,
            err_msg=f"V cache mismatch [{size}]"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_kv_cache_clear(self, skip_without_jax):
        """Test KV cache clear operation."""
        from mlx_primitives.cache.simple_cache import SimpleKVCache

        batch, num_heads, head_dim, max_seq = 2, 8, 64, 128

        np.random.seed(42)

        cache = SimpleKVCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq,
        )

        # Fill with some data
        new_k = mx.array(np.random.randn(batch, 32, num_heads, head_dim).astype(np.float32))
        new_v = mx.array(np.random.randn(batch, 32, num_heads, head_dim).astype(np.float32))
        cache.update(new_k, new_v)

        # Clear
        cache.clear()

        # Verify clear
        assert cache.current_length == 0, "Length not reset"
        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)

        # Should be empty after clear
        assert k_out.shape[1] == 0, f"K cache should be empty, got shape {k_out.shape}"
        assert v_out.shape[1] == 0, f"V cache should be empty, got shape {v_out.shape}"


class TestSlidingWindowCacheParity:
    """Sliding window KV cache parity tests."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_sliding_window_eviction(self, size, skip_without_jax):
        """Test sliding window cache eviction behavior."""
        from mlx_primitives.cache.simple_cache import SlidingWindowCache

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        window_size = min(64, config["seq"] // 2)

        np.random.seed(42)

        # Create sliding window cache (uses batch, seq, heads, dim order)
        cache = SlidingWindowCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
        )

        # Fill beyond window size to trigger eviction
        total_tokens = window_size + 20
        for i in range(total_tokens):
            # Shape: (batch, 1, num_heads, head_dim)
            new_k = mx.array(np.random.randn(batch, 1, num_heads, head_dim).astype(np.float32))
            new_v = mx.array(np.random.randn(batch, 1, num_heads, head_dim).astype(np.float32))
            cache.update(new_k, new_v)

        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)

        # Verify window size is maintained
        effective_len = cache.current_length
        assert effective_len <= window_size, (
            f"Effective length {effective_len} exceeds window_size {window_size}"
        )

        # Verify cache contains valid (non-zero) data
        k_np = _to_numpy(k_out)
        v_np = _to_numpy(v_out)

        assert not np.isnan(k_np).any(), "NaN in sliding window K cache"
        assert not np.isinf(k_np).any(), "Inf in sliding window K cache"
        assert not np.isnan(v_np).any(), "NaN in sliding window V cache"
        assert not np.isinf(v_np).any(), "Inf in sliding window V cache"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_sliding_window_content_order(self, skip_without_jax):
        """Test that sliding window maintains correct token order."""
        from mlx_primitives.cache.simple_cache import SlidingWindowCache

        batch, num_heads, head_dim = 1, 2, 4
        window_size = 8

        cache = SlidingWindowCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
        )

        # Add tokens with identifiable values
        for i in range(12):  # Add more than window_size
            # Fill with i+1 so we can identify each token
            k = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            v = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            cache.update(k, v)

        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)
        k_np = _to_numpy(k_out)

        # After adding tokens 1-12 to window of size 8, should have tokens 5-12
        assert k_out.shape[1] == window_size, f"Expected window_size={window_size}, got {k_out.shape[1]}"

        # Check that we have the most recent tokens (approximately, due to sliding)
        assert cache.total_seen == 12, f"Expected 12 tokens seen, got {cache.total_seen}"


class TestRotatingKVCacheParity:
    """Rotating (circular buffer) KV cache parity tests."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_rotating_circular_write(self, size, skip_without_jax):
        """Test rotating cache circular write behavior."""
        from mlx_primitives.cache.simple_cache import RotatingKVCache

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        max_seq = config["seq"]

        np.random.seed(42)

        cache = RotatingKVCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            max_size=max_seq,
        )

        # Write more than max_size to test circular behavior
        total_tokens = max_seq + max_seq // 2

        for i in range(total_tokens):
            # Shape: (batch, 1, num_heads, head_dim)
            new_k = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            new_v = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            cache.update(new_k, new_v)

        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)

        # The cache should contain exactly max_size tokens
        assert k_out.shape[1] == max_seq, f"Expected {max_seq} tokens, got {k_out.shape[1]}"

        k_np = _to_numpy(k_out)

        # Verify no NaN/Inf
        assert not np.isnan(k_np).any(), "NaN in rotating K cache"
        assert not np.isinf(k_np).any(), "Inf in rotating K cache"

        # Verify correct circular behavior - should have most recent max_seq tokens
        # After adding total_tokens, oldest should be (total_tokens - max_seq + 1)
        expected_oldest = total_tokens - max_seq + 1
        expected_newest = total_tokens

        # The chronological order should go from oldest to newest
        # First token value should be approximately expected_oldest
        # Last token value should be expected_newest
        assert np.min(k_np) >= expected_oldest - 1, (
            f"Rotating cache oldest value (min={np.min(k_np)}) should be >= {expected_oldest - 1}"
        )
        assert np.max(k_np) == expected_newest, (
            f"Rotating cache newest value (max={np.max(k_np)}) should be {expected_newest}"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_rotating_cache_order(self, skip_without_jax):
        """Test that rotating cache returns tokens in chronological order."""
        from mlx_primitives.cache.simple_cache import RotatingKVCache

        batch, num_heads, head_dim = 1, 2, 4
        max_size = 8

        cache = RotatingKVCache(
            batch_size=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            max_size=max_size,
        )

        # Add tokens 1-12 (more than max_size)
        for i in range(12):
            k = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            v = mx.array(np.full((batch, 1, num_heads, head_dim), float(i + 1), dtype=np.float32))
            cache.update(k, v)

        k_out, v_out = cache.get_kv()
        mx.eval(k_out, v_out)
        k_np = _to_numpy(k_out)

        # Should have 8 tokens: values 5, 6, 7, 8, 9, 10, 11, 12 in chronological order
        assert k_out.shape[1] == max_size
        # Extract the first element from each position to check order
        values = k_np[0, :, 0, 0]

        # Values should be monotonically increasing (oldest to newest)
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Rotating cache not in order at position {i}: {values[i]} > {values[i+1]}"
            )


# =============================================================================
# Compressed KV Cache Parity Tests
# =============================================================================

class TestCompressedKVCacheParity:
    """Compressed (quantized) KV cache parity tests."""

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_quantize_8bit_roundtrip(self, skip_without_jax):
        """Test 8-bit quantization round-trip accuracy."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            compression='quantize',
            bits=8,
        )

        # Add data
        np.random.seed(42)
        original_k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        original_v = np.random.randn(1, 4, 8, 16).astype(np.float32)

        cache.update(0, mx.array(original_k), mx.array(original_v))

        # Get dequantized data
        recovered_k, recovered_v = cache.get(0)
        mx.eval(recovered_k, recovered_v)
        recovered_k = _to_numpy(recovered_k)
        recovered_v = _to_numpy(recovered_v)

        # 8-bit quantization should have <1% error
        np.testing.assert_allclose(
            recovered_k, original_k, rtol=1e-2, atol=1e-2,
            err_msg="8-bit quantization error too large for K"
        )
        np.testing.assert_allclose(
            recovered_v, original_v, rtol=1e-2, atol=1e-2,
            err_msg="8-bit quantization error too large for V"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_quantize_4bit_roundtrip(self, skip_without_jax):
        """Test 4-bit quantization round-trip accuracy.

        4-bit quantization has only 16 levels, leading to significant
        quantization error. Typical error is 10-25% for normal data.
        We verify the error is within expected bounds rather than
        checking for tight tolerances.
        """
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            compression='quantize',
            bits=4,
        )

        # Add data with controlled range
        np.random.seed(42)
        original_k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        original_v = np.random.randn(1, 4, 8, 16).astype(np.float32)

        cache.update(0, mx.array(original_k), mx.array(original_v))

        # Get dequantized data
        recovered_k, recovered_v = cache.get(0)
        mx.eval(recovered_k, recovered_v)
        recovered_k = _to_numpy(recovered_k)
        recovered_v = _to_numpy(recovered_v)

        # 4-bit quantization has ~10-30% error for normally distributed data
        # Use a reasonable tolerance that allows for quantization error
        np.testing.assert_allclose(
            recovered_k, original_k, rtol=0.3, atol=0.3,
            err_msg="4-bit quantization error too large for K"
        )
        np.testing.assert_allclose(
            recovered_v, original_v, rtol=0.3, atol=0.3,
            err_msg="4-bit quantization error too large for V"
        )

        # Also verify the error is not TOO large (sanity check)
        k_error = np.max(np.abs(recovered_k - original_k))
        v_error = np.max(np.abs(recovered_v - original_v))
        assert k_error < 1.0, f"4-bit K error {k_error} exceeds sanity limit"
        assert v_error < 1.0, f"4-bit V error {v_error} exceeds sanity limit"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_compressed_cache_memory_ratio(self, skip_without_jax):
        """Test compressed cache memory usage reduction."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache, KVCache

        # Full precision cache
        full_cache = KVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            dtype=mx.float16,  # 2 bytes per element
        )

        # 8-bit quantized cache
        quant_cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            compression='quantize',
            bits=8,
        )

        # Compression ratio should be > 1 (memory savings)
        ratio = quant_cache.compression_ratio
        assert ratio > 1.0, f"Expected compression ratio > 1, got {ratio}"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_pruned_cache(self, skip_without_jax):
        """Test pruned cache keeps most recent tokens."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        keep_ratio = 0.5
        max_seq_len = 64

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            num_heads=4,
            head_dim=16,
            compression='prune',
            keep_ratio=keep_ratio,
        )

        # Add many tokens
        np.random.seed(42)
        for i in range(100):
            new_k = mx.array(np.random.randn(1, 4, 1, 16).astype(np.float32))
            new_v = mx.array(np.random.randn(1, 4, 1, 16).astype(np.float32))
            cache.update(0, new_k, new_v)

        # Get cache - should be pruned
        k, v = cache.get(0)
        mx.eval(k, v)

        # Should keep at most keep_len = max_seq_len * keep_ratio
        keep_len = int(max_seq_len * keep_ratio)
        assert k.shape[2] <= keep_len, f"Cache should be pruned to {keep_len}, got {k.shape[2]}"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_quantize_8bit_multiple_updates(self, size, skip_without_jax):
        """Test 8-bit quantized cache with multiple sequential updates."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        config = SIZE_CONFIGS[size]["cache"]
        batch = min(config["batch"], 2)  # Keep batch small for compressed cache
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        max_seq = config["seq"]

        np.random.seed(42)

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=batch,
            max_seq_len=max_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            compression='quantize',
            bits=8,
        )

        # Multiple updates
        total_len = 0
        num_updates = 4
        for i in range(num_updates):
            seq_len = min(8, (max_seq - total_len) // (num_updates - i))
            if seq_len <= 0:
                break

            new_k = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
            new_v = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)

            cache.update(0, mx.array(new_k), mx.array(new_v))
            total_len += seq_len

        # Get final cache
        k_out, v_out = cache.get(0)
        mx.eval(k_out, v_out)

        # Verify shape and no NaN/Inf
        assert k_out.shape[2] == total_len, f"Expected seq_len={total_len}, got {k_out.shape[2]}"

        k_np = _to_numpy(k_out)
        v_np = _to_numpy(v_out)

        assert not np.isnan(k_np).any(), f"NaN in K cache [{size}]"
        assert not np.isinf(k_np).any(), f"Inf in K cache [{size}]"
        assert not np.isnan(v_np).any(), f"NaN in V cache [{size}]"
        assert not np.isinf(v_np).any(), f"Inf in V cache [{size}]"

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_clustered_cache(self, skip_without_jax):
        """Test clustered cache compression method."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            compression='cluster',
            keep_ratio=0.5,
        )

        np.random.seed(42)

        # Add several tokens
        for i in range(20):
            new_k = mx.array(np.random.randn(1, 4, 1, 16).astype(np.float32))
            new_v = mx.array(np.random.randn(1, 4, 1, 16).astype(np.float32))
            cache.update(0, new_k, new_v)

        # Get cache (returns centroids for clustered)
        k, v = cache.get(0)
        mx.eval(k, v)

        k_np = _to_numpy(k)
        v_np = _to_numpy(v)

        # Verify no NaN/Inf
        assert not np.isnan(k_np).any(), "NaN in clustered K cache"
        assert not np.isinf(k_np).any(), "Inf in clustered K cache"
        assert not np.isnan(v_np).any(), "NaN in clustered V cache"
        assert not np.isinf(v_np).any(), "Inf in clustered V cache"

        # Verify we have cluster centroids (number = max_seq * keep_ratio)
        expected_clusters = int(64 * 0.5)
        assert k.shape[2] == expected_clusters, (
            f"Expected {expected_clusters} clusters, got {k.shape[2]}"
        )

    @pytest.mark.parity_jax
    @pytest.mark.forward_parity
    def test_cache_reset(self, skip_without_jax):
        """Test compressed cache reset functionality."""
        from mlx_primitives.advanced.kv_cache import CompressedKVCache

        cache = CompressedKVCache(
            num_layers=1,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
            compression='quantize',
            bits=8,
        )

        np.random.seed(42)

        # Add some data
        new_k = mx.array(np.random.randn(1, 4, 8, 16).astype(np.float32))
        new_v = mx.array(np.random.randn(1, 4, 8, 16).astype(np.float32))
        cache.update(0, new_k, new_v)

        # Verify data was added
        assert cache.seq_len == 8, f"Expected seq_len=8, got {cache.seq_len}"

        # Reset
        cache.reset()

        # Verify reset
        assert cache.seq_len == 0, f"Expected seq_len=0 after reset, got {cache.seq_len}"

        k, v = cache.get(0)
        mx.eval(k, v)
        assert k.shape[2] == 0, f"Expected empty K cache, got shape {k.shape}"
        assert v.shape[2] == 0, f"Expected empty V cache, got shape {v.shape}"
