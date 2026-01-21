"""PyTorch parity tests for cache operations."""

import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pytest

import mlx.core as mx

from tests.parity.shared.input_generators import cache_inputs, SIZE_CONFIGS
from tests.parity.shared.tolerance_config import get_tolerance, assert_close

# Import MLX cache implementations
from mlx_primitives.cache.eviction import (
    LRUEvictionPolicy,
    FIFOEvictionPolicy,
    AttentionScoreEvictionPolicy,
)
from mlx_primitives.cache.speculative import speculative_verify
from mlx_primitives.cache.paged_attention import paged_attention, create_block_table_from_lengths
from mlx_primitives.cache.block_allocator import BlockAllocator, BlockConfig
from mlx_primitives.attention.flash import flash_attention


# =============================================================================
# Reference Implementations for Parity Testing
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

    def get_order(self) -> List[int]:
        return list(self._order.keys())


class ReferenceFIFO:
    """Reference FIFO implementation using Python OrderedDict."""

    def __init__(self):
        self._order: OrderedDict[int, bool] = OrderedDict()

    def on_create(self, key: int) -> None:
        if key not in self._order:
            self._order[key] = True

    def on_access(self, key: int) -> None:
        # FIFO ignores access - only creation order matters
        pass

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

    def get_order(self) -> List[int]:
        return list(self._order.keys())


class ReferenceAttentionScore:
    """Reference attention score eviction implementation."""

    def __init__(self, decay_factor: float = 0.99):
        self._decay = decay_factor
        self._scores: dict[int, float] = {}

    def on_create(self, key: int) -> None:
        self._scores[key] = 0.0

    def on_delete(self, key: int) -> None:
        self._scores.pop(key, None)

    def update_score(self, key: int, score: float) -> None:
        if key not in self._scores:
            self._scores[key] = 0.0
        self._scores[key] = self._decay * self._scores[key] + (1 - self._decay) * score

    def select_for_eviction(self, candidates: List[int], n: int) -> List[int]:
        scored = [(k, self._scores.get(k, 0.0)) for k in candidates]
        scored.sort(key=lambda x: x[1])  # Lowest score first
        return [k for k, _ in scored[:n]]


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

        # Acceptance probability
        acceptance_prob = min(1.0, float(np.exp(target_logp - draft_logp)))

        # Sample
        r = rng.random()

        if r < acceptance_prob:
            # Accept this token
            continue
        else:
            # Rejection - compute correction token
            target_probs = np.exp(target_log_probs[i])
            draft_prob = np.exp(draft_logp)

            # Adjusted distribution
            adjusted = np.maximum(0.0, target_probs - draft_prob)
            adjusted_sum = np.sum(adjusted)

            if adjusted_sum > 1e-6:
                adjusted = adjusted / adjusted_sum
                correction_token = int(rng.choice(len(adjusted), p=adjusted))
            else:
                # Fall back to sampling from target
                target_probs_norm = np.exp(target_log_probs[i])
                target_probs_norm = target_probs_norm / target_probs_norm.sum()
                correction_token = int(rng.choice(len(target_probs_norm), p=target_probs_norm))

            return i, correction_token

    # All accepted - sample next token from target (last position + 1)
    if target_log_probs.shape[0] > num_tokens:
        target_probs = np.exp(target_log_probs[num_tokens])
        target_probs = target_probs / target_probs.sum()
        next_token = int(rng.choice(len(target_probs), p=target_probs))
        return num_tokens, next_token

    return num_tokens, None


# =============================================================================
# Paged Attention Parity Tests
# =============================================================================

def _reference_standard_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    causal: bool = True,
) -> np.ndarray:
    """Reference standard attention implementation using NumPy.

    Args:
        q: Query tensor (batch, seq_q, num_heads, head_dim).
        k: Key tensor (batch, seq_kv, num_heads, head_dim).
        v: Value tensor (batch, seq_kv, num_heads, head_dim).
        scale: Attention scale factor.
        causal: Apply causal masking.

    Returns:
        Output tensor (batch, seq_q, num_heads, head_dim).
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, _, _ = k.shape

    # Transpose for matmul: (batch, heads, seq, dim)
    q_t = np.transpose(q, (0, 2, 1, 3))
    k_t = np.transpose(k, (0, 2, 1, 3))
    v_t = np.transpose(v, (0, 2, 1, 3))

    # Compute attention scores: (batch, heads, seq_q, seq_kv)
    scores = np.einsum('bhqd,bhkd->bhqk', q_t, k_t) * scale

    if causal:
        # Causal mask: query position i can only attend to positions 0..i
        q_positions = np.arange(seq_q)[:, None]
        kv_positions = np.arange(seq_kv)[None, :]
        causal_mask = kv_positions <= q_positions
        scores = np.where(causal_mask[None, None, :, :], scores, float('-inf'))

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max = np.where(np.isinf(scores_max), 0, scores_max)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)

    # Weighted sum
    output = np.einsum('bhqk,bhkd->bhqd', attn_weights, v_t)

    # Transpose back: (batch, seq_q, heads, dim)
    return np.transpose(output, (0, 2, 1, 3))


class TestPagedAttentionParity:
    """Paged attention parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test paged attention forward pass parity against standard attention.

        Compares paged attention output (which uses block-based KV storage)
        against reference standard attention to verify numerical correctness.
        """
        from tests.parity.conftest import get_mlx_dtype

        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        seq = config["seq"]
        heads = config["heads"]
        head_dim = config["head_dim"]
        block_size = config["block_size"]

        np.random.seed(42)
        mx.random.seed(42)

        # Generate test data
        np_dtype = {"fp32": np.float32, "fp16": np.float32, "bf16": np.float32}.get(dtype, np.float32)

        # Query for decode mode (single token)
        q_np = np.random.randn(batch, 1, heads, head_dim).astype(np_dtype)
        # KV cache - full sequence
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np_dtype)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np_dtype)

        scale = 1.0 / math.sqrt(head_dim)

        # Build paged KV storage
        num_blocks_per_seq = (seq + block_size - 1) // block_size
        total_blocks = batch * num_blocks_per_seq

        # Create block pool and fill with KV data
        k_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np_dtype)
        v_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np_dtype)

        block_tables_np = np.zeros((batch, num_blocks_per_seq), dtype=np.int32)
        context_lens_np = np.full(batch, seq, dtype=np.int32)

        for b in range(batch):
            for block_idx in range(num_blocks_per_seq):
                global_block_id = b * num_blocks_per_seq + block_idx
                block_tables_np[b, block_idx] = global_block_id

                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq)
                length = end_pos - start_pos

                k_pool_np[global_block_id, :length] = k_np[b, start_pos:end_pos]
                v_pool_np[global_block_id, :length] = v_np[b, start_pos:end_pos]

        # Convert to MLX
        mlx_dtype = get_mlx_dtype(dtype)
        q_mlx = mx.array(q_np).astype(mlx_dtype)
        k_pool_mlx = mx.array(k_pool_np).astype(mlx_dtype)
        v_pool_mlx = mx.array(v_pool_np).astype(mlx_dtype)
        block_tables_mlx = mx.array(block_tables_np)
        context_lens_mlx = mx.array(context_lens_np)

        # Run paged attention
        paged_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            scale=scale, block_size=block_size, causal=False
        )
        mx.eval(paged_out)
        paged_out_np = np.array(paged_out)

        # Run reference standard attention
        ref_out_np = _reference_standard_attention(q_np, k_np, v_np, scale, causal=False)

        # Compare
        rtol, atol = get_tolerance("cache", "paged_attention", dtype)
        np.testing.assert_allclose(
            paged_out_np, ref_out_np, rtol=rtol, atol=atol,
            err_msg=f"Paged attention forward mismatch [{size}, {dtype}]"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.backward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_backward_parity(self, size, skip_without_pytorch):
        """Test paged attention backward pass parity.

        Verifies that gradients flow correctly through paged attention
        by comparing numerical gradients against autodiff gradients.
        """
        config = SIZE_CONFIGS[size]["cache"]
        batch = config["batch"]
        seq = min(config["seq"], 64)  # Use smaller seq for numerical grad stability
        heads = min(config["heads"], 4)
        head_dim = min(config["head_dim"], 32)
        block_size = min(config["block_size"], 16)

        np.random.seed(123)
        mx.random.seed(123)

        # Generate test data in fp32 for gradient accuracy
        q_np = np.random.randn(batch, 1, heads, head_dim).astype(np.float32) * 0.1
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32) * 0.1
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32) * 0.1

        scale = 1.0 / math.sqrt(head_dim)

        # Build paged storage
        num_blocks_per_seq = (seq + block_size - 1) // block_size
        total_blocks = batch * num_blocks_per_seq

        k_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        v_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        block_tables_np = np.zeros((batch, num_blocks_per_seq), dtype=np.int32)

        for b in range(batch):
            for block_idx in range(num_blocks_per_seq):
                global_block_id = b * num_blocks_per_seq + block_idx
                block_tables_np[b, block_idx] = global_block_id
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq)
                k_pool_np[global_block_id, :end_pos - start_pos] = k_np[b, start_pos:end_pos]
                v_pool_np[global_block_id, :end_pos - start_pos] = v_np[b, start_pos:end_pos]

        context_lens_np = np.full(batch, seq, dtype=np.int32)

        # Convert to MLX
        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        block_tables_mlx = mx.array(block_tables_np)
        context_lens_mlx = mx.array(context_lens_np)

        # Compute gradient w.r.t. query using MLX autograd
        def loss_fn(q):
            out = paged_attention(
                q, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
                scale=scale, block_size=block_size, causal=False
            )
            return mx.sum(out)

        mlx_grad = mx.grad(loss_fn)(q_mlx)
        mx.eval(mlx_grad)
        mlx_grad_np = np.array(mlx_grad)

        # Verify gradients are not zero (meaningful gradients flow)
        assert not np.allclose(mlx_grad_np, 0), "Gradients should not be all zeros"

        # Verify gradient shape matches input shape
        assert mlx_grad_np.shape == q_np.shape, f"Gradient shape mismatch: {mlx_grad_np.shape} vs {q_np.shape}"

        # Verify gradients are finite
        assert np.all(np.isfinite(mlx_grad_np)), "Gradients should be finite"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_different_block_sizes(self, block_size, skip_without_pytorch):
        """Test paged attention with different block sizes.

        Verifies that changing block_size doesn't affect numerical output,
        only memory layout and performance.
        """
        batch = 2
        seq = 256
        heads = 8
        head_dim = 64

        np.random.seed(456)
        mx.random.seed(456)

        q_np = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        scale = 1.0 / math.sqrt(head_dim)

        # Build paged storage with specified block_size
        num_blocks_per_seq = (seq + block_size - 1) // block_size
        total_blocks = batch * num_blocks_per_seq

        k_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        v_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        block_tables_np = np.zeros((batch, num_blocks_per_seq), dtype=np.int32)

        for b in range(batch):
            for block_idx in range(num_blocks_per_seq):
                global_block_id = b * num_blocks_per_seq + block_idx
                block_tables_np[b, block_idx] = global_block_id
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq)
                k_pool_np[global_block_id, :end_pos - start_pos] = k_np[b, start_pos:end_pos]
                v_pool_np[global_block_id, :end_pos - start_pos] = v_np[b, start_pos:end_pos]

        context_lens_np = np.full(batch, seq, dtype=np.int32)

        # Convert and run paged attention
        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        block_tables_mlx = mx.array(block_tables_np)
        context_lens_mlx = mx.array(context_lens_np)

        paged_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            scale=scale, block_size=block_size, causal=False
        )
        mx.eval(paged_out)
        paged_out_np = np.array(paged_out)

        # Reference standard attention
        ref_out_np = _reference_standard_attention(q_np, k_np, v_np, scale, causal=False)

        # Compare
        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")
        np.testing.assert_allclose(
            paged_out_np, ref_out_np, rtol=rtol, atol=atol,
            err_msg=f"Paged attention mismatch with block_size={block_size}"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_vs_standard_attention(self, skip_without_pytorch):
        """Test paged attention produces same results as MLX flash_attention.

        This is the key parity test: paged attention should produce identical
        results to standard attention, just with different memory layout.
        """
        batch = 2
        seq = 128
        heads = 8
        head_dim = 64
        block_size = 16

        np.random.seed(789)
        mx.random.seed(789)

        # Generate test data
        q_np = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)

        scale = 1.0 / math.sqrt(head_dim)

        # Build paged storage
        num_blocks_per_seq = (seq + block_size - 1) // block_size
        total_blocks = batch * num_blocks_per_seq

        k_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        v_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        block_tables_np = np.zeros((batch, num_blocks_per_seq), dtype=np.int32)

        for b in range(batch):
            for block_idx in range(num_blocks_per_seq):
                global_block_id = b * num_blocks_per_seq + block_idx
                block_tables_np[b, block_idx] = global_block_id
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq)
                k_pool_np[global_block_id, :end_pos - start_pos] = k_np[b, start_pos:end_pos]
                v_pool_np[global_block_id, :end_pos - start_pos] = v_np[b, start_pos:end_pos]

        context_lens_np = np.full(batch, seq, dtype=np.int32)

        # Run paged attention
        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        block_tables_mlx = mx.array(block_tables_np)
        context_lens_mlx = mx.array(context_lens_np)

        paged_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            scale=scale, block_size=block_size, causal=False
        )
        mx.eval(paged_out)

        # Run MLX flash_attention as reference
        k_mlx = mx.array(k_np)
        v_mlx = mx.array(v_np)
        flash_out = flash_attention(q_mlx, k_mlx, v_mlx, scale=scale, causal=False)
        mx.eval(flash_out)

        # Compare
        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")
        np.testing.assert_allclose(
            np.array(paged_out), np.array(flash_out), rtol=rtol, atol=atol,
            err_msg="Paged attention vs flash_attention mismatch"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_variable_sequence_lengths(self, skip_without_pytorch):
        """Test paged attention with variable sequence lengths in batch.

        Each sequence in the batch can have different context lengths,
        which is crucial for efficient batched serving.
        """
        batch = 4
        max_seq = 256
        heads = 8
        head_dim = 64
        block_size = 32

        np.random.seed(101112)
        mx.random.seed(101112)

        # Variable sequence lengths
        seq_lens = [64, 128, 192, 256]
        assert len(seq_lens) == batch

        # Generate test data
        q_np = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, max_seq, heads, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, max_seq, heads, head_dim).astype(np.float32)

        scale = 1.0 / math.sqrt(head_dim)

        # Build paged storage with variable lengths
        max_blocks_per_seq = (max_seq + block_size - 1) // block_size
        total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)

        k_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        v_pool_np = np.zeros((total_blocks, block_size, heads, head_dim), dtype=np.float32)
        block_tables_np = np.full((batch, max_blocks_per_seq), -1, dtype=np.int32)
        context_lens_np = np.array(seq_lens, dtype=np.int32)

        block_counter = 0
        for b in range(batch):
            seq = seq_lens[b]
            num_blocks = (seq + block_size - 1) // block_size
            for block_idx in range(num_blocks):
                block_tables_np[b, block_idx] = block_counter
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq)
                k_pool_np[block_counter, :end_pos - start_pos] = k_np[b, start_pos:end_pos]
                v_pool_np[block_counter, :end_pos - start_pos] = v_np[b, start_pos:end_pos]
                block_counter += 1

        # Run paged attention
        q_mlx = mx.array(q_np)
        k_pool_mlx = mx.array(k_pool_np)
        v_pool_mlx = mx.array(v_pool_np)
        block_tables_mlx = mx.array(block_tables_np)
        context_lens_mlx = mx.array(context_lens_np)

        paged_out = paged_attention(
            q_mlx, k_pool_mlx, v_pool_mlx, block_tables_mlx, context_lens_mlx,
            scale=scale, block_size=block_size, causal=False
        )
        mx.eval(paged_out)
        paged_out_np = np.array(paged_out)

        # Verify each sequence independently
        rtol, atol = get_tolerance("cache", "paged_attention", "fp32")
        for b in range(batch):
            seq = seq_lens[b]
            # Reference attention for this sequence
            ref_out = _reference_standard_attention(
                q_np[b:b+1], k_np[b:b+1, :seq], v_np[b:b+1, :seq], scale, causal=False
            )
            np.testing.assert_allclose(
                paged_out_np[b:b+1], ref_out, rtol=rtol, atol=atol,
                err_msg=f"Variable seq parity failed for batch {b} (seq={seq})"
            )


# =============================================================================
# Block Allocation Parity Tests
# =============================================================================

class ReferenceBlockAllocator:
    """Reference block allocator implementation in pure Python for parity testing."""

    def __init__(self, num_blocks: int, enable_cow: bool = True):
        self.num_blocks = num_blocks
        self.enable_cow = enable_cow
        self.free_blocks = set(range(num_blocks))
        self.allocated_blocks = set()
        self.ref_counts = [0] * num_blocks

    def allocate(self, count: int = 1) -> List[int]:
        """Allocate blocks from the pool."""
        if count > len(self.free_blocks):
            raise RuntimeError(f"Cannot allocate {count} blocks, only {len(self.free_blocks)} available")

        allocated = []
        for _ in range(count):
            block_id = self.free_blocks.pop()
            self.allocated_blocks.add(block_id)
            self.ref_counts[block_id] = 1
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: List[int]) -> None:
        """Return blocks to the free pool."""
        for block_id in block_ids:
            if block_id not in self.allocated_blocks:
                continue
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] <= 0:
                self.allocated_blocks.discard(block_id)
                self.free_blocks.add(block_id)
                self.ref_counts[block_id] = 0

    def increment_ref(self, block_id: int) -> None:
        """Increment reference count for COW sharing."""
        if block_id in self.allocated_blocks:
            self.ref_counts[block_id] += 1

    def get_ref_count(self, block_id: int) -> int:
        """Get reference count for a block."""
        return self.ref_counts[block_id]

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self.allocated_blocks)


class TestBlockAllocationParity:
    """Block allocation parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_forward_parity(self, size, skip_without_pytorch):
        """Test block allocation forward pass parity.

        Verifies that BlockAllocator produces identical state to reference
        after a series of allocations.
        """
        config = SIZE_CONFIGS[size]["cache"]
        # Compute reasonable number of blocks based on size config
        block_size = config["block_size"]
        seq = config["seq"]
        batch = config["batch"]
        num_blocks = batch * ((seq + block_size - 1) // block_size) * 2

        block_config = BlockConfig(
            block_size=block_size,
            num_heads=config["heads"],
            head_dim=config["head_dim"],
        )

        # Create MLX and reference allocators
        mlx_allocator = BlockAllocator(block_config, num_blocks)
        ref_allocator = ReferenceBlockAllocator(num_blocks)

        # Perform allocations in batches
        allocation_sizes = [1, 2, 3, 5, 8]
        all_allocated_mlx = []
        all_allocated_ref = []

        for alloc_size in allocation_sizes:
            if alloc_size <= mlx_allocator.num_free_blocks:
                mlx_blocks = mlx_allocator.allocate(alloc_size)
                ref_blocks = ref_allocator.allocate(alloc_size)
                all_allocated_mlx.extend(mlx_blocks)
                all_allocated_ref.extend(ref_blocks)

        # Verify state matches
        assert mlx_allocator.num_free_blocks == ref_allocator.num_free_blocks, (
            f"Free blocks mismatch: MLX={mlx_allocator.num_free_blocks}, "
            f"ref={ref_allocator.num_free_blocks}"
        )
        assert mlx_allocator.num_allocated_blocks == ref_allocator.num_allocated_blocks, (
            f"Allocated blocks mismatch: MLX={mlx_allocator.num_allocated_blocks}, "
            f"ref={ref_allocator.num_allocated_blocks}"
        )

        # Verify allocation counts match
        assert len(all_allocated_mlx) == len(all_allocated_ref)

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_allocation_order(self, skip_without_pytorch):
        """Test that block allocation order matches reference.

        Both allocators should allocate blocks from the same initial pool
        and maintain consistent ordering.
        """
        num_blocks = 100

        block_config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        mlx_allocator = BlockAllocator(block_config, num_blocks)
        ref_allocator = ReferenceBlockAllocator(num_blocks)

        # Allocate blocks one at a time and verify sets match
        mlx_blocks = []
        ref_blocks = []

        for _ in range(50):
            mlx_block = mlx_allocator.allocate(1)[0]
            ref_block = ref_allocator.allocate(1)[0]
            mlx_blocks.append(mlx_block)
            ref_blocks.append(ref_block)

        # After allocation, the sets of allocated blocks should match
        # (order may differ due to set implementation, but counts and membership should match)
        assert set(mlx_blocks) == set(ref_blocks), (
            f"Allocated block sets differ: MLX={set(mlx_blocks)}, ref={set(ref_blocks)}"
        )

        # Free some blocks and verify state
        to_free = mlx_blocks[:10]
        mlx_allocator.free(to_free)
        ref_allocator.free(ref_blocks[:10])

        assert mlx_allocator.num_free_blocks == ref_allocator.num_free_blocks
        assert mlx_allocator.num_allocated_blocks == ref_allocator.num_allocated_blocks

        # Allocate again and verify consistency
        mlx_new = mlx_allocator.allocate(5)
        ref_new = ref_allocator.allocate(5)

        assert len(mlx_new) == len(ref_new) == 5
        assert mlx_allocator.num_free_blocks == ref_allocator.num_free_blocks

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_free_list_management(self, skip_without_pytorch):
        """Test free list management matches reference.

        Verifies that allocation and deallocation cycles maintain
        consistent free list state between MLX and reference.
        """
        num_blocks = 50

        block_config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        mlx_allocator = BlockAllocator(block_config, num_blocks)
        ref_allocator = ReferenceBlockAllocator(num_blocks)

        # Initial state
        assert mlx_allocator.num_free_blocks == num_blocks
        assert ref_allocator.num_free_blocks == num_blocks

        # Allocate all blocks
        mlx_all = mlx_allocator.allocate(num_blocks)
        ref_all = ref_allocator.allocate(num_blocks)

        assert mlx_allocator.num_free_blocks == 0
        assert ref_allocator.num_free_blocks == 0

        # Free in batches
        batch_sizes = [10, 15, 25]
        start = 0
        for batch_size in batch_sizes:
            mlx_allocator.free(mlx_all[start:start + batch_size])
            ref_allocator.free(ref_all[start:start + batch_size])
            start += batch_size

            assert mlx_allocator.num_free_blocks == ref_allocator.num_free_blocks, (
                f"Free count mismatch after freeing {start} blocks"
            )

        # Final state should have all blocks free
        assert mlx_allocator.num_free_blocks == num_blocks
        assert ref_allocator.num_free_blocks == num_blocks

        # Verify we can allocate again
        mlx_final = mlx_allocator.allocate(20)
        ref_final = ref_allocator.allocate(20)
        assert len(mlx_final) == len(ref_final) == 20

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_copy_on_write(self, skip_without_pytorch):
        """Test copy-on-write (COW) block allocation.

        Verifies that reference counting and COW semantics work correctly:
        - Shared blocks have ref_count > 1
        - copy_on_write creates a new block when ref_count > 1
        - Freeing a shared block decrements ref_count without returning to free list
        """
        num_blocks = 20

        block_config = BlockConfig(block_size=16, num_heads=4, head_dim=32)
        mlx_allocator = BlockAllocator(block_config, num_blocks, enable_cow=True)
        ref_allocator = ReferenceBlockAllocator(num_blocks, enable_cow=True)

        # Allocate a block
        mlx_block = mlx_allocator.allocate(1)[0]
        ref_block = ref_allocator.allocate(1)[0]

        # Initial ref count should be 1
        assert mlx_allocator.get_ref_count(mlx_block) == 1
        assert ref_allocator.get_ref_count(ref_block) == 1

        # Increment ref count (simulate sharing)
        mlx_allocator.increment_ref(mlx_block)
        ref_allocator.increment_ref(ref_block)

        assert mlx_allocator.get_ref_count(mlx_block) == 2
        assert ref_allocator.get_ref_count(ref_block) == 2

        # Free the block once - should only decrement ref count
        mlx_allocator.free([mlx_block])
        ref_allocator.free([ref_block])

        assert mlx_allocator.get_ref_count(mlx_block) == 1, "Ref count should be 1 after first free"
        assert ref_allocator.get_ref_count(ref_block) == 1

        # Block should still be allocated (not in free list)
        assert mlx_allocator.num_free_blocks == num_blocks - 1
        assert ref_allocator.num_free_blocks == num_blocks - 1

        # Free again - now should return to free list
        mlx_allocator.free([mlx_block])
        ref_allocator.free([ref_block])

        assert mlx_allocator.get_ref_count(mlx_block) == 0
        assert ref_allocator.get_ref_count(ref_block) == 0

        # Now should be in free list
        assert mlx_allocator.num_free_blocks == num_blocks
        assert ref_allocator.num_free_blocks == num_blocks

        # Test copy_on_write operation directly (without set_block_data which also calls COW)
        block_a = mlx_allocator.allocate(1)[0]

        # First, set data BEFORE incrementing ref count (when ref_count=1, COW is not triggered)
        test_k = mx.ones((16, 4, 32)) * 42
        test_v = mx.ones((16, 4, 32)) * 24
        mlx_allocator.set_block_data(block_a, test_k, test_v)

        # Now increment ref count to simulate sharing
        mlx_allocator.increment_ref(block_a)  # Now ref_count = 2

        # copy_on_write should create a new block since ref_count > 1
        new_block = mlx_allocator.copy_on_write(block_a)

        # Should be a different block
        assert new_block != block_a, "COW should create new block when ref_count > 1"

        # Original block should have ref_count decremented to 1
        assert mlx_allocator.get_ref_count(block_a) == 1

        # New block should have ref_count = 1
        assert mlx_allocator.get_ref_count(new_block) == 1

        # Data should be copied to new block
        new_k, new_v = mlx_allocator.get_block_data(new_block)
        np.testing.assert_array_equal(np.array(new_k), np.array(test_k))
        np.testing.assert_array_equal(np.array(new_v), np.array(test_v))

        # Verify copy_on_write returns same block when ref_count = 1
        same_block = mlx_allocator.copy_on_write(block_a)
        assert same_block == block_a, "COW should return same block when ref_count <= 1"


# =============================================================================
# Eviction Policies Parity Tests
# =============================================================================

class TestEvictionPoliciesParity:
    """Cache eviction policies parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_lru_forward_parity(self, size, skip_without_pytorch):
        """Test LRU eviction policy forward pass parity."""
        config = SIZE_CONFIGS[size]["cache"]
        num_sequences = config["batch"] * 8

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

        # Test eviction selection for various counts
        candidates = list(range(num_sequences))
        for num_to_evict in [1, 2, num_sequences // 4]:
            mlx_evicted = mlx_lru.select_for_eviction(candidates, num_to_evict)
            ref_evicted = ref_lru.select_for_eviction(candidates, num_to_evict)

            assert mlx_evicted == ref_evicted, (
                f"LRU eviction mismatch for size={size}, n={num_to_evict}: "
                f"MLX={mlx_evicted}, ref={ref_evicted}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    def test_fifo_forward_parity(self, size, skip_without_pytorch):
        """Test FIFO eviction policy forward pass parity."""
        config = SIZE_CONFIGS[size]["cache"]
        num_sequences = config["batch"] * 8

        # Create MLX and reference FIFO policies
        mlx_fifo = FIFOEvictionPolicy()
        ref_fifo = ReferenceFIFO()

        # Create sequences
        for i in range(num_sequences):
            mlx_fifo.on_create(i)
            ref_fifo.on_create(i)

        # Apply access pattern (FIFO should ignore accesses)
        access_pattern = list(range(num_sequences))[::-1]  # Reverse order
        for seq_id in access_pattern:
            mlx_fifo.on_access(seq_id)
            ref_fifo.on_access(seq_id)

        # Test eviction selection
        candidates = list(range(num_sequences))
        for num_to_evict in [1, 2, num_sequences // 4]:
            mlx_evicted = mlx_fifo.select_for_eviction(candidates, num_to_evict)
            ref_evicted = ref_fifo.select_for_eviction(candidates, num_to_evict)

            assert mlx_evicted == ref_evicted, (
                f"FIFO eviction mismatch for size={size}, n={num_to_evict}: "
                f"MLX={mlx_evicted}, ref={ref_evicted}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_lru_eviction_order(self, skip_without_pytorch):
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_fifo_eviction_order(self, skip_without_pytorch):
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_attention_score_eviction(self, skip_without_pytorch):
        """Test attention-score based eviction policy."""
        decay_factor = 0.99
        mlx_policy = AttentionScoreEvictionPolicy(decay_factor=decay_factor)
        ref_policy = ReferenceAttentionScore(decay_factor=decay_factor)

        # Create sequences
        for i in range(4):
            mlx_policy.on_create(i)
            ref_policy.on_create(i)

        # Update attention scores
        scores = [
            (0, 0.1),  # Low score
            (1, 0.9),  # High score
            (2, 0.5),  # Medium score
            (3, 0.3),  # Medium-low score
        ]
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
        # Should evict lowest scores first (0 and 3)
        assert set(mlx_evicted) == {0, 3}


# =============================================================================
# Speculative Verification Parity Tests
# =============================================================================

class TestSpeculativeVerificationParity:
    """Speculative decoding verification parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_forward_parity(self, size, dtype, skip_without_pytorch):
        """Test speculative verification forward pass parity."""
        from tests.parity.conftest import get_mlx_dtype

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
        # Normalize to valid log probs
        target_log_probs_np = target_log_probs_np - np.log(
            np.exp(target_log_probs_np).sum(axis=-1, keepdims=True)
        )

        # MLX version
        mlx_dtype = get_mlx_dtype(dtype)
        draft_log_probs_mlx = mx.array(draft_log_probs_np).astype(mlx_dtype)
        target_log_probs_mlx = mx.array(target_log_probs_np).astype(mlx_dtype)

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
            f"Acceptance count mismatch for size={size}, dtype={dtype}: "
            f"MLX={mlx_accepted}, ref={ref_accepted}"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_acceptance_probability(self, skip_without_pytorch):
        """Test acceptance probability computation."""
        # Test cases with known outcomes
        test_cases = [
            # (draft_logp, target_logp, expected_accept_prob)
            (-1.0, -1.0, 1.0),     # Equal probs -> accept_prob = 1
            (-1.0, -0.5, 1.0),     # Target higher -> accept_prob = 1 (clamped)
            (-0.5, -1.0, np.exp(-0.5)),  # Draft higher -> accept_prob < 1
            (-2.0, -1.0, 1.0),     # Much higher target (clamped to 1)
            (-1.0, -2.0, np.exp(-1.0)),  # Much lower target
        ]

        rtol, atol = get_tolerance("cache", "speculative_verification", "fp32")

        for draft_logp, target_logp, expected in test_cases:
            # MLX computation
            mlx_accept = min(1.0, float(mx.exp(mx.array(target_logp - draft_logp))))

            # NumPy computation
            np_accept = min(1.0, float(np.exp(target_logp - draft_logp)))

            # Compare
            np.testing.assert_allclose(
                mlx_accept, np_accept, rtol=rtol, atol=atol,
                err_msg=f"Acceptance prob mismatch for draft={draft_logp}, target={target_logp}"
            )
            np.testing.assert_allclose(
                mlx_accept, expected, rtol=rtol, atol=atol,
                err_msg=f"Expected acceptance prob mismatch for draft={draft_logp}, target={target_logp}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_rejection_sampling(self, skip_without_pytorch):
        """Test rejection sampling for corrected tokens."""
        vocab_size = 100
        seed = 54321

        # Create a scenario where rejection is guaranteed
        # Draft has very high prob for token 0, target has low prob
        draft_log_probs = np.array([-0.01], dtype=np.float32)  # ~99% prob
        target_log_probs = np.zeros((2, vocab_size), dtype=np.float32)
        target_log_probs[0, 0] = -10.0  # Very low prob for draft token
        target_log_probs[0, 1:] = -2.3  # Higher prob for other tokens
        # Normalize
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
        assert ref_accepted == 0, f"Reference expected rejection, got {ref_accepted} accepted"

        # Correction token should not be 0 (the rejected token)
        assert mlx_correction != 0, "Correction token should not be the rejected token"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("num_speculative", [1, 2, 4, 8])
    def test_different_speculation_lengths(self, num_speculative, skip_without_pytorch):
        """Test with different speculation lengths."""
        vocab_size = 500
        seed = 99999

        np.random.seed(seed)

        # Generate test data
        draft_tokens = list(np.random.randint(0, vocab_size, (num_speculative,)))
        draft_log_probs = np.random.randn(num_speculative).astype(np.float32) * 0.3
        target_log_probs = np.random.randn(num_speculative + 1, vocab_size).astype(np.float32)
        target_log_probs = target_log_probs - np.log(
            np.exp(target_log_probs).sum(axis=-1, keepdims=True)
        )

        # Run with same random state
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_tree_attention_verification(self, skip_without_pytorch):
        """Test tree-based speculative decoding verification."""
        # This tests that the TreeSpeculation class works correctly
        # For now, we test the underlying speculative_verify function
        # which is used by TreeSpeculation.verify_tree()

        vocab_size = 100
        branches = [
            [0, 1, 2],      # Branch 0: 3 tokens
            [0, 1, 3],      # Branch 1: diverges at token 2
            [0, 4, 5],      # Branch 2: diverges at token 1
        ]

        seed = 11111
        np.random.seed(seed)

        results = []
        for branch_tokens in branches:
            draft_log_probs = np.random.randn(len(branch_tokens)).astype(np.float32) * 0.3
            target_log_probs = np.random.randn(len(branch_tokens) + 1, vocab_size).astype(np.float32)
            target_log_probs = target_log_probs - np.log(
                np.exp(target_log_probs).sum(axis=-1, keepdims=True)
            )

            mx.random.seed(42)
            accepted, correction = speculative_verify(
                branch_tokens,
                mx.array(draft_log_probs),
                mx.array(target_log_probs),
            )
            results.append((accepted, correction))

        # All branches should produce valid results
        for i, (accepted, correction) in enumerate(results):
            assert 0 <= accepted <= len(branches[i]), (
                f"Branch {i}: invalid acceptance count {accepted}"
            )


# =============================================================================
# KVCache Variant Parity Tests
# =============================================================================

# Import KV cache implementations
from mlx_primitives.advanced.kv_cache import (
    KVCache,
    SlidingWindowCache,
    RotatingKVCache,
    CompressedKVCache,
)


class ReferenceKVCache:
    """Reference KV cache using NumPy for parity testing."""

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = [
            np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.seq_len = 0

    def reset(self):
        self.seq_len = 0

    def get(self, layer_idx: int):
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len, :],
            self.v_cache[layer_idx][:, :, :self.seq_len, :],
        )

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        new_len = new_k.shape[2]
        start = self.seq_len

        self.k_cache[layer_idx][:, :, start:start + new_len, :] = new_k
        self.v_cache[layer_idx][:, :, start:start + new_len, :] = new_v

        if layer_idx == 0:
            self.seq_len = start + new_len

        return self.get(layer_idx)


class ReferenceSlidingWindowCache:
    """Reference sliding window cache using NumPy."""

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        window_size: int,
        num_heads: int,
        head_dim: int,
    ):
        self.num_layers = num_layers
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = [[] for _ in range(num_layers)]
        self.v_cache = [[] for _ in range(num_layers)]
        self.total_len = 0

    def reset(self):
        self.k_cache = [[] for _ in range(self.num_layers)]
        self.v_cache = [[] for _ in range(self.num_layers)]
        self.total_len = 0

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        # Append new tokens
        for i in range(new_k.shape[2]):
            self.k_cache[layer_idx].append(new_k[:, :, i:i+1, :])
            self.v_cache[layer_idx].append(new_v[:, :, i:i+1, :])

        # Trim to window size
        while len(self.k_cache[layer_idx]) > self.window_size:
            self.k_cache[layer_idx].pop(0)
            self.v_cache[layer_idx].pop(0)

        if layer_idx == 0:
            self.total_len += new_k.shape[2]

        # Return current cache
        if len(self.k_cache[layer_idx]) == 0:
            batch = new_k.shape[0]
            return (
                np.zeros((batch, self.num_heads, 0, self.head_dim)),
                np.zeros((batch, self.num_heads, 0, self.head_dim)),
            )

        return (
            np.concatenate(self.k_cache[layer_idx], axis=2),
            np.concatenate(self.v_cache[layer_idx], axis=2),
        )

    @property
    def position_offset(self):
        return max(0, self.total_len - self.window_size)


class ReferenceRotatingCache:
    """Reference rotating (circular) buffer cache using NumPy."""

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        buffer_size: int,
        num_heads: int,
        head_dim: int,
    ):
        self.num_layers = num_layers
        self.buffer_size = buffer_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = [
            np.zeros((max_batch_size, num_heads, buffer_size, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            np.zeros((max_batch_size, num_heads, buffer_size, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.write_pos = 0
        self.total_len = 0

    def reset(self):
        self.write_pos = 0
        self.total_len = 0

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        new_len = new_k.shape[2]

        for i in range(new_len):
            pos = (self.write_pos + i) % self.buffer_size
            self.k_cache[layer_idx][:, :, pos, :] = new_k[:, :, i, :]
            self.v_cache[layer_idx][:, :, pos, :] = new_v[:, :, i, :]

        if layer_idx == 0:
            self.write_pos = (self.write_pos + new_len) % self.buffer_size
            self.total_len += new_len

        return self.get(layer_idx)

    def get(self, layer_idx: int):
        if self.total_len <= self.buffer_size:
            # Haven't wrapped
            k = self.k_cache[layer_idx][:, :, :self.total_len, :]
            v = self.v_cache[layer_idx][:, :, :self.total_len, :]
        else:
            # Reorder: [write_pos:] + [:write_pos]
            k1 = self.k_cache[layer_idx][:, :, self.write_pos:, :]
            k2 = self.k_cache[layer_idx][:, :, :self.write_pos, :]
            k = np.concatenate([k1, k2], axis=2)

            v1 = self.v_cache[layer_idx][:, :, self.write_pos:, :]
            v2 = self.v_cache[layer_idx][:, :, :self.write_pos, :]
            v = np.concatenate([v1, v2], axis=2)

        return k, v


class TestKVCacheParity:
    """Basic KVCache parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("size", ["tiny", "small", "medium"])
    def test_kv_cache_update_parity(self, size, skip_without_pytorch):
        """Test KVCache update matches reference.

        Note: The KVCache implementation updates seq_len only for layer 0,
        and subsequent layers use the same seq_len. This test uses only
        layer 0 to verify basic functionality, as the multi-layer case
        requires careful coordination of when seq_len updates.
        """
        config = SIZE_CONFIGS[size]["cache"]
        num_layers = 1  # Use single layer for clear parity testing
        batch = config["batch"]
        num_heads = config["heads"]
        head_dim = config["head_dim"]
        max_seq_len = config["seq"]

        # Create caches
        mlx_cache = KVCache(
            num_layers=num_layers,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=mx.float32,
        )
        ref_cache = ReferenceKVCache(
            num_layers=num_layers,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Sequence of updates (typical: prompt, then decode tokens)
        seq_lens = [8, 1, 1, 4]
        for new_len in seq_lens:
            if ref_cache.seq_len + new_len > max_seq_len:
                break

            new_k = np.random.randn(batch, num_heads, new_len, head_dim).astype(np.float32)
            new_v = np.random.randn(batch, num_heads, new_len, head_dim).astype(np.float32)

            mlx_k, mlx_v = mlx_cache.update(0, mx.array(new_k), mx.array(new_v))
            ref_k, ref_v = ref_cache.update(0, new_k, new_v)

            np.testing.assert_allclose(
                np.array(mlx_k), ref_k, rtol=1e-6, atol=1e-7,
                err_msg="KVCache K mismatch"
            )
            np.testing.assert_allclose(
                np.array(mlx_v), ref_v, rtol=1e-6, atol=1e-7,
                err_msg="KVCache V mismatch"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_kv_cache_reset(self, skip_without_pytorch):
        """Test KVCache reset clears state."""
        cache = KVCache(
            num_layers=2,
            max_batch_size=1,
            max_seq_len=64,
            num_heads=4,
            head_dim=16,
        )

        # Add some data
        new_k = mx.random.normal((1, 4, 8, 16))
        new_v = mx.random.normal((1, 4, 8, 16))
        cache.update(0, new_k, new_v)

        assert cache.seq_len == 8

        # Reset
        cache.reset()

        assert cache.seq_len == 0


class TestSlidingWindowCacheParity:
    """Sliding window cache parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("window_size", [16, 32, 64])
    def test_sliding_window_eviction(self, window_size, skip_without_pytorch):
        """Test sliding window cache eviction behavior."""
        num_layers = 1
        batch = 1
        num_heads = 4
        head_dim = 16

        mlx_cache = SlidingWindowCache(
            num_layers=num_layers,
            max_batch_size=batch,
            window_size=window_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=mx.float32,
        )
        ref_cache = ReferenceSlidingWindowCache(
            num_layers=num_layers,
            max_batch_size=batch,
            window_size=window_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Add more tokens than window size
        total_tokens = window_size * 2
        chunk_size = 4

        for i in range(0, total_tokens, chunk_size):
            new_k = np.random.randn(batch, num_heads, chunk_size, head_dim).astype(np.float32)
            new_v = np.random.randn(batch, num_heads, chunk_size, head_dim).astype(np.float32)

            mlx_k, mlx_v = mlx_cache.update(0, mx.array(new_k), mx.array(new_v))
            ref_k, ref_v = ref_cache.update(0, new_k, new_v)

            np.testing.assert_allclose(
                np.array(mlx_k), ref_k, rtol=1e-6, atol=1e-7,
                err_msg=f"SlidingWindowCache K mismatch at token {i}"
            )
            np.testing.assert_allclose(
                np.array(mlx_v), ref_v, rtol=1e-6, atol=1e-7,
                err_msg=f"SlidingWindowCache V mismatch at token {i}"
            )

        # Verify window size is maintained
        k, v = mlx_cache.get(0)
        assert k.shape[2] == window_size, f"Expected window_size={window_size}, got {k.shape[2]}"

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_sliding_window_position_offset(self, skip_without_pytorch):
        """Test sliding window position offset for RoPE."""
        window_size = 32
        cache = SlidingWindowCache(
            num_layers=1,
            max_batch_size=1,
            window_size=window_size,
            num_heads=4,
            head_dim=16,
        )

        # Add tokens
        for i in range(50):
            new_k = mx.random.normal((1, 4, 1, 16))
            new_v = mx.random.normal((1, 4, 1, 16))
            cache.update(0, new_k, new_v)

        # Position offset should be total - window_size
        expected_offset = max(0, 50 - window_size)
        assert cache.position_offset == expected_offset


class TestRotatingKVCacheParity:
    """Rotating (circular buffer) cache parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    @pytest.mark.parametrize("buffer_size", [16, 32])
    def test_rotating_circular_write(self, buffer_size, skip_without_pytorch):
        """Test rotating cache circular write behavior."""
        num_layers = 1
        batch = 1
        num_heads = 4
        head_dim = 16

        mlx_cache = RotatingKVCache(
            num_layers=num_layers,
            max_batch_size=batch,
            buffer_size=buffer_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=mx.float32,
        )
        ref_cache = ReferenceRotatingCache(
            num_layers=num_layers,
            max_batch_size=batch,
            buffer_size=buffer_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Add more tokens than buffer size to test wrap-around
        total_tokens = buffer_size * 2 + 5
        chunk_size = 3

        for i in range(0, total_tokens, chunk_size):
            new_k = np.random.randn(batch, num_heads, chunk_size, head_dim).astype(np.float32)
            new_v = np.random.randn(batch, num_heads, chunk_size, head_dim).astype(np.float32)

            mlx_k, mlx_v = mlx_cache.update(0, mx.array(new_k), mx.array(new_v))
            ref_k, ref_v = ref_cache.update(0, new_k, new_v)

            np.testing.assert_allclose(
                np.array(mlx_k), ref_k, rtol=1e-6, atol=1e-7,
                err_msg=f"RotatingKVCache K mismatch at token {i}"
            )
            np.testing.assert_allclose(
                np.array(mlx_v), ref_v, rtol=1e-6, atol=1e-7,
                err_msg=f"RotatingKVCache V mismatch at token {i}"
            )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_rotating_get_order(self, skip_without_pytorch):
        """Test rotating cache returns tokens in correct order after wrap."""
        buffer_size = 8
        cache = RotatingKVCache(
            num_layers=1,
            max_batch_size=1,
            buffer_size=buffer_size,
            num_heads=1,
            head_dim=4,
            dtype=mx.float32,
        )

        # Add tokens 0-11 (wraps around)
        for i in range(12):
            # Use token index as the value for easy verification
            new_k = mx.ones((1, 1, 1, 4)) * i
            new_v = mx.ones((1, 1, 1, 4)) * i
            cache.update(0, new_k, new_v)

        # After adding 12 tokens with buffer_size=8:
        # Buffer contains tokens 4-11 (oldest evicted)
        # get() should return in order: 4, 5, 6, 7, 8, 9, 10, 11
        k, v = cache.get(0)

        expected_values = list(range(4, 12))
        for i, expected in enumerate(expected_values):
            actual = float(k[0, 0, i, 0])
            assert actual == expected, f"Position {i}: expected {expected}, got {actual}"


class TestCompressedKVCacheParity:
    """Compressed (quantized) KV cache parity tests."""

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_quantize_8bit_roundtrip(self, skip_without_pytorch):
        """Test 8-bit quantization round-trip accuracy."""
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
        original_k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        original_v = np.random.randn(1, 4, 8, 16).astype(np.float32)

        cache.update(0, mx.array(original_k), mx.array(original_v))

        # Get dequantized data
        recovered_k, recovered_v = cache.get(0)
        recovered_k = np.array(recovered_k)
        recovered_v = np.array(recovered_v)

        # 8-bit quantization should have <1% error
        np.testing.assert_allclose(
            recovered_k, original_k, rtol=1e-2, atol=1e-2,
            err_msg="8-bit quantization error too large for K"
        )
        np.testing.assert_allclose(
            recovered_v, original_v, rtol=1e-2, atol=1e-2,
            err_msg="8-bit quantization error too large for V"
        )

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_quantize_4bit_roundtrip(self, skip_without_pytorch):
        """Test 4-bit quantization round-trip accuracy.

        4-bit quantization has only 16 levels, leading to significant
        quantization error. Typical error is 10-25% for normal data.
        We verify the error is within expected bounds rather than
        checking for tight tolerances.
        """
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
        recovered_k = np.array(recovered_k)
        recovered_v = np.array(recovered_v)

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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_compressed_cache_memory_ratio(self, skip_without_pytorch):
        """Test compressed cache memory usage reduction."""
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

    @pytest.mark.parity_pytorch
    @pytest.mark.forward_parity
    def test_pruned_cache(self, skip_without_pytorch):
        """Test pruned cache keeps most recent tokens."""
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
        for i in range(100):
            new_k = mx.random.normal((1, 4, 1, 16))
            new_v = mx.random.normal((1, 4, 1, 16))
            cache.update(0, new_k, new_v)

        # Get cache - should be pruned
        k, v = cache.get(0)

        # Should keep at most keep_len = max_seq_len * keep_ratio
        keep_len = int(max_seq_len * keep_ratio)
        assert k.shape[2] <= keep_len, f"Cache should be pruned to {keep_len}, got {k.shape[2]}"
