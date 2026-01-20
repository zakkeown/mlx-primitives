"""Paged KV Cache for efficient memory management.

Implements vLLM-style paged attention for efficient serving of multiple
sequences with variable lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.advanced.paged_attention.block_manager import (
    BlockManager,
    BlockConfig,
)


@dataclass
class SequenceState:
    """State for a single sequence in the paged cache.

    Tracks the mapping from logical token positions to physical blocks.
    """
    seq_id: int
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def get_block_idx(self, token_pos: int, block_size: int) -> Tuple[int, int]:
        """Get block index and offset for a token position.

        Returns:
            Tuple of (block_list_idx, offset_within_block).
        """
        block_idx = token_pos // block_size
        offset = token_pos % block_size
        return block_idx, offset


class PagedKVCache:
    """Paged KV Cache with efficient memory management.

    Uses block-based allocation to:
    - Eliminate memory fragmentation from variable-length sequences
    - Enable efficient batching with different sequence lengths
    - Support copy-on-write for beam search and speculative decoding
    - Reduce memory waste from pre-allocation

    Args:
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        num_layers: Number of transformer layers.
        block_size: Tokens per block (default: 16).
        max_blocks: Maximum blocks in pool (default: 1024).
        dtype: Data type (default: float16).

    Example:
        >>> cache = PagedKVCache(num_kv_heads=8, head_dim=128, num_layers=32)
        >>> seq_id = cache.create_sequence()
        >>> for layer_idx in range(32):
        ...     k, v = compute_kv(layer_idx)
        ...     cache.append_kv(seq_id, layer_idx, k, v)
        >>> k_full, v_full = cache.get_kv(seq_id, layer_idx=0)
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        block_size: int = 16,
        max_blocks: int = 1024,
        dtype: mx.Dtype = mx.float16,
    ):
        self.config = BlockConfig(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            max_blocks=max_blocks,
            dtype=dtype,
        )

        self._block_manager = BlockManager(self.config)
        self._sequences: Dict[int, SequenceState] = {}
        self._next_seq_id = 0

    @property
    def block_size(self) -> int:
        """Number of tokens per block."""
        return self.config.block_size

    @property
    def num_sequences(self) -> int:
        """Number of active sequences."""
        return len(self._sequences)

    def create_sequence(self, seq_id: Optional[int] = None) -> int:
        """Create a new sequence.

        Args:
            seq_id: Optional specific sequence ID. If None, auto-assigns.

        Returns:
            The sequence ID.
        """
        if seq_id is None:
            seq_id = self._next_seq_id
            self._next_seq_id += 1
        elif seq_id in self._sequences:
            raise ValueError(f"Sequence {seq_id} already exists")

        self._sequences[seq_id] = SequenceState(seq_id=seq_id)
        return seq_id

    def delete_sequence(self, seq_id: int):
        """Delete a sequence and free its blocks.

        Args:
            seq_id: Sequence ID to delete.
        """
        if seq_id not in self._sequences:
            return

        seq_state = self._sequences[seq_id]
        self._block_manager.free_blocks(seq_state.block_ids)
        del self._sequences[seq_id]

    def fork_sequence(self, parent_id: int, child_id: Optional[int] = None) -> int:
        """Fork a sequence for beam search (copy-on-write).

        The child shares blocks with parent until either is modified.

        Args:
            parent_id: Parent sequence ID.
            child_id: Optional child sequence ID.

        Returns:
            Child sequence ID.
        """
        if parent_id not in self._sequences:
            raise ValueError(f"Parent sequence {parent_id} not found")

        parent = self._sequences[parent_id]

        # Create child with same blocks (COW)
        if child_id is None:
            child_id = self._next_seq_id
            self._next_seq_id += 1

        # Increment ref counts for shared blocks
        self._block_manager.increment_ref(parent.block_ids)

        # Create child state
        self._sequences[child_id] = SequenceState(
            seq_id=child_id,
            block_ids=parent.block_ids.copy(),
            num_tokens=parent.num_tokens,
        )

        return child_id

    def _ensure_capacity(self, seq_state: SequenceState, new_tokens: int) -> bool:
        """Ensure sequence has capacity for new tokens.

        Allocates new blocks if needed.

        Returns:
            True if allocation succeeded, False if out of memory.
        """
        current_capacity = len(seq_state.block_ids) * self.config.block_size
        needed = seq_state.num_tokens + new_tokens

        if needed <= current_capacity:
            return True

        # Calculate blocks needed
        new_capacity_needed = needed - current_capacity
        new_blocks_needed = (
            new_capacity_needed + self.config.block_size - 1
        ) // self.config.block_size

        if not self._block_manager.can_allocate(new_blocks_needed):
            return False

        # Allocate new blocks
        new_blocks = self._block_manager.allocate_blocks(new_blocks_needed)
        seq_state.block_ids.extend(new_blocks)
        return True

    def append_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ) -> bool:
        """Append K/V for a single token to the sequence.

        Args:
            seq_id: Sequence ID.
            layer_idx: Layer index.
            k: Key tensor (num_kv_heads, head_dim).
            v: Value tensor (num_kv_heads, head_dim).

        Returns:
            True if successful, False if out of memory.
        """
        if seq_id not in self._sequences:
            raise ValueError(f"Sequence {seq_id} not found")

        seq_state = self._sequences[seq_id]

        # Ensure capacity (only for layer 0, other layers use same position)
        if layer_idx == 0:
            if not self._ensure_capacity(seq_state, 1):
                return False
            token_pos = seq_state.num_tokens
        else:
            # Other layers: write to the position where layer 0 already wrote
            # Layer 0 increments after writing, so we need num_tokens - 1
            token_pos = seq_state.num_tokens - 1
            if token_pos < 0:
                token_pos = 0

        # Get block and offset for this token
        block_list_idx, offset = seq_state.get_block_idx(
            token_pos, self.config.block_size
        )
        block_id = seq_state.block_ids[block_list_idx]

        # Check if we need COW (shared block being modified)
        block = self._block_manager._blocks[block_id]
        if block.ref_count > 1:
            # Copy the block
            new_block_id = self._block_manager.copy_block(block_id)
            seq_state.block_ids[block_list_idx] = new_block_id
            block_id = new_block_id

        # Write K/V
        self._block_manager.write_kv(block_id, layer_idx, offset, k, v)

        # Update token count (only for first layer to avoid double counting)
        if layer_idx == 0:
            seq_state.num_tokens += 1

        return True

    def append_kv_batch(
        self,
        seq_id: int,
        layer_idx: int,
        k_batch: mx.array,
        v_batch: mx.array,
        start_pos: Optional[int] = None,
    ) -> bool:
        """Append K/V for multiple tokens to the sequence.

        Args:
            seq_id: Sequence ID.
            layer_idx: Layer index.
            k_batch: Key tensor (num_tokens, num_kv_heads, head_dim).
            v_batch: Value tensor (num_tokens, num_kv_heads, head_dim).
            start_pos: Optional starting position (for prefill with multiple layers).
                If None, uses current num_tokens for layer 0, or infers for other layers.

        Returns:
            True if successful, False if out of memory.
        """
        if seq_id not in self._sequences:
            raise ValueError(f"Sequence {seq_id} not found")

        seq_state = self._sequences[seq_id]
        num_new_tokens = k_batch.shape[0]

        # Determine starting position
        if start_pos is not None:
            base_pos = start_pos
        elif layer_idx == 0:
            # Layer 0: append after current tokens
            base_pos = seq_state.num_tokens
        else:
            # Other layers: write to positions that layer 0 already reserved
            # Assume we're filling the same tokens as layer 0
            base_pos = seq_state.num_tokens - num_new_tokens
            if base_pos < 0:
                base_pos = 0  # Fallback for first batch

        # Ensure capacity (only needed for layer 0 or explicit start_pos)
        if layer_idx == 0 or start_pos is not None:
            if not self._ensure_capacity(seq_state, num_new_tokens):
                return False

        # Write tokens one by one (could be optimized for bulk writes)
        for i in range(num_new_tokens):
            block_list_idx, offset = seq_state.get_block_idx(
                base_pos + i, self.config.block_size
            )
            block_id = seq_state.block_ids[block_list_idx]

            # Check COW
            block = self._block_manager._blocks[block_id]
            if block.ref_count > 1:
                new_block_id = self._block_manager.copy_block(block_id)
                seq_state.block_ids[block_list_idx] = new_block_id
                block_id = new_block_id

            self._block_manager.write_kv(
                block_id, layer_idx, offset, k_batch[i], v_batch[i]
            )

        # Update token count (only for first layer)
        if layer_idx == 0:
            seq_state.num_tokens += num_new_tokens

        return True

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """Get K/V tensors for a sequence.

        Args:
            seq_id: Sequence ID.
            layer_idx: Layer index.

        Returns:
            Tuple of (K, V) with shape (seq_len, num_kv_heads, head_dim).
        """
        if seq_id not in self._sequences:
            raise ValueError(f"Sequence {seq_id} not found")

        seq_state = self._sequences[seq_id]

        if seq_state.num_tokens == 0:
            # Empty sequence
            return (
                mx.zeros((0, self.config.num_kv_heads, self.config.head_dim)),
                mx.zeros((0, self.config.num_kv_heads, self.config.head_dim)),
            )

        return self._block_manager.read_kv(
            seq_state.block_ids,
            layer_idx,
            seq_state.num_tokens,
        )

    def get_sequence_length(self, seq_id: int) -> int:
        """Get the number of tokens in a sequence."""
        if seq_id not in self._sequences:
            return 0
        return self._sequences[seq_id].num_tokens

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        block_stats = self._block_manager.get_memory_stats()
        return {
            **block_stats,
            "num_sequences": self.num_sequences,
            "block_size": self.config.block_size,
        }

    def reset(self):
        """Reset the cache, deleting all sequences."""
        self._sequences.clear()
        self._block_manager.reset()
        self._next_seq_id = 0


def create_paged_attention_mask(
    seq_lengths: List[int],
    max_seq_len: int,
    causal: bool = True,
) -> mx.array:
    """Create attention mask for paged attention batch.

    Args:
        seq_lengths: List of sequence lengths in the batch.
        max_seq_len: Maximum sequence length.
        causal: Whether to apply causal masking.

    Returns:
        Boolean mask (batch, 1, query_len, kv_len).
    """
    batch_size = len(seq_lengths)

    # Create position indices
    positions = mx.arange(max_seq_len)

    masks = []
    for seq_len in seq_lengths:
        # Valid positions mask
        valid = positions < seq_len

        if causal:
            # Causal mask: can only attend to positions <= current
            causal_mask = positions[None, :] <= positions[:, None]
            # Combine with valid mask
            mask = causal_mask & valid[None, :]
        else:
            mask = valid[None, :].broadcast_to((max_seq_len, max_seq_len))

        masks.append(mask)

    # Stack and add head dimension
    return mx.stack(masks)[:, None, :, :]
