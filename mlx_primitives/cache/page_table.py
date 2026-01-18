"""Page table management for paged attention.

This module maps logical token positions to physical blocks,
enabling non-contiguous KV storage with efficient lookup.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.cache.block_allocator import BlockAllocator


@dataclass
class SequenceMetadata:
    """Metadata for a single sequence in the cache.

    Attributes:
        sequence_id: Unique identifier for this sequence.
        block_table: List of physical block indices in order.
        num_tokens: Current number of cached tokens.
        num_tokens_in_last_block: Tokens used in the last block.
        last_access_time: Timestamp for LRU eviction.
        ref_count: Reference count for prefix sharing.
        is_speculative: Whether this is a speculative branch.
        parent_sequence_id: For tree-based speculation/beam search.
    """

    sequence_id: int
    block_table: List[int] = field(default_factory=list)
    num_tokens: int = 0
    num_tokens_in_last_block: int = 0
    last_access_time: float = field(default_factory=time.time)
    ref_count: int = 1
    is_speculative: bool = False
    parent_sequence_id: Optional[int] = None

    def touch(self) -> None:
        """Update last access time."""
        self.last_access_time = time.time()


class PageTable:
    """Page table mapping sequences to physical blocks.

    Supports:
    - Dynamic sequence growth (allocate new blocks as needed)
    - Prefix sharing via reference counting
    - Efficient lookup for paged attention
    - Batch operations for continuous batching

    Example:
        >>> allocator = BlockAllocator(config, num_blocks=1000)
        >>> page_table = PageTable(allocator)
        >>> seq_id = page_table.create_sequence()
        >>> page_table.extend_sequence(seq_id, num_new_tokens=100)
        >>> block_table = page_table.get_block_table(seq_id)
    """

    def __init__(self, block_allocator: BlockAllocator):
        """Initialize page table.

        Args:
            block_allocator: Block allocator for physical storage.
        """
        self._allocator = block_allocator
        self._sequences: Dict[int, SequenceMetadata] = {}
        self._next_sequence_id = 0

    @property
    def num_sequences(self) -> int:
        """Number of active sequences."""
        return len(self._sequences)

    @property
    def block_size(self) -> int:
        """Tokens per block from the allocator config."""
        return self._allocator.config.block_size

    def create_sequence(
        self,
        initial_tokens: int = 0,
        parent_sequence_id: Optional[int] = None,
        is_speculative: bool = False,
    ) -> int:
        """Create a new sequence.

        Args:
            initial_tokens: Number of tokens to pre-allocate blocks for.
            parent_sequence_id: Parent sequence for forking (beam search).
            is_speculative: Whether this is a speculative branch.

        Returns:
            New sequence ID.
        """
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1

        metadata = SequenceMetadata(
            sequence_id=sequence_id,
            is_speculative=is_speculative,
            parent_sequence_id=parent_sequence_id,
        )

        self._sequences[sequence_id] = metadata

        # Allocate initial blocks if requested
        if initial_tokens > 0:
            self._allocate_blocks_for_tokens(sequence_id, initial_tokens)

        return sequence_id

    def _allocate_blocks_for_tokens(self, sequence_id: int, num_tokens: int) -> None:
        """Allocate blocks to accommodate the given number of tokens.

        Args:
            sequence_id: Sequence to allocate for.
            num_tokens: Total tokens needed (not additional).
        """
        metadata = self._sequences[sequence_id]
        current_capacity = len(metadata.block_table) * self.block_size

        if num_tokens <= current_capacity:
            return

        # Calculate blocks needed
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        new_blocks_needed = blocks_needed - len(metadata.block_table)

        if new_blocks_needed > 0:
            new_blocks = self._allocator.allocate(new_blocks_needed)
            metadata.block_table.extend(new_blocks)

    def extend_sequence(self, sequence_id: int, num_new_tokens: int) -> None:
        """Extend a sequence by adding tokens, allocating blocks as needed.

        Args:
            sequence_id: Sequence to extend.
            num_new_tokens: Number of tokens being added.

        Raises:
            KeyError: If sequence doesn't exist.
        """
        if sequence_id not in self._sequences:
            raise KeyError(f"Sequence {sequence_id} not found")

        metadata = self._sequences[sequence_id]
        metadata.touch()

        new_total = metadata.num_tokens + num_new_tokens
        self._allocate_blocks_for_tokens(sequence_id, new_total)

        # Update token counts
        metadata.num_tokens = new_total
        metadata.num_tokens_in_last_block = new_total % self.block_size
        if metadata.num_tokens_in_last_block == 0 and new_total > 0:
            metadata.num_tokens_in_last_block = self.block_size

    def truncate_sequence(self, sequence_id: int, new_length: int) -> None:
        """Truncate a sequence to a shorter length, freeing excess blocks.

        Args:
            sequence_id: Sequence to truncate.
            new_length: New number of tokens (must be <= current).
        """
        if sequence_id not in self._sequences:
            raise KeyError(f"Sequence {sequence_id} not found")

        metadata = self._sequences[sequence_id]

        if new_length >= metadata.num_tokens:
            return

        # Calculate blocks needed for new length
        blocks_needed = (new_length + self.block_size - 1) // self.block_size
        if new_length == 0:
            blocks_needed = 0

        # Free excess blocks
        excess_blocks = metadata.block_table[blocks_needed:]
        if excess_blocks:
            self._allocator.free(excess_blocks)
            metadata.block_table = metadata.block_table[:blocks_needed]

        # Update token counts
        metadata.num_tokens = new_length
        metadata.num_tokens_in_last_block = new_length % self.block_size
        if metadata.num_tokens_in_last_block == 0 and new_length > 0:
            metadata.num_tokens_in_last_block = self.block_size

    def fork_sequence(self, sequence_id: int, is_speculative: bool = False) -> int:
        """Create a fork of a sequence sharing prefix blocks (for beam search).

        Uses copy-on-write: blocks are shared until modified.

        Args:
            sequence_id: Sequence to fork from.
            is_speculative: Whether the fork is for speculation.

        Returns:
            New forked sequence ID.
        """
        if sequence_id not in self._sequences:
            raise KeyError(f"Sequence {sequence_id} not found")

        parent = self._sequences[sequence_id]

        # Create new sequence with shared block table
        new_id = self._next_sequence_id
        self._next_sequence_id += 1

        # Increment ref counts on shared blocks
        for block_id in parent.block_table:
            self._allocator.increment_ref(block_id)

        new_metadata = SequenceMetadata(
            sequence_id=new_id,
            block_table=parent.block_table.copy(),  # Share same block IDs
            num_tokens=parent.num_tokens,
            num_tokens_in_last_block=parent.num_tokens_in_last_block,
            ref_count=1,
            is_speculative=is_speculative,
            parent_sequence_id=sequence_id,
        )

        self._sequences[new_id] = new_metadata
        return new_id

    def get_block_table(self, sequence_id: int) -> List[int]:
        """Get the block table for a sequence.

        Args:
            sequence_id: Sequence ID.

        Returns:
            List of physical block indices.
        """
        if sequence_id not in self._sequences:
            raise KeyError(f"Sequence {sequence_id} not found")

        metadata = self._sequences[sequence_id]
        metadata.touch()
        return metadata.block_table

    def get_block_table_tensor(
        self,
        sequence_ids: List[int],
        max_blocks: Optional[int] = None,
    ) -> mx.array:
        """Get batched block table tensor for paged attention.

        Args:
            sequence_ids: List of sequence IDs.
            max_blocks: Maximum blocks per sequence. Defaults to max in batch.

        Returns:
            Block table tensor of shape (batch, max_blocks) with -1 padding.
        """
        if not sequence_ids:
            return mx.array([], dtype=mx.int32)

        block_tables = [self.get_block_table(sid) for sid in sequence_ids]

        if max_blocks is None:
            max_blocks = max(len(bt) for bt in block_tables) if block_tables else 0

        # Pad with -1 for invalid blocks
        padded = []
        for bt in block_tables:
            if len(bt) < max_blocks:
                padded.append(bt + [-1] * (max_blocks - len(bt)))
            else:
                padded.append(bt[:max_blocks])

        return mx.array(padded, dtype=mx.int32)

    def get_context_lengths(self, sequence_ids: List[int]) -> mx.array:
        """Get the number of cached tokens for each sequence.

        Args:
            sequence_ids: List of sequence IDs.

        Returns:
            Context lengths tensor of shape (batch,).
        """
        lengths = [self._sequences[sid].num_tokens for sid in sequence_ids]
        return mx.array(lengths, dtype=mx.int32)

    def get_sequence_metadata(self, sequence_id: int) -> SequenceMetadata:
        """Get metadata for a sequence.

        Args:
            sequence_id: Sequence ID.

        Returns:
            Sequence metadata.
        """
        if sequence_id not in self._sequences:
            raise KeyError(f"Sequence {sequence_id} not found")
        return self._sequences[sequence_id]

    def get_token_position(self, sequence_id: int, token_idx: int) -> Tuple[int, int]:
        """Get (block_id, position_in_block) for a token index.

        Args:
            sequence_id: Sequence ID.
            token_idx: Token index within the sequence.

        Returns:
            Tuple of (block_id, position_in_block).
        """
        metadata = self._sequences[sequence_id]
        block_idx = token_idx // self.block_size
        pos_in_block = token_idx % self.block_size
        return metadata.block_table[block_idx], pos_in_block

    def delete_sequence(self, sequence_id: int) -> None:
        """Delete a sequence and free its blocks.

        Args:
            sequence_id: Sequence to delete.
        """
        if sequence_id not in self._sequences:
            return

        metadata = self._sequences[sequence_id]

        # Free all blocks (allocator handles ref counting)
        self._allocator.free(metadata.block_table)

        del self._sequences[sequence_id]

    def get_all_sequence_ids(self) -> List[int]:
        """Get all active sequence IDs.

        Returns:
            List of sequence IDs.
        """
        return list(self._sequences.keys())

    def get_sequences_by_access_time(self, ascending: bool = True) -> List[int]:
        """Get sequences sorted by last access time.

        Args:
            ascending: If True, oldest first (for LRU eviction).

        Returns:
            Sorted list of sequence IDs.
        """
        return sorted(
            self._sequences.keys(),
            key=lambda sid: self._sequences[sid].last_access_time,
            reverse=not ascending,
        )

    def clear(self) -> None:
        """Delete all sequences."""
        for seq_id in list(self._sequences.keys()):
            self.delete_sequence(seq_id)

    def get_stats(self) -> dict:
        """Get page table statistics.

        Returns:
            Dictionary with statistics.
        """
        total_tokens = sum(m.num_tokens for m in self._sequences.values())
        total_blocks = sum(len(m.block_table) for m in self._sequences.values())

        return {
            "num_sequences": self.num_sequences,
            "total_tokens": total_tokens,
            "total_blocks": total_blocks,
            "avg_tokens_per_sequence": total_tokens / self.num_sequences
            if self.num_sequences > 0
            else 0,
            "avg_blocks_per_sequence": total_blocks / self.num_sequences
            if self.num_sequences > 0
            else 0,
        }
