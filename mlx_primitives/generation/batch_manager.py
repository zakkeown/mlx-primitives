"""Batch management for variable-length sequence handling.

This module provides utilities for efficiently batching sequences
of different lengths for transformer inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import mlx.core as mx


class BatchingStrategy(Enum):
    """Strategy for handling variable-length sequences."""

    PADDED = "padded"  # Pad to max length in batch
    PACKED = "packed"  # Pack sequences with position tracking


class PaddingSide(Enum):
    """Side to pad sequences."""

    LEFT = "left"  # Pad on left (for generation)
    RIGHT = "right"  # Pad on right (for training)


@dataclass
class BatchedSequences:
    """A batch of sequences with metadata for unbatching.

    Attributes:
        input_ids: Padded input IDs (batch, max_seq_len).
        attention_mask: Mask where 1=real, 0=pad (batch, max_seq_len).
        position_ids: Position IDs (batch, max_seq_len).
        sequence_lengths: Original lengths (batch,).
        batch_indices: Map batch position -> original request index.
        cu_seqlens: Cumulative sequence lengths (for packed batching).
        max_seqlen: Maximum sequence length in batch.
        padding_side: Which side was padded.
    """

    input_ids: mx.array
    attention_mask: mx.array
    position_ids: mx.array
    sequence_lengths: mx.array
    batch_indices: List[int] = field(default_factory=list)
    cu_seqlens: Optional[mx.array] = None
    max_seqlen: int = 0
    padding_side: PaddingSide = PaddingSide.LEFT

    @property
    def batch_size(self) -> int:
        """Number of sequences in batch."""
        return self.input_ids.shape[0]

    def get_last_token_positions(self) -> mx.array:
        """Get position of last real token for each sequence.

        Returns:
            Indices of last tokens (batch,).
        """
        if self.padding_side == PaddingSide.LEFT:
            # For left padding, last token is at max_seqlen - 1
            return mx.full((self.batch_size,), self.max_seqlen - 1, dtype=mx.int32)
        else:
            # For right padding, last token is at sequence_length - 1
            return self.sequence_lengths - 1

    def get_last_token_logits(self, logits: mx.array) -> mx.array:
        """Extract logits at last token position for each sequence.

        Args:
            logits: Model output (batch, seq_len, vocab_size).

        Returns:
            Logits at last position (batch, vocab_size).
        """
        batch_indices = mx.arange(self.batch_size)
        last_positions = self.get_last_token_positions()
        return logits[batch_indices, last_positions]


class SequenceBatcher:
    """Efficient batching of variable-length sequences.

    Handles padding and attention mask creation for batched inference.

    Example:
        >>> batcher = SequenceBatcher(max_batch_tokens=8192, max_batch_size=32)
        >>> sequences = [mx.array([1, 2, 3]), mx.array([4, 5])]
        >>> batch = batcher.create_batch(sequences)
        >>> # batch.input_ids: [[0, 1, 2, 3], [0, 0, 4, 5]] (left-padded)
    """

    def __init__(
        self,
        max_batch_tokens: int = 8192,
        max_batch_size: int = 64,
        padding_side: PaddingSide = PaddingSide.LEFT,
        pad_token_id: int = 0,
        strategy: BatchingStrategy = BatchingStrategy.PADDED,
    ):
        """Initialize batcher.

        Args:
            max_batch_tokens: Maximum total tokens in a batch.
            max_batch_size: Maximum sequences in a batch.
            padding_side: Side to pad sequences.
            pad_token_id: Token ID to use for padding.
            strategy: Batching strategy.
        """
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_size = max_batch_size
        self._padding_side = padding_side
        self._pad_token_id = pad_token_id
        self._strategy = strategy

    @property
    def max_batch_tokens(self) -> int:
        """Maximum tokens per batch."""
        return self._max_batch_tokens

    @property
    def max_batch_size(self) -> int:
        """Maximum sequences per batch."""
        return self._max_batch_size

    def create_batch(
        self,
        sequences: List[mx.array],
        batch_indices: Optional[List[int]] = None,
    ) -> BatchedSequences:
        """Create a padded batch from variable-length sequences.

        Args:
            sequences: List of token ID arrays.
            batch_indices: Optional mapping to original request indices.

        Returns:
            BatchedSequences with padded tensors.
        """
        if not sequences:
            return BatchedSequences(
                input_ids=mx.zeros((0, 0), dtype=mx.int32),
                attention_mask=mx.zeros((0, 0), dtype=mx.int32),
                position_ids=mx.zeros((0, 0), dtype=mx.int32),
                sequence_lengths=mx.zeros((0,), dtype=mx.int32),
                batch_indices=[],
                max_seqlen=0,
                padding_side=self._padding_side,
            )

        if batch_indices is None:
            batch_indices = list(range(len(sequences)))

        # Get sequence lengths
        lengths = [seq.shape[0] for seq in sequences]
        max_len = max(lengths)

        # Pad sequences
        padded_ids = []
        attention_masks = []
        position_ids_list = []

        for seq in sequences:
            seq_len = seq.shape[0]
            pad_len = max_len - seq_len

            if self._padding_side == PaddingSide.LEFT:
                # Left padding for generation
                if pad_len > 0:
                    padding = mx.full((pad_len,), self._pad_token_id, dtype=mx.int32)
                    padded = mx.concatenate([padding, seq])
                    mask = mx.concatenate([mx.zeros(pad_len), mx.ones(seq_len)])
                    pos_ids = mx.concatenate([
                        mx.zeros(pad_len, dtype=mx.int32),
                        mx.arange(seq_len, dtype=mx.int32),
                    ])
                else:
                    padded = seq
                    mask = mx.ones(seq_len)
                    pos_ids = mx.arange(seq_len, dtype=mx.int32)
            else:
                # Right padding
                if pad_len > 0:
                    padding = mx.full((pad_len,), self._pad_token_id, dtype=mx.int32)
                    padded = mx.concatenate([seq, padding])
                    mask = mx.concatenate([mx.ones(seq_len), mx.zeros(pad_len)])
                    pos_ids = mx.concatenate([
                        mx.arange(seq_len, dtype=mx.int32),
                        mx.zeros(pad_len, dtype=mx.int32),
                    ])
                else:
                    padded = seq
                    mask = mx.ones(seq_len)
                    pos_ids = mx.arange(seq_len, dtype=mx.int32)

            padded_ids.append(padded)
            attention_masks.append(mask)
            position_ids_list.append(pos_ids)

        return BatchedSequences(
            input_ids=mx.stack(padded_ids),
            attention_mask=mx.stack(attention_masks).astype(mx.int32),
            position_ids=mx.stack(position_ids_list),
            sequence_lengths=mx.array(lengths, dtype=mx.int32),
            batch_indices=batch_indices,
            max_seqlen=max_len,
            padding_side=self._padding_side,
        )

    def can_add_sequence(
        self,
        current_batch_tokens: int,
        current_batch_size: int,
        new_sequence_length: int,
    ) -> bool:
        """Check if a sequence can be added to current batch.

        Args:
            current_batch_tokens: Current total tokens in batch.
            current_batch_size: Current number of sequences.
            new_sequence_length: Length of new sequence.

        Returns:
            True if sequence can be added.
        """
        if current_batch_size >= self._max_batch_size:
            return False
        if current_batch_tokens + new_sequence_length > self._max_batch_tokens:
            return False
        return True

    def estimate_batch_memory(self, sequences: List[mx.array]) -> int:
        """Estimate memory usage for a batch.

        Args:
            sequences: Sequences to batch.

        Returns:
            Estimated bytes.
        """
        if not sequences:
            return 0

        max_len = max(seq.shape[0] for seq in sequences)
        batch_size = len(sequences)

        # input_ids + attention_mask + position_ids
        # Each is (batch_size, max_len) of int32
        return batch_size * max_len * 4 * 3


def create_attention_mask(
    sequence_lengths: mx.array,
    max_len: int,
    causal: bool = True,
    padding_side: PaddingSide = PaddingSide.LEFT,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create attention mask handling both padding and causality.

    Args:
        sequence_lengths: Original lengths (batch,).
        max_len: Maximum sequence length.
        causal: Apply causal masking.
        padding_side: Which side is padded.
        dtype: Output dtype.

    Returns:
        Attention mask (batch, 1, max_len, max_len).
        Valid positions are 0.0, masked positions are -inf.
    """
    batch_size = sequence_lengths.shape[0]

    # Create position indices
    positions = mx.arange(max_len)  # (max_len,)

    if padding_side == PaddingSide.LEFT:
        # For left padding, valid positions are at the end
        # Position i is valid if i >= (max_len - seq_len)
        start_positions = max_len - sequence_lengths  # (batch,)
        padding_mask = positions[None, :] >= start_positions[:, None]  # (batch, max_len)
    else:
        # For right padding, valid positions are at the beginning
        # Position i is valid if i < seq_len
        padding_mask = positions[None, :] < sequence_lengths[:, None]  # (batch, max_len)

    # Expand for attention: (batch, 1, 1, max_len)
    padding_mask = padding_mask[:, None, None, :]

    if causal:
        # Causal mask: position i can attend to positions <= i
        causal_mask = mx.tril(mx.ones((max_len, max_len)), k=0)  # (max_len, max_len)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, max_len, max_len)

        # Combined: valid if both padding and causal allow
        # Query at position q can attend to key at position k if:
        # 1. k is a valid (non-padded) position
        # 2. k <= q (causal constraint)
        combined_mask = padding_mask & (causal_mask == 1)
    else:
        # Bidirectional: just padding mask
        combined_mask = padding_mask

    # Convert to additive mask
    return mx.where(combined_mask, mx.array(0.0, dtype=dtype), mx.array(float("-inf"), dtype=dtype))


def create_combined_mask(
    sequence_lengths: mx.array,
    max_len: int,
    causal: bool = True,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create combined padding + causal mask.

    Simplified interface for common use case.

    Args:
        sequence_lengths: Original lengths (batch,).
        max_len: Maximum sequence length.
        causal: Apply causal masking.
        dtype: Output dtype.

    Returns:
        Mask where valid positions are 0.0 and masked are -inf.
        Shape: (batch, 1, max_len, max_len).
    """
    return create_attention_mask(
        sequence_lengths,
        max_len,
        causal=causal,
        padding_side=PaddingSide.LEFT,
        dtype=dtype,
    )


def unbatch_outputs(
    outputs: mx.array,
    batch: BatchedSequences,
) -> List[mx.array]:
    """Unbatch outputs back to individual sequences.

    Args:
        outputs: Batched outputs (batch, max_len, ...).
        batch: BatchedSequences with metadata.

    Returns:
        List of unpadded outputs.
    """
    results = []
    for i in range(batch.batch_size):
        seq_len = int(batch.sequence_lengths[i].item())
        if batch.padding_side == PaddingSide.LEFT:
            # Left padded: take last seq_len positions
            start = batch.max_seqlen - seq_len
            results.append(outputs[i, start:])
        else:
            # Right padded: take first seq_len positions
            results.append(outputs[i, :seq_len])
    return results


class DynamicBatcher:
    """Dynamically forms optimal batches from a stream of requests.

    Groups requests to maximize throughput while respecting constraints.

    Example:
        >>> batcher = DynamicBatcher(max_batch_tokens=8192)
        >>> batcher.add_sequence(seq1, idx=0)
        >>> batcher.add_sequence(seq2, idx=1)
        >>> if batcher.should_flush():
        ...     batch = batcher.flush()
    """

    def __init__(
        self,
        max_batch_tokens: int = 8192,
        max_batch_size: int = 64,
        max_wait_sequences: int = 32,
        padding_side: PaddingSide = PaddingSide.LEFT,
        pad_token_id: int = 0,
    ):
        """Initialize dynamic batcher.

        Args:
            max_batch_tokens: Maximum tokens per batch.
            max_batch_size: Maximum sequences per batch.
            max_wait_sequences: Flush when this many sequences waiting.
            padding_side: Side to pad sequences.
            pad_token_id: Padding token ID.
        """
        self._batcher = SequenceBatcher(
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )
        self._max_wait = max_wait_sequences

        self._pending_sequences: List[mx.array] = []
        self._pending_indices: List[int] = []
        self._pending_tokens: int = 0

    @property
    def num_pending(self) -> int:
        """Number of sequences waiting."""
        return len(self._pending_sequences)

    @property
    def pending_tokens(self) -> int:
        """Total tokens in pending sequences."""
        return self._pending_tokens

    def add_sequence(self, sequence: mx.array, idx: int) -> bool:
        """Add a sequence to the pending batch.

        Args:
            sequence: Token IDs.
            idx: Request index.

        Returns:
            True if added, False if would exceed limits.
        """
        seq_len = sequence.shape[0]

        if not self._batcher.can_add_sequence(
            self._pending_tokens,
            len(self._pending_sequences),
            seq_len,
        ):
            return False

        self._pending_sequences.append(sequence)
        self._pending_indices.append(idx)
        self._pending_tokens += seq_len
        return True

    def should_flush(self) -> bool:
        """Check if batch should be flushed."""
        if not self._pending_sequences:
            return False
        if len(self._pending_sequences) >= self._max_wait:
            return True
        if self._pending_tokens >= self._batcher.max_batch_tokens * 0.8:
            return True
        return False

    def flush(self) -> Optional[BatchedSequences]:
        """Flush pending sequences into a batch.

        Returns:
            BatchedSequences or None if empty.
        """
        if not self._pending_sequences:
            return None

        batch = self._batcher.create_batch(
            self._pending_sequences,
            self._pending_indices,
        )

        self._pending_sequences = []
        self._pending_indices = []
        self._pending_tokens = 0

        return batch

    def clear(self) -> None:
        """Clear pending sequences without batching."""
        self._pending_sequences = []
        self._pending_indices = []
        self._pending_tokens = 0
