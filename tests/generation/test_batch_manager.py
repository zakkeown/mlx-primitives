"""Tests for batch manager with NumPy reference validation."""

import pytest
import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from mlx_primitives.generation import (
    SequenceBatcher,
    BatchedSequences,
    DynamicBatcher,
    PaddingSide,
    create_attention_mask,
    create_combined_mask,
    unbatch_outputs,
)


def numpy_pad_sequences_left(
    sequences: list[NDArray],
    pad_token_id: int = 0,
) -> tuple[NDArray, NDArray, NDArray]:
    """NumPy reference for left-padding sequences.

    Returns:
        input_ids: (batch, max_len)
        attention_mask: (batch, max_len) - 1 for real, 0 for pad
        position_ids: (batch, max_len)
    """
    if not sequences:
        return np.array([]), np.array([]), np.array([])

    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
    position_ids = np.zeros((batch_size, max_len), dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        pad_len = max_len - seq_len
        input_ids[i, pad_len:] = seq
        attention_mask[i, pad_len:] = 1
        position_ids[i, pad_len:] = np.arange(seq_len)

    return input_ids, attention_mask, position_ids


def numpy_pad_sequences_right(
    sequences: list[NDArray],
    pad_token_id: int = 0,
) -> tuple[NDArray, NDArray, NDArray]:
    """NumPy reference for right-padding sequences."""
    if not sequences:
        return np.array([]), np.array([]), np.array([])

    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
    position_ids = np.zeros((batch_size, max_len), dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
        position_ids[i, :seq_len] = np.arange(seq_len)

    return input_ids, attention_mask, position_ids


def numpy_create_causal_mask(max_len: int) -> NDArray:
    """NumPy reference for causal mask."""
    return np.tril(np.ones((max_len, max_len)))


def numpy_create_attention_mask(
    sequence_lengths: NDArray,
    max_len: int,
    causal: bool = True,
    padding_side: str = "left",
) -> NDArray:
    """NumPy reference for combined padding + causal mask.

    Returns:
        Mask (batch, 1, max_len, max_len) where 0.0 = valid, -inf = masked.
    """
    batch_size = len(sequence_lengths)
    positions = np.arange(max_len)

    if padding_side == "left":
        start_positions = max_len - sequence_lengths
        padding_mask = positions[None, :] >= start_positions[:, None]
    else:
        padding_mask = positions[None, :] < sequence_lengths[:, None]

    # Expand to (batch, 1, 1, max_len)
    padding_mask = padding_mask[:, None, None, :]

    if causal:
        causal_mask = numpy_create_causal_mask(max_len)
        causal_mask = causal_mask[None, None, :, :]
        combined_mask = padding_mask & (causal_mask == 1)
    else:
        combined_mask = padding_mask

    return np.where(combined_mask, 0.0, float("-inf"))


class TestSequenceBatcherLeftPadding:
    """Tests for SequenceBatcher with left padding."""

    def test_basic_batching(self) -> None:
        """Test basic sequence batching."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4, 5], dtype=mx.int32),
        ]

        batch = batcher.create_batch(sequences)

        assert batch.batch_size == 2
        assert batch.max_seqlen == 3

    def test_matches_numpy_reference(self) -> None:
        """Test batching matches NumPy reference."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT, pad_token_id=0)
        sequences = [
            mx.array([1, 2, 3, 4], dtype=mx.int32),
            mx.array([5, 6], dtype=mx.int32),
            mx.array([7, 8, 9], dtype=mx.int32),
        ]

        batch = batcher.create_batch(sequences)

        # NumPy reference
        np_seqs = [np.array([1, 2, 3, 4]), np.array([5, 6]), np.array([7, 8, 9])]
        exp_ids, exp_mask, exp_pos = numpy_pad_sequences_left(np_seqs, pad_token_id=0)

        np.testing.assert_array_equal(np.array(batch.input_ids), exp_ids)
        np.testing.assert_array_equal(np.array(batch.attention_mask), exp_mask)
        np.testing.assert_array_equal(np.array(batch.position_ids), exp_pos)

    def test_sequence_lengths_preserved(self) -> None:
        """Test original sequence lengths are preserved."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4], dtype=mx.int32),
            mx.array([5, 6, 7, 8], dtype=mx.int32),
        ]

        batch = batcher.create_batch(sequences)

        np.testing.assert_array_equal(
            np.array(batch.sequence_lengths), [3, 1, 4]
        )

    def test_get_last_token_positions(self) -> None:
        """Test last token position extraction for left padding."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),  # length 3
            mx.array([4, 5], dtype=mx.int32),     # length 2
        ]

        batch = batcher.create_batch(sequences)
        last_positions = batch.get_last_token_positions()

        # For left padding, last token is always at max_seqlen - 1
        assert int(last_positions[0]) == 2
        assert int(last_positions[1]) == 2

    def test_batch_indices_mapping(self) -> None:
        """Test batch indices are preserved."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [mx.array([1, 2], dtype=mx.int32), mx.array([3, 4, 5], dtype=mx.int32)]

        batch = batcher.create_batch(sequences, batch_indices=[5, 10])

        assert batch.batch_indices == [5, 10]


class TestSequenceBatcherRightPadding:
    """Tests for SequenceBatcher with right padding."""

    def test_basic_batching(self) -> None:
        """Test basic sequence batching with right padding."""
        batcher = SequenceBatcher(padding_side=PaddingSide.RIGHT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4, 5], dtype=mx.int32),
        ]

        batch = batcher.create_batch(sequences)

        # Check input_ids padding is on right
        ids = np.array(batch.input_ids)
        assert ids[0, 0] == 1  # First token of first seq
        assert ids[1, 0] == 4  # First token of second seq
        assert ids[1, 2] == 0  # Padding on right

    def test_matches_numpy_reference(self) -> None:
        """Test right padding matches NumPy reference."""
        batcher = SequenceBatcher(padding_side=PaddingSide.RIGHT, pad_token_id=0)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4, 5, 6, 7], dtype=mx.int32),
        ]

        batch = batcher.create_batch(sequences)

        np_seqs = [np.array([1, 2, 3]), np.array([4, 5, 6, 7])]
        exp_ids, exp_mask, exp_pos = numpy_pad_sequences_right(np_seqs, pad_token_id=0)

        np.testing.assert_array_equal(np.array(batch.input_ids), exp_ids)
        np.testing.assert_array_equal(np.array(batch.attention_mask), exp_mask)

    def test_get_last_token_positions(self) -> None:
        """Test last token position extraction for right padding."""
        batcher = SequenceBatcher(padding_side=PaddingSide.RIGHT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),  # length 3
            mx.array([4, 5], dtype=mx.int32),     # length 2
        ]

        batch = batcher.create_batch(sequences)
        last_positions = batch.get_last_token_positions()

        # For right padding, last token is at sequence_length - 1
        assert int(last_positions[0]) == 2  # 3 - 1
        assert int(last_positions[1]) == 1  # 2 - 1


class TestAttentionMask:
    """Tests for attention mask creation."""

    def test_create_attention_mask_left_padding(self) -> None:
        """Test attention mask with left padding."""
        sequence_lengths = mx.array([3, 2], dtype=mx.int32)
        max_len = 4

        mask = create_attention_mask(
            sequence_lengths, max_len,
            causal=True, padding_side=PaddingSide.LEFT
        )

        expected = numpy_create_attention_mask(
            np.array([3, 2]), max_len,
            causal=True, padding_side="left"
        )

        np.testing.assert_allclose(np.array(mask), expected, rtol=1e-5)

    def test_create_attention_mask_right_padding(self) -> None:
        """Test attention mask with right padding."""
        sequence_lengths = mx.array([3, 2], dtype=mx.int32)
        max_len = 4

        mask = create_attention_mask(
            sequence_lengths, max_len,
            causal=True, padding_side=PaddingSide.RIGHT
        )

        expected = numpy_create_attention_mask(
            np.array([3, 2]), max_len,
            causal=True, padding_side="right"
        )

        np.testing.assert_allclose(np.array(mask), expected, rtol=1e-5)

    def test_mask_shape(self) -> None:
        """Test attention mask has correct shape."""
        sequence_lengths = mx.array([5, 3, 4], dtype=mx.int32)
        max_len = 6

        mask = create_attention_mask(
            sequence_lengths, max_len,
            causal=True, padding_side=PaddingSide.LEFT
        )

        assert mask.shape == (3, 1, 6, 6)

    def test_non_causal_mask(self) -> None:
        """Test non-causal (bidirectional) mask."""
        sequence_lengths = mx.array([3, 2], dtype=mx.int32)
        max_len = 4

        mask = create_attention_mask(
            sequence_lengths, max_len,
            causal=False, padding_side=PaddingSide.LEFT
        )

        mask_np = np.array(mask)
        # Non-causal mask has shape (batch, 1, 1, max_len) for key masking
        # Sequence 0 (length 3): positions 1, 2, 3 are valid (left-padded)
        # Sequence 1 (length 2): positions 2, 3 are valid
        assert mask_np[0, 0, 0, 1] == 0.0  # Valid key position for seq 0
        assert mask_np[0, 0, 0, 0] == float("-inf")  # Padded position for seq 0
        assert mask_np[1, 0, 0, 2] == 0.0  # Valid key position for seq 1
        assert mask_np[1, 0, 0, 0] == float("-inf")  # Padded position for seq 1

    def test_create_combined_mask(self) -> None:
        """Test convenience function for combined mask."""
        sequence_lengths = mx.array([4, 2, 3], dtype=mx.int32)
        max_len = 5

        mask = create_combined_mask(sequence_lengths, max_len, causal=True)

        assert mask.shape == (3, 1, 5, 5)


class TestBatchedSequences:
    """Tests for BatchedSequences methods."""

    def test_get_last_token_logits(self) -> None:
        """Test extracting last token logits."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4, 5], dtype=mx.int32),
        ]
        batch = batcher.create_batch(sequences)

        # Fake logits: (batch=2, seq=3, vocab=4)
        logits = mx.array([
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [1.0, 2.0, 3.0, 4.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]],
        ])

        last_logits = batch.get_last_token_logits(logits)

        # Both should get position 2 (last position for left padding)
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])
        np.testing.assert_array_equal(np.array(last_logits), expected)


class TestUnbatchOutputs:
    """Tests for unbatching outputs."""

    def test_unbatch_left_padded(self) -> None:
        """Test unbatching left-padded outputs."""
        batcher = SequenceBatcher(padding_side=PaddingSide.LEFT)
        sequences = [
            mx.array([1, 2, 3], dtype=mx.int32),  # length 3
            mx.array([4, 5], dtype=mx.int32),     # length 2
        ]
        batch = batcher.create_batch(sequences)

        # Fake outputs: (batch=2, max_len=3, hidden=2)
        outputs = mx.array([
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # seq 0: [pad, tok1, tok2, tok3] -> last 3
            [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0]],  # seq 1: [pad, pad, tok1, tok2] -> last 2
        ])

        unbatched = unbatch_outputs(outputs, batch)

        assert len(unbatched) == 2
        assert unbatched[0].shape == (3, 2)  # Original length 3
        assert unbatched[1].shape == (2, 2)  # Original length 2

    def test_unbatch_right_padded(self) -> None:
        """Test unbatching right-padded outputs."""
        batcher = SequenceBatcher(padding_side=PaddingSide.RIGHT)
        sequences = [
            mx.array([1, 2], dtype=mx.int32),   # length 2
            mx.array([3, 4, 5], dtype=mx.int32),  # length 3
        ]
        batch = batcher.create_batch(sequences)

        # Fake outputs: (batch=2, max_len=3, hidden=2)
        outputs = mx.array([
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],  # seq 0: first 2 are real
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],  # seq 1: all 3 are real
        ])

        unbatched = unbatch_outputs(outputs, batch)

        assert len(unbatched) == 2
        assert unbatched[0].shape == (2, 2)
        assert unbatched[1].shape == (3, 2)
        np.testing.assert_array_equal(np.array(unbatched[0]), [[1.0, 1.0], [2.0, 2.0]])


class TestDynamicBatcher:
    """Tests for DynamicBatcher."""

    def test_add_sequence(self) -> None:
        """Test adding sequences to dynamic batcher."""
        batcher = DynamicBatcher(max_batch_tokens=100, max_batch_size=4)

        result = batcher.add_sequence(mx.array([1, 2, 3], dtype=mx.int32), idx=0)

        assert result is True
        assert batcher.num_pending == 1
        assert batcher.pending_tokens == 3

    def test_batch_size_limit(self) -> None:
        """Test batch size limit is respected."""
        batcher = DynamicBatcher(max_batch_tokens=1000, max_batch_size=2)

        batcher.add_sequence(mx.array([1, 2], dtype=mx.int32), idx=0)
        batcher.add_sequence(mx.array([3, 4], dtype=mx.int32), idx=1)
        result = batcher.add_sequence(mx.array([5, 6], dtype=mx.int32), idx=2)

        assert result is False
        assert batcher.num_pending == 2

    def test_token_limit(self) -> None:
        """Test token limit is respected."""
        batcher = DynamicBatcher(max_batch_tokens=10, max_batch_size=100)

        batcher.add_sequence(mx.array([1, 2, 3, 4, 5], dtype=mx.int32), idx=0)
        batcher.add_sequence(mx.array([6, 7, 8, 9, 10], dtype=mx.int32), idx=1)
        result = batcher.add_sequence(mx.array([11], dtype=mx.int32), idx=2)

        assert result is False
        assert batcher.pending_tokens == 10

    def test_flush(self) -> None:
        """Test flushing pending sequences."""
        batcher = DynamicBatcher(max_batch_tokens=100, max_batch_size=10)

        batcher.add_sequence(mx.array([1, 2], dtype=mx.int32), idx=0)
        batcher.add_sequence(mx.array([3, 4, 5], dtype=mx.int32), idx=1)

        batch = batcher.flush()

        assert batch is not None
        assert batch.batch_size == 2
        assert batcher.num_pending == 0
        assert batcher.pending_tokens == 0

    def test_should_flush(self) -> None:
        """Test flush decision logic."""
        batcher = DynamicBatcher(
            max_batch_tokens=100,
            max_batch_size=10,
            max_wait_sequences=3,
        )

        assert not batcher.should_flush()  # Empty

        batcher.add_sequence(mx.array([1, 2], dtype=mx.int32), idx=0)
        batcher.add_sequence(mx.array([3, 4], dtype=mx.int32), idx=1)
        assert not batcher.should_flush()  # Not at max_wait

        batcher.add_sequence(mx.array([5, 6], dtype=mx.int32), idx=2)
        assert batcher.should_flush()  # At max_wait_sequences

    def test_clear(self) -> None:
        """Test clearing without batching."""
        batcher = DynamicBatcher(max_batch_tokens=100, max_batch_size=10)

        batcher.add_sequence(mx.array([1, 2], dtype=mx.int32), idx=0)
        batcher.add_sequence(mx.array([3, 4], dtype=mx.int32), idx=1)
        batcher.clear()

        assert batcher.num_pending == 0
        assert batcher.pending_tokens == 0


class TestSequenceBatcherConstraints:
    """Tests for batcher constraint checking."""

    def test_can_add_sequence_batch_size(self) -> None:
        """Test batch size constraint checking."""
        batcher = SequenceBatcher(max_batch_size=2)

        assert batcher.can_add_sequence(0, 0, 10) is True
        assert batcher.can_add_sequence(10, 1, 10) is True
        assert batcher.can_add_sequence(20, 2, 10) is False  # At limit

    def test_can_add_sequence_token_limit(self) -> None:
        """Test token limit constraint checking."""
        batcher = SequenceBatcher(max_batch_tokens=50)

        assert batcher.can_add_sequence(0, 0, 30) is True
        assert batcher.can_add_sequence(30, 1, 20) is True
        assert batcher.can_add_sequence(30, 2, 25) is False  # Would exceed

    def test_estimate_batch_memory(self) -> None:
        """Test memory estimation."""
        batcher = SequenceBatcher()
        sequences = [
            mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
            mx.array([6, 7, 8], dtype=mx.int32),
        ]

        memory = batcher.estimate_batch_memory(sequences)

        # 2 sequences, max_len=5, 3 arrays of int32
        expected = 2 * 5 * 4 * 3  # batch * max_len * sizeof(int32) * num_arrays
        assert memory == expected


class TestEmptyBatch:
    """Tests for edge cases with empty batches."""

    def test_empty_batch_creation(self) -> None:
        """Test creating batch from empty sequence list."""
        batcher = SequenceBatcher()

        batch = batcher.create_batch([])

        assert batch.batch_size == 0
        assert batch.max_seqlen == 0

    def test_flush_empty(self) -> None:
        """Test flushing empty dynamic batcher."""
        batcher = DynamicBatcher()

        batch = batcher.flush()

        assert batch is None
