"""Text and sequence transforms for MLX.

This module provides transforms for text/sequence data:
- RandomMask: BERT-style random token masking
- SpanMask: T5-style span masking
- TokenDropout: Random token dropping
- pad_sequence: Pad sequences to same length
- pack_sequences: Efficient sequence packing
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx


def pad_sequence(
    sequences: List[mx.array],
    batch_first: bool = True,
    padding_value: float = 0.0,
    max_length: Optional[int] = None,
) -> mx.array:
    """Pad a list of variable-length sequences.

    Args:
        sequences: List of sequences (1D or 2D arrays).
        batch_first: If True, output is (batch, seq_len, ...).
        padding_value: Value for padding elements (default: 0).
        max_length: Maximum sequence length. If None, uses longest sequence.

    Returns:
        Padded tensor.

    Example:
        >>> seqs = [mx.array([1, 2, 3]), mx.array([4, 5]), mx.array([6])]
        >>> padded = pad_sequence(seqs)
        >>> padded.shape
        (3, 3)
    """
    # Get maximum length
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max_length if max_length is not None else max(lengths)

    # Get other dimensions if sequences are multi-dimensional
    if sequences[0].ndim > 1:
        trailing_dims = sequences[0].shape[1:]
    else:
        trailing_dims = ()

    # Create output tensor
    batch_size = len(sequences)
    if batch_first:
        output_shape = (batch_size, max_len) + trailing_dims
    else:
        output_shape = (max_len, batch_size) + trailing_dims

    # Initialize with padding value
    output = mx.full(output_shape, padding_value, dtype=sequences[0].dtype)

    # Fill in sequences
    padded_list = []
    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], max_len)
        if batch_first:
            # For each sequence, create a row with padding
            if length < max_len:
                pad_width = [(0, max_len - length)] + [(0, 0)] * len(trailing_dims)
                padded = mx.pad(seq[:length], pad_width, constant_values=padding_value)
            else:
                padded = seq[:max_len]
            padded_list.append(padded)
        else:
            pad_width = [(0, max_len - length)] + [(0, 0)] * len(trailing_dims)
            padded = mx.pad(seq[:length], pad_width, constant_values=padding_value)
            padded_list.append(padded)

    if batch_first:
        return mx.stack(padded_list, axis=0)
    else:
        return mx.stack(padded_list, axis=1)


def pad_to_length(
    sequence: mx.array,
    length: int,
    padding_value: float = 0.0,
    side: str = "right",
) -> mx.array:
    """Pad a single sequence to a specific length.

    Args:
        sequence: Input sequence.
        length: Target length.
        padding_value: Value for padding.
        side: 'left' or 'right' padding.

    Returns:
        Padded sequence.
    """
    current_len = sequence.shape[0]

    if current_len >= length:
        return sequence[:length]

    pad_amount = length - current_len

    if side == "right":
        pad_width = [(0, pad_amount)] + [(0, 0)] * (sequence.ndim - 1)
    else:
        pad_width = [(pad_amount, 0)] + [(0, 0)] * (sequence.ndim - 1)

    return mx.pad(sequence, pad_width, constant_values=padding_value)


def create_attention_mask(
    lengths: Union[List[int], mx.array],
    max_length: Optional[int] = None,
    dtype: mx.Dtype = mx.bool_,
) -> mx.array:
    """Create attention mask from sequence lengths.

    Args:
        lengths: List of sequence lengths.
        max_length: Maximum length (uses max(lengths) if None).
        dtype: Output dtype.

    Returns:
        Attention mask of shape (batch_size, max_length).

    Example:
        >>> lengths = [3, 5, 2]
        >>> mask = create_attention_mask(lengths, max_length=5)
        >>> # mask[0] = [True, True, True, False, False]
    """
    if isinstance(lengths, mx.array):
        lengths = lengths.tolist()

    max_len = max_length if max_length is not None else max(lengths)
    batch_size = len(lengths)

    # Create position indices
    positions = mx.arange(max_len)[None, :]  # (1, max_len)
    lengths_arr = mx.array(lengths)[:, None]  # (batch, 1)

    mask = positions < lengths_arr

    return mask.astype(dtype)


def create_causal_mask(
    seq_length: int,
    dtype: mx.Dtype = mx.bool_,
) -> mx.array:
    """Create causal (triangular) attention mask.

    Args:
        seq_length: Sequence length.
        dtype: Output dtype.

    Returns:
        Causal mask of shape (seq_length, seq_length).

    Example:
        >>> mask = create_causal_mask(4)
        >>> # [[True, False, False, False],
        >>> #  [True, True, False, False],
        >>> #  [True, True, True, False],
        >>> #  [True, True, True, True]]
    """
    return mx.tril(mx.ones((seq_length, seq_length), dtype=dtype))


class RandomMask:
    """BERT-style random token masking.

    Randomly selects tokens and:
    - 80% of the time: replace with mask_token_id
    - 10% of the time: replace with random token
    - 10% of the time: keep original

    Args:
        mask_prob: Probability of masking a token (default: 0.15).
        mask_token_id: ID of the [MASK] token.
        vocab_size: Vocabulary size for random replacement.
        special_token_ids: IDs to never mask (e.g., [CLS], [SEP], [PAD]).

    Example:
        >>> masker = RandomMask(mask_prob=0.15, mask_token_id=103, vocab_size=30522)
        >>> masked_ids, labels = masker(input_ids)
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 103,
        vocab_size: int = 30522,
        special_token_ids: Optional[Sequence[int]] = None,
    ):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids or [])

    def __call__(
        self, input_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Apply random masking.

        Args:
            input_ids: Token IDs of shape (seq_length,) or (batch, seq_length).

        Returns:
            Tuple of (masked_input_ids, labels) where labels is -100 for
            non-masked positions.
        """
        # Create mask for which tokens to mask
        mask_candidates = mx.random.uniform(shape=input_ids.shape) < self.mask_prob

        # Don't mask special tokens
        for special_id in self.special_token_ids:
            mask_candidates = mask_candidates & (input_ids != special_id)

        # Create labels (-100 for positions we don't predict)
        labels = mx.where(mask_candidates, input_ids, mx.array(-100))

        # Decide replacement strategy
        replace_probs = mx.random.uniform(shape=input_ids.shape)

        # 80% mask token
        use_mask_token = mask_candidates & (replace_probs < 0.8)
        # 10% random token
        use_random = mask_candidates & (replace_probs >= 0.8) & (replace_probs < 0.9)
        # 10% keep original (no change needed)

        # Apply replacements
        masked_ids = input_ids
        masked_ids = mx.where(use_mask_token, self.mask_token_id, masked_ids)

        random_tokens = mx.random.randint(0, self.vocab_size, shape=input_ids.shape)
        masked_ids = mx.where(use_random, random_tokens, masked_ids)

        return masked_ids, labels


class SpanMask:
    """T5-style span masking.

    Masks contiguous spans of tokens and replaces each span with
    a single sentinel token.

    Args:
        mask_ratio: Fraction of tokens to mask (default: 0.15).
        mean_span_length: Average span length (default: 3).
        sentinel_start_id: Starting ID for sentinel tokens.

    Example:
        >>> masker = SpanMask(mask_ratio=0.15, mean_span_length=3)
        >>> masked_ids, labels = masker(input_ids)
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mean_span_length: float = 3.0,
        sentinel_start_id: int = 32000,
    ):
        self.mask_ratio = mask_ratio
        self.mean_span_length = mean_span_length
        self.sentinel_start_id = sentinel_start_id

    def __call__(
        self, input_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Apply span masking.

        Args:
            input_ids: Token IDs of shape (seq_length,).

        Returns:
            Tuple of (masked_input_ids, target_ids).
        """
        seq_length = input_ids.shape[0]
        num_to_mask = int(seq_length * self.mask_ratio)

        # Generate span lengths (geometric distribution approximation)
        spans = []
        total_masked = 0
        sentinel_id = self.sentinel_start_id

        while total_masked < num_to_mask:
            # Random span length
            span_length = max(1, int(mx.random.exponential() * self.mean_span_length))
            span_length = min(span_length, num_to_mask - total_masked)

            # Random start position
            max_start = seq_length - span_length - total_masked
            if max_start <= 0:
                break
            start = int(mx.random.randint(0, max_start))

            spans.append((start, span_length, sentinel_id))
            total_masked += span_length
            sentinel_id += 1

        # Sort spans by start position
        spans.sort(key=lambda x: x[0])

        # Build masked sequence and targets
        masked_tokens = []
        target_tokens = []
        prev_end = 0

        for start, length, sent_id in spans:
            # Add unmasked tokens
            if start > prev_end:
                masked_tokens.extend(input_ids[prev_end:start].tolist())

            # Add sentinel for masked span
            masked_tokens.append(sent_id)

            # Add sentinel and span to targets
            target_tokens.append(sent_id)
            target_tokens.extend(input_ids[start : start + length].tolist())

            prev_end = start + length

        # Add remaining tokens
        masked_tokens.extend(input_ids[prev_end:].tolist())

        return mx.array(masked_tokens), mx.array(target_tokens)


class TokenDropout:
    """Randomly drop tokens from sequence.

    Args:
        drop_prob: Probability of dropping each token.
        min_length: Minimum sequence length to preserve (default: 1).

    Example:
        >>> dropout = TokenDropout(drop_prob=0.1)
        >>> dropped = dropout(input_ids)
    """

    def __init__(self, drop_prob: float = 0.1, min_length: int = 1):
        self.drop_prob = drop_prob
        self.min_length = min_length

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Apply token dropout.

        Args:
            input_ids: Token IDs of shape (seq_length,).

        Returns:
            Token IDs with some tokens dropped.
        """
        seq_length = input_ids.shape[0]

        # Create keep mask
        keep_mask = mx.random.uniform(shape=(seq_length,)) >= self.drop_prob

        # Ensure minimum length
        num_kept = int(mx.sum(keep_mask))
        if num_kept < self.min_length:
            # Force keep random tokens to meet minimum length
            perm = mx.random.permutation(seq_length)
            keep_indices = perm[: self.min_length]
            # Create mask from indices
            keep_mask = mx.zeros((seq_length,), dtype=mx.bool_)
            for idx in keep_indices.tolist():
                keep_mask = mx.where(
                    mx.arange(seq_length) == idx,
                    mx.array(True),
                    keep_mask
                )

        # Filter tokens using boolean indexing
        kept_indices = mx.array([i for i in range(seq_length) if keep_mask[i]])
        return input_ids[kept_indices]


def pack_sequences(
    sequences: List[mx.array],
    max_length: int,
    padding_value: int = 0,
    eos_token_id: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """Pack multiple sequences into fixed-length chunks.

    Efficiently combines short sequences to minimize padding waste.

    Args:
        sequences: List of token ID sequences.
        max_length: Maximum chunk length.
        padding_value: Padding token ID.
        eos_token_id: If provided, add between concatenated sequences.

    Returns:
        Tuple of (packed_sequences, attention_mask).

    Example:
        >>> seqs = [mx.array([1, 2, 3]), mx.array([4, 5]), mx.array([6, 7, 8, 9])]
        >>> packed, mask = pack_sequences(seqs, max_length=6)
        >>> # packed[0] might be [1, 2, 3, 4, 5, 0] (two sequences packed)
    """
    packed_chunks = []
    current_chunk = []
    current_length = 0

    for seq in sequences:
        seq_list = seq.tolist()

        # Add EOS if needed
        if eos_token_id is not None:
            seq_list = seq_list + [eos_token_id]

        seq_length = len(seq_list)

        # Check if sequence fits in current chunk
        if current_length + seq_length <= max_length:
            current_chunk.extend(seq_list)
            current_length += seq_length
        else:
            # Save current chunk and start new one
            if current_chunk:
                packed_chunks.append(current_chunk)
            current_chunk = seq_list
            current_length = seq_length

    # Don't forget last chunk
    if current_chunk:
        packed_chunks.append(current_chunk)

    # Pad all chunks to max_length
    padded_chunks = []
    masks = []

    for chunk in packed_chunks:
        pad_length = max_length - len(chunk)
        padded = chunk + [padding_value] * pad_length
        mask = [1] * len(chunk) + [0] * pad_length

        padded_chunks.append(padded)
        masks.append(mask)

    return mx.array(padded_chunks), mx.array(masks, dtype=mx.bool_)


def truncate_sequences(
    sequences: List[mx.array],
    max_length: int,
    truncation_side: str = "right",
) -> List[mx.array]:
    """Truncate sequences to maximum length.

    Args:
        sequences: List of sequences.
        max_length: Maximum length.
        truncation_side: 'left' or 'right'.

    Returns:
        List of truncated sequences.
    """
    truncated = []
    for seq in sequences:
        if seq.shape[0] > max_length:
            if truncation_side == "right":
                truncated.append(seq[:max_length])
            else:
                truncated.append(seq[-max_length:])
        else:
            truncated.append(seq)
    return truncated
