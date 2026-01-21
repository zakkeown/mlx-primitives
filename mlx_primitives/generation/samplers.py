"""Token sampling strategies for text generation.

This module provides efficient batched token sampling with support
for various sampling strategies including greedy, top-k, top-p (nucleus),
temperature scaling, and repetition penalty.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.generation.requests import SamplingConfig


class TokenSampler:
    """Efficient batched token sampling.

    Supports multiple sampling strategies in a vectorized manner.

    Example:
        >>> sampler = TokenSampler()
        >>> logits = model(input_ids)  # (batch, vocab_size)
        >>> configs = [SamplingConfig(temperature=0.7) for _ in range(batch)]
        >>> tokens = sampler(logits, configs)  # (batch,)
    """

    def __init__(self, vocab_size: Optional[int] = None):
        """Initialize sampler.

        Args:
            vocab_size: Vocabulary size (for validation).
        """
        self._vocab_size = vocab_size

    def __call__(
        self,
        logits: mx.array,
        configs: List[SamplingConfig],
        generated_tokens: Optional[List[List[int]]] = None,
    ) -> mx.array:
        """Sample next tokens for entire batch.

        Args:
            logits: Logits from model (batch, vocab_size).
            configs: Sampling config for each batch element.
            generated_tokens: Previously generated tokens (for repetition penalty).

        Returns:
            Sampled token IDs (batch,).
        """
        batch_size = logits.shape[0]

        if len(configs) != batch_size:
            raise ValueError(
                f"Number of configs ({len(configs)}) must match batch size ({batch_size})"
            )

        # Fast path: all greedy
        if all(c.is_greedy() for c in configs):
            return mx.argmax(logits, axis=-1)

        # Check if all configs are the same (common case)
        if self._all_configs_equal(configs):
            return self._sample_uniform_config(
                logits, configs[0], generated_tokens
            )

        # Mixed configs: process individually
        return self._sample_mixed_configs(logits, configs, generated_tokens)

    def _all_configs_equal(self, configs: List[SamplingConfig]) -> bool:
        """Check if all configs are equal."""
        if len(configs) <= 1:
            return True
        first = configs[0]
        return all(
            c.temperature == first.temperature
            and c.top_k == first.top_k
            and c.top_p == first.top_p
            and c.repetition_penalty == first.repetition_penalty
            and c.presence_penalty == first.presence_penalty
            and c.frequency_penalty == first.frequency_penalty
            for c in configs[1:]
        )

    def _sample_uniform_config(
        self,
        logits: mx.array,
        config: SamplingConfig,
        generated_tokens: Optional[List[List[int]]],
    ) -> mx.array:
        """Sample with uniform config across batch."""
        # Apply repetition penalty if needed
        if config.repetition_penalty != 1.0 and generated_tokens:
            logits = apply_repetition_penalty_batch(
                logits, generated_tokens, config.repetition_penalty
            )

        # Apply presence penalty (batched version)
        if config.presence_penalty != 0.0 and generated_tokens:
            logits = apply_presence_penalty_batch(
                logits, generated_tokens, config.presence_penalty
            )

        # Apply frequency penalty (batched version)
        if config.frequency_penalty != 0.0 and generated_tokens:
            logits = apply_frequency_penalty_batch(
                logits, generated_tokens, config.frequency_penalty
            )

        # Apply temperature
        if config.temperature > 0:
            logits = logits / config.temperature

        # Apply top-k
        if config.top_k > 0:
            logits = apply_top_k(logits, config.top_k)

        # Apply top-p
        if config.top_p < 1.0:
            logits = apply_top_p(logits, config.top_p)

        # Sample
        if config.temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits)

    def _sample_mixed_configs(
        self,
        logits: mx.array,
        configs: List[SamplingConfig],
        generated_tokens: Optional[List[List[int]]],
    ) -> mx.array:
        """Sample with different configs per batch element."""
        batch_size = logits.shape[0]
        tokens = []

        for i in range(batch_size):
            config = configs[i]
            sample_logits = logits[i : i + 1]  # Keep batch dim
            gen_toks = [generated_tokens[i]] if generated_tokens and i < len(generated_tokens) else [[]]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0 and gen_toks[0]:
                sample_logits = apply_repetition_penalty_batch(
                    sample_logits, gen_toks, config.repetition_penalty,
                )

            # Apply presence penalty
            if config.presence_penalty != 0.0 and gen_toks[0]:
                sample_logits = apply_presence_penalty_batch(
                    sample_logits, gen_toks, config.presence_penalty,
                )

            # Apply frequency penalty
            if config.frequency_penalty != 0.0 and gen_toks[0]:
                sample_logits = apply_frequency_penalty_batch(
                    sample_logits, gen_toks, config.frequency_penalty,
                )

            # Apply temperature
            if config.temperature > 0:
                sample_logits = sample_logits / config.temperature

            # Apply top-k
            if config.top_k > 0:
                sample_logits = apply_top_k(sample_logits, config.top_k)

            # Apply top-p
            if config.top_p < 1.0:
                sample_logits = apply_top_p(sample_logits, config.top_p)

            # Sample
            if config.temperature == 0:
                token = mx.argmax(sample_logits, axis=-1)
            else:
                token = mx.random.categorical(sample_logits)

            tokens.append(token.squeeze())

        return mx.stack(tokens)

    def sample_with_scores(
        self,
        logits: mx.array,
        config: SamplingConfig,
        generated_tokens: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """Sample a single token and return its log probability.

        Args:
            logits: Logits for single position (vocab_size,).
            config: Sampling configuration.
            generated_tokens: Previously generated tokens.

        Returns:
            Tuple of (token_id, log_prob).
        """
        logits = logits[None, :]  # Add batch dim

        if config.repetition_penalty != 1.0 and generated_tokens:
            logits = apply_repetition_penalty_batch(
                logits, [generated_tokens], config.repetition_penalty
            )

        if config.temperature > 0:
            logits = logits / config.temperature

        if config.top_k > 0:
            logits = apply_top_k(logits, config.top_k)

        if config.top_p < 1.0:
            logits = apply_top_p(logits, config.top_p)

        # Compute log probs
        log_probs = mx.log_softmax(logits, axis=-1)

        # Sample
        if config.temperature == 0:
            token = int(mx.argmax(logits, axis=-1).item())
        else:
            token = int(mx.random.categorical(logits).item())

        log_prob = float(log_probs[0, token].item())

        return token, log_prob


def apply_temperature(logits: mx.array, temperature: float) -> mx.array:
    """Apply temperature scaling to logits.

    Args:
        logits: Input logits (batch, vocab_size).
        temperature: Temperature value. Higher = more random.

    Returns:
        Scaled logits.
    """
    if temperature == 1.0:
        return logits
    if temperature == 0.0:
        return logits  # Will use argmax anyway
    return logits / temperature


def apply_top_k(logits: mx.array, k: int) -> mx.array:
    """Apply top-k filtering to logits.

    Sets logits outside top-k to -inf.

    Args:
        logits: Input logits (batch, vocab_size).
        k: Number of top tokens to keep.

    Returns:
        Filtered logits.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits

    # Get k-th largest value for each batch
    # Use ascending sort and take the k-th element from the end
    sorted_logits = mx.sort(logits, axis=-1)  # Ascending
    # Note: [:, -k:-k+1] fails when k=1 (becomes [:, -1:0] which is empty)
    # Instead, use [:, -k] and reshape to preserve the batch dimension
    threshold = sorted_logits[:, -k : sorted_logits.shape[-1] - k + 1]  # k-th largest, shape (batch, 1)

    # Mask values below threshold
    return mx.where(logits >= threshold, logits, mx.array(float("-inf")))


def apply_top_p(logits: mx.array, p: float) -> mx.array:
    """Apply nucleus (top-p) sampling to logits.

    Keeps smallest set of tokens with cumulative probability >= p.

    Args:
        logits: Input logits (batch, vocab_size).
        p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits

    # Sort by descending probability
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]  # Descending
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Compute cumulative probabilities in fp32 for numerical stability
    # bf16/fp16 cumsum accumulates significant error over large vocab sizes
    sorted_probs = mx.softmax(sorted_logits.astype(mx.float32), axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Find cutoff: keep tokens until cumulative prob exceeds p
    # Shift cumulative probs to include first token always
    shifted_cumulative = mx.concatenate(
        [mx.zeros((logits.shape[0], 1)), cumulative_probs[:, :-1]], axis=-1
    )
    sorted_mask = shifted_cumulative < p

    # Ensure at least the top token is always kept to prevent all-inf logits
    # This handles edge cases like p=0 or numerical precision issues
    first_token_mask = mx.zeros_like(sorted_mask)
    first_token_mask = first_token_mask.at[:, 0].add(True)
    sorted_mask = mx.logical_or(sorted_mask, first_token_mask)

    # Set filtered positions to -inf
    sorted_logits = mx.where(sorted_mask, sorted_logits, mx.array(float("-inf")))

    # Unsort back to original order
    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    return mx.take_along_axis(sorted_logits, inverse_indices, axis=-1)


def apply_repetition_penalty(
    logits: mx.array,
    token_ids: List[int],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to logits.

    Reduces probability of tokens that have already been generated.
    Uses vectorized MLX operations for efficiency.

    Args:
        logits: Input logits (batch, vocab_size) or (vocab_size,).
        token_ids: Previously generated token IDs.
        penalty: Penalty factor. >1 reduces repetition.

    Returns:
        Penalized logits.
    """
    if penalty == 1.0 or not token_ids:
        return logits

    # Ensure 2D for consistent handling
    was_1d = logits.ndim == 1
    if was_1d:
        logits = logits[None, :]

    vocab_size = logits.shape[-1]

    # Get unique tokens within vocab range
    unique_tokens = [t for t in set(token_ids) if 0 <= t < vocab_size]
    if not unique_tokens:
        return logits.squeeze(0) if was_1d else logits

    # Create mask for tokens to penalize (vocab_size,)
    token_indices = mx.array(unique_tokens, dtype=mx.int32)
    penalty_mask = mx.zeros((vocab_size,), dtype=mx.bool_)
    penalty_mask = penalty_mask.at[token_indices].add(True)

    # Compute what the row should be after penalty
    # Positive logits: divide by penalty
    # Negative logits: multiply by penalty
    positive_row = logits[0] / penalty
    negative_row = logits[0] * penalty
    penalized_row = mx.where(logits[0] > 0, positive_row, negative_row)

    # Apply only at penalized positions
    new_row = mx.where(penalty_mask, penalized_row, logits[0])
    result = new_row[None, :]

    if was_1d:
        result = result.squeeze(0)

    return result


def apply_repetition_penalty_batch(
    logits: mx.array,
    generated_tokens: List[List[int]],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to batched logits.

    Uses vectorized MLX operations for efficiency.

    Args:
        logits: Input logits (batch, vocab_size).
        generated_tokens: Previously generated tokens for each batch.
        penalty: Penalty factor.

    Returns:
        Penalized logits.
    """
    if penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Build a mask of shape (batch_size, vocab_size) indicating which tokens to penalize
    # For each batch element, mark the tokens that appear in its generated_tokens
    penalty_mask = mx.zeros((batch_size, vocab_size), dtype=mx.bool_)

    for i in range(batch_size):
        if i < len(generated_tokens) and generated_tokens[i]:
            unique_tokens = [t for t in set(generated_tokens[i]) if 0 <= t < vocab_size]
            if unique_tokens:
                token_indices = mx.array(unique_tokens, dtype=mx.int32)
                # Set mask to True at these token positions for batch element i
                row_mask = mx.zeros((vocab_size,), dtype=mx.bool_)
                row_mask = row_mask.at[token_indices].add(True)
                penalty_mask = penalty_mask.at[i].add(row_mask)

    # Compute penalized logits for all positions
    # Positive logits: divide by penalty
    # Negative logits: multiply by penalty
    penalized_positive = logits / penalty
    penalized_negative = logits * penalty
    penalized_logits = mx.where(logits > 0, penalized_positive, penalized_negative)

    # Apply penalty only where mask is True
    result = mx.where(penalty_mask, penalized_logits, logits)

    return result


def apply_presence_penalty(
    logits: mx.array,
    token_ids: List[int],
    penalty: float,
) -> mx.array:
    """Apply presence penalty to logits.

    Subtracts a fixed penalty from tokens that have appeared.

    Args:
        logits: Input logits (vocab_size,).
        token_ids: Previously generated token IDs.
        penalty: Penalty to subtract.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0 or not token_ids:
        return logits

    vocab_size = logits.shape[-1]
    unique_tokens = [t for t in set(token_ids) if 0 <= t < vocab_size]
    if not unique_tokens:
        return logits

    # Create mask for tokens to penalize
    token_indices = mx.array(unique_tokens, dtype=mx.int32)
    penalty_mask = mx.zeros((vocab_size,), dtype=mx.bool_)
    penalty_mask = penalty_mask.at[token_indices].add(True)

    # Subtract penalty where mask is True
    penalty_values = mx.where(penalty_mask, mx.array(penalty), mx.array(0.0))
    return logits - penalty_values


def apply_presence_penalty_batch(
    logits: mx.array,
    generated_tokens: List[List[int]],
    penalty: float,
) -> mx.array:
    """Apply presence penalty to batched logits.

    Uses vectorized MLX operations for efficiency.

    Args:
        logits: Input logits (batch, vocab_size).
        generated_tokens: Previously generated tokens for each batch.
        penalty: Penalty to subtract.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Build a mask of shape (batch_size, vocab_size) indicating which tokens to penalize
    penalty_mask = mx.zeros((batch_size, vocab_size), dtype=mx.bool_)

    for i in range(batch_size):
        if i < len(generated_tokens) and generated_tokens[i]:
            unique_tokens = [t for t in set(generated_tokens[i]) if 0 <= t < vocab_size]
            if unique_tokens:
                token_indices = mx.array(unique_tokens, dtype=mx.int32)
                row_mask = mx.zeros((vocab_size,), dtype=mx.bool_)
                row_mask = row_mask.at[token_indices].add(True)
                penalty_mask = penalty_mask.at[i].add(row_mask)

    # Subtract penalty where mask is True
    penalty_values = mx.where(penalty_mask, mx.array(penalty), mx.array(0.0))
    return logits - penalty_values


def apply_frequency_penalty(
    logits: mx.array,
    token_ids: List[int],
    penalty: float,
) -> mx.array:
    """Apply frequency penalty to logits.

    Subtracts penalty proportional to token frequency.

    Args:
        logits: Input logits (vocab_size,).
        token_ids: Previously generated token IDs.
        penalty: Penalty per occurrence.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0 or not token_ids:
        return logits

    vocab_size = logits.shape[-1]

    # Count frequencies
    freq: Dict[int, int] = {}
    for token_id in token_ids:
        if 0 <= token_id < vocab_size:
            freq[token_id] = freq.get(token_id, 0) + 1

    if not freq:
        return logits

    # Create frequency penalty array
    freq_penalties = mx.zeros((vocab_size,), dtype=logits.dtype)
    token_indices = mx.array(list(freq.keys()), dtype=mx.int32)
    counts = mx.array(list(freq.values()), dtype=logits.dtype)
    freq_penalties = freq_penalties.at[token_indices].add(counts * penalty)

    return logits - freq_penalties


def apply_frequency_penalty_batch(
    logits: mx.array,
    generated_tokens: List[List[int]],
    penalty: float,
) -> mx.array:
    """Apply frequency penalty to batched logits.

    Uses vectorized MLX operations for efficiency.

    Args:
        logits: Input logits (batch, vocab_size).
        generated_tokens: Previously generated tokens for each batch.
        penalty: Penalty per occurrence.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Build frequency penalty matrix of shape (batch_size, vocab_size)
    freq_penalties = mx.zeros((batch_size, vocab_size), dtype=logits.dtype)

    for i in range(batch_size):
        if i < len(generated_tokens) and generated_tokens[i]:
            # Count frequencies for this batch element
            freq: Dict[int, int] = {}
            for token_id in generated_tokens[i]:
                if 0 <= token_id < vocab_size:
                    freq[token_id] = freq.get(token_id, 0) + 1

            if freq:
                token_indices = mx.array(list(freq.keys()), dtype=mx.int32)
                counts = mx.array(list(freq.values()), dtype=logits.dtype)
                row_penalties = mx.zeros((vocab_size,), dtype=logits.dtype)
                row_penalties = row_penalties.at[token_indices].add(counts * penalty)
                freq_penalties = freq_penalties.at[i].add(row_penalties)

    return logits - freq_penalties


def sample_greedy(logits: mx.array) -> mx.array:
    """Greedy (argmax) sampling.

    Args:
        logits: Input logits (batch, vocab_size).

    Returns:
        Token IDs (batch,).
    """
    return mx.argmax(logits, axis=-1)


def sample_multinomial(logits: mx.array) -> mx.array:
    """Multinomial sampling from logits.

    Args:
        logits: Input logits (batch, vocab_size).

    Returns:
        Token IDs (batch,).
    """
    return mx.random.categorical(logits)


def apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    """Apply min-p filtering to logits.

    Keeps tokens with probability >= min_p * max_prob, where max_prob
    is the maximum probability in each row. This dynamically adjusts
    the threshold based on the confidence of the top prediction.

    Args:
        logits: Input logits (batch, vocab_size).
        min_p: Minimum probability ratio threshold (0 to 1).
            Higher values are more selective.

    Returns:
        Filtered logits with low-probability tokens set to -inf.
    """
    if min_p <= 0.0:
        return logits

    # Convert logits to probabilities using softmax
    # Use float32 for numerical stability
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)

    # Get max probability for each batch element
    max_prob = mx.max(probs, axis=-1, keepdims=True)

    # Compute threshold: min_p * max_prob
    threshold = min_p * max_prob

    # Create mask for tokens to keep (prob >= threshold)
    keep_mask = probs >= threshold

    # Ensure at least the top token is always kept
    # This handles edge cases where min_p is very high
    top_mask = probs == max_prob
    keep_mask = mx.logical_or(keep_mask, top_mask)

    # Set filtered tokens to -inf
    return mx.where(keep_mask, logits, mx.array(float("-inf")))
