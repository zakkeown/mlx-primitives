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

            # Apply repetition penalty
            if config.repetition_penalty != 1.0 and generated_tokens:
                sample_logits = apply_repetition_penalty_batch(
                    sample_logits,
                    [generated_tokens[i]] if i < len(generated_tokens) else [[]],
                    config.repetition_penalty,
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
    # Use ascending sort and negative index to avoid MLX slicing bug
    sorted_logits = mx.sort(logits, axis=-1)  # Ascending
    threshold = sorted_logits[:, -k : -k + 1]  # k-th largest, shape (batch, 1)

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

    # Compute cumulative probabilities
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
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

    result = logits
    unique_tokens = list(set(token_ids))
    for token_id in unique_tokens:
        if 0 <= token_id < logits.shape[-1]:
            # Apply penalty: divide positive logits, multiply negative
            token_logit = float(logits[0, token_id])
            if token_logit > 0:
                new_val = token_logit / penalty
            else:
                new_val = token_logit * penalty
            # Use index assignment
            result_np = result.tolist()
            result_np[0][token_id] = new_val
            result = mx.array(result_np, dtype=logits.dtype)

    if was_1d:
        result = result.squeeze(0)

    return result


def apply_repetition_penalty_batch(
    logits: mx.array,
    generated_tokens: List[List[int]],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to batched logits.

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
    result_np = logits.tolist()

    for i in range(batch_size):
        if i < len(generated_tokens) and generated_tokens[i]:
            unique_tokens = list(set(generated_tokens[i]))
            for token_id in unique_tokens:
                if 0 <= token_id < vocab_size:
                    token_logit = result_np[i][token_id]
                    if token_logit > 0:
                        result_np[i][token_id] = token_logit / penalty
                    else:
                        result_np[i][token_id] = token_logit * penalty

    return mx.array(result_np, dtype=logits.dtype)


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

    result_np = logits.tolist()
    unique_tokens = list(set(token_ids))
    for token_id in unique_tokens:
        if 0 <= token_id < len(result_np):
            result_np[token_id] = result_np[token_id] - penalty

    return mx.array(result_np, dtype=logits.dtype)


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

    result_np = logits.tolist()

    # Count frequencies
    freq: Dict[int, int] = {}
    for token_id in token_ids:
        freq[token_id] = freq.get(token_id, 0) + 1

    for token_id, count in freq.items():
        if 0 <= token_id < len(result_np):
            result_np[token_id] = result_np[token_id] - penalty * count

    return mx.array(result_np, dtype=logits.dtype)


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


def sample_beam(
    logits: mx.array,
    beam_width: int,
    length_penalty: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Get top-k tokens for beam search.

    Args:
        logits: Input logits (batch * beam, vocab_size).
        beam_width: Number of beams.
        length_penalty: Length penalty for scoring.

    Returns:
        Tuple of (token_ids, scores) each (batch * beam, beam_width).
    """
    # Get top-k for each beam
    top_k_values, top_k_indices = mx.topk(logits, k=beam_width, axis=-1)
    return top_k_indices, top_k_values
