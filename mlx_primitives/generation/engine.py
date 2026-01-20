"""Generation engine for batched text generation.

This module provides the high-level GenerationEngine that orchestrates
request scheduling, batch management, model inference, and token sampling.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx

from mlx_primitives.generation.batch_manager import (
    BatchedSequences,
    PaddingSide,
    SequenceBatcher,
)
from mlx_primitives.generation.requests import (
    GenerationRequest,
    RequestStatus,
    SamplingConfig,
    StopCondition,
    StopConditionType,
    create_request,
)
from mlx_primitives.generation.samplers import TokenSampler
from mlx_primitives.generation.scheduler import (
    RequestScheduler,
    SchedulerConfig,
)


@dataclass
class EngineConfig:
    """Configuration for generation engine.

    Attributes:
        vocab_size: Vocabulary size.
        eos_token_id: End of sequence token.
        pad_token_id: Padding token.
        max_batch_tokens: Maximum tokens per batch.
        max_batch_size: Maximum sequences per batch.
        enable_continuous_batching: Allow adding requests mid-generation.
            Note: Without KV cache, the engine recomputes the full sequence
            each step (O(n²) complexity). For efficient inference, use a
            model with KV cache management.
        eval_frequency: Evaluate MLX graph every N steps.
    """

    vocab_size: int
    eos_token_id: int = 2
    pad_token_id: int = 0
    max_batch_tokens: int = 8192
    max_batch_size: int = 32
    enable_continuous_batching: bool = True
    eval_frequency: int = 1


class GenerationEngine:
    """High-level generation engine with continuous batching.

    Coordinates request scheduling, batch management, model inference,
    and token sampling for efficient batched generation.

    Example:
        >>> def model_forward(input_ids, attention_mask):
        ...     return model(input_ids, attention_mask)
        >>>
        >>> engine = GenerationEngine(
        ...     model_forward_fn=model_forward,
        ...     config=EngineConfig(vocab_size=32000, eos_token_id=2),
        ... )
        >>>
        >>> # Submit requests
        >>> req1 = engine.submit(input_ids1, max_new_tokens=100)
        >>> req2 = engine.submit(input_ids2, max_new_tokens=50, priority=1)
        >>>
        >>> # Generate with streaming
        >>> for request_id, token in engine.generate_stream():
        ...     print(f"Request {request_id}: {token}")
    """

    def __init__(
        self,
        model_forward_fn: Callable[[mx.array, Optional[mx.array]], mx.array],
        config: EngineConfig,
    ):
        """Initialize generation engine.

        Args:
            model_forward_fn: Function that takes (input_ids, attention_mask)
                and returns logits (batch, seq_len, vocab_size).
            config: Engine configuration.

        Note:
            This engine does not currently support KV caching. Each generation
            step recomputes attention over the full sequence. For long sequences,
            consider using a model with built-in KV cache management.
        """
        self._model_forward = model_forward_fn
        self._config = config

        # Initialize components
        scheduler_config = SchedulerConfig(
            max_batch_tokens=config.max_batch_tokens,
            max_batch_size=config.max_batch_size,
            enable_continuous_batching=config.enable_continuous_batching,
            pad_token_id=config.pad_token_id,
        )
        self._scheduler = RequestScheduler(scheduler_config)
        self._sampler = TokenSampler(config.vocab_size)

        # Streaming output queue
        self._output_queue: queue.Queue[Tuple[str, int]] = queue.Queue()
        self._step_count = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._running = False

    @property
    def config(self) -> EngineConfig:
        """Get engine configuration."""
        return self._config

    @property
    def scheduler(self) -> RequestScheduler:
        """Get the request scheduler."""
        return self._scheduler

    def submit(
        self,
        input_ids: Union[mx.array, List[int]],
        sampling_config: Optional[SamplingConfig] = None,
        max_new_tokens: int = 100,
        stop_token_ids: Optional[List[int]] = None,
        priority: int = 0,
        request_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> GenerationRequest:
        """Submit a generation request.

        Returns immediately with a request handle.

        Args:
            input_ids: Input token IDs.
            sampling_config: Sampling configuration.
            max_new_tokens: Maximum tokens to generate.
            stop_token_ids: Additional stop token IDs.
            priority: Request priority (higher = earlier).
            request_id: Optional custom request ID.
            metadata: Optional user metadata.

        Returns:
            Submitted request.

        Raises:
            ValueError: If sampling_config has invalid values.
        """
        # Validate sampling config if provided
        if sampling_config is not None:
            sampling_config.validate()

        request = create_request(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=sampling_config.temperature if sampling_config else 1.0,
            top_k=sampling_config.top_k if sampling_config else 0,
            top_p=sampling_config.top_p if sampling_config else 1.0,
            repetition_penalty=sampling_config.repetition_penalty
            if sampling_config
            else 1.0,
            eos_token_id=self._config.eos_token_id,
            stop_token_ids=stop_token_ids,
            priority=priority,
            request_id=request_id,
            metadata=metadata,
        )

        self._scheduler.submit(request)
        return request

    def step(self) -> Dict[str, List[int]]:
        """Execute one generation step for the current batch.

        Returns:
            Dict mapping request_id -> newly generated tokens.
        """
        # Get batch from scheduler
        requests, batch = self._scheduler.get_batch()
        if not requests or batch is None:
            return {}

        # Forward pass
        logits = self._forward_batch(batch)

        # Get logits at last position for each sequence
        last_logits = batch.get_last_token_logits(logits)

        # Sample tokens
        configs = [r.sampling_config for r in requests]
        generated = [r.generated_tokens for r in requests]
        new_tokens = self._sampler(last_logits, configs, generated)

        # Force evaluation at step boundary
        self._step_count += 1
        if self._step_count % self._config.eval_frequency == 0:
            mx.eval(new_tokens)

        # Update request states
        results: Dict[str, List[int]] = {}
        tokens_list = new_tokens.tolist()

        for i, (request, token) in enumerate(zip(requests, tokens_list)):
            # Add token to request
            self._scheduler.update_request(request.request_id, [token])
            results[request.request_id] = [token]

            # Queue for streaming output
            self._output_queue.put((request.request_id, token))

            # Check stop conditions
            if self._should_stop(request):
                self._scheduler.complete_request(request.request_id)

        return results

    def _forward_batch(self, batch: BatchedSequences) -> mx.array:
        """Run model forward pass on batched sequences.

        Args:
            batch: Batched input sequences.

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        return self._model_forward(batch.input_ids, batch.attention_mask)

    def _should_stop(self, request: GenerationRequest) -> bool:
        """Check if generation should stop for a request.

        Args:
            request: The request to check.

        Returns:
            True if should stop.
        """
        return request.check_stop_conditions()

    def generate_stream(self) -> Iterator[Tuple[str, int]]:
        """Generate tokens with streaming output.

        Yields (request_id, token) tuples as tokens are generated.
        Thread-safe: can be stopped from another thread via stop().

        Example:
            >>> for request_id, token in engine.generate_stream():
            ...     print(f"Request {request_id}: {token}")
        """
        with self._lock:
            self._running = True

        try:
            while True:
                # Check if we should stop (thread-safe read)
                with self._lock:
                    if not self._running:
                        break

                if not self._scheduler.has_work():
                    break

                # Run generation step
                self.step()

                # Yield any queued outputs using timeout to avoid TOCTOU race
                while True:
                    try:
                        item = self._output_queue.get(timeout=0.001)
                        yield item
                    except queue.Empty:
                        break
        finally:
            with self._lock:
                self._running = False

    def generate(
        self,
        input_ids: Union[mx.array, List[int]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> List[int]:
        """Generate tokens for a single request (blocking).

        Args:
            input_ids: Input token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty.

        Returns:
            Generated token IDs.
        """
        config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        request = self.submit(
            input_ids=input_ids,
            sampling_config=config,
            max_new_tokens=max_new_tokens,
        )

        # Generate until complete
        while not request.is_finished:
            self.step()

        return request.generated_tokens

    def generate_batch(
        self,
        inputs: List[Union[mx.array, List[int]]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> List[List[int]]:
        """Generate tokens for multiple requests (blocking).

        Args:
            inputs: List of input token ID sequences.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.

        Returns:
            List of generated token sequences.
        """
        config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        requests = []
        for input_ids in inputs:
            req = self.submit(
                input_ids=input_ids,
                sampling_config=config,
                max_new_tokens=max_new_tokens,
            )
            requests.append(req)

        # Generate until all complete
        while any(not r.is_finished for r in requests):
            self.step()

        return [r.generated_tokens for r in requests]

    def stop(self) -> None:
        """Stop the generation loop.

        Thread-safe: can be called from any thread.
        """
        with self._lock:
            self._running = False

    def clear(self) -> None:
        """Clear all pending and active requests."""
        self._scheduler.clear()
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID.

        Args:
            request_id: Request ID.

        Returns:
            Request or None.
        """
        return self._scheduler.get_request(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request.

        Args:
            request_id: Request to cancel.

        Returns:
            True if cancelled.
        """
        return self._scheduler.cancel(request_id)

    def get_stats(self) -> dict:
        """Get engine statistics.

        Returns:
            Dictionary with statistics.
        """
        scheduler_stats = self._scheduler.get_stats()
        with self._lock:
            running = self._running
        return {
            **scheduler_stats,
            "step_count": self._step_count,
            "output_queue_size": self._output_queue.qsize(),
            "running": running,
        }


class StreamingOutput:
    """Helper for collecting streaming output.

    Example:
        >>> output = StreamingOutput()
        >>> for request_id, token in engine.generate_stream():
        ...     output.add(request_id, token)
        >>> results = output.get_all()
    """

    def __init__(self):
        """Initialize streaming output collector."""
        self._tokens: Dict[str, List[int]] = {}

    def add(self, request_id: str, token: int) -> None:
        """Add a token for a request.

        Args:
            request_id: Request ID.
            token: Token ID.
        """
        if request_id not in self._tokens:
            self._tokens[request_id] = []
        self._tokens[request_id].append(token)

    def get(self, request_id: str) -> List[int]:
        """Get tokens for a request.

        Args:
            request_id: Request ID.

        Returns:
            List of tokens.
        """
        return self._tokens.get(request_id, [])

    def get_all(self) -> Dict[str, List[int]]:
        """Get all tokens.

        Returns:
            Dict mapping request_id -> tokens.
        """
        return dict(self._tokens)

    def clear(self) -> None:
        """Clear all collected tokens."""
        self._tokens.clear()


def create_generation_engine(
    model_forward_fn: Callable[[mx.array, Optional[mx.array]], mx.array],
    vocab_size: int,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
    max_batch_tokens: int = 8192,
    max_batch_size: int = 32,
    enable_continuous_batching: bool = True,
) -> GenerationEngine:
    """Create a generation engine with common defaults.

    Args:
        model_forward_fn: Model forward function.
        vocab_size: Vocabulary size.
        eos_token_id: EOS token ID.
        pad_token_id: Padding token ID.
        max_batch_tokens: Max tokens per batch.
        max_batch_size: Max sequences per batch.
        enable_continuous_batching: Enable continuous batching.

    Returns:
        Configured GenerationEngine.
    """
    config = EngineConfig(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
        enable_continuous_batching=enable_continuous_batching,
    )
    return GenerationEngine(model_forward_fn, config)
