"""Request management for batched generation.

This module provides data structures for managing generation requests,
including request state, stop conditions, and sampling configuration.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Set, Union

import mlx.core as mx


class RequestStatus(Enum):
    """Status of a generation request."""

    PENDING = "pending"  # Waiting in queue
    PREFILL = "prefill"  # Processing initial prompt
    GENERATING = "generating"  # Active token generation
    PAUSED = "paused"  # Temporarily paused (for reordering)
    COMPLETED = "completed"  # Hit stop condition
    CANCELLED = "cancelled"  # User cancelled
    ERROR = "error"  # Processing error


class StopConditionType(Enum):
    """Type of stop condition."""

    EOS_TOKEN = "eos_token"  # Stop on specific token(s)
    MAX_LENGTH = "max_length"  # Maximum sequence length
    MAX_NEW_TOKENS = "max_new_tokens"  # Maximum new tokens
    STOP_STRINGS = "stop_strings"  # Stop on string patterns
    CUSTOM = "custom"  # Custom callable


@dataclass
class StopCondition:
    """Configurable stop condition for generation.

    Attributes:
        condition_type: Type of stop condition.
        value: Condition-specific value.
    """

    condition_type: StopConditionType
    value: Union[int, List[int], Set[int], Callable[[List[int]], bool]]

    def check(
        self,
        generated_tokens: List[int],
        total_length: int,
        new_tokens: int,
    ) -> bool:
        """Check if stop condition is met.

        Args:
            generated_tokens: All generated tokens so far.
            total_length: Total sequence length (prompt + generated).
            new_tokens: Number of newly generated tokens.

        Returns:
            True if generation should stop.
        """
        if self.condition_type == StopConditionType.EOS_TOKEN:
            if not generated_tokens:
                return False
            last_token = generated_tokens[-1]
            if isinstance(self.value, (list, set)):
                return last_token in self.value
            return last_token == self.value

        elif self.condition_type == StopConditionType.MAX_LENGTH:
            assert isinstance(self.value, int)
            return total_length >= self.value

        elif self.condition_type == StopConditionType.MAX_NEW_TOKENS:
            assert isinstance(self.value, int)
            return new_tokens >= self.value

        elif self.condition_type == StopConditionType.CUSTOM:
            assert callable(self.value)
            return self.value(generated_tokens)

        return False


@dataclass
class SamplingConfig:
    """Configuration for token sampling.

    Attributes:
        temperature: Sampling temperature. 0 = greedy.
        top_k: Keep only top-k tokens. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        repetition_penalty: Penalty for repeated tokens. 1.0 = disabled.
        presence_penalty: Penalty for tokens already present. 0.0 = disabled.
        frequency_penalty: Penalty based on frequency. 0.0 = disabled.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def is_greedy(self) -> bool:
        """Check if sampling is deterministic (greedy)."""
        return self.temperature == 0.0 or self.top_k == 1

    def validate(self) -> None:
        """Validate configuration values."""
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


@dataclass
class GenerationRequest:
    """A single generation request with its state and configuration.

    Attributes:
        request_id: Unique identifier for this request.
        input_ids: Input token IDs.
        sampling_config: Sampling configuration.
        stop_conditions: List of stop conditions.
        max_new_tokens: Maximum new tokens to generate.
        priority: Request priority (higher = more urgent).
        status: Current request status.
        generated_tokens: Tokens generated so far.
        current_position: Current position in generation.
        kv_cache_handle: Handle to KV cache (if using).
        arrival_time: When request was submitted.
        start_time: When processing started.
        finish_time: When processing completed.
        error: Error message if status is ERROR.
        metadata: Optional user metadata.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: mx.array = field(default_factory=lambda: mx.array([], dtype=mx.int32))
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    stop_conditions: List[StopCondition] = field(default_factory=list)
    max_new_tokens: int = 100
    priority: int = 0

    # Runtime state
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: List[int] = field(default_factory=list)
    current_position: int = 0
    kv_cache_handle: Optional[Any] = None

    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    # Error handling
    error: Optional[str] = None

    # User metadata
    metadata: Optional[dict] = None

    @property
    def prompt_length(self) -> int:
        """Length of the input prompt."""
        return self.input_ids.shape[0] if self.input_ids.size > 0 else 0

    @property
    def total_length(self) -> int:
        """Total length including generated tokens."""
        return self.prompt_length + len(self.generated_tokens)

    @property
    def num_generated(self) -> int:
        """Number of tokens generated."""
        return len(self.generated_tokens)

    @property
    def is_active(self) -> bool:
        """Check if request is actively processing."""
        return self.status in (RequestStatus.PREFILL, RequestStatus.GENERATING)

    @property
    def is_finished(self) -> bool:
        """Check if request processing is complete."""
        return self.status in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.ERROR,
        )

    @property
    def latency_ms(self) -> Optional[float]:
        """Total latency in milliseconds."""
        if self.start_time is None or self.finish_time is None:
            return None
        return (self.finish_time - self.start_time) * 1000

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Generation throughput."""
        latency = self.latency_ms
        if latency is None or latency == 0 or self.num_generated == 0:
            return None
        return self.num_generated / (latency / 1000)

    def start(self) -> None:
        """Mark request as started."""
        self.start_time = time.time()
        self.status = RequestStatus.PREFILL

    def begin_generation(self) -> None:
        """Transition from prefill to generation."""
        self.status = RequestStatus.GENERATING

    def add_token(self, token_id: int) -> None:
        """Add a generated token."""
        self.generated_tokens.append(token_id)
        self.current_position += 1

    def complete(self) -> None:
        """Mark request as completed."""
        self.finish_time = time.time()
        self.status = RequestStatus.COMPLETED

    def cancel(self) -> None:
        """Cancel the request."""
        self.finish_time = time.time()
        self.status = RequestStatus.CANCELLED

    def fail(self, error_message: str) -> None:
        """Mark request as failed."""
        self.finish_time = time.time()
        self.status = RequestStatus.ERROR
        self.error = error_message

    def check_stop_conditions(self) -> bool:
        """Check if any stop condition is met.

        Returns:
            True if generation should stop.
        """
        for condition in self.stop_conditions:
            if condition.check(
                self.generated_tokens,
                self.total_length,
                self.num_generated,
            ):
                return True
        return False

    def get_all_tokens(self) -> mx.array:
        """Get all tokens (prompt + generated)."""
        if not self.generated_tokens:
            return self.input_ids

        generated = mx.array(self.generated_tokens, dtype=mx.int32)
        return mx.concatenate([self.input_ids, generated])


def create_request(
    input_ids: Union[mx.array, List[int]],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    priority: int = 0,
    request_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> GenerationRequest:
    """Create a generation request with common parameters.

    Args:
        input_ids: Input token IDs.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        repetition_penalty: Repetition penalty.
        eos_token_id: EOS token(s) to stop on.
        stop_token_ids: Additional stop tokens.
        priority: Request priority.
        request_id: Optional custom request ID.
        metadata: Optional user metadata.

    Returns:
        Configured GenerationRequest.
    """
    if isinstance(input_ids, list):
        input_ids = mx.array(input_ids, dtype=mx.int32)

    sampling_config = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    stop_conditions = [
        StopCondition(StopConditionType.MAX_NEW_TOKENS, max_new_tokens)
    ]

    # Add EOS stop condition
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            stop_conditions.append(
                StopCondition(StopConditionType.EOS_TOKEN, eos_token_id)
            )
        else:
            stop_conditions.append(
                StopCondition(StopConditionType.EOS_TOKEN, set(eos_token_id))
            )

    # Add additional stop tokens
    if stop_token_ids:
        stop_conditions.append(
            StopCondition(StopConditionType.EOS_TOKEN, set(stop_token_ids))
        )

    return GenerationRequest(
        request_id=request_id or str(uuid.uuid4()),
        input_ids=input_ids,
        sampling_config=sampling_config,
        stop_conditions=stop_conditions,
        max_new_tokens=max_new_tokens,
        priority=priority,
        metadata=metadata,
    )
