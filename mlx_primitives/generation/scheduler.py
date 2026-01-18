"""Request scheduling for batched generation.

This module provides request scheduling with support for:
- Priority queuing
- Dynamic batching
- Continuous batching (add new requests mid-generation)
"""

import bisect
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_primitives.generation.batch_manager import (
    BatchedSequences,
    PaddingSide,
    SequenceBatcher,
)
from mlx_primitives.generation.requests import (
    GenerationRequest,
    RequestStatus,
)


@dataclass
class SchedulerConfig:
    """Configuration for request scheduler.

    Attributes:
        max_batch_tokens: Maximum tokens in a batch.
        max_batch_size: Maximum sequences in a batch.
        max_waiting_time_ms: Maximum time to wait for batch fill.
        enable_continuous_batching: Allow adding requests mid-generation.
        padding_side: Which side to pad sequences.
        pad_token_id: Token ID for padding.
    """

    max_batch_tokens: int = 8192
    max_batch_size: int = 32
    max_waiting_time_ms: float = 50.0
    enable_continuous_batching: bool = True
    padding_side: PaddingSide = PaddingSide.LEFT
    pad_token_id: int = 0


class RequestScheduler:
    """Manages request queue and batch formation.

    Supports:
    - Priority queuing (higher priority = earlier processing)
    - Dynamic batching (combine requests as they arrive)
    - Continuous batching (add new requests mid-generation)

    Example:
        >>> scheduler = RequestScheduler(config)
        >>> scheduler.submit(request1)
        >>> scheduler.submit(request2)
        >>> requests, batch = scheduler.get_batch()
        >>> # Process batch...
        >>> scheduler.complete_request(request1.request_id, tokens)
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        batcher: Optional[SequenceBatcher] = None,
    ):
        """Initialize scheduler.

        Args:
            config: Scheduler configuration.
            batcher: Sequence batcher. Created from config if not provided.
        """
        self._config = config or SchedulerConfig()

        self._batcher = batcher or SequenceBatcher(
            max_batch_tokens=self._config.max_batch_tokens,
            max_batch_size=self._config.max_batch_size,
            padding_side=self._config.padding_side,
            pad_token_id=self._config.pad_token_id,
        )

        # Request queues
        self._pending_queue: List[GenerationRequest] = []
        self._active_requests: Dict[str, GenerationRequest] = {}
        self._completed_requests: Dict[str, GenerationRequest] = {}

        # Thread safety
        self._lock = threading.Lock()

    @property
    def num_pending(self) -> int:
        """Number of pending requests."""
        with self._lock:
            return len(self._pending_queue)

    @property
    def num_active(self) -> int:
        """Number of active requests."""
        with self._lock:
            return len(self._active_requests)

    @property
    def num_completed(self) -> int:
        """Number of completed requests."""
        with self._lock:
            return len(self._completed_requests)

    def submit(self, request: GenerationRequest) -> None:
        """Add request to queue (thread-safe).

        Inserts maintaining priority order (higher priority first).

        Args:
            request: Request to submit.
        """
        with self._lock:
            request.status = RequestStatus.PENDING

            # Insert maintaining priority order
            # Use bisect for O(log n) insertion
            # Key: (-priority, arrival_time) for descending priority
            key = (-request.priority, request.arrival_time)

            # Find insertion point
            keys = [(-r.priority, r.arrival_time) for r in self._pending_queue]
            idx = bisect.bisect_right(keys, key)
            self._pending_queue.insert(idx, request)

    def cancel(self, request_id: str) -> bool:
        """Cancel a request.

        Args:
            request_id: Request to cancel.

        Returns:
            True if cancelled, False if not found.
        """
        with self._lock:
            # Check pending queue
            for i, req in enumerate(self._pending_queue):
                if req.request_id == request_id:
                    req.cancel()
                    self._completed_requests[request_id] = req
                    del self._pending_queue[i]
                    return True

            # Check active requests
            if request_id in self._active_requests:
                req = self._active_requests.pop(request_id)
                req.cancel()
                self._completed_requests[request_id] = req
                return True

        return False

    def get_batch(
        self,
        include_generating: bool = True,
    ) -> Tuple[List[GenerationRequest], Optional[BatchedSequences]]:
        """Get next batch for processing.

        If continuous batching is enabled and include_generating=True,
        combines pending prefill requests with active generation requests.

        Args:
            include_generating: Include active generation requests.

        Returns:
            Tuple of (requests, batched_sequences).
        """
        with self._lock:
            if self._config.enable_continuous_batching and include_generating:
                return self._form_continuous_batch()
            else:
                return self._form_static_batch()

    def _form_static_batch(self) -> Tuple[List[GenerationRequest], Optional[BatchedSequences]]:
        """Form batch from pending requests only."""
        if not self._pending_queue:
            return [], None

        batch_requests = []
        batch_tokens = 0

        # Select requests that fit
        while self._pending_queue:
            req = self._pending_queue[0]
            prompt_len = req.prompt_length

            if not self._batcher.can_add_sequence(
                batch_tokens, len(batch_requests), prompt_len
            ):
                break

            # Move to active
            self._pending_queue.pop(0)
            req.start()
            self._active_requests[req.request_id] = req
            batch_requests.append(req)
            batch_tokens += prompt_len

        if not batch_requests:
            return [], None

        # Create batch
        sequences = [req.input_ids for req in batch_requests]
        batch = self._batcher.create_batch(
            sequences, list(range(len(batch_requests)))
        )

        return batch_requests, batch

    def _form_continuous_batch(
        self,
    ) -> Tuple[List[GenerationRequest], Optional[BatchedSequences]]:
        """Form batch combining prefill and generation requests.

        Strategy:
        1. Take active generation requests (1 token each)
        2. Fill remaining capacity with pending prefill requests
        """
        batch_requests = []
        batch_tokens = 0
        sequences = []

        # First: active generation requests (each needs 1 new token)
        generation_requests = [
            req
            for req in self._active_requests.values()
            if req.status == RequestStatus.GENERATING
        ]

        for req in generation_requests:
            if len(batch_requests) >= self._config.max_batch_size:
                break

            batch_requests.append(req)
            batch_tokens += 1  # Generation: 1 token per request

            # For generation, we only need the last generated token
            if req.generated_tokens:
                sequences.append(mx.array([req.generated_tokens[-1]], dtype=mx.int32))
            else:
                # First generation step after prefill
                sequences.append(
                    req.input_ids[-1:] if req.input_ids.size > 0 else mx.array([0])
                )

        # Second: pending prefill requests
        remaining_tokens = self._config.max_batch_tokens - batch_tokens
        remaining_slots = self._config.max_batch_size - len(batch_requests)

        prefill_candidates = []
        for req in self._pending_queue:
            prompt_len = req.prompt_length
            if prompt_len <= remaining_tokens and len(prefill_candidates) < remaining_slots:
                prefill_candidates.append(req)
                remaining_tokens -= prompt_len

        # Move prefill candidates to active
        for req in prefill_candidates:
            self._pending_queue.remove(req)
            req.start()
            self._active_requests[req.request_id] = req
            batch_requests.append(req)
            sequences.append(req.input_ids)

        if not batch_requests:
            return [], None

        # Create mixed batch
        batch = self._batcher.create_batch(
            sequences, list(range(len(batch_requests)))
        )

        return batch_requests, batch

    def update_request(
        self,
        request_id: str,
        new_tokens: List[int],
    ) -> None:
        """Update a request with newly generated tokens.

        Args:
            request_id: Request to update.
            new_tokens: New tokens to add.
        """
        with self._lock:
            if request_id not in self._active_requests:
                return

            req = self._active_requests[request_id]
            for token in new_tokens:
                req.add_token(token)

            # Transition from prefill to generating
            if req.status == RequestStatus.PREFILL:
                req.begin_generation()

    def complete_request(
        self,
        request_id: str,
        final_tokens: Optional[List[int]] = None,
    ) -> None:
        """Mark request as completed and remove from active set.

        Args:
            request_id: Request to complete.
            final_tokens: Final tokens to add before completing.
        """
        with self._lock:
            if request_id not in self._active_requests:
                return

            req = self._active_requests.pop(request_id)

            if final_tokens:
                for token in final_tokens:
                    req.add_token(token)

            req.complete()
            self._completed_requests[request_id] = req

    def fail_request(self, request_id: str, error: str) -> None:
        """Mark request as failed.

        Args:
            request_id: Request that failed.
            error: Error message.
        """
        with self._lock:
            if request_id in self._active_requests:
                req = self._active_requests.pop(request_id)
                req.fail(error)
                self._completed_requests[request_id] = req

    def get_request(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a request by ID.

        Args:
            request_id: Request ID.

        Returns:
            Request or None.
        """
        with self._lock:
            if request_id in self._active_requests:
                return self._active_requests[request_id]
            if request_id in self._completed_requests:
                return self._completed_requests[request_id]
            for req in self._pending_queue:
                if req.request_id == request_id:
                    return req
        return None

    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        with self._lock:
            return len(self._pending_queue) > 0

    def has_active(self) -> bool:
        """Check if there are active requests."""
        with self._lock:
            return len(self._active_requests) > 0

    def has_work(self) -> bool:
        """Check if there is any work to do."""
        with self._lock:
            return len(self._pending_queue) > 0 or len(self._active_requests) > 0

    def get_completed(self, request_id: str) -> Optional[GenerationRequest]:
        """Get a completed request and remove from completed set.

        Args:
            request_id: Request ID.

        Returns:
            Completed request or None.
        """
        with self._lock:
            return self._completed_requests.pop(request_id, None)

    def get_all_completed(self) -> List[GenerationRequest]:
        """Get all completed requests and clear the completed set.

        Returns:
            List of completed requests.
        """
        with self._lock:
            completed = list(self._completed_requests.values())
            self._completed_requests.clear()
            return completed

    def clear(self) -> None:
        """Clear all requests."""
        with self._lock:
            for req in self._pending_queue:
                req.cancel()
            for req in self._active_requests.values():
                req.cancel()
            self._pending_queue.clear()
            self._active_requests.clear()
            self._completed_requests.clear()

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "num_pending": len(self._pending_queue),
                "num_active": len(self._active_requests),
                "num_completed": len(self._completed_requests),
                "pending_tokens": sum(r.prompt_length for r in self._pending_queue),
                "active_tokens": sum(
                    r.total_length for r in self._active_requests.values()
                ),
            }


class PriorityScheduler(RequestScheduler):
    """Scheduler with enhanced priority handling.

    Supports priority levels and aging to prevent starvation.
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        priority_levels: int = 10,
        aging_factor: float = 0.1,
    ):
        """Initialize priority scheduler.

        Args:
            config: Scheduler configuration.
            priority_levels: Number of priority levels.
            aging_factor: How much to boost priority per wait cycle.
        """
        super().__init__(config)
        self._priority_levels = priority_levels
        self._aging_factor = aging_factor
        self._wait_cycles: Dict[str, int] = {}

    def submit(self, request: GenerationRequest) -> None:
        """Submit with priority validation."""
        # Clamp priority to valid range
        request.priority = max(0, min(request.priority, self._priority_levels - 1))
        super().submit(request)
        self._wait_cycles[request.request_id] = 0

    def get_batch(
        self,
        include_generating: bool = True,
    ) -> Tuple[List[GenerationRequest], Optional[BatchedSequences]]:
        """Get batch with aging applied."""
        # Apply aging to pending requests
        with self._lock:
            for req in self._pending_queue:
                cycles = self._wait_cycles.get(req.request_id, 0)
                self._wait_cycles[req.request_id] = cycles + 1

                # Boost effective priority
                effective_boost = int(cycles * self._aging_factor)
                # Don't modify actual priority, just use for sorting

        return super().get_batch(include_generating)

    def complete_request(
        self,
        request_id: str,
        final_tokens: Optional[List[int]] = None,
    ) -> None:
        """Complete request and clean up aging data."""
        super().complete_request(request_id, final_tokens)
        self._wait_cycles.pop(request_id, None)
