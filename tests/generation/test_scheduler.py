"""Tests for request scheduler with state machine and concurrency validation."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.generation.scheduler import (
    RequestScheduler,
    PriorityScheduler,
    SchedulerConfig,
)
from mlx_primitives.generation.requests import (
    GenerationRequest,
    RequestStatus,
    create_request,
)
from mlx_primitives.generation.batch_manager import PaddingSide


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> SchedulerConfig:
    """Standard scheduler configuration."""
    return SchedulerConfig(
        max_batch_tokens=100,
        max_batch_size=4,
        enable_continuous_batching=True,
    )


@pytest.fixture
def scheduler(default_config: SchedulerConfig) -> RequestScheduler:
    """Create scheduler with small limits for testing."""
    return RequestScheduler(default_config)


@pytest.fixture
def static_scheduler() -> RequestScheduler:
    """Create scheduler without continuous batching."""
    config = SchedulerConfig(
        max_batch_tokens=100,
        max_batch_size=4,
        enable_continuous_batching=False,
    )
    return RequestScheduler(config)


@pytest.fixture
def priority_scheduler() -> PriorityScheduler:
    """Create priority scheduler with aging."""
    config = SchedulerConfig(max_batch_tokens=50, max_batch_size=2)
    return PriorityScheduler(
        config=config,
        priority_levels=10,
        aging_factor=0.5,
    )


def make_request(
    prompt_length: int = 5,
    priority: int = 0,
    request_id: str = None,
) -> GenerationRequest:
    """Create a test generation request."""
    return create_request(
        input_ids=list(range(prompt_length)),
        priority=priority,
        request_id=request_id,
    )


# =============================================================================
# TestSchedulerConfig
# =============================================================================


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test all defaults match expected values."""
        config = SchedulerConfig()

        assert config.max_batch_tokens == 8192
        assert config.max_batch_size == 32
        assert config.max_waiting_time_ms == 50.0
        assert config.enable_continuous_batching is True
        assert config.padding_side == PaddingSide.LEFT
        assert config.pad_token_id == 0

    def test_custom_configuration(self) -> None:
        """Test custom values are properly stored."""
        config = SchedulerConfig(
            max_batch_tokens=4096,
            max_batch_size=16,
            max_waiting_time_ms=100.0,
            enable_continuous_batching=False,
            padding_side=PaddingSide.RIGHT,
            pad_token_id=1,
        )

        assert config.max_batch_tokens == 4096
        assert config.max_batch_size == 16
        assert config.enable_continuous_batching is False
        assert config.padding_side == PaddingSide.RIGHT
        assert config.pad_token_id == 1

    def test_padding_side_enum(self) -> None:
        """Test both padding sides work."""
        config_left = SchedulerConfig(padding_side=PaddingSide.LEFT)
        config_right = SchedulerConfig(padding_side=PaddingSide.RIGHT)

        assert config_left.padding_side == PaddingSide.LEFT
        assert config_right.padding_side == PaddingSide.RIGHT


# =============================================================================
# TestRequestSchedulerStateTransitions
# =============================================================================


class TestRequestSchedulerStateTransitions:
    """Tests for correct state machine behavior."""

    def test_submit_sets_pending_status(self, scheduler: RequestScheduler) -> None:
        """Request enters PENDING state on submit."""
        request = make_request()
        scheduler.submit(request)

        assert request.status == RequestStatus.PENDING

    def test_get_batch_transitions_to_prefill(self, scheduler: RequestScheduler) -> None:
        """First get_batch() moves request to PREFILL."""
        request = make_request()
        scheduler.submit(request)

        requests, batch = scheduler.get_batch()

        assert len(requests) == 1
        assert requests[0].status == RequestStatus.PREFILL

    def test_update_request_transitions_to_generating(
        self, scheduler: RequestScheduler
    ) -> None:
        """Calling update_request() after prefill transitions to GENERATING."""
        request = make_request()
        scheduler.submit(request)
        scheduler.get_batch()

        # Update with new tokens
        scheduler.update_request(request.request_id, [100])

        assert request.status == RequestStatus.GENERATING

    def test_complete_request_transitions_to_completed(
        self, scheduler: RequestScheduler
    ) -> None:
        """complete_request() sets COMPLETED status."""
        request = make_request()
        scheduler.submit(request)
        scheduler.get_batch()
        scheduler.update_request(request.request_id, [100])

        scheduler.complete_request(request.request_id)

        assert request.status == RequestStatus.COMPLETED
        assert scheduler.num_active == 0
        assert scheduler.num_completed == 1

    def test_cancel_from_pending(self, scheduler: RequestScheduler) -> None:
        """Cancel removes from pending queue, sets CANCELLED."""
        request = make_request()
        scheduler.submit(request)

        result = scheduler.cancel(request.request_id)

        assert result is True
        assert request.status == RequestStatus.CANCELLED
        assert scheduler.num_pending == 0
        assert scheduler.num_completed == 1

    def test_cancel_from_active(self, scheduler: RequestScheduler) -> None:
        """Cancel removes from active, sets CANCELLED."""
        request = make_request()
        scheduler.submit(request)
        scheduler.get_batch()

        result = scheduler.cancel(request.request_id)

        assert result is True
        assert request.status == RequestStatus.CANCELLED
        assert scheduler.num_active == 0
        assert scheduler.num_completed == 1

    def test_cancel_nonexistent_returns_false(
        self, scheduler: RequestScheduler
    ) -> None:
        """Cancel on invalid request_id returns False."""
        result = scheduler.cancel("nonexistent")

        assert result is False

    def test_fail_request_sets_error_status(self, scheduler: RequestScheduler) -> None:
        """fail_request() sets ERROR status with message."""
        request = make_request()
        scheduler.submit(request)
        scheduler.get_batch()

        scheduler.fail_request(request.request_id, "test error")

        assert request.status == RequestStatus.ERROR
        assert request.error == "test error"
        assert scheduler.num_active == 0
        assert scheduler.num_completed == 1

    def test_full_lifecycle_pending_to_complete(
        self, scheduler: RequestScheduler
    ) -> None:
        """Full happy path through all states."""
        request = make_request()

        # Submit -> PENDING
        scheduler.submit(request)
        assert request.status == RequestStatus.PENDING

        # Get batch -> PREFILL
        scheduler.get_batch()
        assert request.status == RequestStatus.PREFILL

        # Update -> GENERATING
        scheduler.update_request(request.request_id, [100])
        assert request.status == RequestStatus.GENERATING

        # More updates stay in GENERATING
        scheduler.update_request(request.request_id, [101, 102])
        assert request.status == RequestStatus.GENERATING

        # Complete -> COMPLETED
        scheduler.complete_request(request.request_id, [103])
        assert request.status == RequestStatus.COMPLETED

    def test_tokens_added_correctly(self, scheduler: RequestScheduler) -> None:
        """update_request() adds tokens to request."""
        request = make_request()
        scheduler.submit(request)
        scheduler.get_batch()

        scheduler.update_request(request.request_id, [100, 101])
        scheduler.update_request(request.request_id, [102])

        assert request.generated_tokens == [100, 101, 102]
        assert request.num_generated == 3


# =============================================================================
# TestBatchFormation
# =============================================================================


class TestBatchFormation:
    """Tests for batch formation respects limits."""

    def test_batch_respects_max_batch_size(self) -> None:
        """Never exceed max_batch_size."""
        config = SchedulerConfig(max_batch_tokens=1000, max_batch_size=2)
        scheduler = RequestScheduler(config)

        for i in range(5):
            scheduler.submit(make_request(prompt_length=3, request_id=f"req-{i}"))

        requests, batch = scheduler.get_batch()

        assert len(requests) <= 2
        assert scheduler.num_pending == 3

    def test_batch_respects_max_batch_tokens(self) -> None:
        """Token count stays within limit."""
        config = SchedulerConfig(max_batch_tokens=15, max_batch_size=10)
        scheduler = RequestScheduler(config)

        # Each request has 5 tokens
        for i in range(5):
            scheduler.submit(make_request(prompt_length=5, request_id=f"req-{i}"))

        requests, batch = scheduler.get_batch()

        total_tokens = sum(r.prompt_length for r in requests)
        assert total_tokens <= 15
        assert len(requests) == 3  # 5+5+5=15

    def test_batch_formation_single_request(
        self, scheduler: RequestScheduler
    ) -> None:
        """Single request forms valid batch."""
        scheduler.submit(make_request())

        requests, batch = scheduler.get_batch()

        assert len(requests) == 1
        assert batch is not None
        assert batch.batch_size == 1

    def test_batch_formation_empty_queue(self, scheduler: RequestScheduler) -> None:
        """Empty queue returns empty tuple."""
        requests, batch = scheduler.get_batch()

        assert requests == []
        assert batch is None

    def test_batch_partial_fill(self) -> None:
        """Requests that don't fit remain in queue."""
        config = SchedulerConfig(max_batch_tokens=10, max_batch_size=10)
        scheduler = RequestScheduler(config)

        scheduler.submit(make_request(prompt_length=8, request_id="req-0"))
        scheduler.submit(make_request(prompt_length=8, request_id="req-1"))

        requests, batch = scheduler.get_batch()

        assert len(requests) == 1
        assert scheduler.num_pending == 1

    def test_large_request_processed_alone(self) -> None:
        """Single request > typical batch gets processed alone."""
        config = SchedulerConfig(max_batch_tokens=100, max_batch_size=10)
        scheduler = RequestScheduler(config)

        scheduler.submit(make_request(prompt_length=50, request_id="large"))

        requests, batch = scheduler.get_batch()

        assert len(requests) == 1
        assert requests[0].request_id == "large"


# =============================================================================
# TestContinuousBatching
# =============================================================================


class TestContinuousBatching:
    """Tests for prefill + generation mixing."""

    def test_generation_requests_included_first(
        self, scheduler: RequestScheduler
    ) -> None:
        """Active generating requests take priority."""
        # Submit and start first request
        req1 = make_request(prompt_length=5, request_id="gen-1")
        scheduler.submit(req1)
        scheduler.get_batch()
        scheduler.update_request(req1.request_id, [100])  # Now GENERATING

        # Submit new request
        req2 = make_request(prompt_length=5, request_id="prefill-1")
        scheduler.submit(req2)

        # Next batch should include both
        requests, batch = scheduler.get_batch()

        assert len(requests) == 2
        # Generation request should be first
        assert requests[0].status == RequestStatus.GENERATING
        assert requests[1].status == RequestStatus.PREFILL

    def test_generation_uses_full_sequence_length(
        self, scheduler: RequestScheduler
    ) -> None:
        """Generation requests use full sequence length (prompt + generated).

        Without KV cache, the scheduler counts the full sequence for batch
        token limits to ensure correct model inference.
        """
        config = SchedulerConfig(
            max_batch_tokens=20,  # Enough for both: 6 + 8 = 14 < 20
            max_batch_size=10,
            enable_continuous_batching=True,
        )
        scheduler = RequestScheduler(config)

        # Create active generation request
        req1 = make_request(prompt_length=5, request_id="gen-1")
        scheduler.submit(req1)
        scheduler.get_batch()
        scheduler.update_request(req1.request_id, [100])

        # Submit request with 8 tokens
        # gen-1 now has 5 + 1 = 6 tokens total (prompt + generated)
        req2 = make_request(prompt_length=8, request_id="prefill-1")
        scheduler.submit(req2)

        requests, batch = scheduler.get_batch()

        # Without KV cache, generation uses full sequence: 6 + 8 = 14 < 20
        assert len(requests) == 2

    def test_continuous_batching_disabled(
        self, static_scheduler: RequestScheduler
    ) -> None:
        """When disabled, only prefill in batch."""
        # Create active generation request
        req1 = make_request(prompt_length=5, request_id="gen-1")
        static_scheduler.submit(req1)
        static_scheduler.get_batch()
        static_scheduler.update_request(req1.request_id, [100])

        # Submit new prefill request
        req2 = make_request(prompt_length=5, request_id="prefill-1")
        static_scheduler.submit(req2)

        # Next batch should only have prefill request
        requests, batch = static_scheduler.get_batch()

        assert len(requests) == 1
        assert requests[0].request_id == "prefill-1"

    def test_new_requests_join_mid_generation(
        self, scheduler: RequestScheduler
    ) -> None:
        """Submit during generation adds to next batch."""
        req1 = make_request(prompt_length=5, request_id="gen-1")
        scheduler.submit(req1)
        scheduler.get_batch()
        scheduler.update_request(req1.request_id, [100])

        # Submit new request mid-generation
        req2 = make_request(prompt_length=5, request_id="new-1")
        scheduler.submit(req2)

        assert scheduler.num_pending == 1
        assert scheduler.num_active == 1


# =============================================================================
# TestPriorityOrdering
# =============================================================================


class TestPriorityOrdering:
    """Tests for priority-based scheduling."""

    def test_higher_priority_processed_first(
        self, scheduler: RequestScheduler
    ) -> None:
        """Priority 10 before priority 5."""
        req_low = make_request(priority=5, request_id="low")
        req_high = make_request(priority=10, request_id="high")

        scheduler.submit(req_low)
        scheduler.submit(req_high)

        requests, batch = scheduler.get_batch()

        assert requests[0].request_id == "high"
        assert requests[1].request_id == "low"

    def test_equal_priority_fifo(self, scheduler: RequestScheduler) -> None:
        """Same priority uses arrival order."""
        req1 = make_request(priority=5, request_id="first")
        time.sleep(0.001)  # Ensure different arrival times
        req2 = make_request(priority=5, request_id="second")

        scheduler.submit(req1)
        scheduler.submit(req2)

        requests, batch = scheduler.get_batch()

        assert requests[0].request_id == "first"
        assert requests[1].request_id == "second"

    def test_mixed_priorities_correct_order(
        self, scheduler: RequestScheduler
    ) -> None:
        """Mix of priorities sorted correctly."""
        requests_to_submit = [
            make_request(priority=3, request_id="p3"),
            make_request(priority=7, request_id="p7"),
            make_request(priority=1, request_id="p1"),
            make_request(priority=5, request_id="p5"),
        ]

        for req in requests_to_submit:
            scheduler.submit(req)

        batch_requests, batch = scheduler.get_batch()

        priorities = [r.priority for r in batch_requests]
        assert priorities == [7, 5, 3, 1]


# =============================================================================
# TestPrioritySchedulerAging
# =============================================================================


class TestPrioritySchedulerAging:
    """Tests for aging mechanism in PriorityScheduler."""

    def test_priority_clamped_to_levels(
        self, priority_scheduler: PriorityScheduler
    ) -> None:
        """Priority capped at priority_levels - 1."""
        req = make_request(priority=100)  # Way above 10 levels

        priority_scheduler.submit(req)

        assert req.priority == 9  # Clamped to max level (10 - 1)

    def test_priority_clamped_to_zero(
        self, priority_scheduler: PriorityScheduler
    ) -> None:
        """Negative priority clamped to 0."""
        req = make_request(priority=-5)

        priority_scheduler.submit(req)

        assert req.priority == 0

    def test_aging_increments_wait_cycles(
        self, priority_scheduler: PriorityScheduler
    ) -> None:
        """Each get_batch() increments cycle count."""
        # Create request that won't fit in batch
        req = make_request(prompt_length=100, request_id="waiting")
        priority_scheduler.submit(req)

        # Multiple get_batch calls (empty batches due to token limit)
        for _ in range(3):
            priority_scheduler.get_batch()

        assert priority_scheduler._wait_cycles.get(req.request_id, 0) == 3

    def test_aging_clears_on_complete(
        self, priority_scheduler: PriorityScheduler
    ) -> None:
        """Completed requests removed from aging dict."""
        req = make_request(prompt_length=5)
        priority_scheduler.submit(req)
        priority_scheduler.get_batch()
        priority_scheduler.update_request(req.request_id, [100])

        priority_scheduler.complete_request(req.request_id)

        assert req.request_id not in priority_scheduler._wait_cycles


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Tests for concurrent access behavior."""

    def test_concurrent_submit(self, scheduler: RequestScheduler) -> None:
        """Multiple threads can submit safely."""
        num_threads = 10
        requests_per_thread = 20
        all_ids: List[str] = []
        lock = threading.Lock()

        def submit_requests(thread_id: int) -> None:
            for i in range(requests_per_thread):
                req_id = f"thread-{thread_id}-req-{i}"
                req = make_request(request_id=req_id)
                scheduler.submit(req)
                with lock:
                    all_ids.append(req_id)

        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=submit_requests, args=(t,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        assert scheduler.num_pending == num_threads * requests_per_thread
        assert len(all_ids) == num_threads * requests_per_thread

    def test_concurrent_get_batch(self, scheduler: RequestScheduler) -> None:
        """Multiple threads calling get_batch() don't crash."""
        # Submit many requests
        for i in range(50):
            scheduler.submit(make_request(prompt_length=2, request_id=f"req-{i}"))

        results: List[int] = []
        lock = threading.Lock()

        def get_batches() -> None:
            for _ in range(10):
                requests, batch = scheduler.get_batch()
                with lock:
                    results.append(len(requests))

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_batches)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # All requests should be processed exactly once
        total_processed = sum(results)
        assert total_processed == 50

    def test_submit_and_get_batch_concurrent(
        self, scheduler: RequestScheduler
    ) -> None:
        """Submit while get_batch in progress."""
        processed = []
        lock = threading.Lock()

        def submit_loop() -> None:
            for i in range(20):
                scheduler.submit(make_request(prompt_length=3, request_id=f"submit-{i}"))
                time.sleep(0.001)

        def get_loop() -> None:
            for _ in range(30):
                requests, batch = scheduler.get_batch()
                with lock:
                    processed.extend([r.request_id for r in requests])
                time.sleep(0.001)

        t1 = threading.Thread(target=submit_loop)
        t2 = threading.Thread(target=get_loop)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # All submitted should eventually be processed
        # Drain remaining
        while scheduler.has_pending():
            requests, batch = scheduler.get_batch()
            processed.extend([r.request_id for r in requests])

        assert len(set(processed)) == 20

    def test_concurrent_complete(self) -> None:
        """Multiple completions don't corrupt state."""
        # Use static scheduler to avoid continuous batching complexity
        config = SchedulerConfig(
            max_batch_tokens=1000,
            max_batch_size=10,
            enable_continuous_batching=False,
        )
        scheduler = RequestScheduler(config)

        # Submit and start requests
        for i in range(20):
            req = make_request(prompt_length=2, request_id=f"req-{i}")
            scheduler.submit(req)

        # Process all into active state (static batching, so only pending -> active)
        while scheduler.has_pending():
            requests, batch = scheduler.get_batch()
            for r in requests:
                scheduler.update_request(r.request_id, [100])

        # Concurrently complete
        def complete_requests(ids: List[str]) -> None:
            for req_id in ids:
                scheduler.complete_request(req_id)

        ids = [f"req-{i}" for i in range(20)]
        threads = []
        for chunk_start in range(0, 20, 5):
            chunk = ids[chunk_start : chunk_start + 5]
            thread = threading.Thread(target=complete_requests, args=(chunk,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        assert scheduler.num_active == 0
        assert scheduler.num_completed == 20


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and unusual scenarios."""

    def test_empty_queue_get_batch(self, scheduler: RequestScheduler) -> None:
        """get_batch on empty scheduler."""
        requests, batch = scheduler.get_batch()

        assert requests == []
        assert batch is None

    def test_single_request_workflow(self, scheduler: RequestScheduler) -> None:
        """Complete workflow with one request."""
        req = make_request()
        scheduler.submit(req)

        requests, batch = scheduler.get_batch()
        assert len(requests) == 1

        scheduler.update_request(req.request_id, [100])
        scheduler.complete_request(req.request_id)

        assert scheduler.num_pending == 0
        assert scheduler.num_active == 0
        assert scheduler.num_completed == 1

    def test_request_not_found_update(self, scheduler: RequestScheduler) -> None:
        """update_request with invalid ID doesn't crash."""
        scheduler.update_request("nonexistent", [100, 101])
        # Should not raise

    def test_request_not_found_complete(self, scheduler: RequestScheduler) -> None:
        """complete_request with invalid ID doesn't crash."""
        scheduler.complete_request("nonexistent")
        # Should not raise

    def test_complete_already_completed(self, scheduler: RequestScheduler) -> None:
        """Complete request twice is idempotent."""
        req = make_request()
        scheduler.submit(req)
        scheduler.get_batch()
        scheduler.update_request(req.request_id, [100])

        scheduler.complete_request(req.request_id)
        scheduler.complete_request(req.request_id)  # Second call

        assert scheduler.num_completed == 1

    def test_clear_scheduler(self, scheduler: RequestScheduler) -> None:
        """clear() removes all requests."""
        for i in range(5):
            scheduler.submit(make_request(request_id=f"req-{i}"))

        scheduler.get_batch()  # Move some to active

        scheduler.clear()

        assert scheduler.num_pending == 0
        assert scheduler.num_active == 0
        assert scheduler.num_completed == 0

    def test_get_request_from_all_states(self, scheduler: RequestScheduler) -> None:
        """get_request finds in pending/active/completed."""
        req1 = make_request(request_id="pending")
        req2 = make_request(request_id="active")
        req3 = make_request(request_id="completed")

        scheduler.submit(req1)
        scheduler.submit(req2)
        scheduler.submit(req3)

        # Move req2 and req3 to active
        scheduler.get_batch()

        # Complete req3
        scheduler.complete_request("completed")

        # Submit new pending
        req4 = make_request(request_id="new-pending")
        scheduler.submit(req4)

        assert scheduler.get_request("new-pending") is not None
        assert scheduler.get_request("active") is not None
        assert scheduler.get_request("completed") is not None

    def test_get_all_completed_clears(self, scheduler: RequestScheduler) -> None:
        """get_all_completed empties completed set."""
        for i in range(3):
            req = make_request(request_id=f"req-{i}")
            scheduler.submit(req)
            scheduler.get_batch()
            scheduler.complete_request(req.request_id)

        completed = scheduler.get_all_completed()

        assert len(completed) == 3
        assert scheduler.num_completed == 0

    def test_has_work_states(self, scheduler: RequestScheduler) -> None:
        """has_work() reflects pending OR active."""
        assert scheduler.has_work() is False

        scheduler.submit(make_request())
        assert scheduler.has_work() is True
        assert scheduler.has_pending() is True
        assert scheduler.has_active() is False

        scheduler.get_batch()
        assert scheduler.has_work() is True
        assert scheduler.has_pending() is False
        assert scheduler.has_active() is True

    def test_zero_token_request(self, scheduler: RequestScheduler) -> None:
        """Request with empty input_ids."""
        req = create_request(input_ids=[], request_id="empty")
        scheduler.submit(req)

        requests, batch = scheduler.get_batch()

        assert len(requests) == 1

    def test_get_completed_removes_from_set(self, scheduler: RequestScheduler) -> None:
        """get_completed retrieves and removes."""
        req = make_request(request_id="test")
        scheduler.submit(req)
        scheduler.get_batch()
        scheduler.complete_request(req.request_id)

        retrieved = scheduler.get_completed("test")
        assert retrieved is not None
        assert retrieved.request_id == "test"

        # Second retrieval returns None
        assert scheduler.get_completed("test") is None


# =============================================================================
# TestSchedulerStats
# =============================================================================


class TestSchedulerStats:
    """Tests for statistics reporting."""

    def test_stats_empty_scheduler(self, scheduler: RequestScheduler) -> None:
        """Stats reflect empty state."""
        stats = scheduler.get_stats()

        assert stats["num_pending"] == 0
        assert stats["num_active"] == 0
        assert stats["num_completed"] == 0
        assert stats["pending_tokens"] == 0
        assert stats["active_tokens"] == 0

    def test_stats_with_pending(self, scheduler: RequestScheduler) -> None:
        """Pending count and tokens correct."""
        scheduler.submit(make_request(prompt_length=5))
        scheduler.submit(make_request(prompt_length=10))

        stats = scheduler.get_stats()

        assert stats["num_pending"] == 2
        assert stats["pending_tokens"] == 15

    def test_stats_with_active(self, scheduler: RequestScheduler) -> None:
        """Active count and tokens correct."""
        scheduler.submit(make_request(prompt_length=5))
        scheduler.submit(make_request(prompt_length=10))
        scheduler.get_batch()

        stats = scheduler.get_stats()

        assert stats["num_active"] == 2
        assert stats["num_pending"] == 0

    def test_stats_with_completed(self, scheduler: RequestScheduler) -> None:
        """Completed count correct."""
        req = make_request()
        scheduler.submit(req)
        scheduler.get_batch()
        scheduler.complete_request(req.request_id)

        stats = scheduler.get_stats()

        assert stats["num_completed"] == 1


# =============================================================================
# TestBatchingStrategies
# =============================================================================


class TestBatchingStrategies:
    """Tests for static vs continuous batching modes."""

    def test_static_batch_only_pending(
        self, static_scheduler: RequestScheduler
    ) -> None:
        """Static mode ignores active requests."""
        req1 = make_request(request_id="first")
        static_scheduler.submit(req1)
        static_scheduler.get_batch()
        static_scheduler.update_request(req1.request_id, [100])

        req2 = make_request(request_id="second")
        static_scheduler.submit(req2)

        requests, batch = static_scheduler.get_batch()

        # Only new prefill, not the generating one
        assert len(requests) == 1
        assert requests[0].request_id == "second"

    def test_include_generating_false(self, scheduler: RequestScheduler) -> None:
        """include_generating=False excludes active."""
        req1 = make_request(request_id="gen")
        scheduler.submit(req1)
        scheduler.get_batch()
        scheduler.update_request(req1.request_id, [100])

        req2 = make_request(request_id="prefill")
        scheduler.submit(req2)

        requests, batch = scheduler.get_batch(include_generating=False)

        assert len(requests) == 1
        assert requests[0].request_id == "prefill"
