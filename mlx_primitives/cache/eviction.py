"""Cache eviction policies for KV cache management.

This module provides eviction policies to manage memory when the KV cache
reaches capacity. Follows patterns from model_cache.py for LRU implementation.

Lock Ordering Protocol:
    EvictionPolicy locks are the lowest priority in the cache lock hierarchy.
    See page_table.py for the full lock ordering documentation.

    IMPORTANT: EvictionPolicy methods must NOT call PageTable or BlockAllocator
    methods while holding their internal _lock. Doing so can cause deadlocks.

    Allowed call patterns:
    - PageTable -> EvictionPolicy (OK)
    - BlockAllocator -> EvictionPolicy (OK)
    - EvictionPolicy -> PageTable (FORBIDDEN while holding _lock)
    - EvictionPolicy -> BlockAllocator (FORBIDDEN while holding _lock)
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from mlx_primitives.utils.lock_validator import ordered_lock

if TYPE_CHECKING:
    from mlx_primitives.cache.page_table import PageTable


class EvictionPolicy(ABC):
    """Base class for cache eviction policies.

    Policies track sequence access patterns and select sequences
    for eviction when memory pressure requires freeing space.
    """

    @abstractmethod
    def on_access(self, sequence_id: int) -> None:
        """Called when a sequence is accessed.

        Args:
            sequence_id: ID of the accessed sequence.
        """
        pass

    @abstractmethod
    def on_create(self, sequence_id: int) -> None:
        """Called when a new sequence is created.

        Args:
            sequence_id: ID of the new sequence.
        """
        pass

    @abstractmethod
    def on_delete(self, sequence_id: int) -> None:
        """Called when a sequence is deleted.

        Args:
            sequence_id: ID of the deleted sequence.
        """
        pass

    @abstractmethod
    def select_for_eviction(
        self,
        candidates: List[int],
        num_to_evict: int,
    ) -> List[int]:
        """Select sequences to evict to free memory.

        Args:
            candidates: List of sequence IDs that can be evicted.
            num_to_evict: Number of sequences to select.

        Returns:
            List of sequence IDs to evict.
        """
        pass

    @abstractmethod
    def get_priority(self, sequence_id: int) -> float:
        """Get eviction priority for a sequence.

        Lower priority = evict first.

        Args:
            sequence_id: Sequence ID.

        Returns:
            Priority score (lower = evict first).
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy.

    Uses OrderedDict for O(1) access tracking and efficient eviction selection.
    Thread-safe via internal locking.

    Example:
        >>> policy = LRUEvictionPolicy()
        >>> policy.on_create(1)
        >>> policy.on_create(2)
        >>> policy.on_access(1)  # 1 is now most recent
        >>> policy.select_for_eviction([1, 2], 1)  # Returns [2]
    """

    def __init__(self):
        """Initialize LRU policy."""
        # OrderedDict provides O(1) move_to_end and maintains insertion order
        self._access_order: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()

    def on_access(self, sequence_id: int) -> None:
        """Update access order - move to end (most recent). O(1) operation."""
        with ordered_lock("EvictionPolicy", self._lock):
            if sequence_id in self._access_order:
                self._access_order.move_to_end(sequence_id)
            self._access_order[sequence_id] = time.time()

    def on_create(self, sequence_id: int) -> None:
        """Add new sequence to tracking."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._access_order[sequence_id] = time.time()

    def on_delete(self, sequence_id: int) -> None:
        """Remove sequence from tracking."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._access_order.pop(sequence_id, None)

    def select_for_eviction(
        self,
        candidates: List[int],
        num_to_evict: int,
    ) -> List[int]:
        """Select least recently used sequences."""
        candidate_set = set(candidates)

        with ordered_lock("EvictionPolicy", self._lock):
            # OrderedDict iteration is from oldest to newest
            to_evict = []
            for seq_id in self._access_order:
                if seq_id in candidate_set:
                    to_evict.append(seq_id)
                    if len(to_evict) >= num_to_evict:
                        break

        return to_evict

    def get_priority(self, sequence_id: int) -> float:
        """Priority based on access time (older = lower priority)."""
        with ordered_lock("EvictionPolicy", self._lock):
            return self._access_order.get(sequence_id, 0.0)


class FIFOEvictionPolicy(EvictionPolicy):
    """First In First Out eviction policy.

    Simple queue-based eviction suitable for streaming workloads
    where older sequences are naturally less relevant.
    Thread-safe via internal locking.

    Example:
        >>> policy = FIFOEvictionPolicy()
        >>> policy.on_create(1)
        >>> policy.on_create(2)
        >>> policy.select_for_eviction([1, 2], 1)  # Returns [1] (oldest)
    """

    def __init__(self):
        """Initialize FIFO policy."""
        # Use OrderedDict for O(1) deletion instead of list.remove()
        self._creation_order: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()

    def on_access(self, sequence_id: int) -> None:
        """FIFO ignores access - only creation order matters."""
        pass

    def on_create(self, sequence_id: int) -> None:
        """Add new sequence to queue."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._creation_order[sequence_id] = time.time()

    def on_delete(self, sequence_id: int) -> None:
        """Remove sequence from queue."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._creation_order.pop(sequence_id, None)

    def select_for_eviction(
        self,
        candidates: List[int],
        num_to_evict: int,
    ) -> List[int]:
        """Select oldest sequences."""
        candidate_set = set(candidates)

        with ordered_lock("EvictionPolicy", self._lock):
            to_evict = []
            for seq_id in self._creation_order:
                if seq_id in candidate_set:
                    to_evict.append(seq_id)
                    if len(to_evict) >= num_to_evict:
                        break

        return to_evict

    def get_priority(self, sequence_id: int) -> float:
        """Priority based on creation time (older = lower priority)."""
        with ordered_lock("EvictionPolicy", self._lock):
            return self._creation_order.get(sequence_id, 0.0)


class AttentionScoreEvictionPolicy(EvictionPolicy):
    """Eviction based on accumulated attention scores.

    Tracks which sequences receive attention and evicts those
    with lowest total attention. Useful for long-context scenarios
    where some prefixes are more relevant than others.

    Attention scores decay over time to favor recent relevance.
    Thread-safe via internal locking.

    Note on decay_factor: Default 0.99 means ~99% weight on history, ~1% on new.
    After 100 updates, the contribution of the first score is ~0.99^100 ≈ 0.37.
    Use lower values (e.g., 0.9) for faster adaptation to recent patterns.

    Example:
        >>> policy = AttentionScoreEvictionPolicy(decay_factor=0.99)
        >>> policy.on_create(1)
        >>> policy.update_attention_score(1, 0.8)
        >>> policy.update_attention_score(1, 0.6)  # Score accumulates with decay
    """

    def __init__(self, decay_factor: float = 0.99):
        """Initialize attention score policy.

        Args:
            decay_factor: Decay factor for exponential moving average (0.0-1.0).
                Higher values (closer to 1) give more weight to history.
                Default 0.99 balances recent relevance with stability.
        """
        self._decay_factor = decay_factor
        self._attention_scores: Dict[int, float] = {}
        self._access_times: Dict[int, float] = {}
        self._lock = threading.Lock()

    def on_access(self, sequence_id: int) -> None:
        """Update access time."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._access_times[sequence_id] = time.time()

    def on_create(self, sequence_id: int) -> None:
        """Initialize tracking for new sequence."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._attention_scores[sequence_id] = 0.0
            self._access_times[sequence_id] = time.time()

    def on_delete(self, sequence_id: int) -> None:
        """Remove sequence from tracking."""
        with ordered_lock("EvictionPolicy", self._lock):
            self._attention_scores.pop(sequence_id, None)
            self._access_times.pop(sequence_id, None)

    def update_attention_score(self, sequence_id: int, score: float) -> None:
        """Update attention score for a sequence.

        Uses exponential moving average with decay.

        Args:
            sequence_id: Sequence ID.
            score: New attention score to incorporate.
        """
        with ordered_lock("EvictionPolicy", self._lock):
            if sequence_id not in self._attention_scores:
                self._attention_scores[sequence_id] = 0.0

            # Exponential moving average
            old_score = self._attention_scores[sequence_id]
            self._attention_scores[sequence_id] = (
                self._decay_factor * old_score + (1 - self._decay_factor) * score
            )
            self._access_times[sequence_id] = time.time()

    def select_for_eviction(
        self,
        candidates: List[int],
        num_to_evict: int,
    ) -> List[int]:
        """Select sequences with lowest attention scores."""
        with ordered_lock("EvictionPolicy", self._lock):
            # Sort by attention score (lowest first)
            scored = [
                (seq_id, self._attention_scores.get(seq_id, 0.0))
                for seq_id in candidates
            ]
        scored.sort(key=lambda x: x[1])

        return [seq_id for seq_id, _ in scored[:num_to_evict]]

    def get_priority(self, sequence_id: int) -> float:
        """Priority based on attention score (higher = keep longer)."""
        with ordered_lock("EvictionPolicy", self._lock):
            return self._attention_scores.get(sequence_id, 0.0)


class CompositeEvictionPolicy(EvictionPolicy):
    """Combines multiple policies with configurable weights.

    Useful for balancing different eviction criteria, e.g.,
    70% LRU + 30% attention score.

    Example:
        >>> lru = LRUEvictionPolicy()
        >>> attn = AttentionScoreEvictionPolicy()
        >>> composite = CompositeEvictionPolicy([(lru, 0.7), (attn, 0.3)])
    """

    def __init__(self, policies: List[Tuple[EvictionPolicy, float]]):
        """Initialize composite policy.

        Args:
            policies: List of (policy, weight) tuples.
        """
        self._policies = policies
        total_weight = sum(w for _, w in policies)
        self._normalized_weights = [w / total_weight for _, w in policies]

    def on_access(self, sequence_id: int) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_access(sequence_id)

    def on_create(self, sequence_id: int) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_create(sequence_id)

    def on_delete(self, sequence_id: int) -> None:
        """Forward to all policies."""
        for policy, _ in self._policies:
            policy.on_delete(sequence_id)

    def select_for_eviction(
        self,
        candidates: List[int],
        num_to_evict: int,
    ) -> List[int]:
        """Select based on weighted priority scores."""
        # Compute weighted priorities
        priorities = {seq_id: 0.0 for seq_id in candidates}

        for (policy, _), weight in zip(self._policies, self._normalized_weights):
            # Get priorities from each policy
            policy_priorities = {
                seq_id: policy.get_priority(seq_id) for seq_id in candidates
            }

            # Normalize to [0, 1] range
            if policy_priorities:
                min_p = min(policy_priorities.values())
                max_p = max(policy_priorities.values())
                range_p = max_p - min_p if max_p > min_p else 1.0

                for seq_id in candidates:
                    normalized = (policy_priorities[seq_id] - min_p) / range_p
                    priorities[seq_id] += weight * normalized

        # Sort by combined priority (lowest first)
        sorted_candidates = sorted(candidates, key=lambda x: priorities[x])
        return sorted_candidates[:num_to_evict]

    def get_priority(self, sequence_id: int) -> float:
        """Get weighted priority."""
        priority = 0.0
        for (policy, _), weight in zip(self._policies, self._normalized_weights):
            priority += weight * policy.get_priority(sequence_id)
        return priority


@dataclass
class CacheMemoryStats:
    """Statistics about cache memory usage.

    Attributes:
        total_bytes: Total memory allocated for cache.
        used_bytes: Memory currently in use.
        num_sequences: Number of active sequences.
        num_blocks_allocated: Total allocated blocks.
        num_blocks_free: Available blocks.
        utilization: Fraction of capacity used.
    """

    total_bytes: int
    used_bytes: int
    num_sequences: int
    num_blocks_allocated: int
    num_blocks_free: int

    @property
    def utilization(self) -> float:
        """Cache utilization as a fraction."""
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0

    @property
    def total_mb(self) -> float:
        """Total memory in MB."""
        return self.total_bytes / (1024 * 1024)

    @property
    def used_mb(self) -> float:
        """Used memory in MB."""
        return self.used_bytes / (1024 * 1024)


class MemoryBudgetManager:
    """Enforces memory budget for KV cache.

    Monitors memory usage and triggers eviction when necessary.
    Works with eviction policies to select what to remove.

    Example:
        >>> manager = MemoryBudgetManager(
        ...     max_memory_bytes=8 * 1024**3,  # 8GB
        ...     eviction_policy=LRUEvictionPolicy(),
        ... )
        >>> # Check if we need to evict before allocation
        >>> if not manager.can_allocate(needed_bytes, page_table):
        ...     manager.evict_to_free(needed_bytes, page_table)
    """

    def __init__(
        self,
        max_memory_bytes: int,
        eviction_policy: Optional[EvictionPolicy] = None,
        soft_limit_fraction: float = 0.9,
    ):
        """Initialize memory budget manager.

        Args:
            max_memory_bytes: Maximum memory for KV cache.
            eviction_policy: Policy for selecting sequences to evict.
            soft_limit_fraction: Fraction at which to start warning/proactive eviction.
        """
        self._max_memory = max_memory_bytes
        self._soft_limit = int(max_memory_bytes * soft_limit_fraction)
        self._policy = eviction_policy or LRUEvictionPolicy()

    @property
    def max_memory_bytes(self) -> int:
        """Maximum allowed memory."""
        return self._max_memory

    @property
    def soft_limit_bytes(self) -> int:
        """Soft limit for warnings."""
        return self._soft_limit

    @property
    def eviction_policy(self) -> EvictionPolicy:
        """The eviction policy."""
        return self._policy

    def get_memory_stats(self, page_table: "PageTable") -> CacheMemoryStats:
        """Get current memory statistics.

        Args:
            page_table: Page table to query.

        Returns:
            Memory statistics.
        """
        allocator = page_table.allocator  # Use public property
        return CacheMemoryStats(
            total_bytes=allocator.total_memory_bytes,
            used_bytes=allocator.memory_usage_bytes,
            num_sequences=page_table.num_sequences,
            num_blocks_allocated=allocator.num_allocated_blocks,
            num_blocks_free=allocator.num_free_blocks,
        )

    def can_allocate(self, requested_bytes: int, page_table: "PageTable") -> bool:
        """Check if allocation would exceed budget.

        Args:
            requested_bytes: Bytes to allocate.
            page_table: Page table to check.

        Returns:
            True if allocation is possible within budget.
        """
        stats = self.get_memory_stats(page_table)
        return stats.used_bytes + requested_bytes <= self._max_memory

    def is_at_soft_limit(self, page_table: "PageTable") -> bool:
        """Check if at soft limit.

        Args:
            page_table: Page table to check.

        Returns:
            True if at or above soft limit.
        """
        stats = self.get_memory_stats(page_table)
        return stats.used_bytes >= self._soft_limit

    def evict_to_free(
        self,
        bytes_needed: int,
        page_table: "PageTable",
        protected_sequences: Optional[List[int]] = None,
    ) -> int:
        """Evict sequences to free the requested memory.

        Args:
            bytes_needed: Bytes that need to be freed.
            page_table: Page table to evict from.
            protected_sequences: Sequences that cannot be evicted.

        Returns:
            Number of sequences evicted.
        """
        protected = set(protected_sequences or [])
        allocator = page_table.allocator  # Use public property
        block_bytes = allocator.config.block_bytes

        # Get eviction candidates
        all_sequences = page_table.get_all_sequence_ids()
        candidates = [seq_id for seq_id in all_sequences if seq_id not in protected]

        if not candidates:
            return 0

        bytes_freed = 0
        sequences_evicted = 0

        while bytes_freed < bytes_needed and candidates:
            # Select one sequence to evict
            to_evict = self._policy.select_for_eviction(candidates, 1)
            if not to_evict:
                break

            seq_id = to_evict[0]
            metadata = page_table.get_sequence_metadata(seq_id)
            seq_bytes = len(metadata.block_table) * block_bytes

            # Delete the sequence
            page_table.delete_sequence(seq_id)
            self._policy.on_delete(seq_id)

            bytes_freed += seq_bytes
            sequences_evicted += 1
            candidates.remove(seq_id)

        return sequences_evicted

    def check_and_evict(
        self,
        page_table: "PageTable",
        protected_sequences: Optional[List[int]] = None,
    ) -> int:
        """Check memory and evict if above soft limit.

        Args:
            page_table: Page table to manage.
            protected_sequences: Sequences that cannot be evicted.

        Returns:
            Number of sequences evicted.
        """
        stats = self.get_memory_stats(page_table)

        if stats.used_bytes <= self._soft_limit:
            return 0

        # Evict to get below soft limit
        bytes_to_free = stats.used_bytes - self._soft_limit
        return self.evict_to_free(bytes_to_free, page_table, protected_sequences)
