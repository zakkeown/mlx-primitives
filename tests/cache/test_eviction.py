"""Tests for cache eviction policies."""
import time

import pytest

from mlx_primitives.cache.eviction import (
    AttentionScoreEvictionPolicy,
    CacheMemoryStats,
    CompositeEvictionPolicy,
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
)


class TestLRUEvictionPolicy:
    """Tests for LRU eviction policy."""

    def test_basic_lru_order(self) -> None:
        """Test that access updates LRU order."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)
        policy.on_create(2)
        policy.on_create(3)

        # Access 1, making it most recent
        policy.on_access(1)

        # Evict should return 2 (oldest accessed)
        to_evict = policy.select_for_eviction([1, 2, 3], 1)
        assert to_evict == [2]

    def test_lru_select_multiple(self) -> None:
        """Test selecting multiple for eviction."""
        policy = LRUEvictionPolicy()
        for i in range(5):
            policy.on_create(i)
            time.sleep(0.001)  # Ensure different timestamps

        to_evict = policy.select_for_eviction([0, 1, 2, 3, 4], 3)
        assert len(to_evict) == 3
        assert to_evict == [0, 1, 2]  # Oldest first

    def test_lru_delete_removes_tracking(self) -> None:
        """Test that delete removes sequence from tracking."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)
        policy.on_create(2)
        policy.on_delete(1)

        to_evict = policy.select_for_eviction([1, 2], 2)
        # 1 was deleted, should only return 2
        assert to_evict == [2]

    def test_lru_access_updates_priority(self) -> None:
        """Test that accessing updates priority."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)
        time.sleep(0.001)
        policy.on_create(2)

        # Initially, 1 has lower priority (older)
        p1_before = policy.get_priority(1)
        p2_before = policy.get_priority(2)
        assert p1_before < p2_before

        # Access 1, making it more recent
        policy.on_access(1)
        p1_after = policy.get_priority(1)
        assert p1_after > p2_before

    def test_lru_empty_candidates(self) -> None:
        """Test with empty candidate list."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)

        to_evict = policy.select_for_eviction([], 1)
        assert to_evict == []

    def test_lru_request_more_than_available(self) -> None:
        """Test requesting more evictions than candidates."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)
        policy.on_create(2)

        to_evict = policy.select_for_eviction([1, 2], 5)
        assert len(to_evict) == 2


class TestFIFOEvictionPolicy:
    """Tests for FIFO eviction policy."""

    def test_fifo_ignores_access(self) -> None:
        """Test that FIFO ignores access order."""
        policy = FIFOEvictionPolicy()
        policy.on_create(1)
        time.sleep(0.001)
        policy.on_create(2)

        # Access 1 - should not change eviction order
        policy.on_access(1)

        to_evict = policy.select_for_eviction([1, 2], 1)
        assert to_evict == [1]  # Still oldest by creation

    def test_fifo_creation_order(self) -> None:
        """Test FIFO maintains creation order."""
        policy = FIFOEvictionPolicy()
        for i in range(5):
            policy.on_create(i)
            time.sleep(0.001)

        to_evict = policy.select_for_eviction([0, 1, 2, 3, 4], 3)
        assert to_evict == [0, 1, 2]

    def test_fifo_delete(self) -> None:
        """Test FIFO handles deletion."""
        policy = FIFOEvictionPolicy()
        policy.on_create(1)
        policy.on_create(2)
        policy.on_create(3)
        policy.on_delete(2)

        to_evict = policy.select_for_eviction([1, 3], 2)
        assert to_evict == [1, 3]

    def test_fifo_priority(self) -> None:
        """Test FIFO priority based on creation time."""
        policy = FIFOEvictionPolicy()
        policy.on_create(1)
        time.sleep(0.001)
        policy.on_create(2)

        assert policy.get_priority(1) < policy.get_priority(2)


class TestAttentionScoreEvictionPolicy:
    """Tests for attention-score-based eviction."""

    def test_attention_score_update(self) -> None:
        """Test that attention scores accumulate with decay."""
        policy = AttentionScoreEvictionPolicy(decay_factor=0.5)
        policy.on_create(1)

        policy.update_attention_score(1, 1.0)
        # Score: 0.5 * 0 + 0.5 * 1.0 = 0.5
        assert abs(policy.get_priority(1) - 0.5) < 0.01

        policy.update_attention_score(1, 1.0)
        # Score: 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        assert abs(policy.get_priority(1) - 0.75) < 0.01

    def test_attention_score_eviction(self) -> None:
        """Test eviction selects lowest attention scores."""
        policy = AttentionScoreEvictionPolicy(decay_factor=0.9)
        policy.on_create(1)
        policy.on_create(2)
        policy.on_create(3)

        # Give different attention scores
        policy.update_attention_score(1, 0.1)
        policy.update_attention_score(2, 0.9)
        policy.update_attention_score(3, 0.5)

        # Evict 2 - should get 1 first (lowest score)
        to_evict = policy.select_for_eviction([1, 2, 3], 2)
        assert to_evict[0] == 1  # Lowest score first
        assert 2 not in to_evict  # Highest score kept

    def test_attention_score_delete(self) -> None:
        """Test deletion removes from tracking."""
        policy = AttentionScoreEvictionPolicy()
        policy.on_create(1)
        policy.update_attention_score(1, 0.5)
        policy.on_delete(1)

        assert policy.get_priority(1) == 0.0


class TestCompositeEvictionPolicy:
    """Tests for composite policies."""

    def test_weighted_combination(self) -> None:
        """Test weighted policy combination."""
        lru = LRUEvictionPolicy()
        fifo = FIFOEvictionPolicy()
        composite = CompositeEvictionPolicy([(lru, 0.7), (fifo, 0.3)])

        composite.on_create(1)
        composite.on_create(2)

        # Both should track the sequences
        assert lru.get_priority(1) > 0
        assert fifo.get_priority(1) > 0

    def test_composite_forwards_events(self) -> None:
        """Test that events are forwarded to all policies."""
        lru = LRUEvictionPolicy()
        attn = AttentionScoreEvictionPolicy()
        composite = CompositeEvictionPolicy([(lru, 0.5), (attn, 0.5)])

        composite.on_create(1)
        composite.on_access(1)
        composite.on_delete(1)

        # After delete, both should have removed tracking
        to_evict = lru.select_for_eviction([1], 1)
        assert to_evict == []

    def test_composite_eviction_order(self) -> None:
        """Test composite eviction considers all policies."""
        lru = LRUEvictionPolicy()
        fifo = FIFOEvictionPolicy()
        composite = CompositeEvictionPolicy([(lru, 0.5), (fifo, 0.5)])

        composite.on_create(1)
        time.sleep(0.001)
        composite.on_create(2)
        time.sleep(0.001)
        composite.on_create(3)

        # Access 1 to make it recent in LRU
        composite.on_access(1)

        to_evict = composite.select_for_eviction([1, 2, 3], 1)
        # 1 is old in FIFO but recent in LRU
        # 2 is old in both - likely to be evicted
        assert 2 in to_evict


class TestCacheMemoryStats:
    """Tests for CacheMemoryStats dataclass."""

    def test_utilization(self) -> None:
        """Test utilization calculation."""
        stats = CacheMemoryStats(
            total_bytes=1000,
            used_bytes=500,
            num_sequences=5,
            num_blocks_allocated=10,
            num_blocks_free=10,
        )
        assert abs(stats.utilization - 0.5) < 0.001

    def test_utilization_zero_total(self) -> None:
        """Test utilization with zero total."""
        stats = CacheMemoryStats(
            total_bytes=0,
            used_bytes=0,
            num_sequences=0,
            num_blocks_allocated=0,
            num_blocks_free=0,
        )
        assert stats.utilization == 0.0

    def test_memory_mb_conversion(self) -> None:
        """Test MB conversion."""
        stats = CacheMemoryStats(
            total_bytes=1024 * 1024 * 100,  # 100 MB
            used_bytes=1024 * 1024 * 50,  # 50 MB
            num_sequences=10,
            num_blocks_allocated=50,
            num_blocks_free=50,
        )
        assert abs(stats.total_mb - 100.0) < 0.001
        assert abs(stats.used_mb - 50.0) < 0.001


class TestEvictionPolicyEdgeCases:
    """Edge case tests for eviction policies."""

    def test_duplicate_access(self) -> None:
        """Test handling duplicate access calls."""
        policy = LRUEvictionPolicy()
        policy.on_create(1)
        policy.on_access(1)
        policy.on_access(1)
        policy.on_access(1)

        # Should still work correctly
        to_evict = policy.select_for_eviction([1], 1)
        assert to_evict == [1]

    def test_delete_nonexistent(self) -> None:
        """Test deleting non-existent sequence doesn't crash."""
        policy = LRUEvictionPolicy()
        policy.on_delete(999)  # Should not raise

    def test_priority_nonexistent(self) -> None:
        """Test priority of non-existent sequence."""
        policy = LRUEvictionPolicy()
        priority = policy.get_priority(999)
        assert priority == 0.0
