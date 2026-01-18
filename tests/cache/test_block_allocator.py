"""Tests for block allocator and page table."""

import pytest
import mlx.core as mx
import numpy as np

from mlx_primitives.cache import (
    BlockAllocator,
    BlockConfig,
    PageTable,
    get_optimal_block_size,
    calculate_num_blocks,
)


class TestBlockConfig:
    """Tests for BlockConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = BlockConfig()
        assert config.block_size == 16
        assert config.num_heads == 32
        assert config.head_dim == 128

    def test_block_bytes_fp16(self) -> None:
        """Test memory calculation for fp16."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64, dtype=mx.float16)
        # K + V: 2 * 16 * 8 * 64 * 2 bytes = 32768
        assert config.block_bytes == 32768

    def test_block_bytes_fp32(self) -> None:
        """Test memory calculation for fp32."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64, dtype=mx.float32)
        # K + V: 2 * 16 * 8 * 64 * 4 bytes = 65536
        assert config.block_bytes == 65536

    def test_kv_shape(self) -> None:
        """Test KV shape property."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        assert config.kv_shape == (16, 8, 64)


class TestBlockAllocator:
    """Tests for BlockAllocator."""

    def test_allocation_basic(self) -> None:
        """Test basic block allocation."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)

        blocks = allocator.allocate(5)
        assert len(blocks) == 5
        assert allocator.num_allocated_blocks == 5
        assert allocator.num_free_blocks == 95

    def test_allocation_all_blocks(self) -> None:
        """Test allocating all blocks."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10)

        blocks = allocator.allocate(10)
        assert len(blocks) == 10
        assert allocator.num_free_blocks == 0

    def test_allocation_exceeds_capacity(self) -> None:
        """Test allocation when capacity exceeded."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=5)

        with pytest.raises(RuntimeError, match="Cannot allocate"):
            allocator.allocate(10)

    def test_free_blocks(self) -> None:
        """Test freeing blocks."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)

        blocks = allocator.allocate(10)
        allocator.free(blocks[:5])

        assert allocator.num_allocated_blocks == 5
        assert allocator.num_free_blocks == 95

    def test_free_and_reallocate(self) -> None:
        """Test that freed blocks can be reallocated."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10)

        blocks1 = allocator.allocate(10)
        allocator.free(blocks1[:5])
        blocks2 = allocator.allocate(3)

        assert len(blocks2) == 3
        assert allocator.num_allocated_blocks == 8

    def test_get_set_block_data(self) -> None:
        """Test reading and writing block data."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10)

        blocks = allocator.allocate(1)
        block_id = blocks[0]

        # Write data
        k = mx.random.normal((8, 8, 64))  # 8 tokens
        v = mx.random.normal((8, 8, 64))
        allocator.set_block_data(block_id, k, v, start_pos=0, length=8)

        # Read data
        k_out, v_out = allocator.get_block_data(block_id)

        # Use rtol=1e-3 for float16 tolerance
        np.testing.assert_allclose(
            np.array(k_out[:8]), np.array(k), rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            np.array(v_out[:8]), np.array(v), rtol=1e-3, atol=1e-3
        )

    def test_copy_on_write(self) -> None:
        """Test copy-on-write functionality."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10, enable_cow=True)

        blocks = allocator.allocate(1)
        block_id = blocks[0]

        # Write initial data
        k = mx.ones((16, 8, 64))
        v = mx.ones((16, 8, 64))
        allocator.set_block_data(block_id, k, v)

        # Increment ref count (simulating fork)
        allocator.increment_ref(block_id)
        assert allocator.get_ref_count(block_id) == 2

        # COW should create new block
        new_block = allocator.copy_on_write(block_id)
        assert new_block != block_id
        assert allocator.get_ref_count(block_id) == 1  # Decremented

    def test_get_pools(self) -> None:
        """Test getting pool tensors."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10)

        k_pool, v_pool = allocator.get_pools()

        assert k_pool.shape == (10, 16, 8, 64)
        assert v_pool.shape == (10, 16, 8, 64)

    def test_clear(self) -> None:
        """Test clearing allocator."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=10)

        allocator.allocate(5)
        allocator.clear()

        assert allocator.num_allocated_blocks == 0
        assert allocator.num_free_blocks == 10

    def test_stats(self) -> None:
        """Test statistics."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)

        allocator.allocate(25)
        stats = allocator.get_stats()

        assert stats["num_blocks"] == 100
        assert stats["num_allocated"] == 25
        assert stats["num_free"] == 75
        assert stats["utilization"] == 0.25


class TestPageTable:
    """Tests for PageTable."""

    def test_create_sequence(self) -> None:
        """Test creating a sequence."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence()
        assert seq_id == 0
        assert page_table.num_sequences == 1

    def test_create_sequence_with_initial_tokens(self) -> None:
        """Test creating sequence with pre-allocated blocks."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence(initial_tokens=48)  # 3 blocks
        metadata = page_table.get_sequence_metadata(seq_id)

        assert len(metadata.block_table) == 3

    def test_extend_sequence(self) -> None:
        """Test extending a sequence."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence()
        page_table.extend_sequence(seq_id, 20)

        metadata = page_table.get_sequence_metadata(seq_id)
        assert metadata.num_tokens == 20
        assert len(metadata.block_table) == 2  # 20 tokens = 2 blocks

    def test_extend_sequence_incremental(self) -> None:
        """Test extending sequence incrementally."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence()

        # Add tokens one by one (simulating generation)
        for i in range(32):
            page_table.extend_sequence(seq_id, 1)

        metadata = page_table.get_sequence_metadata(seq_id)
        assert metadata.num_tokens == 32
        assert len(metadata.block_table) == 2

    def test_truncate_sequence(self) -> None:
        """Test truncating a sequence."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence(initial_tokens=48)  # 3 blocks
        page_table.extend_sequence(seq_id, 48)  # Now 48 tokens

        # Truncate to 20 tokens (2 blocks)
        page_table.truncate_sequence(seq_id, 20)

        metadata = page_table.get_sequence_metadata(seq_id)
        assert metadata.num_tokens == 20
        assert len(metadata.block_table) == 2

    def test_fork_sequence(self) -> None:
        """Test forking a sequence (for beam search)."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100, enable_cow=True)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence(initial_tokens=32)
        page_table.extend_sequence(seq_id, 32)

        forked_id = page_table.fork_sequence(seq_id)

        assert forked_id != seq_id
        assert page_table.num_sequences == 2

        # Both should have same blocks (shared)
        orig_metadata = page_table.get_sequence_metadata(seq_id)
        fork_metadata = page_table.get_sequence_metadata(forked_id)
        assert orig_metadata.block_table == fork_metadata.block_table

    def test_delete_sequence(self) -> None:
        """Test deleting a sequence."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq_id = page_table.create_sequence(initial_tokens=32)
        page_table.extend_sequence(seq_id, 32)
        initial_allocated = allocator.num_allocated_blocks

        page_table.delete_sequence(seq_id)

        assert page_table.num_sequences == 0
        assert allocator.num_allocated_blocks < initial_allocated

    def test_get_block_table_tensor(self) -> None:
        """Test getting batched block table tensor."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        # Create sequences of different lengths
        seq1 = page_table.create_sequence(initial_tokens=16)
        page_table.extend_sequence(seq1, 16)
        seq2 = page_table.create_sequence(initial_tokens=48)
        page_table.extend_sequence(seq2, 48)

        block_table = page_table.get_block_table_tensor([seq1, seq2])

        assert block_table.shape == (2, 3)  # 2 seqs, max 3 blocks
        assert int(block_table[0, 1]) == -1  # Padding for seq1

    def test_get_context_lengths(self) -> None:
        """Test getting context lengths."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config, num_blocks=100)
        page_table = PageTable(allocator)

        seq1 = page_table.create_sequence()
        page_table.extend_sequence(seq1, 20)
        seq2 = page_table.create_sequence()
        page_table.extend_sequence(seq2, 35)

        context_lens = page_table.get_context_lengths([seq1, seq2])

        np.testing.assert_array_equal(np.array(context_lens), [20, 35])


class TestUtilities:
    """Tests for utility functions."""

    def test_get_optimal_block_size(self) -> None:
        """Test optimal block size calculation."""
        block_size = get_optimal_block_size(
            head_dim=128, num_heads=32, dtype=mx.float16
        )
        # Should be a power of 2 between 8 and 64
        assert block_size in [8, 16, 32, 64]
        assert block_size & (block_size - 1) == 0  # Power of 2

    def test_calculate_num_blocks(self) -> None:
        """Test block count calculation."""
        config = BlockConfig(block_size=16, num_heads=8, head_dim=64)
        num_blocks = calculate_num_blocks(0.1, config)  # 100MB

        assert num_blocks > 0
        # Verify calculation: 0.1GB * 0.9 / block_bytes
        expected = int(0.1 * 1e9 * 0.9 / config.block_bytes)
        assert num_blocks == expected
