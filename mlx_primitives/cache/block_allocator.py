"""Block-based memory allocation for KV cache.

This module provides efficient fixed-size block allocation for KV storage,
enabling paged attention with non-contiguous memory and zero fragmentation.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import mlx.core as mx

from mlx_primitives.constants import DEFAULT_L2_CACHE_MB


@dataclass(frozen=True)
class BlockConfig:
    """Configuration for KV cache blocks.

    Attributes:
        block_size: Number of tokens per block (16-64 typical).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        dtype: Data type for storage.
    """

    block_size: int = 16
    num_heads: int = 32
    head_dim: int = 128
    dtype: mx.Dtype = mx.float16

    @property
    def tokens_per_block(self) -> int:
        """Alias for block_size."""
        return self.block_size

    @property
    def block_bytes(self) -> int:
        """Memory per block (K + V combined)."""
        dtype_size = 2 if self.dtype in (mx.float16, mx.bfloat16) else 4
        return 2 * self.block_size * self.num_heads * self.head_dim * dtype_size

    @property
    def kv_shape(self) -> Tuple[int, int, int]:
        """Shape of a single K or V block: (block_size, num_heads, head_dim)."""
        return (self.block_size, self.num_heads, self.head_dim)


class BlockAllocator:
    """Fixed-size block allocator for KV cache.

    Manages a pool of equally-sized blocks, enabling:
    - O(1) allocation and deallocation
    - Zero fragmentation
    - Copy-on-write support for prefix sharing

    Design Considerations for Apple Silicon:
    - Blocks are pre-allocated contiguously for cache efficiency
    - Block size tuned for L2 cache (8-48MB depending on chip tier)
    - Unified memory enables zero-copy block references

    Example:
        >>> config = BlockConfig(block_size=16, num_heads=32, head_dim=128)
        >>> allocator = BlockAllocator(config, num_blocks=1000)
        >>> block_ids = allocator.allocate(5)
        >>> k, v = allocator.get_block_data(block_ids[0])
        >>> allocator.free(block_ids)
    """

    def __init__(
        self,
        config: BlockConfig,
        num_blocks: int,
        enable_cow: bool = True,
    ):
        """Initialize the block allocator.

        Args:
            config: Block configuration.
            num_blocks: Total number of blocks to pre-allocate.
            enable_cow: Enable copy-on-write for prefix sharing.
        """
        self._config = config
        self._num_blocks = num_blocks
        self._enable_cow = enable_cow

        # Pre-allocate block pools as contiguous memory
        # K blocks: (num_blocks, block_size, num_heads, head_dim)
        # V blocks: (num_blocks, block_size, num_heads, head_dim)
        block_shape = (num_blocks,) + config.kv_shape
        self._k_pool = mx.zeros(block_shape, dtype=config.dtype)
        self._v_pool = mx.zeros(block_shape, dtype=config.dtype)

        # Free list for O(1) allocation
        self._free_blocks: Set[int] = set(range(num_blocks))

        # Reference counts for copy-on-write
        self._ref_counts: List[int] = [0] * num_blocks

        # Track which blocks are allocated
        self._allocated_blocks: Set[int] = set()

    @property
    def config(self) -> BlockConfig:
        """Get the block configuration."""
        return self._config

    @property
    def num_blocks(self) -> int:
        """Total number of blocks in the pool."""
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of blocks currently in use."""
        return len(self._allocated_blocks)

    @property
    def memory_usage_bytes(self) -> int:
        """Total memory used by allocated blocks."""
        return self.num_allocated_blocks * self._config.block_bytes

    @property
    def total_memory_bytes(self) -> int:
        """Total memory of the block pool."""
        return self._num_blocks * self._config.block_bytes

    def allocate(self, count: int = 1) -> List[int]:
        """Allocate blocks from the pool.

        Args:
            count: Number of blocks to allocate.

        Returns:
            List of allocated block indices.

        Raises:
            RuntimeError: If not enough blocks are available.
        """
        if count > len(self._free_blocks):
            raise RuntimeError(
                f"Cannot allocate {count} blocks, only {len(self._free_blocks)} available"
            )

        allocated = []
        for _ in range(count):
            block_id = self._free_blocks.pop()
            self._allocated_blocks.add(block_id)
            self._ref_counts[block_id] = 1
            allocated.append(block_id)

        return allocated

    def _validate_block_id(self, block_id: int, context: str = "") -> None:
        """Validate that a block_id is within bounds.

        Args:
            block_id: Block ID to validate.
            context: Optional context string for error message.

        Raises:
            ValueError: If block_id is out of bounds.
        """
        if not isinstance(block_id, int) or block_id < 0 or block_id >= self._num_blocks:
            ctx = f" in {context}" if context else ""
            raise ValueError(
                f"block_id {block_id} out of bounds [0, {self._num_blocks}){ctx}"
            )

    def free(self, block_ids: List[int]) -> None:
        """Return blocks to the free pool.

        For COW blocks, decrements reference count and only frees when count reaches 0.

        Args:
            block_ids: List of block indices to free.

        Raises:
            ValueError: If any block_id is out of bounds.
        """
        for block_id in block_ids:
            self._validate_block_id(block_id, "free")

            if block_id not in self._allocated_blocks:
                continue

            self._ref_counts[block_id] -= 1

            if self._ref_counts[block_id] <= 0:
                self._allocated_blocks.discard(block_id)
                self._free_blocks.add(block_id)
                self._ref_counts[block_id] = 0

    def increment_ref(self, block_id: int) -> None:
        """Increment reference count for a block (for COW sharing).

        Args:
            block_id: Block index to increment.

        Raises:
            ValueError: If block_id is out of bounds.
        """
        self._validate_block_id(block_id, "increment_ref")
        if block_id in self._allocated_blocks:
            self._ref_counts[block_id] += 1

    def get_ref_count(self, block_id: int) -> int:
        """Get reference count for a block.

        Args:
            block_id: Block index.

        Returns:
            Current reference count.

        Raises:
            ValueError: If block_id is out of bounds.
        """
        self._validate_block_id(block_id, "get_ref_count")
        return self._ref_counts[block_id]

    def copy_on_write(self, block_id: int) -> int:
        """Create a copy of a block for modification.

        If the block has ref_count > 1, allocates a new block and copies data.
        Otherwise, returns the same block_id (safe to modify in place).

        Args:
            block_id: Block to potentially copy.

        Returns:
            Block ID safe for modification (may be same or new).

        Raises:
            ValueError: If block_id is out of bounds.
            RuntimeError: If no blocks available for copy.
        """
        self._validate_block_id(block_id, "copy_on_write")

        if not self._enable_cow:
            return block_id

        if self._ref_counts[block_id] <= 1:
            # Safe to modify in place
            return block_id

        # Need to copy
        new_blocks = self.allocate(1)
        new_block_id = new_blocks[0]

        # Copy data
        self._k_pool[new_block_id] = self._k_pool[block_id]
        self._v_pool[new_block_id] = self._v_pool[block_id]

        # Decrement ref count on original
        self._ref_counts[block_id] -= 1

        return new_block_id

    def get_block_data(self, block_id: int) -> Tuple[mx.array, mx.array]:
        """Get K and V data for a block as views (zero-copy).

        Args:
            block_id: Block index.

        Returns:
            Tuple of (K, V) arrays, each shape (block_size, num_heads, head_dim).

        Raises:
            ValueError: If block_id is out of bounds.
        """
        self._validate_block_id(block_id, "get_block_data")
        return self._k_pool[block_id], self._v_pool[block_id]

    def set_block_data(
        self,
        block_id: int,
        k: mx.array,
        v: mx.array,
        start_pos: int = 0,
        length: Optional[int] = None,
    ) -> None:
        """Set K and V data for a block.

        Args:
            block_id: Block index to write to.
            k: Key data, shape (length, num_heads, head_dim).
            v: Value data, shape (length, num_heads, head_dim).
            start_pos: Starting position within the block.
            length: Number of tokens to write. Defaults to k.shape[0].

        Raises:
            ValueError: If block_id is out of bounds.
        """
        self._validate_block_id(block_id, "set_block_data")
        if length is None:
            length = k.shape[0]

        end_pos = start_pos + length

        # Handle COW if needed
        actual_block_id = self.copy_on_write(block_id) if self._enable_cow else block_id

        # Update the block data using direct assignment
        self._k_pool[actual_block_id, start_pos:end_pos] = k
        self._v_pool[actual_block_id, start_pos:end_pos] = v

    def get_pools(self) -> Tuple[mx.array, mx.array]:
        """Get the full K and V block pools.

        Returns:
            Tuple of (K_pool, V_pool), each shape (num_blocks, block_size, num_heads, head_dim).
        """
        return self._k_pool, self._v_pool

    def clear(self) -> None:
        """Reset all blocks to free state."""
        self._free_blocks = set(range(self._num_blocks))
        self._allocated_blocks.clear()
        self._ref_counts = [0] * self._num_blocks

    def get_stats(self) -> dict:
        """Get allocator statistics.

        Returns:
            Dictionary with allocation statistics.
        """
        return {
            "num_blocks": self._num_blocks,
            "num_free": self.num_free_blocks,
            "num_allocated": self.num_allocated_blocks,
            "memory_used_mb": self.memory_usage_bytes / (1024 * 1024),
            "memory_total_mb": self.total_memory_bytes / (1024 * 1024),
            "utilization": self.num_allocated_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0,
            "cow_enabled": self._enable_cow,
        }


def get_optimal_block_size(
    head_dim: int,
    num_heads: int,
    dtype: mx.Dtype = mx.float16,
    target_l2_fraction: float = 0.5,
    l2_cache_mb: Optional[float] = None,
) -> int:
    """Select optimal block size for current hardware.

    Targets fitting 2-4 K+V block pairs in the specified L2 cache fraction.

    Args:
        head_dim: Dimension per attention head.
        num_heads: Number of attention heads.
        dtype: Data type for storage.
        target_l2_fraction: Fraction of L2 to target (default 50%).
        l2_cache_mb: L2 cache size in MB. If None, auto-detects from hardware.

    Returns:
        Optimal block size (power of 2, between 8 and 64).
    """
    # Auto-detect L2 cache size from hardware if not specified
    if l2_cache_mb is None:
        try:
            from mlx_primitives.hardware import get_chip_info

            chip_info = get_chip_info()
            l2_cache_mb = float(chip_info.l2_cache_mb)
        except Exception:
            # Fallback to default if hardware detection fails
            l2_cache_mb = DEFAULT_L2_CACHE_MB

    dtype_size = 2 if dtype in (mx.float16, mx.bfloat16) else 4
    kv_per_token = 2 * num_heads * head_dim * dtype_size

    # Target: 4 blocks fit in target fraction of L2
    target_bytes = l2_cache_mb * 1024 * 1024 * target_l2_fraction / 4
    block_size = int(target_bytes / kv_per_token)

    # Round down to power of 2
    import math

    block_size = 2 ** int(math.log2(max(8, block_size)))

    # Clamp to reasonable range
    return min(max(8, block_size), 64)


def calculate_num_blocks(
    max_memory_gb: float,
    config: BlockConfig,
    reserve_fraction: float = 0.1,
) -> int:
    """Calculate number of blocks for a memory budget.

    Args:
        max_memory_gb: Maximum memory for KV cache in GB.
        config: Block configuration.
        reserve_fraction: Fraction to reserve for overhead.

    Returns:
        Number of blocks that fit in the budget.
    """
    available_bytes = max_memory_gb * 1e9 * (1 - reserve_fraction)
    return int(available_bytes / config.block_bytes)
