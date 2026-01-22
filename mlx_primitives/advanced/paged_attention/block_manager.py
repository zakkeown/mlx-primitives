"""Block Manager for Paged KV Cache.

Manages physical memory blocks for efficient KV cache allocation.
Based on vLLM's PagedAttention design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import mlx.core as mx

from mlx_primitives.utils.dtypes import get_dtype_size


@dataclass
class BlockConfig:
    """Configuration for paged attention blocks.

    Args:
        block_size: Number of tokens per block (default: 16).
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        num_layers: Number of transformer layers.
        max_blocks: Maximum number of blocks in the pool.
        dtype: Data type for KV cache (default: float16).
    """
    block_size: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 32
    max_blocks: int = 1024
    dtype: mx.Dtype = mx.float16

    @property
    def block_memory_bytes(self) -> int:
        """Memory per block in bytes."""
        # K and V for all layers
        # Use centralized dtype size utility to correctly handle all dtypes
        # (float16, bfloat16, int8, etc.)
        dtype_size = get_dtype_size(self.dtype)
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.num_layers
            * 2  # K and V
            * dtype_size
        )

    @property
    def total_memory_bytes(self) -> int:
        """Total pool memory in bytes."""
        return self.max_blocks * self.block_memory_bytes


@dataclass
class PhysicalBlock:
    """A physical memory block in the KV cache pool."""
    block_id: int
    ref_count: int = 0
    is_allocated: bool = False


class BlockManager:
    """Manages allocation of physical blocks for paged KV cache.

    Uses a free list to efficiently allocate and deallocate blocks.
    Supports copy-on-write (COW) for beam search and speculative decoding.

    Args:
        config: Block configuration.

    Example:
        >>> config = BlockConfig(block_size=16, num_kv_heads=8, head_dim=128)
        >>> manager = BlockManager(config)
        >>> blocks = manager.allocate_blocks(num_blocks=4)
        >>> manager.free_blocks(blocks)
    """

    def __init__(self, config: BlockConfig):
        self.config = config

        # Initialize physical blocks
        self._blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i)
            for i in range(config.max_blocks)
        ]

        # Free block list
        self._free_blocks: Set[int] = set(range(config.max_blocks))

        # Physical memory pool: [num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]
        # Dimension order optimized for layer-by-layer access pattern
        self._kv_pool: Optional[mx.array] = None
        self._initialized = False

    def _initialize_pool(self):
        """Lazily initialize the physical memory pool."""
        if self._initialized:
            return

        # Create the KV cache pool
        # Shape: [max_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]
        # The '2' dimension is for K (index 0) and V (index 1)
        self._kv_pool = mx.zeros(
            (
                self.config.max_blocks,
                self.config.num_layers,
                2,  # K=0, V=1
                self.config.num_kv_heads,
                self.config.block_size,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
        )
        self._initialized = True

    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of allocated blocks."""
        return self.config.max_blocks - len(self._free_blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate the requested number of blocks."""
        return num_blocks <= len(self._free_blocks)

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate physical blocks.

        Args:
            num_blocks: Number of blocks to allocate.

        Returns:
            List of allocated block IDs.

        Raises:
            RuntimeError: If not enough free blocks available.
        """
        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks, "
                f"only {len(self._free_blocks)} free"
            )

        self._initialize_pool()

        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop()
            block = self._blocks[block_id]
            block.is_allocated = True
            block.ref_count = 1
            allocated.append(block_id)

        return allocated

    def free_blocks(self, block_ids: List[int]):
        """Free physical blocks.

        Decrements reference count and returns to free list when ref_count = 0.

        Args:
            block_ids: List of block IDs to free.
        """
        for block_id in block_ids:
            block = self._blocks[block_id]
            if not block.is_allocated:
                continue

            block.ref_count -= 1
            if block.ref_count <= 0:
                block.is_allocated = False
                block.ref_count = 0
                self._free_blocks.add(block_id)

    def _validate_block_id(self, block_id: int, context: str = "") -> None:
        """Validate that a block_id is within bounds.

        Args:
            block_id: Block ID to validate.
            context: Optional context string for error message.

        Raises:
            ValueError: If block_id is out of bounds.
        """
        if not isinstance(block_id, int) or block_id < 0 or block_id >= self.config.max_blocks:
            ctx = f" in {context}" if context else ""
            raise ValueError(
                f"block_id {block_id} out of bounds [0, {self.config.max_blocks}){ctx}"
            )

    def increment_ref(self, block_ids: List[int]):
        """Increment reference count for blocks (for COW).

        Args:
            block_ids: Block IDs to increment ref count.

        Raises:
            ValueError: If any block_id is out of bounds.
        """
        for block_id in block_ids:
            self._validate_block_id(block_id, "increment_ref")
            self._blocks[block_id].ref_count += 1

    def copy_block(self, src_block_id: int) -> int:
        """Copy a block for copy-on-write.

        Args:
            src_block_id: Source block ID.

        Returns:
            New block ID with copied data.

        Raises:
            ValueError: If src_block_id is out of bounds.
        """
        self._validate_block_id(src_block_id, "copy_block")
        self._initialize_pool()

        # Allocate new block
        new_blocks = self.allocate_blocks(1)
        new_block_id = new_blocks[0]

        # Copy data
        self._kv_pool[new_block_id] = self._kv_pool[src_block_id]

        # Decrement source ref count
        src_block = self._blocks[src_block_id]
        src_block.ref_count -= 1
        if src_block.ref_count <= 0:
            src_block.is_allocated = False
            self._free_blocks.add(src_block_id)

        return new_block_id

    def write_kv(
        self,
        block_id: int,
        layer_idx: int,
        token_idx: int,
        k: mx.array,
        v: mx.array,
    ):
        """Write K/V values to a specific position in a block.

        Args:
            block_id: Physical block ID.
            layer_idx: Layer index.
            token_idx: Token position within the block.
            k: Key tensor (num_kv_heads, head_dim).
            v: Value tensor (num_kv_heads, head_dim).

        Raises:
            ValueError: If block_id is out of bounds.
        """
        self._validate_block_id(block_id, "write_kv")
        self._initialize_pool()

        self._kv_pool[block_id, layer_idx, 0, :, token_idx, :] = k
        self._kv_pool[block_id, layer_idx, 1, :, token_idx, :] = v

    def read_kv(
        self,
        block_ids: List[int],
        layer_idx: int,
        num_tokens: Optional[int] = None,
    ) -> tuple[mx.array, mx.array]:
        """Read K/V from multiple blocks.

        Args:
            block_ids: List of physical block IDs.
            layer_idx: Layer index.
            num_tokens: Total number of valid tokens (for partial last block).

        Returns:
            Tuple of (K, V) arrays with shape (seq_len, num_kv_heads, head_dim).

        Raises:
            ValueError: If any block_id is out of bounds.
        """
        for block_id in block_ids:
            self._validate_block_id(block_id, "read_kv")

        self._initialize_pool()

        num_blocks = len(block_ids)
        full_seq_len = num_blocks * self.config.block_size

        # Gather blocks
        block_indices = mx.array(block_ids, dtype=mx.int32)

        # Read all blocks: (num_blocks, num_kv_heads, block_size, head_dim)
        k_blocks = self._kv_pool[block_indices, layer_idx, 0]
        v_blocks = self._kv_pool[block_indices, layer_idx, 1]

        # Reshape to (seq_len, num_kv_heads, head_dim)
        # First transpose to (num_blocks, block_size, num_kv_heads, head_dim)
        k_blocks = k_blocks.transpose(0, 2, 1, 3)
        v_blocks = v_blocks.transpose(0, 2, 1, 3)

        # Reshape to (seq_len, num_kv_heads, head_dim)
        k = k_blocks.reshape(-1, self.config.num_kv_heads, self.config.head_dim)
        v = v_blocks.reshape(-1, self.config.num_kv_heads, self.config.head_dim)

        # Trim to actual token count if specified
        if num_tokens is not None and num_tokens < full_seq_len:
            k = k[:num_tokens]
            v = v[:num_tokens]

        return k, v

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        allocated = self.num_allocated_blocks
        free = self.num_free_blocks
        total = self.config.max_blocks

        return {
            "allocated_blocks": allocated,
            "free_blocks": free,
            "total_blocks": total,
            "utilization": allocated / total if total > 0 else 0,
            "allocated_bytes": allocated * self.config.block_memory_bytes,
            "total_bytes": self.config.total_memory_bytes,
        }

    def reset(self):
        """Reset the block manager, freeing all blocks."""
        for block in self._blocks:
            block.is_allocated = False
            block.ref_count = 0
        self._free_blocks = set(range(self.config.max_blocks))
